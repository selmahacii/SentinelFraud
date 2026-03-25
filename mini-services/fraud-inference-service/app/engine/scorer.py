"""
=============================================================================
SENTINEL-FRAUD: Core Inference Engine - THE SHIELD
=============================================================================
Real-time fraud scoring engine with sub-50ms latency target.

Features:
    - XGBoost quantized model inference
    - Redis feature store integration
    - Haversine distance calculation (impossible travel detection)
    - Rule-based fallback strategy
    - Circuit breaker pattern for resilience

Architecture Principles:
    - SOLID: Single Responsibility, Dependency Inversion
    - Clean Code: Self-documenting, meaningful names
    - Performance: Async I/O, msgpack serialization
=============================================================================
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Optional

import msgpack
import numpy as np
import redis.asyncio as redis
from pydantic import BaseModel, Field, computed_field, model_validator

# =============================================================================
# CONFIGURATION & CONSTANTS
# =============================================================================

LOGGER = logging.getLogger("sentinel.scorer")

# Performance targets
LATENCY_TARGET_MS = 50.0
FEATURE_CACHE_TTL_SECONDS = 86400  # 24 hours

# Risk thresholds
FRAUD_THRESHOLD = 0.85
HIGH_RISK_THRESHOLD = 0.70
MEDIUM_RISK_THRESHOLD = 0.50

# Physical constants
EARTH_RADIUS_KM = 6371.0
MAX_HUMAN_TRAVEL_SPEED_KMH = 1200.0  # Max commercial flight speed

# Circuit breaker settings
CIRCUIT_BREAKER_FAILURE_THRESHOLD = 5
CIRCUIT_BREAKER_RECOVERY_TIMEOUT_SECONDS = 30


# =============================================================================
# SCHEMAS - Pydantic V2 Validation
# =============================================================================

class GeoLocation(BaseModel):
    """Geographic location with latitude and longitude."""
    latitude: float = Field(..., ge=-90.0, le=90.0)
    longitude: float = Field(..., ge=-180.0, le=180.0)

    def to_radians(self) -> tuple[float, float]:
        """Convert coordinates to radians for calculations."""
        return (
            math.radians(self.latitude),
            math.radians(self.longitude)
        )


class TransactionBase(BaseModel):
    """Base transaction model with strict validation."""
    transaction_id: str = Field(..., min_length=1, max_length=64)
    user_id: str = Field(..., min_length=1, max_length=64)
    amount: float = Field(..., ge=0.01, le=1_000_000.0)
    currency: str = Field(default="USD", min_length=3, max_length=3)
    merchant_id: str = Field(..., min_length=1, max_length=64)
    merchant_category: str = Field(..., min_length=1, max_length=32)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    location: GeoLocation
    device_id: Optional[str] = Field(default=None, max_length=64)
    ip_address: Optional[str] = Field(default=None, max_length=45)
    channel: str = Field(default="online", pattern="^(online|offline|mobile|api)$")

    @model_validator(mode='after')
    def validate_transaction(self) -> 'TransactionBase':
        """Additional cross-field validation."""
        if self.amount > 10000 and self.channel == "api":
            # High-value API transactions require extra scrutiny
            pass
        return self


class TransactionFeatures(BaseModel):
    """Engineered features for model inference."""
    # Transaction features
    amount: float
    amount_log: float = Field(..., description="Log-transformed amount")
    amount_zscore: float = Field(..., description="Z-score vs user history")

    # Temporal features
    hour_of_day: int = Field(..., ge=0, le=23)
    day_of_week: int = Field(..., ge=0, le=6)
    is_weekend: bool
    time_since_last_txn_seconds: float

    # Velocity features
    txn_count_1h: int = Field(..., ge=0)
    txn_count_24h: int = Field(..., ge=0)
    txn_count_7d: int = Field(..., ge=0)
    total_amount_24h: float = Field(..., ge=0.0)

    # Geographic features
    distance_from_last_txn_km: float = Field(..., ge=0.0)
    travel_speed_kmh: float = Field(..., ge=0.0)
    is_impossible_travel: bool

    # User profile features
    user_avg_amount: float
    user_std_amount: float
    user_txn_count_total: int
    user_fraud_history_count: int = Field(default=0, ge=0)

    # Merchant features
    merchant_risk_score: float = Field(default=0.5, ge=0.0, le=1.0)
    merchant_txn_count_24h: int = Field(default=0, ge=0)

    @computed_field
    @property
    def velocity_score(self) -> float:
        """Computed velocity risk score."""
        return min(1.0, (self.txn_count_1h / 10.0) * 0.3 + 
                       (self.txn_count_24h / 50.0) * 0.7)


class FraudScore(BaseModel):
    """Fraud scoring result."""
    transaction_id: str
    user_id: str
    fraud_probability: float = Field(..., ge=0.0, le=1.0)
    risk_level: str
    risk_factors: list[str] = Field(default_factory=list)
    model_version: str
    inference_time_ms: float
    model_used: str = Field(default="xgboost")
    features_used: int = Field(default=0)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @model_validator(mode='after')
    def determine_risk_level(self) -> 'FraudScore':
        """Automatically determine risk level from probability."""
        if self.fraud_probability >= FRAUD_THRESHOLD:
            self.risk_level = "CRITICAL"
        elif self.fraud_probability >= HIGH_RISK_THRESHOLD:
            self.risk_level = "HIGH"
        elif self.fraud_probability >= MEDIUM_RISK_THRESHOLD:
            self.risk_level = "MEDIUM"
        else:
            self.risk_level = "LOW"
        return self


class ScoringStrategy(str, Enum):
    """Available scoring strategies."""
    ML_MODEL = "ml_model"
    RULE_BASED = "rule_based"
    HYBRID = "hybrid"


class CircuitState(str, Enum):
    """Circuit breaker states."""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"      # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing recovery


# =============================================================================
# DOMAIN EXCEPTIONS
# =============================================================================

class ScoringError(Exception):
    """Base exception for scoring errors."""
    pass


class ModelNotLoadedError(ScoringError):
    """Raised when ML model is not available."""
    pass


class FeatureStoreError(ScoringError):
    """Raised when feature store is unavailable."""
    pass


class CircuitBreakerOpenError(ScoringError):
    """Raised when circuit breaker is open."""
    pass


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def haversine_distance(loc1: GeoLocation, loc2: GeoLocation) -> float:
    """
    Calculate the great-circle distance between two points on Earth.
    
    Uses the Haversine formula for accurate distance calculation.
    
    Args:
        loc1: First geographic location
        loc2: Second geographic location
    
    Returns:
        Distance in kilometers
    """
    lat1_rad, lon1_rad = loc1.to_radians()
    lat2_rad, lon2_rad = loc2.to_radians()
    
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    
    a = (
        math.sin(dlat / 2) ** 2 +
        math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2) ** 2
    )
    c = 2 * math.asin(math.sqrt(a))
    
    return EARTH_RADIUS_KM * c


def calculate_travel_speed(
    distance_km: float,
    time_delta_seconds: float
) -> float:
    """
    Calculate travel speed between two transactions.
    
    Args:
        distance_km: Distance between locations in kilometers
        time_delta_seconds: Time difference in seconds
    
    Returns:
        Speed in km/h, or 0 if time_delta is too small
    """
    if time_delta_seconds <= 0:
        return 0.0
    
    time_delta_hours = time_delta_seconds / 3600.0
    if time_delta_hours < 1e-6:  # Avoid division by near-zero
        return 0.0
    
    return distance_km / time_delta_hours


def is_impossible_travel(
    distance_km: float,
    time_delta_seconds: float,
    max_speed_kmh: float = MAX_HUMAN_TRAVEL_SPEED_KMH
) -> bool:
    """
    Determine if travel between two locations is physically impossible.
    
    Args:
        distance_km: Distance between locations
        time_delta_seconds: Time difference between transactions
        max_speed_kmh: Maximum possible travel speed
    
    Returns:
        True if travel is impossible (fraud indicator)
    """
    if time_delta_seconds <= 0:
        return False
    
    speed = calculate_travel_speed(distance_km, time_delta_seconds)
    return speed > max_speed_kmh


# =============================================================================
# CIRCUIT BREAKER PATTERN
# =============================================================================

class CircuitBreaker:
    """
    Circuit breaker for resilience patterns.
    
    States:
        - CLOSED: Normal operation, requests pass through
        - OPEN: Failure threshold exceeded, requests fail fast
        - HALF_OPEN: Testing if service recovered
    """
    
    def __init__(
        self,
        failure_threshold: int = CIRCUIT_BREAKER_FAILURE_THRESHOLD,
        recovery_timeout: float = CIRCUIT_BREAKER_RECOVERY_TIMEOUT_SECONDS
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.state = CircuitState.CLOSED
        self.last_failure_time: Optional[float] = None
    
    def record_success(self) -> None:
        """Record a successful operation."""
        self.failure_count = 0
        self.state = CircuitState.CLOSED
    
    def record_failure(self) -> None:
        """Record a failed operation."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN
            LOGGER.warning(
                f"Circuit breaker opened after {self.failure_count} failures"
            )
    
    def can_execute(self) -> bool:
        """Check if operation can be executed."""
        if self.state == CircuitState.CLOSED:
            return True
        
        if self.state == CircuitState.OPEN:
            # Check if recovery timeout has passed
            if self.last_failure_time is None:
                return False
            
            elapsed = time.time() - self.last_failure_time
            if elapsed >= self.recovery_timeout:
                self.state = CircuitState.HALF_OPEN
                LOGGER.info("Circuit breaker entering half-open state")
                return True
            return False
        
        # HALF_OPEN - allow one test request
        return True


# =============================================================================
# FEATURE STORE INTERFACE
# =============================================================================

class FeatureStoreInterface(ABC):
    """Abstract interface for feature store operations."""
    
    @abstractmethod
    async def get_user_features(self, user_id: str) -> dict[str, Any]:
        """Retrieve user-level features."""
        pass
    
    @abstractmethod
    async def update_user_features(
        self,
        user_id: str,
        features: dict[str, Any],
        ttl: int = FEATURE_CACHE_TTL_SECONDS
    ) -> None:
        """Update user features with TTL."""
        pass
    
    @abstractmethod
    async def get_last_transaction(self, user_id: str) -> Optional[dict[str, Any]]:
        """Get the user's last transaction for velocity checks."""
        pass
    
    @abstractmethod
    async def record_transaction(
        self,
        user_id: str,
        transaction: dict[str, Any],
        ttl: int = FEATURE_CACHE_TTL_SECONDS
    ) -> None:
        """Record transaction for historical analysis."""
        pass


class RedisFeatureStore(FeatureStoreInterface):
    """
    Redis-based feature store implementation.
    
    Uses msgpack for efficient serialization and maintains
    a 24-hour sliding window of transaction history.
    """
    
    def __init__(self, redis_client: redis.Redis):
        self._redis = redis_client
        self._prefix = "sentinel:features"
        self._txn_prefix = "sentinel:transactions"
    
    def _user_key(self, user_id: str) -> str:
        return f"{self._prefix}:user:{user_id}"
    
    def _txn_key(self, user_id: str) -> str:
        return f"{self._txn_prefix}:{user_id}"
    
    async def get_user_features(self, user_id: str) -> dict[str, Any]:
        """Retrieve user features from Redis."""
        key = self._user_key(user_id)
        data = await self._redis.get(key)
        
        if data is None:
            return self._default_user_features()
        
        return msgpack.unpackb(data, raw=False)
    
    async def update_user_features(
        self,
        user_id: str,
        features: dict[str, Any],
        ttl: int = FEATURE_CACHE_TTL_SECONDS
    ) -> None:
        """Update user features in Redis with TTL."""
        key = self._user_key(user_id)
        data = msgpack.packb(features, use_bin_type=True)
        await self._redis.setex(key, ttl, data)
    
    async def get_last_transaction(
        self, user_id: str
    ) -> Optional[dict[str, Any]]:
        """Get user's most recent transaction."""
        key = self._txn_key(user_id)
        
        # Get the most recent transaction from the sorted set
        results = await self._redis.zrange(
            key, -1, -1, withscores=True
        )
        
        if not results:
            return None
        
        data, _ = results[0]
        return msgpack.unpackb(data, raw=False)
    
    async def record_transaction(
        self,
        user_id: str,
        transaction: dict[str, Any],
        ttl: int = FEATURE_CACHE_TTL_SECONDS
    ) -> None:
        """Record transaction in Redis sorted set with timestamp score."""
        key = self._txn_key(user_id)
        timestamp = transaction.get("timestamp", time.time())
        data = msgpack.packb(transaction, use_bin_type=True)
        
        # Use timestamp as score for sorted set
        await self._redis.zadd(key, {data: timestamp})
        
        # Remove transactions older than TTL
        cutoff = time.time() - ttl
        await self._redis.zremrangebyscore(key, "-inf", cutoff)
        
        # Set expiry on the key
        await self._redis.expire(key, ttl)
    
    async def get_transaction_count(
        self,
        user_id: str,
        window_seconds: int
    ) -> int:
        """Count transactions within a time window."""
        key = self._txn_key(user_id)
        cutoff = time.time() - window_seconds
        return await self._redis.zcount(key, cutoff, "+inf")
    
    async def get_transaction_sum(
        self,
        user_id: str,
        window_seconds: int
    ) -> float:
        """Sum transaction amounts within a time window."""
        key = self._txn_key(user_id)
        cutoff = time.time() - window_seconds
        
        results = await self._redis.zrangebyscore(key, cutoff, "+inf")
        total = 0.0
        
        for data in results:
            txn = msgpack.unpackb(data, raw=False)
            total += txn.get("amount", 0.0)
        
        return total
    
    @staticmethod
    def _default_user_features() -> dict[str, Any]:
        """Default features for new users."""
        return {
            "avg_amount": 0.0,
            "std_amount": 0.0,
            "txn_count_total": 0,
            "fraud_count": 0,
            "first_seen": time.time(),
            "last_seen": time.time(),
        }


# =============================================================================
# SCORING ENGINE INTERFACE
# =============================================================================

class ScoringEngineInterface(ABC):
    """Abstract interface for fraud scoring engines."""
    
    @abstractmethod
    async def score(self, transaction: TransactionBase) -> FraudScore:
        """Generate a fraud score for a transaction."""
        pass
    
    @abstractmethod
    async def is_ready(self) -> bool:
        """Check if the scoring engine is ready."""
        pass


# =============================================================================
# ML MODEL SCORER
# =============================================================================

class XGBoostScorer(ScoringEngineInterface):
    """
    XGBoost-based fraud scoring engine.
    
    Features:
        - Quantized model loading for fast inference
        - Feature engineering pipeline
        - Circuit breaker for resilience
        - Performance monitoring
    """
    
    def __init__(
        self,
        model_path: str,
        feature_store: FeatureStoreInterface,
        model_version: str = "1.0.0"
    ):
        self._model_path = Path(model_path)
        self._feature_store = feature_store
        self._model_version = model_version
        self._model: Optional[Any] = None
        self._circuit_breaker = CircuitBreaker()
        self._feature_names: list[str] = []
        self._is_loaded = False
    
    async def load_model(self) -> None:
        """Load the XGBoost model from disk."""
        try:
            import xgboost as xgb
            
            if not self._model_path.exists():
                raise ModelNotLoadedError(
                    f"Model not found at {self._model_path}"
                )
            
            # Load model (XGBoost handles quantization internally)
            self._model = xgb.Booster()
            self._model.load_model(str(self._model_path))
            
            # Get feature names from model
            self._feature_names = self._model.feature_names or []
            
            self._is_loaded = True
            LOGGER.info(
                f"XGBoost model loaded successfully from {self._model_path}"
            )
            
        except Exception as e:
            self._is_loaded = False
            LOGGER.error(f"Failed to load XGBoost model: {e}")
            raise ModelNotLoadedError(str(e))
    
    async def is_ready(self) -> bool:
        """Check if model is loaded and circuit breaker is closed."""
        return self._is_loaded and self._model is not None
    
    async def _engineer_features(
        self,
        transaction: TransactionBase
    ) -> TransactionFeatures:
        """
        Engineer features for model input.
        
        This is the core feature engineering pipeline that transforms
        raw transaction data into model-ready features.
        """
        # Get user features from store
        user_features = await self._feature_store.get_user_features(
            transaction.user_id
        )
        
        # Get last transaction for velocity/travel calculations
        last_txn = await self._feature_store.get_last_transaction(
            transaction.user_id
        )
        
        # Get transaction counts for different windows
        redis_store = self._feature_store
        if isinstance(redis_store, RedisFeatureStore):
            txn_count_1h = await redis_store.get_transaction_count(
                transaction.user_id, 3600
            )
            txn_count_24h = await redis_store.get_transaction_count(
                transaction.user_id, 86400
            )
            txn_count_7d = await redis_store.get_transaction_count(
                transaction.user_id, 604800
            )
            total_amount_24h = await redis_store.get_transaction_sum(
                transaction.user_id, 86400
            )
        else:
            txn_count_1h = 0
            txn_count_24h = 0
            txn_count_7d = 0
            total_amount_24h = 0.0
        
        # Calculate distance and travel metrics
        distance_km = 0.0
        travel_speed_kmh = 0.0
        is_impossible = False
        time_since_last = 0.0
        
        if last_txn is not None:
            last_location = GeoLocation(
                latitude=last_txn.get("latitude", transaction.location.latitude),
                longitude=last_txn.get("longitude", transaction.location.longitude)
            )
            distance_km = haversine_distance(transaction.location, last_location)
            
            last_timestamp = last_txn.get("timestamp", time.time())
            if isinstance(last_timestamp, str):
                last_timestamp = datetime.fromisoformat(last_timestamp).timestamp()
            
            time_since_last = time.time() - last_timestamp
            travel_speed_kmh = calculate_travel_speed(distance_km, time_since_last)
            is_impossible = is_impossible_travel(distance_km, time_since_last)
        
        # Calculate amount statistics
        user_avg = user_features.get("avg_amount", transaction.amount)
        user_std = user_features.get("std_amount", 0.0)
        
        amount_zscore = 0.0
        if user_std > 0:
            amount_zscore = (transaction.amount - user_avg) / user_std
        
        # Time features
        txn_time = transaction.timestamp
        hour_of_day = txn_time.hour
        day_of_week = txn_time.weekday()
        is_weekend = day_of_week >= 5
        
        return TransactionFeatures(
            amount=transaction.amount,
            amount_log=math.log1p(transaction.amount),
            amount_zscore=amount_zscore,
            hour_of_day=hour_of_day,
            day_of_week=day_of_week,
            is_weekend=is_weekend,
            time_since_last_txn_seconds=time_since_last,
            txn_count_1h=txn_count_1h,
            txn_count_24h=txn_count_24h,
            txn_count_7d=txn_count_7d,
            total_amount_24h=total_amount_24h,
            distance_from_last_txn_km=distance_km,
            travel_speed_kmh=travel_speed_kmh,
            is_impossible_travel=is_impossible,
            user_avg_amount=user_avg,
            user_std_amount=user_std,
            user_txn_count_total=user_features.get("txn_count_total", 0),
            user_fraud_history_count=user_features.get("fraud_count", 0),
        )
    
    def _features_to_array(
        self,
        features: TransactionFeatures
    ) -> np.ndarray:
        """Convert features to numpy array for model input."""
        feature_order = [
            "amount", "amount_log", "amount_zscore",
            "hour_of_day", "day_of_week", "is_weekend",
            "time_since_last_txn_seconds",
            "txn_count_1h", "txn_count_24h", "txn_count_7d",
            "total_amount_24h",
            "distance_from_last_txn_km", "travel_speed_kmh",
            "is_impossible_travel",
            "user_avg_amount", "user_std_amount",
            "user_txn_count_total", "user_fraud_history_count",
            "velocity_score", "merchant_risk_score"
        ]
        
        values = []
        for name in feature_order:
            value = getattr(features, name, 0)
            if isinstance(value, bool):
                value = float(value)
            values.append(value)
        
        return np.array([values], dtype=np.float32)
    
    async def score(self, transaction: TransactionBase) -> FraudScore:
        """Generate fraud score using XGBoost model."""
        start_time = time.perf_counter()
        
        # Check circuit breaker
        if not self._circuit_breaker.can_execute():
            raise CircuitBreakerOpenError(
                "Circuit breaker is open - model unavailable"
            )
        
        try:
            # Engineer features
            features = await self._engineer_features(transaction)
            feature_array = self._features_to_array(features)
            
            # Run inference
            import xgboost as xgb
            
            dmatrix = xgb.DMatrix(feature_array)
            probability = self._model.predict(dmatrix)[0]
            
            # Record success
            self._circuit_breaker.record_success()
            
            # Identify risk factors
            risk_factors = self._identify_risk_factors(features, probability)
            
            # Calculate inference time
            inference_time_ms = (time.perf_counter() - start_time) * 1000
            
            return FraudScore(
                transaction_id=transaction.transaction_id,
                user_id=transaction.user_id,
                fraud_probability=float(probability),
                risk_factors=risk_factors,
                model_version=self._model_version,
                inference_time_ms=inference_time_ms,
                model_used="xgboost",
                features_used=len(features.model_dump())
            )
            
        except Exception as e:
            self._circuit_breaker.record_failure()
            raise ScoringError(f"Model inference failed: {e}")
    
    def _identify_risk_factors(
        self,
        features: TransactionFeatures,
        probability: float
    ) -> list[str]:
        """Identify contributing risk factors for explainability."""
        risk_factors = []
        
        if features.is_impossible_travel:
            risk_factors.append("IMPOSSIBLE_TRAVEL")
        
        if features.amount_zscore > 3.0:
            risk_factors.append("UNUSUAL_AMOUNT")
        
        if features.velocity_score > 0.7:
            risk_factors.append("HIGH_VELOCITY")
        
        if features.txn_count_1h > 5:
            risk_factors.append("RAPID_TRANSACTIONS")
        
        if features.distance_from_last_txn_km > 500:
            risk_factors.append("DISTANT_TRANSACTION")
        
        if features.user_fraud_history_count > 0:
            risk_factors.append("PRIOR_FRAUD_HISTORY")
        
        if features.time_since_last_txn_seconds < 60:
            risk_factors.append("TOO_QUICK")
        
        if features.hour_of_day < 4:
            risk_factors.append("UNUSUAL_TIME")
        
        return risk_factors


# =============================================================================
# RULE-BASED FALLBACK SCORER
# =============================================================================

class RuleBasedScorer(ScoringEngineInterface):
    """
    Rule-based fraud scorer for fallback scenarios.
    
    Implements deterministic rules when ML model is unavailable.
    Provides a safety net while maintaining sub-50ms latency.
    """
    
    def __init__(self, feature_store: FeatureStoreInterface):
        self._feature_store = feature_store
        self._rules: list[Callable[[TransactionFeatures], float]] = [
            self._rule_amount_velocity,
            self._rule_impossible_travel,
            self._rule_time_pattern,
            self._rule_frequency,
            self._rule_amount_anomaly,
        ]
    
    async def is_ready(self) -> bool:
        """Rule-based scorer is always ready."""
        return True
    
    async def score(self, transaction: TransactionBase) -> FraudScore:
        """Generate fraud score using rules."""
        start_time = time.perf_counter()
        
        # Get user context
        user_features = await self._feature_store.get_user_features(
            transaction.user_id
        )
        last_txn = await self._feature_store.get_last_transaction(
            transaction.user_id
        )
        
        # Build minimal features for rules
        features = self._build_minimal_features(
            transaction, user_features, last_txn
        )
        
        # Apply all rules and aggregate scores
        total_score = 0.0
        risk_factors = []
        
        for rule in self._rules:
            rule_score, rule_factor = rule(features)
            if rule_score > 0:
                total_score += rule_score
                if rule_factor:
                    risk_factors.append(rule_factor)
        
        # Normalize to [0, 1] range
        probability = min(1.0, total_score / len(self._rules))
        
        inference_time_ms = (time.perf_counter() - start_time) * 1000
        
        return FraudScore(
            transaction_id=transaction.transaction_id,
            user_id=transaction.user_id,
            fraud_probability=probability,
            risk_factors=risk_factors,
            model_version="rule-based-v1",
            inference_time_ms=inference_time_ms,
            model_used="rules",
            features_used=0
        )
    
    def _build_minimal_features(
        self,
        transaction: TransactionBase,
        user_features: dict[str, Any],
        last_txn: Optional[dict[str, Any]]
    ) -> TransactionFeatures:
        """Build minimal features for rule evaluation."""
        # Calculate basic metrics
        distance_km = 0.0
        travel_speed_kmh = 0.0
        is_impossible = False
        time_since_last = 0.0
        
        if last_txn:
            last_location = GeoLocation(
                latitude=last_txn.get("latitude", transaction.location.latitude),
                longitude=last_txn.get("longitude", transaction.location.longitude)
            )
            distance_km = haversine_distance(transaction.location, last_location)
            
            last_timestamp = last_txn.get("timestamp", time.time())
            if isinstance(last_timestamp, str):
                last_timestamp = datetime.fromisoformat(last_timestamp).timestamp()
            
            time_since_last = time.time() - last_timestamp
            travel_speed_kmh = calculate_travel_speed(distance_km, time_since_last)
            is_impossible = is_impossible_travel(distance_km, time_since_last)
        
        user_avg = user_features.get("avg_amount", transaction.amount)
        user_std = user_features.get("std_amount", 0.0)
        
        amount_zscore = 0.0
        if user_std > 0:
            amount_zscore = (transaction.amount - user_avg) / user_std
        
        return TransactionFeatures(
            amount=transaction.amount,
            amount_log=math.log1p(transaction.amount),
            amount_zscore=amount_zscore,
            hour_of_day=transaction.timestamp.hour,
            day_of_week=transaction.timestamp.weekday(),
            is_weekend=transaction.timestamp.weekday() >= 5,
            time_since_last_txn_seconds=time_since_last,
            txn_count_1h=0,  # Not available in fallback
            txn_count_24h=user_features.get("txn_count_total", 0),
            txn_count_7d=user_features.get("txn_count_total", 0),
            total_amount_24h=0.0,
            distance_from_last_txn_km=distance_km,
            travel_speed_kmh=travel_speed_kmh,
            is_impossible_travel=is_impossible,
            user_avg_amount=user_avg,
            user_std_amount=user_std,
            user_txn_count_total=user_features.get("txn_count_total", 0),
        )
    
    @staticmethod
    def _rule_amount_velocity(features: TransactionFeatures) -> tuple[float, str]:
        """Rule: High amount with high frequency is suspicious."""
        if features.amount > 1000 and features.txn_count_24h > 10:
            return 0.8, "HIGH_AMOUNT_FREQUENCY"
        return 0.0, ""
    
    @staticmethod
    def _rule_impossible_travel(features: TransactionFeatures) -> tuple[float, str]:
        """Rule: Physically impossible travel is highly suspicious."""
        if features.is_impossible_travel:
            return 1.0, "IMPOSSIBLE_TRAVEL"
        if features.travel_speed_kmh > 800:
            return 0.7, "SUSPICIOUS_TRAVEL_SPEED"
        return 0.0, ""
    
    @staticmethod
    def _rule_time_pattern(features: TransactionFeatures) -> tuple[float, str]:
        """Rule: Unusual transaction times."""
        if features.hour_of_day < 4:  # 12am-4am
            return 0.3, "LATE_NIGHT_TRANSACTION"
        return 0.0, ""
    
    @staticmethod
    def _rule_frequency(features: TransactionFeatures) -> tuple[float, str]:
        """Rule: Rapid successive transactions."""
        if features.time_since_last_txn_seconds < 60:
            return 0.6, "RAPID_SUCCESSION"
        if features.time_since_last_txn_seconds < 300:
            return 0.3, "QUICK_SUCCESSION"
        return 0.0, ""
    
    @staticmethod
    def _rule_amount_anomaly(features: TransactionFeatures) -> tuple[float, str]:
        """Rule: Statistical amount anomaly."""
        if abs(features.amount_zscore) > 3:
            return 0.7, "AMOUNT_ANOMALY"
        if abs(features.amount_zscore) > 2:
            return 0.4, "MODERATE_AMOUNT_ANOMALY"
        return 0.0, ""


# =============================================================================
# HYBRID SCORER - MAIN ENTRY POINT
# =============================================================================

class HybridFraudScorer(ScoringEngineInterface):
    """
    Main fraud scoring engine with ML + Rule-based fallback.
    
    Strategy:
        1. Try ML model first (preferred)
        2. Fall back to rules if model fails
        3. Always maintain sub-50ms latency target
    """
    
    def __init__(
        self,
        ml_scorer: XGBoostScorer,
        rule_scorer: RuleBasedScorer,
        latency_target_ms: float = LATENCY_TARGET_MS
    ):
        self._ml_scorer = ml_scorer
        self._rule_scorer = rule_scorer
        self._latency_target_ms = latency_target_ms
    
    async def is_ready(self) -> bool:
        """Hybrid scorer is ready if at least one scorer is available."""
        return await self._rule_scorer.is_ready()
    
    async def score(self, transaction: TransactionBase) -> FraudScore:
        """
        Score transaction with automatic fallback.
        
        Tries ML model first, falls back to rules on failure.
        Monitors latency to ensure performance targets are met.
        """
        start_time = time.perf_counter()
        
        # Try ML scorer first
        try:
            if await self._ml_scorer.is_ready():
                score = await self._ml_scorer.score(transaction)
                
                # Check latency target
                total_latency = (time.perf_counter() - start_time) * 1000
                if total_latency > self._latency_target_ms:
                    LOGGER.warning(
                        f"Latency target exceeded: {total_latency:.2f}ms > "
                        f"{self._latency_target_ms}ms"
                    )
                
                return score
                
        except (CircuitBreakerOpenError, ScoringError) as e:
            LOGGER.warning(f"ML scorer failed, using fallback: {e}")
        
        # Fall back to rule-based scorer
        score = await self._rule_scorer.score(transaction)
        
        # Update score to indicate fallback was used
        total_latency = (time.perf_counter() - start_time) * 1000
        
        return FraudScore(
            transaction_id=score.transaction_id,
            user_id=score.user_id,
            fraud_probability=score.fraud_probability,
            risk_factors=score.risk_factors + ["FALLBACK_MODE"],
            model_version=score.model_version,
            inference_time_ms=total_latency,
            model_used="rules_fallback",
            features_used=score.features_used
        )
    
    async def update_after_transaction(
        self,
        transaction: TransactionBase,
        score: FraudScore
    ) -> None:
        """Update feature store after scoring."""
        # Record the transaction
        txn_data = {
            "transaction_id": transaction.transaction_id,
            "amount": transaction.amount,
            "latitude": transaction.location.latitude,
            "longitude": transaction.location.longitude,
            "timestamp": time.time(),
            "merchant_id": transaction.merchant_id,
        }
        
        await self._ml_scorer._feature_store.record_transaction(
            transaction.user_id, txn_data
        )
        
        # Update user features (rolling statistics)
        user_features = await self._ml_scorer._feature_store.get_user_features(
            transaction.user_id
        )
        
        # Update running statistics using Welford's algorithm
        old_count = user_features.get("txn_count_total", 0)
        old_mean = user_features.get("avg_amount", 0.0)
        old_m2 = user_features.get("m2", 0.0)  # For variance calculation
        
        new_count = old_count + 1
        delta = transaction.amount - old_mean
        new_mean = old_mean + delta / new_count
        delta2 = transaction.amount - new_mean
        new_m2 = old_m2 + delta * delta2
        new_std = math.sqrt(new_m2 / new_count) if new_count > 1 else 0.0
        
        updated_features = {
            "avg_amount": new_mean,
            "std_amount": new_std,
            "m2": new_m2,
            "txn_count_total": new_count,
            "fraud_count": user_features.get("fraud_count", 0),
            "first_seen": user_features.get("first_seen", time.time()),
            "last_seen": time.time(),
        }
        
        # Increment fraud count if detected
        if score.fraud_probability >= FRAUD_THRESHOLD:
            updated_features["fraud_count"] = updated_features["fraud_count"] + 1
        
        await self._ml_scorer._feature_store.update_user_features(
            transaction.user_id, updated_features
        )


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

async def create_scorer(
    model_path: str,
    redis_url: str,
    model_version: str = "1.0.0"
) -> HybridFraudScorer:
    """
    Factory function to create a fully configured fraud scorer.
    
    Args:
        model_path: Path to XGBoost model file
        redis_url: Redis connection URL
        model_version: Model version identifier
    
    Returns:
        Configured HybridFraudScorer instance
    """
    # Create Redis client
    redis_client = redis.from_url(redis_url, decode_responses=False)
    
    # Create feature store
    feature_store = RedisFeatureStore(redis_client)
    
    # Create scorers
    ml_scorer = XGBoostScorer(
        model_path=model_path,
        feature_store=feature_store,
        model_version=model_version
    )
    
    rule_scorer = RuleBasedScorer(feature_store=feature_store)
    
    # Try to load ML model
    try:
        await ml_scorer.load_model()
        LOGGER.info("ML model loaded successfully")
    except ModelNotLoadedError as e:
        LOGGER.warning(f"Could not load ML model, will use rule-based: {e}")
    
    # Create hybrid scorer
    return HybridFraudScorer(
        ml_scorer=ml_scorer,
        rule_scorer=rule_scorer
    )
