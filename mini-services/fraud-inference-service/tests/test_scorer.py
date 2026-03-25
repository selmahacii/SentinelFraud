"""
=============================================================================
SENTINEL-FRAUD: Unit Tests - Transaction Storm Attack Simulation
=============================================================================
Comprehensive tests simulating various attack scenarios including:
- Transaction Storm (high-volume attack)
- Impossible Travel Attack
- Velocity Attack
- Model Fallback Behavior
- Circuit Breaker Pattern
=============================================================================
"""

import asyncio
import time
import uuid
from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import numpy as np

# Import modules to test
import sys
sys.path.insert(0, "/app")

from app.engine.scorer import (
    CircuitBreaker,
    CircuitState,
    FraudScore,
    GeoLocation,
    Haversine_distance,
    calculate_travel_speed,
    is_impossible_travel,
    TransactionBase,
    TransactionFeatures,
    XGBoostScorer,
    RuleBasedScorer,
    HybridFraudScorer,
    RedisFeatureStore,
    ScoringError,
    ModelNotLoadedError,
    CircuitBreakerOpenError,
    FRAUD_THRESHOLD,
    LATENCY_TARGET_MS,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def sample_location():
    """Sample geographic location."""
    return GeoLocation(latitude=40.7128, longitude=-74.0060)


@pytest.fixture
def sample_transaction(sample_location):
    """Sample transaction for testing."""
    return TransactionBase(
        transaction_id=f"TXN_{uuid.uuid4().hex[:16]}",
        user_id="USER_00000001",
        amount=500.00,
        currency="USD",
        merchant_id="MERCH_000001",
        merchant_category="online",
        location=sample_location,
        device_id="DEV_0001",
        ip_address="192.168.1.1",
        channel="online",
    )


@pytest.fixture
def suspicious_transaction():
    """High-risk transaction for testing."""
    return TransactionBase(
        transaction_id=f"TXN_{uuid.uuid4().hex[:16]}",
        user_id="USER_00000001",
        amount=50000.00,  # Very high amount
        currency="USD",
        merchant_id="MERCH_000001",
        merchant_category="gaming",  # High-risk category
        location=GeoLocation(latitude=-33.8688, longitude=151.2093),  # Sydney
        device_id="DEV_NEW_0001",  # New device
        ip_address="10.0.0.1",
        channel="api",  # API channel
    )


@pytest.fixture
def mock_redis():
    """Mock Redis client."""
    redis_mock = AsyncMock()
    redis_mock.get = AsyncMock(return_value=None)
    redis_mock.setex = AsyncMock(return_value=True)
    redis_mock.ping = AsyncMock(return_value=True)
    redis_mock.zrange = AsyncMock(return_value=[])
    redis_mock.zadd = AsyncMock(return_value=1)
    redis_mock.zremrangebyscore = AsyncMock(return_value=1)
    redis_mock.zcount = AsyncMock(return_value=0)
    redis_mock.zrangebyscore = AsyncMock(return_value=[])
    redis_mock.expire = AsyncMock(return_value=True)
    return redis_mock


@pytest.fixture
def mock_feature_store(mock_redis):
    """Mock feature store."""
    return RedisFeatureStore(mock_redis)


@pytest.fixture
def circuit_breaker():
    """Circuit breaker instance."""
    return CircuitBreaker(
        failure_threshold=3,
        recovery_timeout=5.0
    )


# =============================================================================
# GEOLOCATION TESTS
# =============================================================================

class TestGeoLocation:
    """Tests for geographic calculations."""

    def test_haversine_distance_same_location(self, sample_location):
        """Distance to same location should be zero."""
        distance = haversine_distance(sample_location, sample_location)
        assert distance == pytest.approx(0.0, abs=0.001)

    def test_haversine_distance_new_york_to_london(self):
        """Test distance between New York and London."""
        ny = GeoLocation(latitude=40.7128, longitude=-74.0060)
        london = GeoLocation(latitude=51.5074, longitude=-0.1278)
        
        distance = haversine_distance(ny, london)
        
        # Actual distance is approximately 5570 km
        assert distance == pytest.approx(5570, rel=0.01)

    def test_haversine_distance_antipodes(self):
        """Test distance to antipodal point (opposite side of Earth)."""
        # Madrid and Weber, New Zealand are roughly antipodal
        madrid = GeoLocation(latitude=40.4168, longitude=-3.7038)
        
        distance = haversine_distance(madrid, 
            GeoLocation(latitude=-40.4168, longitude=176.2962))
        
        # Should be approximately half Earth's circumference
        assert distance > 19000  # km

    def test_travel_speed_calculation(self):
        """Test travel speed calculation."""
        distance_km = 500  # 500 km
        time_seconds = 3600  # 1 hour
        
        speed = calculate_travel_speed(distance_km, time_seconds)
        
        assert speed == pytest.approx(500.0, abs=0.1)

    def test_travel_speed_zero_time(self):
        """Travel speed with zero time should be zero."""
        speed = calculate_travel_speed(500, 0)
        assert speed == 0.0

    def test_impossible_travel_detection(self):
        """Test impossible travel detection."""
        # 1000 km in 1 hour = 1000 km/h (possible by plane)
        assert not is_impossible_travel(1000, 3600, max_speed_kmh=1200)
        
        # 2000 km in 1 hour = 2000 km/h (impossible)
        assert is_impossible_travel(2000, 3600, max_speed_kmh=1200)
        
        # 100 km in 1 minute = 6000 km/h (impossible)
        assert is_impossible_travel(100, 60, max_speed_kmh=1200)


# =============================================================================
# CIRCUIT BREAKER TESTS
# =============================================================================

class TestCircuitBreaker:
    """Tests for circuit breaker pattern."""

    def test_initial_state_closed(self, circuit_breaker):
        """Circuit breaker should start in CLOSED state."""
        assert circuit_breaker.state == CircuitState.CLOSED
        assert circuit_breaker.can_execute() is True

    def test_opens_after_threshold_failures(self, circuit_breaker):
        """Circuit breaker should open after failure threshold."""
        # Record failures up to threshold
        for _ in range(circuit_breaker.failure_threshold):
            circuit_breaker.record_failure()
        
        assert circuit_breaker.state == CircuitState.OPEN
        assert circuit_breaker.can_execute() is False

    def test_resets_on_success(self, circuit_breaker):
        """Circuit breaker should reset on success."""
        # Record some failures
        circuit_breaker.record_failure()
        circuit_breaker.record_failure()
        
        # Record success
        circuit_breaker.record_success()
        
        assert circuit_breaker.state == CircuitState.CLOSED
        assert circuit_breaker.failure_count == 0

    def test_half_open_after_recovery_timeout(self, circuit_breaker):
        """Circuit breaker should transition to HALF_OPEN after timeout."""
        circuit_breaker.recovery_timeout = 0.1  # 100ms
        
        # Open the circuit
        for _ in range(circuit_breaker.failure_threshold):
            circuit_breaker.record_failure()
        
        assert circuit_breaker.state == CircuitState.OPEN
        
        # Wait for recovery timeout
        time.sleep(0.15)
        
        # Should now allow execution (HALF_OPEN)
        assert circuit_breaker.can_execute() is True
        assert circuit_breaker.state == CircuitState.HALF_OPEN


# =============================================================================
# TRANSACTION VALIDATION TESTS
# =============================================================================

class TestTransactionValidation:
    """Tests for Pydantic V2 transaction validation."""

    def test_valid_transaction(self, sample_transaction):
        """Valid transaction should pass validation."""
        assert sample_transaction.transaction_id.startswith("TXN_")
        assert sample_transaction.amount > 0

    def test_invalid_amount_negative(self):
        """Negative amount should fail validation."""
        with pytest.raises(Exception):  # Pydantic ValidationError
            TransactionBase(
                transaction_id="TXN_001",
                user_id="USER_001",
                amount=-100.00,  # Invalid
                merchant_id="MERCH_001",
                merchant_category="retail",
                location=GeoLocation(latitude=40.0, longitude=-74.0),
            )

    def test_invalid_amount_too_high(self):
        """Amount over limit should fail validation."""
        with pytest.raises(Exception):
            TransactionBase(
                transaction_id="TXN_001",
                user_id="USER_001",
                amount=2_000_000.00,  # Over limit
                merchant_id="MERCH_001",
                merchant_category="retail",
                location=GeoLocation(latitude=40.0, longitude=-74.0),
            )

    def test_invalid_latitude(self):
        """Invalid latitude should fail validation."""
        with pytest.raises(Exception):
            GeoLocation(latitude=100.0, longitude=-74.0)  # Invalid latitude

    def test_invalid_longitude(self):
        """Invalid longitude should fail validation."""
        with pytest.raises(Exception):
            GeoLocation(latitude=40.0, longitude=200.0)  # Invalid longitude

    def test_invalid_channel(self):
        """Invalid channel should fail validation."""
        with pytest.raises(Exception):
            TransactionBase(
                transaction_id="TXN_001",
                user_id="USER_001",
                amount=100.00,
                merchant_id="MERCH_001",
                merchant_category="retail",
                location=GeoLocation(latitude=40.0, longitude=-74.0),
                channel="invalid_channel",  # Invalid
            )


# =============================================================================
# RULE-BASED SCORER TESTS
# =============================================================================

class TestRuleBasedScorer:
    """Tests for rule-based fallback scorer."""

    @pytest.mark.asyncio
    async def test_scorer_always_ready(self, mock_feature_store):
        """Rule-based scorer should always be ready."""
        scorer = RuleBasedScorer(mock_feature_store)
        assert await scorer.is_ready() is True

    @pytest.mark.asyncio
    async def test_scores_normal_transaction(self, mock_feature_store, sample_transaction):
        """Should score normal transaction with low risk."""
        scorer = RuleBasedScorer(mock_feature_store)
        
        score = await scorer.score(sample_transaction)
        
        assert isinstance(score, FraudScore)
        assert score.fraud_probability >= 0.0
        assert score.fraud_probability <= 1.0
        assert score.model_used == "rules"

    @pytest.mark.asyncio
    async def test_detects_impossible_travel(self, mock_feature_store):
        """Should detect impossible travel pattern."""
        # Mock last transaction in New York
        mock_feature_store.get_last_transaction = AsyncMock(return_value={
            "latitude": 40.7128,
            "longitude": -74.0060,
            "timestamp": time.time() - 1800,  # 30 minutes ago
            "amount": 100.0,
        })
        
        scorer = RuleBasedScorer(mock_feature_store)
        
        # Current transaction in Sydney (impossible travel)
        txn = TransactionBase(
            transaction_id="TXN_001",
            user_id="USER_001",
            amount=1000.00,
            merchant_id="MERCH_001",
            merchant_category="online",
            location=GeoLocation(latitude=-33.8688, longitude=151.2093),  # Sydney
        )
        
        score = await scorer.score(txn)
        
        assert "IMPOSSIBLE_TRAVEL" in score.risk_factors

    @pytest.mark.asyncio
    async def test_latency_target(self, mock_feature_store, sample_transaction):
        """Should meet latency target of <50ms."""
        scorer = RuleBasedScorer(mock_feature_store)
        
        start_time = time.perf_counter()
        await scorer.score(sample_transaction)
        latency_ms = (time.perf_counter() - start_time) * 1000
        
        assert latency_ms < LATENCY_TARGET_MS


# =============================================================================
# TRANSACTION STORM ATTACK TEST
# =============================================================================

class TestTransactionStormAttack:
    """
    Test simulating a Transaction Storm attack.
    
    A Transaction Storm is when an attacker attempts to:
    1. Flood the system with high-volume transactions
    2. Test multiple stolen cards simultaneously
    3. Overwhelm fraud detection systems
    """

    @pytest.mark.asyncio
    async def test_storm_detection_velocity(self, mock_feature_store):
        """Should detect high-velocity transaction storm."""
        # Simulate high transaction count
        mock_feature_store.get_transaction_count = AsyncMock(return_value=50)  # 50 txns in 1h
        
        scorer = RuleBasedScorer(mock_feature_store)
        
        # Storm transaction
        txn = TransactionBase(
            transaction_id="TXN_STORM_001",
            user_id="USER_STORM",
            amount=100.00,
            merchant_id="MERCH_001",
            merchant_category="online",
            location=GeoLocation(latitude=40.0, longitude=-74.0),
        )
        
        score = await scorer.score(txn)
        
        # High velocity should be flagged
        assert score.fraud_probability > 0.5 or "HIGH_VELOCITY" in score.risk_factors

    @pytest.mark.asyncio
    async def test_storm_batch_processing(self, mock_feature_store):
        """Should handle batch of storm transactions efficiently."""
        scorer = RuleBasedScorer(mock_feature_store)
        
        # Generate storm of transactions
        storm_transactions = [
            TransactionBase(
                transaction_id=f"TXN_STORM_{i:04d}",
                user_id=f"USER_STORM_{i % 10:04d}",  # 10 different users
                amount=100.00 + (i * 10),
                merchant_id="MERCH_STORM",
                merchant_category="online",
                location=GeoLocation(
                    latitude=40.0 + (i * 0.01),
                    longitude=-74.0 + (i * 0.01)
                ),
            )
            for i in range(100)  # 100 transactions
        ]
        
        start_time = time.perf_counter()
        
        # Process all transactions
        tasks = [scorer.score(txn) for txn in storm_transactions]
        results = await asyncio.gather(*tasks)
        
        total_latency_ms = (time.perf_counter() - start_time) * 1000
        avg_latency_ms = total_latency_ms / len(storm_transactions)
        
        # All should be processed
        assert len(results) == 100
        
        # Average latency should be reasonable
        assert avg_latency_ms < 10  # Should be very fast with rules

    @pytest.mark.asyncio
    async def test_storm_amount_anomaly_detection(self, mock_feature_store):
        """Should detect amount anomalies during storm."""
        # User's average is $100
        mock_feature_store.get_user_features = AsyncMock(return_value={
            "avg_amount": 100.0,
            "std_amount": 50.0,
            "txn_count_total": 100,
            "fraud_count": 0,
        })
        
        scorer = RuleBasedScorer(mock_feature_store)
        
        # Storm transaction with anomalous amount
        txn = TransactionBase(
            transaction_id="TXN_STORM_HIGH",
            user_id="USER_001",
            amount=10000.00,  # 100x normal
            merchant_id="MERCH_001",
            merchant_category="online",
            location=GeoLocation(latitude=40.0, longitude=-74.0),
        )
        
        score = await scorer.score(txn)
        
        # High amount anomaly should be detected
        assert "AMOUNT_ANOMALY" in score.risk_factors or score.fraud_probability > 0.5

    @pytest.mark.asyncio
    async def test_storm_time_pattern_detection(self, mock_feature_store):
        """Should detect unusual time patterns during storm."""
        scorer = RuleBasedScorer(mock_feature_store)
        
        # Create transaction at unusual hour (3 AM)
        from datetime import datetime, timezone, timedelta
        
        unusual_time = datetime.now(timezone.utc).replace(hour=3, minute=0)
        
        txn = TransactionBase(
            transaction_id="TXN_STORM_NIGHT",
            user_id="USER_001",
            amount=500.00,
            merchant_id="MERCH_001",
            merchant_category="online",
            location=GeoLocation(latitude=40.0, longitude=-74.0),
            timestamp=unusual_time,
        )
        
        score = await scorer.score(txn)
        
        # Late night transaction should be flagged
        assert "LATE_NIGHT_TRANSACTION" in score.risk_factors

    @pytest.mark.asyncio
    async def test_storm_rapid_succession_detection(self, mock_feature_store):
        """Should detect rapid succession transactions."""
        # Mock last transaction 30 seconds ago
        mock_feature_store.get_last_transaction = AsyncMock(return_value={
            "latitude": 40.0,
            "longitude": -74.0,
            "timestamp": time.time() - 30,  # 30 seconds ago
            "amount": 100.0,
        })
        
        scorer = RuleBasedScorer(mock_feature_store)
        
        txn = TransactionBase(
            transaction_id="TXN_RAPID_001",
            user_id="USER_001",
            amount=500.00,
            merchant_id="MERCH_001",
            merchant_category="online",
            location=GeoLocation(latitude=40.0, longitude=-74.0),
        )
        
        score = await scorer.score(txn)
        
        # Rapid succession should be detected
        assert "RAPID_SUCCESSION" in score.risk_factors or score.fraud_probability > 0.3


# =============================================================================
# CIRCUIT BREAKER INTEGRATION TEST
# =============================================================================

class TestCircuitBreakerIntegration:
    """Tests for circuit breaker integration with scorer."""

    @pytest.mark.asyncio
    async def test_fallback_on_model_failure(self, mock_feature_store):
        """Should fall back to rules when model fails."""
        # Create mock XGBoost scorer that fails
        ml_scorer = MagicMock(spec=XGBoostScorer)
        ml_scorer.is_ready = AsyncMock(return_value=True)
        ml_scorer.score = AsyncMock(side_effect=ScoringError("Model failed"))
        ml_scorer._feature_store = mock_feature_store
        ml_scorer._circuit_breaker = CircuitBreaker()
        
        # Create rule-based scorer
        rule_scorer = RuleBasedScorer(mock_feature_store)
        
        # Create hybrid scorer
        hybrid = HybridFraudScorer(ml_scorer, rule_scorer)
        
        # Score should use fallback
        txn = TransactionBase(
            transaction_id="TXN_001",
            user_id="USER_001",
            amount=100.00,
            merchant_id="MERCH_001",
            merchant_category="online",
            location=GeoLocation(latitude=40.0, longitude=-74.0),
        )
        
        score = await hybrid.score(txn)
        
        assert "FALLBACK_MODE" in score.risk_factors
        assert score.model_used == "rules_fallback"

    @pytest.mark.asyncio
    async def test_circuit_breaker_opens_on_failures(self, mock_feature_store):
        """Circuit breaker should open after repeated failures."""
        circuit = CircuitBreaker(failure_threshold=3, recovery_timeout=60)
        
        # Simulate failures
        for _ in range(3):
            circuit.record_failure()
        
        assert circuit.state == CircuitState.OPEN
        assert circuit.can_execute() is False


# =============================================================================
# PERFORMANCE TESTS
# =============================================================================

class TestPerformance:
    """Performance tests for fraud detection."""

    @pytest.mark.asyncio
    async def test_latency_target_met(self, mock_feature_store, sample_transaction):
        """Should meet 50ms latency target."""
        scorer = RuleBasedScorer(mock_feature_store)
        
        latencies = []
        for _ in range(100):
            start = time.perf_counter()
            await scorer.score(sample_transaction)
            latency = (time.perf_counter() - start) * 1000
            latencies.append(latency)
        
        avg_latency = sum(latencies) / len(latencies)
        p95_latency = sorted(latencies)[int(len(latencies) * 0.95)]
        
        assert avg_latency < LATENCY_TARGET_MS
        assert p95_latency < LATENCY_TARGET_MS * 2  # P95 can be higher

    @pytest.mark.asyncio
    async def test_concurrent_request_handling(self, mock_feature_store, sample_transaction):
        """Should handle concurrent requests efficiently."""
        scorer = RuleBasedScorer(mock_feature_store)
        
        # Simulate 100 concurrent requests
        tasks = [
            scorer.score(sample_transaction)
            for _ in range(100)
        ]
        
        start = time.perf_counter()
        results = await asyncio.gather(*tasks)
        total_time = (time.perf_counter() - start) * 1000
        
        assert len(results) == 100
        assert all(isinstance(r, FraudScore) for r in results)
        assert total_time < 1000  # All 100 should complete in < 1 second


# =============================================================================
# RUN TESTS
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
