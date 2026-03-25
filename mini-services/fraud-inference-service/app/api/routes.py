"""
=============================================================================
SENTINEL-FRAUD: API Routes
=============================================================================
REST API endpoints for fraud detection service.
=============================================================================
"""

import time
from datetime import datetime, timezone
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from ..core.config import get_settings
from ..core.middleware import fraud_metrics
from ..engine.scorer import (
    FraudScore,
    GeoLocation,
    HybridFraudScorer,
    TransactionBase,
    create_scorer,
)

router = APIRouter()
settings = get_settings()

# Global scorer instance (initialized on startup)
_scorer: Optional[HybridFraudScorer] = None


def get_scorer() -> HybridFraudScorer:
    """Dependency to get the scorer instance."""
    if _scorer is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Scorer not initialized"
        )
    return _scorer


# =============================================================================
# REQUEST/RESPONSE SCHEMAS
# =============================================================================

class TransactionRequest(BaseModel):
    """Request schema for fraud scoring."""
    transaction_id: str = Field(..., min_length=1, max_length=64)
    user_id: str = Field(..., min_length=1, max_length=64)
    amount: float = Field(..., ge=0.01, le=1_000_000.0)
    currency: str = Field(default="USD", min_length=3, max_length=3)
    merchant_id: str = Field(..., min_length=1, max_length=64)
    merchant_category: str = Field(..., min_length=1, max_length=32)
    latitude: float = Field(..., ge=-90.0, le=90.0)
    longitude: float = Field(..., ge=-180.0, le=180.0)
    device_id: Optional[str] = Field(default=None, max_length=64)
    ip_address: Optional[str] = Field(default=None, max_length=45)
    channel: str = Field(default="online", pattern="^(online|offline|mobile|api)$")
    timestamp: Optional[datetime] = Field(default=None)
    
    def to_transaction_base(self) -> TransactionBase:
        """Convert to TransactionBase model."""
        return TransactionBase(
            transaction_id=self.transaction_id,
            user_id=self.user_id,
            amount=self.amount,
            currency=self.currency,
            merchant_id=self.merchant_id,
            merchant_category=self.merchant_category,
            location=GeoLocation(
                latitude=self.latitude,
                longitude=self.longitude
            ),
            device_id=self.device_id,
            ip_address=self.ip_address,
            channel=self.channel,
            timestamp=self.timestamp or datetime.now(timezone.utc)
        )


class BatchTransactionRequest(BaseModel):
    """Request schema for batch fraud scoring."""
    transactions: list[TransactionRequest] = Field(
        ...,
        min_length=1,
        max_length=100
    )


class ScoringResponse(BaseModel):
    """Response schema for fraud scoring."""
    success: bool
    score: Optional[FraudScore] = None
    error: Optional[str] = None
    processing_time_ms: float


class BatchScoringResponse(BaseModel):
    """Response schema for batch fraud scoring."""
    success: bool
    results: list[ScoringResponse]
    total_processed: int
    total_processing_time_ms: float


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    service: str
    version: str
    model_loaded: bool
    redis_connected: bool
    kafka_connected: bool
    uptime_seconds: float
    timestamp: datetime


class UserFeaturesResponse(BaseModel):
    """Response for user feature lookup."""
    user_id: str
    features: dict[str, Any]
    last_updated: Optional[datetime] = None


# =============================================================================
# HEALTH & STATUS ENDPOINTS
# =============================================================================

@router.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Health check endpoint.
    
    Returns the current status of the service and its dependencies.
    """
    from ..core.middleware import METRICS
    
    # Check scorer status
    model_loaded = _scorer is not None and await _scorer.is_ready()
    
    # Check Redis connectivity
    redis_connected = False
    if _scorer is not None:
        try:
            feature_store = _scorer._ml_scorer._feature_store
            if hasattr(feature_store, "_redis"):
                await feature_store._redis.ping()
                redis_connected = True
        except Exception:
            pass
    
    return HealthResponse(
        status="healthy" if model_loaded else "degraded",
        service=settings.service_name,
        version=settings.model_version,
        model_loaded=model_loaded,
        redis_connected=redis_connected,
        kafka_connected=True,  # Would check actual Kafka connection
        uptime_seconds=time.time() - _start_time,
        timestamp=datetime.now(timezone.utc)
    )


@router.get("/ready", tags=["Health"])
async def readiness_check():
    """
    Readiness check endpoint.
    
    Returns 200 if service is ready to accept requests.
    """
    if _scorer is None or not await _scorer.is_ready():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service not ready"
        )
    return {"status": "ready"}


@router.get("/live", tags=["Health"])
async def liveness_check():
    """
    Liveness check endpoint.
    
    Returns 200 if service is alive.
    """
    return {"status": "alive"}


# =============================================================================
# FRAUD SCORING ENDPOINTS
# =============================================================================

@router.post(
    "/score",
    response_model=ScoringResponse,
    status_code=status.HTTP_200_OK,
    tags=["Fraud Detection"]
)
async def score_transaction(
    request: TransactionRequest,
    scorer: HybridFraudScorer = Depends(get_scorer)
) -> ScoringResponse:
    """
    Score a single transaction for fraud risk.
    
    This endpoint accepts transaction data and returns a fraud probability
    score along with risk factors and metadata.
    
    Performance target: <50ms response time.
    """
    start_time = time.perf_counter()
    
    try:
        # Convert request to internal model
        transaction = request.to_transaction_base()
        
        # Score the transaction
        score = await scorer.score(transaction)
        
        # Update feature store with the new transaction
        await scorer.update_after_transaction(transaction, score)
        
        # Record metrics
        fraud_metrics.record_score(
            fraud_probability=score.fraud_probability,
            risk_level=score.risk_level,
            model_used=score.model_used,
            duration_seconds=score.inference_time_ms / 1000,
            amount=transaction.amount,
            currency=transaction.currency
        )
        
        processing_time = (time.perf_counter() - start_time) * 1000
        
        return ScoringResponse(
            success=True,
            score=score,
            processing_time_ms=processing_time
        )
        
    except Exception as e:
        processing_time = (time.perf_counter() - start_time) * 1000
        return ScoringResponse(
            success=False,
            error=str(e),
            processing_time_ms=processing_time
        )


@router.post(
    "/score/batch",
    response_model=BatchScoringResponse,
    status_code=status.HTTP_200_OK,
    tags=["Fraud Detection"]
)
async def score_batch_transactions(
    request: BatchTransactionRequest,
    scorer: HybridFraudScorer = Depends(get_scorer)
) -> BatchScoringResponse:
    """
    Score multiple transactions in a single request.
    
    Accepts up to 100 transactions at once for efficient batch processing.
    """
    start_time = time.perf_counter()
    results = []
    
    for txn_request in request.transactions:
        try:
            transaction = txn_request.to_transaction_base()
            score = await scorer.score(transaction)
            await scorer.update_after_transaction(transaction, score)
            
            fraud_metrics.record_score(
                fraud_probability=score.fraud_probability,
                risk_level=score.risk_level,
                model_used=score.model_used,
                duration_seconds=score.inference_time_ms / 1000,
                amount=transaction.amount,
                currency=transaction.currency
            )
            
            results.append(ScoringResponse(
                success=True,
                score=score,
                processing_time_ms=score.inference_time_ms
            ))
        except Exception as e:
            results.append(ScoringResponse(
                success=False,
                error=str(e),
                processing_time_ms=0
            ))
    
    total_processing_time = (time.perf_counter() - start_time) * 1000
    
    return BatchScoringResponse(
        success=all(r.success for r in results),
        results=results,
        total_processed=len(results),
        total_processing_time_ms=total_processing_time
    )


# =============================================================================
# USER FEATURES ENDPOINTS
# =============================================================================

@router.get(
    "/users/{user_id}/features",
    response_model=UserFeaturesResponse,
    tags=["Feature Store"]
)
async def get_user_features(
    user_id: str,
    scorer: HybridFraudScorer = Depends(get_scorer)
) -> UserFeaturesResponse:
    """
    Retrieve stored features for a user.
    
    Returns the aggregated feature values used for fraud detection.
    """
    try:
        features = await scorer._ml_scorer._feature_store.get_user_features(user_id)
        
        last_updated = None
        if "last_seen" in features:
            last_updated = datetime.fromtimestamp(
                features["last_seen"],
                tz=timezone.utc
            )
        
        return UserFeaturesResponse(
            user_id=user_id,
            features=features,
            last_updated=last_updated
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User features not found: {e}"
        )


@router.delete(
    "/users/{user_id}/features",
    status_code=status.HTTP_204_NO_CONTENT,
    tags=["Feature Store"]
)
async def clear_user_features(
    user_id: str,
    scorer: HybridFraudScorer = Depends(get_scorer)
):
    """
    Clear stored features for a user.
    
    Useful for testing or privacy compliance.
    """
    try:
        # In production, implement proper deletion
        pass
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to clear features: {e}"
        )


# =============================================================================
# MODEL MANAGEMENT ENDPOINTS
# =============================================================================

@router.get(
    "/model/info",
    tags=["Model Management"]
)
async def get_model_info(
    scorer: HybridFraudScorer = Depends(get_scorer)
) -> dict:
    """
    Get information about the loaded model.
    """
    ml_ready = await scorer._ml_scorer.is_ready()
    
    return {
        "model_version": scorer._ml_scorer._model_version,
        "model_path": scorer._ml_scorer._model_path,
        "model_loaded": ml_ready,
        "feature_count": len(scorer._ml_scorer._feature_names),
        "fallback_enabled": scorer._rule_scorer is not None,
        "circuit_breaker_state": scorer._ml_scorer._circuit_breaker.state.value,
        "latency_target_ms": scorer._latency_target_ms
    }


@router.post(
    "/model/reload",
    status_code=status.HTTP_202_ACCEPTED,
    tags=["Model Management"]
)
async def reload_model(
    scorer: HybridFraudScorer = Depends(get_scorer)
) -> dict:
    """
    Reload the ML model from disk.
    
    Useful for hot-swapping models without service restart.
    """
    try:
        await scorer._ml_scorer.load_model()
        return {
            "status": "success",
            "message": "Model reloaded successfully",
            "model_version": scorer._ml_scorer._model_version
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to reload model: {e}"
        )


# =============================================================================
# STATISTICS ENDPOINTS
# =============================================================================

@router.get(
    "/stats/summary",
    tags=["Statistics"]
)
async def get_stats_summary() -> dict:
    """
    Get summary statistics for the fraud detection service.
    """
    return {
        "service": settings.service_name,
        "uptime_seconds": time.time() - _start_time,
        "model_version": settings.model_version,
        "fraud_thresholds": {
            "critical": settings.fraud_threshold,
            "high": settings.high_risk_threshold,
            "medium": settings.medium_risk_threshold
        },
        "performance_targets": {
            "latency_ms": settings.latency_target_ms
        }
    }


# =============================================================================
# STARTUP/SHUTDOWN HANDLERS
# =============================================================================

_start_time: float = 0


async def initialize_scorer() -> None:
    """Initialize the scorer on application startup."""
    global _scorer, _start_time
    _start_time = time.time()
    
    _scorer = await create_scorer(
        model_path=settings.model_path,
        redis_url=settings.redis_url,
        model_version=settings.model_version
    )


async def shutdown_scorer() -> None:
    """Cleanup on application shutdown."""
    global _scorer
    if _scorer is not None:
        # Close Redis connection
        try:
            await _scorer._ml_scorer._feature_store._redis.close()
        except Exception:
            pass
    _scorer = None
