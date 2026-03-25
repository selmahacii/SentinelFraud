"""
=============================================================================
SENTINEL-FRAUD: Prometheus Metrics Middleware
=============================================================================
Custom Prometheus metrics for fraud detection monitoring.
=============================================================================
"""

import time
from contextlib import asynccontextmanager
from typing import Callable

from fastapi import Request, Response
from prometheus_client import (
    REGISTRY,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
)
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from ..core.config import get_settings


# Create custom registry to avoid duplicates
METRICS_REGISTRY = CollectorRegistry(auto_describe=True)


# =============================================================================
# METRIC DEFINITIONS
# =============================================================================

def create_metrics(prefix: str = "sentinel") -> dict:
    """Create Prometheus metrics with given prefix."""
    return {
        # Request metrics
        "http_requests_total": Counter(
            f"{prefix}_http_requests_total",
            "Total HTTP requests",
            ["method", "endpoint", "status_code"],
            registry=METRICS_REGISTRY,
        ),
        "http_request_duration_seconds": Histogram(
            f"{prefix}_http_request_duration_seconds",
            "HTTP request latency",
            ["method", "endpoint"],
            buckets=[0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 1.0],
            registry=METRICS_REGISTRY,
        ),
        
        # Fraud detection metrics
        "fraud_score_total": Counter(
            f"{prefix}_fraud_score_total",
            "Total fraud scoring requests",
            ["risk_level", "model_used"],
            registry=METRICS_REGISTRY,
        ),
        "fraud_score_duration_seconds": Histogram(
            f"{prefix}_fraud_score_duration_seconds",
            "Fraud scoring latency",
            ["model_used"],
            buckets=[0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.075, 0.1, 0.25],
            registry=METRICS_REGISTRY,
        ),
        "fraud_amount_total": Counter(
            f"{prefix}_fraud_amount_total",
            "Total transaction amount processed",
            ["currency", "risk_level"],
            registry=METRICS_REGISTRY,
        ),
        "fraud_rate": Gauge(
            f"{prefix}_fraud_rate",
            "Current fraud detection rate",
            registry=METRICS_REGISTRY,
        ),
        
        # Model metrics
        "model_predictions_total": Counter(
            f"{prefix}_model_predictions_total",
            "Total model predictions",
            ["model_version", "status"],
            registry=METRICS_REGISTRY,
        ),
        "model_latency_seconds": Histogram(
            f"{prefix}_model_latency_seconds",
            "Model inference latency",
            ["model_version"],
            buckets=[0.001, 0.005, 0.01, 0.02, 0.03, 0.05, 0.1],
            registry=METRICS_REGISTRY,
        ),
        
        # Feature store metrics
        "feature_store_operations_total": Counter(
            f"{prefix}_feature_store_operations_total",
            "Feature store operations",
            ["operation", "status"],
            registry=METRICS_REGISTRY,
        ),
        "feature_store_latency_seconds": Histogram(
            f"{prefix}_feature_store_latency_seconds",
            "Feature store latency",
            ["operation"],
            buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1],
            registry=METRICS_REGISTRY,
        ),
        
        # Circuit breaker metrics
        "circuit_breaker_state": Gauge(
            f"{prefix}_circuit_breaker_state",
            "Circuit breaker state (0=closed, 1=open, 2=half_open)",
            ["service"],
            registry=METRICS_REGISTRY,
        ),
        "circuit_breaker_failures_total": Counter(
            f"{prefix}_circuit_breaker_failures_total",
            "Total circuit breaker failures",
            ["service"],
            registry=METRICS_REGISTRY,
        ),
        
        # Kafka metrics
        "kafka_messages_consumed_total": Counter(
            f"{prefix}_kafka_messages_consumed_total",
            "Total Kafka messages consumed",
            ["topic", "status"],
            registry=METRICS_REGISTRY,
        ),
        "kafka_consumer_lag": Gauge(
            f"{prefix}_kafka_consumer_lag",
            "Kafka consumer lag",
            ["topic", "partition"],
            registry=METRICS_REGISTRY,
        ),
        
        # Alert metrics
        "fraud_alerts_total": Counter(
            f"{prefix}_fraud_alerts_total",
            "Total fraud alerts generated",
            ["risk_level"],
            registry=METRICS_REGISTRY,
        ),
        "high_value_fraud_total": Counter(
            f"{prefix}_high_value_fraud_total",
            "Total high-value fraud detected",
            ["currency"],
            registry=METRICS_REGISTRY,
        ),
    }


# Initialize metrics
settings = get_settings()
METRICS = create_metrics(settings.metrics_prefix)


# =============================================================================
# MIDDLEWARE
# =============================================================================

class PrometheusMiddleware(BaseHTTPMiddleware):
    """
    Middleware for collecting HTTP request metrics.
    
    Tracks request count, latency, and status codes.
    """
    
    def __init__(self, app: ASGIApp, prefix: str = "sentinel"):
        super().__init__(app)
        self.prefix = prefix
        self.metrics = METRICS
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request and collect metrics."""
        # Skip metrics endpoint to avoid recursion
        if request.url.path == "/metrics":
            return await call_next(request)
        
        # Normalize endpoint for labeling
        endpoint = self._normalize_endpoint(request.url.path)
        method = request.method
        
        # Time the request
        start_time = time.perf_counter()
        
        try:
            response = await call_next(request)
            status_code = str(response.status_code)
        except Exception as e:
            status_code = "500"
            raise e
        finally:
            # Record metrics
            duration = time.perf_counter() - start_time
            
            self.metrics["http_requests_total"].labels(
                method=method,
                endpoint=endpoint,
                status_code=status_code
            ).inc()
            
            self.metrics["http_request_duration_seconds"].labels(
                method=method,
                endpoint=endpoint
            ).observe(duration)
        
        return response
    
    @staticmethod
    def _normalize_endpoint(path: str) -> str:
        """Normalize endpoint path for metric labels."""
        # Replace dynamic segments with placeholders
        segments = path.strip("/").split("/")
        normalized = []
        
        for segment in segments:
            # Check if segment looks like an ID (UUID, numeric, etc.)
            if len(segment) == 36 and "-" in segment:  # UUID
                normalized.append("{id}")
            elif segment.isdigit():
                normalized.append("{id}")
            elif len(segment) > 20:  # Likely an ID
                normalized.append("{id}")
            else:
                normalized.append(segment)
        
        return "/" + "/".join(normalized) if normalized else "/"


# =============================================================================
# FRAUD METRICS HELPERS
# =============================================================================

class FraudMetrics:
    """Helper class for recording fraud-specific metrics."""
    
    def __init__(self):
        self.metrics = METRICS
        self._fraud_count = 0
        self._total_count = 0
    
    def record_score(
        self,
        fraud_probability: float,
        risk_level: str,
        model_used: str,
        duration_seconds: float,
        amount: float = 0.0,
        currency: str = "USD"
    ) -> None:
        """Record fraud scoring metrics."""
        # Increment counters
        self.metrics["fraud_score_total"].labels(
            risk_level=risk_level,
            model_used=model_used
        ).inc()
        
        # Record latency
        self.metrics["fraud_score_duration_seconds"].labels(
            model_used=model_used
        ).observe(duration_seconds)
        
        # Record amount
        if amount > 0:
            self.metrics["fraud_amount_total"].labels(
                currency=currency,
                risk_level=risk_level
            ).inc(amount)
        
        # Update fraud rate
        self._total_count += 1
        if risk_level in ("CRITICAL", "HIGH"):
            self._fraud_count += 1
            self.metrics["fraud_alerts_total"].labels(
                risk_level=risk_level
            ).inc()
            
            if amount > 10000:
                self.metrics["high_value_fraud_total"].labels(
                    currency=currency
                ).inc()
        
        if self._total_count > 0:
            self.metrics["fraud_rate"].set(
                self._fraud_count / self._total_count
            )
    
    def record_model_prediction(
        self,
        model_version: str,
        status: str,
        latency_seconds: float
    ) -> None:
        """Record model prediction metrics."""
        self.metrics["model_predictions_total"].labels(
            model_version=model_version,
            status=status
        ).inc()
        
        if status == "success":
            self.metrics["model_latency_seconds"].labels(
                model_version=model_version
            ).observe(latency_seconds)
    
    def record_feature_store_operation(
        self,
        operation: str,
        status: str,
        latency_seconds: float
    ) -> None:
        """Record feature store operation metrics."""
        self.metrics["feature_store_operations_total"].labels(
            operation=operation,
            status=status
        ).inc()
        
        self.metrics["feature_store_latency_seconds"].labels(
            operation=operation
        ).observe(latency_seconds)
    
    def record_circuit_breaker_change(
        self,
        service: str,
        state: str
    ) -> None:
        """Record circuit breaker state change."""
        state_values = {"closed": 0, "open": 1, "half_open": 2}
        self.metrics["circuit_breaker_state"].labels(
            service=service
        ).set(state_values.get(state, 0))
        
        if state == "open":
            self.metrics["circuit_breaker_failures_total"].labels(
                service=service
            ).inc()
    
    def record_kafka_message(
        self,
        topic: str,
        status: str
    ) -> None:
        """Record Kafka message consumption."""
        self.metrics["kafka_messages_consumed_total"].labels(
            topic=topic,
            status=status
        ).inc()


# Global metrics instance
fraud_metrics = FraudMetrics()


# =============================================================================
# METRICS ENDPOINT
# =============================================================================

async def metrics_endpoint() -> Response:
    """FastAPI endpoint for Prometheus metrics scraping."""
    metrics_output = generate_latest(METRICS_REGISTRY)
    return Response(
        content=metrics_output,
        media_type=CONTENT_TYPE_LATEST,
        headers={
            "Cache-Control": "no-cache",
            "Pragma": "no-cache",
        }
    )
