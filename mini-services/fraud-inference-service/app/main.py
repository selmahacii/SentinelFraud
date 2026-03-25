"""
=============================================================================
SENTINEL-FRAUD: FastAPI Application Entry Point
=============================================================================
Main FastAPI application with async support, CORS, and middleware.
=============================================================================
"""

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError

from .core.config import get_settings
from .core.logging import setup_logging, get_logger
from .core.middleware import PrometheusMiddleware, metrics_endpoint
from .api.routes import router as api_router, initialize_scorer, shutdown_scorer

# Setup logging
setup_logging()
logger = get_logger(__name__)
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Application lifespan manager.
    
    Handles startup and shutdown events for the FastAPI application.
    """
    # Startup
    logger.info(f"Starting {settings.service_name}...")
    logger.info(f"Environment: {settings.environment}")
    logger.info(f"Log level: {settings.log_level}")
    
    try:
        # Initialize the fraud scorer
        await initialize_scorer()
        logger.info("Fraud scorer initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize scorer: {e}")
        # Continue anyway - fallback to rules
    
    logger.info(f"{settings.service_name} is ready to accept requests")
    
    yield
    
    # Shutdown
    logger.info(f"Shutting down {settings.service_name}...")
    await shutdown_scorer()
    logger.info("Shutdown complete")


# =============================================================================
# APPLICATION INSTANCE
# =============================================================================

app = FastAPI(
    title="Sentinel-Fraud Detection API",
    description="""
    ## Real-time Fraud Detection API
    
    Production-grade fraud detection service with sub-50ms latency target.
    
    ### Features
    - **ML-based scoring** using XGBoost with automatic fallback
    - **Feature engineering** with 20+ transaction features
    - **Impossible travel detection** using Haversine distance
    - **Real-time velocity checks** for transaction frequency
    - **Circuit breaker** for resilience
    
    ### Risk Levels
    - **CRITICAL**: Probability >= 0.85
    - **HIGH**: Probability >= 0.70
    - **MEDIUM**: Probability >= 0.50
    - **LOW**: Probability < 0.50
    """,
    version="1.0.0",
    docs_url="/docs" if settings.debug else None,
    redoc_url="/redoc" if settings.debug else None,
    openapi_url="/openapi.json" if settings.debug else None,
    lifespan=lifespan,
)


# =============================================================================
# MIDDLEWARE
# =============================================================================

# CORS middleware
allowed_origins = settings.cors_origins.split(",") if settings.cors_origins else ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Process-Time", "X-Request-ID"],
)

# Prometheus metrics middleware
if settings.prometheus_enabled:
    app.add_middleware(
        PrometheusMiddleware,
        prefix=settings.metrics_prefix
    )


# =============================================================================
# EXCEPTION HANDLERS
# =============================================================================

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request,
    exc: RequestValidationError
) -> JSONResponse:
    """Handle validation errors with detailed messages."""
    errors = []
    for error in exc.errors():
        errors.append({
            "field": ".".join(str(loc) for loc in error["loc"]),
            "message": error["msg"],
            "type": error["type"]
        })
    
    return JSONResponse(
        status_code=422,
        content={
            "detail": "Validation error",
            "errors": errors
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(
    request: Request,
    exc: Exception
) -> JSONResponse:
    """Handle unexpected exceptions."""
    logger.exception(f"Unhandled exception: {exc}")
    
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Internal server error",
            "message": str(exc) if settings.debug else "An error occurred"
        }
    )


# =============================================================================
# ROUTES
# =============================================================================

# Include API routes
app.include_router(api_router, prefix="/api/v1")

# Metrics endpoint for Prometheus
if settings.prometheus_enabled:
    app.get("/metrics", include_in_schema=False)(metrics_endpoint)


# Root endpoint
@app.get("/", tags=["Root"])
async def root() -> dict:
    """Root endpoint with service information."""
    return {
        "service": settings.service_name,
        "version": "1.0.0",
        "status": "operational",
        "docs": "/docs" if settings.debug else "disabled",
        "health": "/api/v1/health"
    }


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "app.main:app",
        host=settings.api_host,
        port=settings.api_port,
        workers=settings.api_workers,
        log_level=settings.log_level.lower(),
        reload=settings.debug,
        access_log=settings.debug
    )
