"""
Virtual Chemistry Lab API - Main Application
Production-ready FastAPI application with async support, WebSockets, and comprehensive error handling.
"""

import time
import uuid
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from app.config import settings
from app.api.v1 import experiments, calculations, batch, export, websocket as ws_router, health
from app.core.utils.exceptions import (
    ChemLabException,
    ValidationError,
    CalculationError,
    EngineNotAvailableError,
    ResourceNotFoundError,
)
from app.db.session import engine
from app.models.base import Base

# Prometheus metrics
REQUEST_COUNT = Counter("http_requests_total", "Total HTTP requests", ["method", "endpoint", "status"])
REQUEST_DURATION = Histogram("http_request_duration_seconds", "HTTP request duration")
ACTIVE_WEBSOCKETS = Counter("active_websockets", "Number of active WebSocket connections")

# Rate limiter
limiter = Limiter(key_func=get_remote_address)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events."""
    logger.info("🧪 Virtual Chemistry Lab API starting up...")

    # Create database tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    logger.info("✅ Database tables created/verified")
    logger.info(f"🔧 Default engine: {settings.DEFAULT_ENGINE}")
    logger.info(f"📊 Rate limit: {settings.RATE_LIMIT_PER_MINUTE} req/min")

    yield

    logger.info("🛑 Virtual Chemistry Lab API shutting down...")
    await engine.dispose()
    logger.info("✅ Cleanup complete")


app = FastAPI(
    title="Virtual Chemistry Lab API",
    description="""
    Production-ready virtual chemistry laboratory surpassing ChemGymRL, Arrows, and VirtualChemLab.

    ## Features
    - **Molecular Calculations**: DFT, kinetics, spectra, docking, MD
    - **ML Predictions**: Yield, pKa, LogP, solvent recommendation
    - **Multi-Engine**: RDKit, Psi4, Open Babel, XGBoost
    - **Real-time**: WebSocket streaming for experiment progress
    - **Batch Processing**: Submit multiple experiments at once
    - **Export**: CIF, PDB, SDF, XYZ, JSON formats

    ## Calculation Types
    - DFT Optimization (PBE0/def2-SVP)
    - Single Point Energy
    - IR Spectra Simulation
    - NMR Spectra (1H, 13C)
    - Reaction Kinetics
    - Molecular Docking (AutoDock Vina)
    - Molecular Dynamics
    - pKa Prediction
    - LogP Prediction
    - Solvent Recommendation
    - Yield Prediction
    - Crystallization Prediction
    - Electrochemistry (CV Simulation)

    ## Authentication
    All endpoints require an API key in the `X-API-Key` header.
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan,
)

# Rate limiting
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    """Middleware to collect Prometheus metrics."""
    start_time = time.time()
    response = await call_next(request)
    duration = time.time() - start_time

    REQUEST_COUNT.labels(
        method=request.method,
        endpoint=request.url.path,
        status=response.status_code,
    ).inc()
    REQUEST_DURATION.observe(duration)

    # Add request ID
    response.headers["X-Request-ID"] = str(uuid.uuid4())
    return response


# Custom exception handlers
@app.exception_handler(ChemLabException)
async def chem_lab_exception_handler(request: Request, exc: ChemLabException):
    """Handle custom ChemLab exceptions."""
    logger.error(f"ChemLabException: {exc.detail} | Status: {exc.status_code}")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.error_code,
            "message": exc.detail,
            "request_id": getattr(request.state, "request_id", str(uuid.uuid4())),
        },
    )


@app.exception_handler(ValidationError)
async def validation_error_handler(request: Request, exc: ValidationError):
    """Handle validation errors."""
    logger.warning(f"ValidationError: {exc.detail}")
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": "VALIDATION_ERROR",
            "message": exc.detail,
            "field": exc.field if hasattr(exc, "field") else None,
        },
    )


@app.exception_handler(CalculationError)
async def calculation_error_handler(request: Request, exc: CalculationError):
    """Handle calculation errors."""
    logger.error(f"CalculationError: {exc.detail}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "CALCULATION_ERROR",
            "message": exc.detail,
            "engine": exc.engine if hasattr(exc, "engine") else None,
        },
    )


@app.exception_handler(EngineNotAvailableError)
async def engine_not_available_handler(request: Request, exc: EngineNotAvailableError):
    """Handle engine not available errors."""
    logger.error(f"EngineNotAvailableError: {exc.detail}")
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={
            "error": "ENGINE_UNAVAILABLE",
            "message": exc.detail,
            "available_engines": ["rdkit", "openbabel"],
        },
    )


@app.exception_handler(ResourceNotFoundError)
async def resource_not_found_handler(request: Request, exc: ResourceNotFoundError):
    """Handle resource not found errors."""
    logger.warning(f"ResourceNotFoundError: {exc.detail}")
    return JSONResponse(
        status_code=status.HTTP_404_NOT_FOUND,
        content={
            "error": "NOT_FOUND",
            "message": exc.detail,
            "resource_type": exc.resource_type if hasattr(exc, "resource_type") else None,
        },
    )


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    """Handle all unhandled exceptions."""
    logger.exception(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "INTERNAL_SERVER_ERROR",
            "message": "An unexpected error occurred. Please try again later.",
            "request_id": str(uuid.uuid4()),
        },
    )


# Health check endpoint
@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint for load balancers and monitoring."""
    return {
        "status": "healthy",
        "version": "1.0.0",
        "timestamp": time.time(),
        "services": {
            "api": "up",
            "database": "up",
            "redis": "up",
        },
    }


# Metrics endpoint for Prometheus
@app.get("/metrics", tags=["Monitoring"])
async def metrics():
    """Prometheus metrics endpoint."""
    from starlette.responses import Response
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


# Include routers
app.include_router(health.router, prefix="/api/v1", tags=["Health"])
app.include_router(experiments.router, prefix="/api/v1", tags=["Experiments"])
app.include_router(calculations.router, prefix="/api/v1", tags=["Calculations"])
app.include_router(batch.router, prefix="/api/v1", tags=["Batch Jobs"])
app.include_router(export.router, prefix="/api/v1", tags=["Export"])
app.include_router(ws_router.router, prefix="/ws", tags=["WebSocket"])


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Virtual Chemistry Lab API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "metrics": "/metrics",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
