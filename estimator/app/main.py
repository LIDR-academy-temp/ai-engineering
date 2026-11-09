import asyncio

import structlog
from contextlib import asynccontextmanager
from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from slowapi.errors import RateLimitExceeded

from app.config import get_settings
from app.api.embeddings import router as embeddings_router
from app.api.search import router as search_router
from app.api import config as config_api
from app.api import estimations, ingestion, sessions
from app.api.rate_limiting import limiter, rate_limit_exceeded_handler
from app.api.routers.estimate import router as estimate_router
from app.api.routers.estimate_stages import router as estimate_stages_router
from app.api.routers.retrieval import router as retrieval_router


def configure_logging() -> None:
    """Set up structlog: JSON in production, human-readable in development."""
    settings = get_settings()

    if settings.APP_ENV == "production":
        renderer = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer()

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            renderer,
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown lifecycle."""
    configure_logging()
    log = structlog.get_logger()
    settings = get_settings()
    # Session 6: fail fast on a malformed catalog rather than at the first
    # ingestion request. Catalogs are versioned in git; a broken one is a
    # deploy-time problem, not a request-time one.
    try:
        from app.dependencies import get_catalog

        catalog = get_catalog()
        log.info(
            "catalog_loaded",
            version=catalog.version,
            sources_total=len(catalog.sources),
            sources_included=len(catalog.included_sources()),
        )
    except Exception as exc:  # noqa: BLE001
        log.error("catalog_load_failed", error=str(exc)[:400])
    # Session 10: load + warm up the cross-encoder ONCE, here, so no request pays
    # the model load. Off the event loop (torch load is blocking). A failure is
    # logged and surfaced by /health (reranker.loaded=false) instead of crashing
    # the service: vector/hybrid search keep working without the reranker.
    if settings.RERANKER_PRELOAD:
        from app.dependencies import get_reranker

        reranker = get_reranker()
        try:
            await asyncio.to_thread(reranker.warmup)
            log.info("reranker_ready", model=reranker.model_name)
        except Exception as exc:  # noqa: BLE001
            log.error(
                "reranker_preload_failed",
                model=reranker.model_name,
                error_type=type(exc).__name__,
            )
    log.info("application_started", environment=settings.APP_ENV)
    yield
    log.info("application_shutdown")


app = FastAPI(
    title="Software Estimation Service",
    description="AI-powered software estimation service with typed input and versioned prompts",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Session 9: per-API-key rate limiting (slowapi). The decorators on the routers
# read ``app.state.limiter``; a custom handler turns the limit breach into a
# JSON 429 with a ``Retry-After`` header.
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, rate_limit_exceeded_handler)


@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    """Assign a correlation id per request, bind it for structlog, and echo it
    back on the ``X-Request-ID`` response header so failures are debuggable."""
    request_id = request.headers.get("X-Request-ID") or str(uuid4())
    request.state.request_id = request_id
    structlog.contextvars.bind_contextvars(request_id=request_id)
    try:
        response = await call_next(request)
    finally:
        structlog.contextvars.unbind_contextvars("request_id")
    response.headers["X-Request-ID"] = request_id
    return response


app.include_router(estimations.router)
app.include_router(sessions.router)
app.include_router(ingestion.router)
app.include_router(embeddings_router)
app.include_router(search_router)
app.include_router(config_api.router)
# Session 9 — RAG retrieval + grounded estimation (each independently secured).
app.include_router(retrieval_router)
app.include_router(estimate_router)
# Per-stage endpoints exposing each pipeline step (wizard / live-session aid).
app.include_router(estimate_stages_router)


@app.get("/health")
async def health_check() -> dict:
    """Return service health status."""
    settings = get_settings()
    from app.dependencies import get_reranker

    # Reports readiness of the Session 10 cross-encoder without triggering a
    # load. Liveness ("status") does not depend on it: retrieval works without it.
    reranker = get_reranker()
    return {
        "status": "healthy",
        "version": "0.1.0",
        "environment": settings.APP_ENV,
        "reranker": {"model": reranker.model_name, "loaded": reranker.is_loaded},
    }
