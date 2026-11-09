from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables and .env file."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- Session 2 fields (kept for backwards compatibility with the live demos) ---
    OPENAI_API_KEY: str | None = None
    ANTHROPIC_API_KEY: str | None = None
    LLM_PROVIDER: Literal["openai", "anthropic"] = "anthropic"
    LLM_MODEL: str = "claude-haiku-4-5"
    APP_ENV: Literal["development", "staging", "production"] = "development"
    LOG_LEVEL: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "DEBUG"

    # --- Session 3 fields (LiteLLM wrapper, Redis cache, Streamlit transport) ---
    PRIMARY_MODEL: str = "gpt-4o-mini"
    FALLBACK_MODEL: str = "claude-haiku-4-5-20251001"
    LLM_TIMEOUT: int = 30
    LLM_RETRIES: int = 2
    # Catalog of models selectable at runtime via PUT /api/v1/config/models
    # (kept aligned with MODEL_COSTS in app/foundation/llm/wrapper.py). The
    # endpoint filters this list by the API keys actually configured.
    AVAILABLE_MODELS: list[str] = [
        "gpt-4o-mini",
        "gpt-4o",
        "gpt-5",
        "gpt-5-mini",
        "claude-haiku-4-5-20251001",
        "claude-sonnet-4-5",
    ]

    REDIS_URL: str = "redis://localhost:6379"
    CACHE_TTL: int = 86400

    # --- Session 4 fields (semantic cache) ---
    EMBEDDING_MODEL: str = "text-embedding-3-small"
    SEMANTIC_CACHE_THRESHOLD: float = 0.85
    SEMANTIC_CACHE_TTL: int = 86400
    # When True, the semantic cache LOGS potential hits but does NOT serve them.
    # Used to gather metrics before flipping the cache on in production.
    SEMANTIC_CACHE_LOG_ONLY: bool = False

    ESTIMATOR_API_BASE_URL: str = "http://localhost:8000"

    # --- Session 5 fields (conversational memory + attachments) ---
    # MAX_CONVERSATION_TURNS counts user+assistant pairs. The system prompt is
    # always preserved as an invariant on top of the window.
    MAX_CONVERSATION_TURNS: int = 6
    # Hard cap per extracted attachment (in characters) to protect the context
    # window. Real chunking enters in module 3.
    MAX_ATTACHMENT_CHARS: int = 60_000
    # The metadata extractor runs once per turn; a small/cheap model is enough.
    METADATA_EXTRACTOR_MODEL: str = "gpt-4o-mini"

    # --- Session 5 live: compression + tier + ACB ---
    # Anchor detector: "heuristic" (regex over key phrases) or "llm" (binary
    # classifier via Instructor). Heuristic is the default for cost.
    ANCHOR_DETECTION_MODE: Literal["heuristic", "llm"] = "heuristic"
    # Cheap model used by the cumulative summarizer (history compression).
    COMPRESSION_MODEL: str = "gpt-4o-mini"
    # Conversational prompt version used by ``estimate_conversational``.
    # v2 = pre-live-session baseline. v3 = adds <audience> block driven by tier
    # and an optional <critic_feedback> block consumed by the Boss.
    CONVERSATIONAL_PROMPT_VERSION: str = "v3"
    # Critic model (read-only auditor; cheap is fine).
    CRITIC_MODEL: str = "gpt-4o-mini"
    # Max iterations the Boss can drive (each iteration = 1 actor + 1 critic call).
    # Three is the practical floor: one initial draft + two directed retries.
    # With only two iterations the actor often cannot address all flagged issues
    # in the single available retry, and the loop falls back without converging.
    BOSS_MAX_ITERATIONS: int = 3

    # --- Session 6 fields (data-driven AI: persistence + ingestion + PII) ---
    # Postgres connection string. pgvector/pgvector:pg16 image; the extension
    # is dormant in S06 (no CREATE EXTENSION vector) and only activates in S07.
    DATABASE_URL: str = "postgresql+psycopg://estimator:estimator@localhost:5433/estimator"
    # Where the YAML catalog lives. Resolved relative to the working directory.
    CATALOG_PATH: Path = Path("data/catalog/catalog.yaml")
    # Root where ``CatalogSource.location`` entries are resolved against.
    INGESTION_DATA_ROOT: Path = Path("data/seed")
    # spaCy model loaded by the Presidio AnalyzerEngine. Must be the Spanish
    # one for the live session; ``es_core_news_md`` is the recommended size.
    PRESIDIO_SPACY_MODEL: str = "es_core_news_md"
    # Locale used by Faker to generate consistent pseudonyms per entity_type.
    PSEUDONYM_FAKER_LOCALE: str = "es_ES"
    # HMAC salt. Stored in env so it can be rotated independently of the code.
    PSEUDONYM_HASH_SALT: str = "change-me-in-prod"

    # --- Session 7 live fields (chunking strategies that call external APIs) ---
    # LLM that decomposes a component into atomic propositions (one call per
    # component). A small/cheap model is enough.
    PROPOSITIONAL_CHUNKER_MODEL: str = "gpt-4o-mini"
    # Claude model used by Contextual Retrieval to situate each chunk inside its
    # parent budget. Prompt caching makes the (large) parent document cheap to
    # reuse across the chunks of the same budget.
    CONTEXTUAL_CHUNKER_MODEL: str = "claude-sonnet-4-5"

    # --- Session 9 fields (RAG estimation: transcript → grounded estimate) ---
    # Query understanding distills a transcript into an EstimationQuery; a small
    # model is enough. Generation reasons over retrieved budgets, so it uses the
    # strongest model with medium reasoning effort. Both go through LLMWrapper.
    REFORMULATION_MODEL: str = "gpt-5-mini"
    GENERATION_MODEL: str = "gpt-5"
    # "high" drives a deeper, more consistent module→task decomposition (the S09
    # article used "medium"; we raise it for the granular modular breakdown).
    GENERATION_REASONING_EFFORT: Literal["minimal", "low", "medium", "high"] = "high"
    # Token ceiling (reasoning + output) for the RAG structured calls. gpt-5 is a
    # reasoning model: its reasoning tokens count against this budget, so the
    # 4000 wrapper default leaves nothing for the JSON and the call truncates
    # (finish_reason='length'). Generous headroom so high-effort reasoning can
    # finish AND emit the larger nested (modules→tasks) Estimate. It is a CAP,
    # not a target — the model only spends what it needs, so a high value adds no
    # latency on its own.
    GENERATION_MAX_TOKENS: int = 64000
    # Retrieval knobs (locked defaults from the Session 9 articles).
    RETRIEVAL_TOP_K: int = 10
    RETRIEVAL_DISTANCE_THRESHOLD: float = 0.6
    # Token budget for the assembled <source> context block (tiktoken cl100k_base).
    MAX_CONTEXT_TOKENS: int = 16384
    # Idempotency cache for POST /v1/estimate/from-transcript (seconds; 24h).
    IDEMPOTENCY_TTL: int = 86400
    # API keys for the two Session 9 routers. None disables the router (401 on
    # every request) — set them in .env to enable the endpoints.
    RETRIEVAL_API_KEY: str | None = None
    ESTIMATE_API_KEY: str | None = None

    # --- Session 10 fields (hybrid search + cross-encoder reranking) ---
    # Default retrieval mode. "vector" reproduces the Session 9 baseline; "hybrid"
    # fuses the dense and lexical (full-text) branches with RRF. Switchable per
    # request (RetrievalRequest.search_mode) and at runtime (RuntimeRetrievalConfig).
    RETRIEVAL_SEARCH_MODE: Literal["vector", "hybrid"] = "vector"
    # Whether the cross-encoder reranks by default. Off keeps the baseline cheap;
    # the recall-then-rerank path turns on per request / at runtime.
    RERANKER_ENABLED: bool = False
    # Multilingual cross-encoder (ES+EN), small enough for CPU at teaching latency.
    RERANKER_MODEL: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
    # Load + warm up the cross-encoder ONCE in the app lifespan, so no request
    # pays the model load (~seconds + ~500MB RAM). On by default because reranking
    # can be switched on per request / at runtime even when RERANKER_ENABLED=false.
    # The test suite turns it off (no torch weights in unit tests).
    RERANKER_PRELOAD: bool = True
    # Recall width before reranking/fusion (recall-then-rerank): retrieve this many
    # candidates cheaply, then the cross-encoder rescores them down to RERANK_TOP_N.
    # Bounded so nobody can ask for thousands of candidates and turn the
    # cross-encoder (O(candidates) forward passes) into a denial of service.
    RETRIEVAL_RECALL_TOP_K: int = Field(default=50, ge=1, le=200)
    RERANK_TOP_N: int = Field(default=5, ge=1, le=50)
    # RRF smoothing constant (Cormack et al. default). Larger = a document must
    # rank well in BOTH branches to win; smaller = a single #1 can dominate.
    RRF_K: int = Field(default=60, gt=0, le=1000)
    # Candidates each hybrid branch returns BEFORE fusion; the fused ranking is
    # then capped at RETRIEVAL_RECALL_TOP_K.
    VECTOR_CANDIDATE_LIMIT: int = Field(default=50, ge=1, le=200)
    LEXICAL_CANDIDATE_LIMIT: int = Field(default=50, ge=1, le=200)

    @model_validator(mode="after")
    def validate_at_least_one_api_key(self) -> "Settings":
        """LiteLLM may try either provider via fallback, so we require at least one key."""
        if not self.OPENAI_API_KEY and not self.ANTHROPIC_API_KEY:
            raise ValueError("At least one of OPENAI_API_KEY or ANTHROPIC_API_KEY must be set")
        return self


@lru_cache
def get_settings() -> Settings:
    """Return cached application settings (singleton)."""
    return Settings()
