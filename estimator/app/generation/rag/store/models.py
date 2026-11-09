"""SQLAlchemy ORM models for the vector store (Session 8).

Two tables, a real one-to-many:

* ``documents`` — one row per ingested source (a historical budget). Owns
  provenance: where it came from, when, and document-level metadata.
* ``chunks`` — N rows per document, each carrying the embeddable text and its
  1536-dim vector. ``ON DELETE CASCADE`` means deleting a budget removes all
  its chunks — referential integrity instead of denormalized duplication.

Design notes (defended in the README):

* ``metadata`` is a JSONB column on both tables. Stable fields live in typed
  columns; whatever the chunker enriches (sector, technologies, hours) goes to
  JSONB, queryable via the GIN index without a migration per new key.
* ``embedding`` is **nullable**: it allows inserting a chunk first and filling
  the vector later (async ingestion, future sessions). Session 8 ingests
  chunk+embedding atomically and never exercises that path.
* ``Vector(1536)`` is hardcoded to ``text-embedding-3-small``'s dimensionality;
  changing it means re-embedding the whole corpus, so it is not configuration.
* **No vector index on purpose** — the live session adds HNSW and measures the
  before/after against this sequential-scan baseline.

``metadata`` is a reserved attribute on SQLAlchemy declarative models, so the
Python attribute is ``metadata_`` mapped onto the ``"metadata"`` column.
"""

from __future__ import annotations

from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import BigInteger, Computed, ForeignKey, Index, String, Text, func, text
from sqlalchemy.dialects.postgresql import JSONB, TSVECTOR
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.types import DateTime

from app.foundation.persistence.models import Base

EMBEDDING_DIMENSIONS = 1536  # text-embedding-3-small

# Postgres text-search configuration shared by the generated ``content_tsv``
# column (migration 0003) and the lexical query. They MUST match or the GIN
# index is bypassed and matching silently changes. ``english`` because the
# shipped corpus is in English (see the migration docstring for the rationale).
TEXT_SEARCH_CONFIG = "english"


class DocumentRow(Base):
    __tablename__ = "documents"
    __table_args__ = (Index("ix_documents_source_path", "source_path"),)

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    source_path: Mapped[str] = mapped_column(Text, nullable=False)
    document_type: Mapped[str] = mapped_column(String(50), nullable=False)
    ingested_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    metadata_: Mapped[dict] = mapped_column(
        "metadata", JSONB, nullable=False, server_default=text("'{}'::jsonb")
    )

    chunks: Mapped[list[ChunkRow]] = relationship(
        back_populates="document", cascade="all, delete-orphan", passive_deletes=True
    )


class ChunkRow(Base):
    __tablename__ = "chunks"
    __table_args__ = (
        Index("ix_chunks_document_id", "document_id"),
        Index("ix_chunks_chunk_type", "chunk_type"),
        Index("ix_chunks_metadata_gin", "metadata", postgresql_using="gin"),
        Index("ix_chunks_content_tsv", "content_tsv", postgresql_using="gin"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    document_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("documents.id", ondelete="CASCADE"), nullable=False
    )
    chunk_type: Mapped[str] = mapped_column(String(50), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    embedding: Mapped[list[float] | None] = mapped_column(
        Vector(EMBEDDING_DIMENSIONS), nullable=True
    )
    # Session 10: STORED generated tsvector backing the lexical (keyword) branch
    # of hybrid search. Read-only at the ORM level — Postgres maintains it from
    # ``content``. ``english`` config matches the corpus (see migration 0003).
    content_tsv: Mapped[str | None] = mapped_column(
        TSVECTOR,
        Computed(f"to_tsvector('{TEXT_SEARCH_CONFIG}', content)", persisted=True),
        nullable=True,
    )
    metadata_: Mapped[dict] = mapped_column(
        "metadata", JSONB, nullable=False, server_default=text("'{}'::jsonb")
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    document: Mapped[DocumentRow] = relationship(back_populates="chunks")
