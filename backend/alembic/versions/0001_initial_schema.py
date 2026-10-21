"""Initial schema: agents, documents, chunks (pgvector + tsvector), chats, usage.

Revision ID: 0001
Revises:
Create Date: 2026-10-22 05:40:26.891848
"""

from collections.abc import Sequence

import sqlalchemy as sa
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects import postgresql

from alembic import op
from app.core.config import get_settings

revision: str = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.create_table(
        "agents",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("name", sa.String(length=120), nullable=False),
        sa.Column("description", sa.Text(), nullable=False),
        sa.Column("instructions", sa.Text(), nullable=False),
        sa.Column("provider", sa.String(length=50), nullable=False),
        sa.Column("model", sa.String(length=200), nullable=False),
        sa.Column("fallback_provider", sa.String(length=50), nullable=True),
        sa.Column("fallback_model", sa.String(length=200), nullable=True),
        sa.Column("temperature", sa.Float(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_table(
        "conversations",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("agent_id", sa.UUID(), nullable=False),
        sa.Column("title", sa.String(length=200), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["agent_id"], ["agents.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_conversations_agent_id", "conversations", ["agent_id"], unique=False)
    op.create_table(
        "documents",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("agent_id", sa.UUID(), nullable=False),
        sa.Column("filename", sa.String(length=255), nullable=False),
        sa.Column("file_type", sa.String(length=10), nullable=False),
        sa.Column("content_type", sa.String(length=120), nullable=False),
        sa.Column("size_bytes", sa.BigInteger(), nullable=False),
        sa.Column("sha256", sa.String(length=64), nullable=False),
        sa.Column("storage_path", sa.String(length=500), nullable=False),
        sa.Column("status", sa.String(length=20), nullable=False),
        sa.Column("stage", sa.String(length=20), nullable=False),
        sa.Column("error_code", sa.String(length=50), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("page_count", sa.Integer(), nullable=True),
        sa.Column("ocr_page_count", sa.Integer(), nullable=False),
        sa.Column("chunk_count", sa.Integer(), nullable=False),
        sa.Column("char_count", sa.Integer(), nullable=False),
        sa.Column("embedding_model", sa.String(length=200), nullable=True),
        sa.Column("attempts", sa.Integer(), nullable=False),
        sa.Column("locked_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("processed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["agent_id"], ["agents.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_documents_agent_id", "documents", ["agent_id"], unique=False)
    op.create_index(
        "ix_documents_status_locked", "documents", ["status", "locked_at"], unique=False
    )
    op.create_table(
        "chunks",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("agent_id", sa.UUID(), nullable=False),
        sa.Column("document_id", sa.UUID(), nullable=False),
        sa.Column("chunk_index", sa.Integer(), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("page_number", sa.Integer(), nullable=True),
        sa.Column("section", sa.String(length=500), nullable=True),
        sa.Column("token_count", sa.Integer(), nullable=False),
        sa.Column("embedding", Vector(get_settings().embedding_dimensions), nullable=False),
        sa.Column("tsv", postgresql.TSVECTOR(), nullable=False),
        sa.Column("metadata", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["agent_id"], ["agents.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["document_id"], ["documents.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_chunks_agent_id", "chunks", ["agent_id"], unique=False)
    op.create_index("ix_chunks_document_id", "chunks", ["document_id"], unique=False)
    op.create_index(
        "ix_chunks_embedding_hnsw",
        "chunks",
        ["embedding"],
        unique=False,
        postgresql_using="hnsw",
        postgresql_ops={"embedding": "vector_cosine_ops"},
    )
    op.create_index("ix_chunks_tsv", "chunks", ["tsv"], unique=False, postgresql_using="gin")
    op.create_table(
        "messages",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("conversation_id", sa.UUID(), nullable=False),
        sa.Column("role", sa.String(length=20), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("status", sa.String(length=20), nullable=False),
        sa.Column("citations", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("sources", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("provider", sa.String(length=50), nullable=True),
        sa.Column("model", sa.String(length=200), nullable=True),
        sa.Column("rewritten_query", sa.Text(), nullable=True),
        sa.Column("fallback_events", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("error", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("prompt_tokens", sa.Integer(), nullable=False),
        sa.Column("completion_tokens", sa.Integer(), nullable=False),
        sa.Column("cost_usd", sa.Numeric(precision=12, scale=6), nullable=True),
        sa.Column("latency_ms", sa.Integer(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["conversation_id"], ["conversations.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_messages_conversation_created",
        "messages",
        ["conversation_id", "created_at"],
        unique=False,
    )
    op.create_table(
        "usage_records",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("agent_id", sa.UUID(), nullable=False),
        sa.Column("conversation_id", sa.UUID(), nullable=True),
        sa.Column("kind", sa.String(length=30), nullable=False),
        sa.Column("provider", sa.String(length=50), nullable=False),
        sa.Column("model", sa.String(length=200), nullable=False),
        sa.Column("prompt_tokens", sa.Integer(), nullable=False),
        sa.Column("completion_tokens", sa.Integer(), nullable=False),
        sa.Column("cost_usd", sa.Numeric(precision=12, scale=6), nullable=False),
        sa.Column("cost_source", sa.String(length=20), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["agent_id"], ["agents.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["conversation_id"], ["conversations.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_usage_agent_created", "usage_records", ["agent_id", "created_at"], unique=False
    )


def downgrade() -> None:
    op.drop_index("ix_usage_agent_created", table_name="usage_records")
    op.drop_table("usage_records")
    op.drop_index("ix_messages_conversation_created", table_name="messages")
    op.drop_table("messages")
    op.drop_index("ix_chunks_tsv", table_name="chunks", postgresql_using="gin")
    op.drop_index(
        "ix_chunks_embedding_hnsw",
        table_name="chunks",
        postgresql_using="hnsw",
        postgresql_ops={"embedding": "vector_cosine_ops"},
    )
    op.drop_index("ix_chunks_document_id", table_name="chunks")
    op.drop_index("ix_chunks_agent_id", table_name="chunks")
    op.drop_table("chunks")
    op.drop_index("ix_documents_status_locked", table_name="documents")
    op.drop_index("ix_documents_agent_id", table_name="documents")
    op.drop_table("documents")
    op.drop_index("ix_conversations_agent_id", table_name="conversations")
    op.drop_table("conversations")
    op.drop_table("agents")
