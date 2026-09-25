-- Production PostgreSQL Schema Migration: 001_initial_multi_tenant_schema.sql
-- Enables pgvector extension, multi-tenant isolation, and HNSW vector index.

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "vector";

-- Tenants registry
CREATE TABLE IF NOT EXISTS tenants (
    tenant_id VARCHAR(64) PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    plan_tier VARCHAR(64) NOT NULL DEFAULT 'standard',
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Episodic history: Episodes
CREATE TABLE IF NOT EXISTS episodes (
    episode_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id VARCHAR(64) NOT NULL REFERENCES tenants(tenant_id) ON DELETE CASCADE,
    session_id VARCHAR(128) NOT NULL,
    total_cost_usd NUMERIC(10, 6) NOT NULL DEFAULT 0.0,
    total_tokens INTEGER NOT NULL DEFAULT 0,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_episodes_tenant_session ON episodes(tenant_id, session_id);
CREATE INDEX IF NOT EXISTS idx_episodes_tenant_created ON episodes(tenant_id, created_at DESC);

-- Episodic history: Turns
CREATE TABLE IF NOT EXISTS turns (
    turn_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id VARCHAR(64) NOT NULL REFERENCES tenants(tenant_id) ON DELETE CASCADE,
    session_id VARCHAR(128) NOT NULL,
    turn_index INTEGER NOT NULL,
    user_input TEXT NOT NULL,
    model_output TEXT NOT NULL,
    tool_calls_json JSONB NOT NULL DEFAULT '[]'::jsonb,
    tool_results_json JSONB NOT NULL DEFAULT '[]'::jsonb,
    tokens_in INTEGER NOT NULL DEFAULT 0,
    tokens_out INTEGER NOT NULL DEFAULT 0,
    cost_usd NUMERIC(10, 6) NOT NULL DEFAULT 0.0,
    latency_ms NUMERIC(10, 2) NOT NULL DEFAULT 0.0,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_turns_tenant_session_index ON turns(tenant_id, session_id, turn_index);
CREATE INDEX IF NOT EXISTS idx_turns_tenant_created ON turns(tenant_id, created_at DESC);

-- Memory Gate Step 1 & 2: Fact Quarantine
CREATE TABLE IF NOT EXISTS fact_quarantine (
    candidate_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id VARCHAR(64) NOT NULL REFERENCES tenants(tenant_id) ON DELETE CASCADE,
    source_episode_id VARCHAR(128) NOT NULL,
    session_id VARCHAR(128) NOT NULL,
    subject VARCHAR(255) NOT NULL,
    predicate VARCHAR(255) NOT NULL,
    object TEXT NOT NULL,
    confidence NUMERIC(4, 3) NOT NULL,
    extractor_model VARCHAR(128) NOT NULL,
    status VARCHAR(32) NOT NULL DEFAULT 'quarantined',
    rejection_reason VARCHAR(64),
    rejection_detail TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_quarantine_tenant_status ON fact_quarantine(tenant_id, status);
CREATE INDEX IF NOT EXISTS idx_quarantine_tenant_session ON fact_quarantine(tenant_id, session_id);

-- Memory Gate Step 3: Verified Semantic Facts with pgvector
CREATE TABLE IF NOT EXISTS semantic_facts (
    fact_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id VARCHAR(64) NOT NULL REFERENCES tenants(tenant_id) ON DELETE CASCADE,
    candidate_id VARCHAR(128) NOT NULL,
    source_episode_id VARCHAR(128) NOT NULL,
    session_id VARCHAR(128) NOT NULL,
    subject VARCHAR(255) NOT NULL,
    predicate VARCHAR(255) NOT NULL,
    object TEXT NOT NULL,
    confidence NUMERIC(4, 3) NOT NULL,
    embedding vector(1536),
    valid_from TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    valid_until TIMESTAMPTZ,
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    retired_at TIMESTAMPTZ,
    promoted_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    provenance_json JSONB NOT NULL DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS idx_semantic_tenant_subject_active ON semantic_facts(tenant_id, subject, is_active);
CREATE INDEX IF NOT EXISTS idx_semantic_tenant_session ON semantic_facts(tenant_id, session_id);

-- pgvector HNSW Index for sub-millisecond approximate nearest neighbor search
CREATE INDEX IF NOT EXISTS idx_semantic_facts_embedding_hnsw
    ON semantic_facts
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);
