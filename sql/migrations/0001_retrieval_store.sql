CREATE TABLE traffic_rag.retrieval_dataset (
    dataset_id text PRIMARY KEY CHECK (length(dataset_id) = 64),
    release_name text NOT NULL,
    status text NOT NULL CHECK (status IN ('BUILDING', 'ACTIVE', 'RETIRED', 'FAILED')),
    corpus_version text NOT NULL,
    document_digests jsonb NOT NULL,
    article_manifest_digest text NOT NULL CHECK (length(article_manifest_digest) = 64),
    fine_manifest_digest text NOT NULL CHECK (length(fine_manifest_digest) = 64),
    strategy_lock_sha256 text NOT NULL CHECK (length(strategy_lock_sha256) = 64),
    embedding_model text NOT NULL,
    embedding_revision text NOT NULL,
    embedding_dimensions integer NOT NULL CHECK (embedding_dimensions = 1024),
    normalized_embeddings boolean NOT NULL CHECK (normalized_embeddings),
    release_manifest jsonb NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now(),
    activated_at timestamptz,
    CHECK ((status = 'ACTIVE' AND activated_at IS NOT NULL) OR status <> 'ACTIVE')
);

CREATE UNIQUE INDEX retrieval_dataset_one_active
    ON traffic_rag.retrieval_dataset ((status)) WHERE status = 'ACTIVE';

CREATE TABLE traffic_rag.legal_node (
    dataset_id text NOT NULL REFERENCES traffic_rag.retrieval_dataset(dataset_id) ON DELETE CASCADE,
    node_id text NOT NULL,
    document_id text NOT NULL,
    node_type text NOT NULL,
    hierarchy jsonb NOT NULL,
    title text,
    text_content text NOT NULL,
    parent_node_id text,
    source jsonb,
    content_hash text NOT NULL,
    node_status text NOT NULL,
    metadata jsonb NOT NULL,
    source_ordinal integer NOT NULL CHECK (source_ordinal >= 0),
    PRIMARY KEY (dataset_id, node_id),
    UNIQUE (dataset_id, document_id, source_ordinal),
    FOREIGN KEY (dataset_id, parent_node_id)
        REFERENCES traffic_rag.legal_node(dataset_id, node_id)
        DEFERRABLE INITIALLY DEFERRED
);

CREATE TABLE traffic_rag.legal_node_child (
    dataset_id text NOT NULL,
    parent_node_id text NOT NULL,
    child_node_id text NOT NULL,
    ordinal integer NOT NULL CHECK (ordinal >= 0),
    PRIMARY KEY (dataset_id, parent_node_id, ordinal),
    UNIQUE (dataset_id, parent_node_id, child_node_id),
    FOREIGN KEY (dataset_id, parent_node_id)
        REFERENCES traffic_rag.legal_node(dataset_id, node_id) ON DELETE CASCADE,
    FOREIGN KEY (dataset_id, child_node_id)
        REFERENCES traffic_rag.legal_node(dataset_id, node_id) ON DELETE CASCADE
);

CREATE TABLE traffic_rag.retrieval_passage (
    dataset_id text NOT NULL REFERENCES traffic_rag.retrieval_dataset(dataset_id) ON DELETE CASCADE,
    passage_id text NOT NULL,
    strategy text NOT NULL CHECK (strategy IN ('B1_article', 'B4e_document_article_clause_point')),
    document_id text NOT NULL,
    primary_node_id text NOT NULL,
    article_node_id text NOT NULL,
    hierarchy jsonb NOT NULL,
    index_text text NOT NULL CHECK (length(btrim(index_text)) > 0),
    evidence_text text NOT NULL CHECK (length(btrim(evidence_text)) > 0),
    display_text text NOT NULL CHECK (length(btrim(display_text)) > 0),
    token_count_index integer NOT NULL CHECK (token_count_index > 0),
    token_count_evidence integer NOT NULL CHECK (token_count_evidence > 0),
    passage_content_hash text NOT NULL,
    index_text_sha256 text NOT NULL CHECK (length(index_text_sha256) = 64),
    embedding_input_sha256 text NOT NULL CHECK (length(embedding_input_sha256) = 64),
    PRIMARY KEY (dataset_id, passage_id),
    FOREIGN KEY (dataset_id, primary_node_id)
        REFERENCES traffic_rag.legal_node(dataset_id, node_id),
    FOREIGN KEY (dataset_id, article_node_id)
        REFERENCES traffic_rag.legal_node(dataset_id, node_id)
);

CREATE INDEX retrieval_passage_strategy_article
    ON traffic_rag.retrieval_passage (dataset_id, strategy, article_node_id, passage_id);

CREATE TABLE traffic_rag.passage_node_link (
    dataset_id text NOT NULL,
    passage_id text NOT NULL,
    relation_type text NOT NULL CHECK (relation_type IN ('source', 'index', 'context', 'citation')),
    node_id text NOT NULL,
    ordinal integer NOT NULL CHECK (ordinal >= 0),
    PRIMARY KEY (dataset_id, passage_id, relation_type, ordinal),
    UNIQUE (dataset_id, passage_id, relation_type, node_id),
    FOREIGN KEY (dataset_id, passage_id)
        REFERENCES traffic_rag.retrieval_passage(dataset_id, passage_id) ON DELETE CASCADE,
    FOREIGN KEY (dataset_id, node_id)
        REFERENCES traffic_rag.legal_node(dataset_id, node_id)
);

CREATE TABLE traffic_rag.passage_embedding (
    dataset_id text NOT NULL,
    passage_id text NOT NULL,
    model_name text NOT NULL,
    model_revision text NOT NULL,
    dimensions integer NOT NULL CHECK (dimensions = 1024),
    normalized boolean NOT NULL CHECK (normalized),
    embedding_input_sha256 text NOT NULL CHECK (length(embedding_input_sha256) = 64),
    embedding vector(1024) NOT NULL,
    status text NOT NULL DEFAULT 'READY' CHECK (status IN ('READY', 'FAILED')),
    created_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (dataset_id, passage_id, model_name, model_revision),
    FOREIGN KEY (dataset_id, passage_id)
        REFERENCES traffic_rag.retrieval_passage(dataset_id, passage_id) ON DELETE CASCADE
);

CREATE INDEX passage_embedding_lookup
    ON traffic_rag.passage_embedding (dataset_id, model_name, model_revision, passage_id)
    WHERE status = 'READY';

GRANT USAGE ON SCHEMA traffic_rag TO traffic_rag_ingest, traffic_rag_runtime;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA traffic_rag TO traffic_rag_ingest;
GRANT SELECT ON ALL TABLES IN SCHEMA traffic_rag TO traffic_rag_runtime;
ALTER DEFAULT PRIVILEGES FOR ROLE traffic_rag_owner IN SCHEMA traffic_rag
    GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO traffic_rag_ingest;
ALTER DEFAULT PRIVILEGES FOR ROLE traffic_rag_owner IN SCHEMA traffic_rag
    GRANT SELECT ON TABLES TO traffic_rag_runtime;
