CREATE TABLE IF NOT EXISTS legal_graph_release (
    release_id text PRIMARY KEY,
    status text NOT NULL CHECK (status IN ('BUILDING', 'VALIDATING', 'READY', 'ACTIVE', 'RETIRED', 'FAILED')),
    workspace text NOT NULL UNIQUE,
    canonical_digest char(64) NOT NULL,
    b7_dataset_id text NOT NULL,
    source_manifest_hash char(64) NOT NULL,
    config_hash char(64) NOT NULL,
    extraction_provider text,
    extraction_model text,
    metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE UNIQUE INDEX IF NOT EXISTS uq_legal_graph_one_active
    ON legal_graph_release ((status)) WHERE status = 'ACTIVE';

CREATE TABLE IF NOT EXISTS legal_graph_source_document (
    release_id text NOT NULL REFERENCES legal_graph_release(release_id) ON DELETE CASCADE,
    article_node_id text NOT NULL,
    canonical_document_id text NOT NULL,
    title text NOT NULL,
    text text NOT NULL,
    extraction_text text NOT NULL,
    source_node_ids jsonb NOT NULL,
    node_texts jsonb NOT NULL,
    token_count integer NOT NULL CHECK (token_count > 0),
    content_hash char(64) NOT NULL,
    PRIMARY KEY (release_id, article_node_id)
);

CREATE TABLE IF NOT EXISTS legal_graph_node (
    release_id text NOT NULL REFERENCES legal_graph_release(release_id) ON DELETE CASCADE,
    graph_node_id text NOT NULL,
    node_kind text NOT NULL CHECK (node_kind IN ('canonical', 'semantic', 'authority')),
    node_type text NOT NULL,
    label text NOT NULL,
    document_id text,
    canonical_node_id text,
    properties jsonb NOT NULL DEFAULT '{}'::jsonb,
    PRIMARY KEY (release_id, graph_node_id),
    CHECK (node_kind <> 'canonical' OR canonical_node_id = graph_node_id)
);

CREATE INDEX IF NOT EXISTS ix_legal_graph_node_label
    ON legal_graph_node (release_id, lower(label));
CREATE INDEX IF NOT EXISTS ix_legal_graph_node_type
    ON legal_graph_node (release_id, node_kind, node_type);

CREATE TABLE IF NOT EXISTS legal_graph_edge (
    release_id text NOT NULL REFERENCES legal_graph_release(release_id) ON DELETE CASCADE,
    edge_id text NOT NULL,
    source_node_id text NOT NULL,
    target_node_id text NOT NULL,
    relation_type text NOT NULL,
    description text NOT NULL DEFAULT '',
    confidence double precision NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
    extraction_version text NOT NULL,
    properties jsonb NOT NULL DEFAULT '{}'::jsonb,
    PRIMARY KEY (release_id, edge_id),
    FOREIGN KEY (release_id, source_node_id)
        REFERENCES legal_graph_node(release_id, graph_node_id) ON DELETE CASCADE,
    FOREIGN KEY (release_id, target_node_id)
        REFERENCES legal_graph_node(release_id, graph_node_id) ON DELETE CASCADE,
    CHECK (source_node_id <> target_node_id)
);

CREATE INDEX IF NOT EXISTS ix_legal_graph_edge_source
    ON legal_graph_edge (release_id, source_node_id, relation_type);
CREATE INDEX IF NOT EXISTS ix_legal_graph_edge_target
    ON legal_graph_edge (release_id, target_node_id, relation_type);

CREATE TABLE IF NOT EXISTS legal_graph_provenance (
    release_id text NOT NULL,
    edge_id text NOT NULL,
    ordinal integer NOT NULL CHECK (ordinal >= 0),
    canonical_node_id text NOT NULL,
    evidence_text text NOT NULL,
    start_offset integer NOT NULL CHECK (start_offset >= 0),
    end_offset integer NOT NULL CHECK (end_offset > start_offset),
    PRIMARY KEY (release_id, edge_id, ordinal),
    FOREIGN KEY (release_id, edge_id)
        REFERENCES legal_graph_edge(release_id, edge_id) ON DELETE CASCADE,
    FOREIGN KEY (release_id, canonical_node_id)
        REFERENCES legal_graph_node(release_id, graph_node_id) ON DELETE RESTRICT
);

CREATE TABLE IF NOT EXISTS legal_graph_alias (
    release_id text NOT NULL REFERENCES legal_graph_release(release_id) ON DELETE CASCADE,
    graph_node_id text NOT NULL,
    alias text NOT NULL,
    normalized_alias text NOT NULL,
    review_status text NOT NULL DEFAULT 'approved',
    PRIMARY KEY (release_id, graph_node_id, normalized_alias),
    FOREIGN KEY (release_id, graph_node_id)
        REFERENCES legal_graph_node(release_id, graph_node_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS legal_graph_mirror_map (
    release_id text NOT NULL REFERENCES legal_graph_release(release_id) ON DELETE CASCADE,
    object_kind text NOT NULL CHECK (object_kind IN ('chunk', 'entity', 'edge')),
    legal_object_id text NOT NULL,
    lightrag_object_id text NOT NULL,
    workspace text NOT NULL,
    mirror_status text NOT NULL CHECK (mirror_status IN ('READY', 'FAILED')),
    metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
    PRIMARY KEY (release_id, object_kind, legal_object_id)
);

GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO traffic_rag_lightrag_app;
