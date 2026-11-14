-- =============================================================================
-- timescaledb/003_add_pipeline_tables.sql
-- CRIP Streaming Detection Pipeline Hypertables
-- =============================================================================

CREATE TABLE IF NOT EXISTS cost_events (
    sim_ts              TIMESTAMPTZ         NOT NULL,
    wall_ts_ms          BIGINT              NOT NULL,
    run_id              TEXT                NOT NULL,
    event_id            TEXT                NOT NULL,
    tick_index          INT                 NOT NULL,
    service             TEXT                NOT NULL,
    request_count       DOUBLE PRECISION    NOT NULL DEFAULT 0,
    usage_amount        DOUBLE PRECISION    NOT NULL DEFAULT 0,
    unblended_cost      DOUBLE PRECISION    NOT NULL DEFAULT 0,
    unit_price          DOUBLE PRECISION    NOT NULL DEFAULT 0,
    created_at          TIMESTAMPTZ         NOT NULL DEFAULT NOW(),
    CONSTRAINT pk_cost_events PRIMARY KEY (sim_ts, event_id)
);

SELECT create_hypertable(
    'cost_events',
    'sim_ts',
    chunk_time_interval => INTERVAL '1 day',
    if_not_exists       => TRUE
);

CREATE INDEX IF NOT EXISTS idx_cost_events_service_sim_ts
    ON cost_events (service, sim_ts DESC);

CREATE INDEX IF NOT EXISTS idx_cost_events_run_id
    ON cost_events (run_id, sim_ts DESC);


CREATE TABLE IF NOT EXISTS detections (
    detect_ts               TIMESTAMPTZ         NOT NULL DEFAULT NOW(),
    detection_id            TEXT                NOT NULL,
    run_id                  TEXT                NOT NULL,
    service                 TEXT                NOT NULL,
    sim_ts                  TIMESTAMPTZ         NOT NULL,
    detect_wall_ts_ms       BIGINT              NOT NULL,
    first_event_wall_ts_ms  BIGINT              NOT NULL,
    severity                DOUBLE PRECISION    NOT NULL,
    detector_votes          JSONB               NOT NULL DEFAULT '[]'::jsonb,
    rca                     JSONB               NOT NULL DEFAULT '{}'::jsonb,
    status                  TEXT                NOT NULL DEFAULT 'open',
    CONSTRAINT pk_detections PRIMARY KEY (detect_ts, detection_id)
);

SELECT create_hypertable(
    'detections',
    'detect_ts',
    chunk_time_interval => INTERVAL '1 day',
    if_not_exists       => TRUE
);

CREATE INDEX IF NOT EXISTS idx_detections_service_sim_ts
    ON detections (service, sim_ts DESC);

CREATE INDEX IF NOT EXISTS idx_detections_run_id
    ON detections (run_id, detect_ts DESC);


CREATE TABLE IF NOT EXISTS detector_scores (
    sim_ts          TIMESTAMPTZ         NOT NULL,
    run_id          TEXT                NOT NULL,
    service         TEXT                NOT NULL,
    cost            DOUBLE PRECISION    NOT NULL DEFAULT 0,
    expected_cost   DOUBLE PRECISION    NOT NULL DEFAULT 0,
    score_zscore    DOUBLE PRECISION    NOT NULL DEFAULT 0,
    score_ewma      DOUBLE PRECISION    NOT NULL DEFAULT 0,
    score_cpr       DOUBLE PRECISION    NOT NULL DEFAULT 0
);

SELECT create_hypertable(
    'detector_scores',
    'sim_ts',
    chunk_time_interval => INTERVAL '1 hour',
    if_not_exists       => TRUE
);

CREATE INDEX IF NOT EXISTS idx_detector_scores_service_sim_ts
    ON detector_scores (service, sim_ts DESC);

SELECT add_retention_policy(
    'detector_scores',
    INTERVAL '7 days',
    if_not_exists => TRUE
);

