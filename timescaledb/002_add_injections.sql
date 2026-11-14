-- =============================================================================
-- timescaledb/002_add_injections.sql
-- CRIP Phase 4 — Ground-Truth Injection Table
-- =============================================================================
-- Apply manually or via your migration runner:
--   psql -h localhost -U crip_user -d crip -f timescaledb/002_add_injections.sql
-- =============================================================================

-- ---------------------------------------------------------------------------
-- injections
--   Stores every anomaly injection as ground truth for offline evaluation.
--   One row per injection (upserted on conflict to update sim_end_ts etc.).
--
--   NOTE: This is NOT a hypertable — injection cardinality is low and we
--   want simple UPSERT semantics.  If you expect millions of rows, convert
--   to a hypertable on sim_start_ts.
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS injections (
    -- Identity
    injection_id    TEXT            NOT NULL PRIMARY KEY,
    run_id          TEXT            NOT NULL,

    -- Injection description
    type            TEXT            NOT NULL,
    service         TEXT            NOT NULL,
    magnitude       DOUBLE PRECISION NOT NULL,
    duration_min    DOUBLE PRECISION NOT NULL,

    -- Simulated time window
    sim_start_ts    TIMESTAMPTZ,
    sim_end_ts      TIMESTAMPTZ,

    -- Real-wall-clock timestamps (epoch ms)
    wall_start_ms   BIGINT,
    wall_end_ms     BIGINT,

    -- Full injection parameters as JSONB for flexibility
    params          JSONB,

    -- Audit
    created_at      TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

-- Index: look up all injections for a given run
CREATE INDEX IF NOT EXISTS idx_injections_run_id
    ON injections (run_id, sim_start_ts DESC);

-- Index: filter by service and type for detection evaluation
CREATE INDEX IF NOT EXISTS idx_injections_service_type
    ON injections (service, type, sim_start_ts DESC);

-- Trigger: auto-update updated_at on upsert
CREATE OR REPLACE FUNCTION _update_injections_updated_at()
RETURNS TRIGGER LANGUAGE plpgsql AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$;

DROP TRIGGER IF EXISTS trg_injections_updated_at ON injections;
CREATE TRIGGER trg_injections_updated_at
    BEFORE UPDATE ON injections
    FOR EACH ROW EXECUTE FUNCTION _update_injections_updated_at();

COMMENT ON TABLE injections IS
    'Ground-truth record of every anomaly injection made during a generator '
    'run.  Used by offline evaluation scripts and RCA experiments.';

COMMENT ON COLUMN injections.params IS
    'Full JSON payload of the injection request including ramp_min and any '
    'other optional parameters.';

