-- =============================================================================
-- timescaledb/add_anomaly_table.sql
-- CRIP Phase 3 Upgrade — Persistent Anomaly Storage
-- =============================================================================
-- Run this after init.sql has already created the base schema.
-- Safe to re-run: all objects use IF NOT EXISTS.
--
-- Execute via:
--   docker exec -i timescaledb psql -U crip_user -d crip \
--       < timescaledb/add_anomaly_table.sql
-- =============================================================================

-- ---------------------------------------------------------------------------
-- 1. detected_anomalies hypertable
--    Stores every flagged cost anomaly with ensemble vote counts,
--    severity scores, top contributing service, and RCA findings.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS detected_anomalies (
    -- Time dimension (hypertable key)
    window_start            TIMESTAMPTZ         NOT NULL,
    window_end              TIMESTAMPTZ         NOT NULL,

    -- Detection metadata
    detected_at             TIMESTAMPTZ         NOT NULL DEFAULT NOW(),
    run_id                  TEXT,               -- UUID per script run (dedup across reruns)

    -- Cost measurements
    actual_cost             DOUBLE PRECISION    NOT NULL,
    expected_cost           DOUBLE PRECISION,   -- Prophet yhat
    cost_upper_bound        DOUBLE PRECISION,   -- Prophet yhat_upper
    cost_delta              DOUBLE PRECISION,   -- actual - upper_bound
    pct_over_bound          DOUBLE PRECISION,   -- (cost_delta / upper_bound) * 100

    -- Ensemble detection
    vote_count              SMALLINT            NOT NULL DEFAULT 0,
    methods_flagged         TEXT[]              NOT NULL DEFAULT '{}',
    --   e.g. ARRAY['zscore','isolation_forest','prophet']

    -- Severity (composite 0-1 score)
    severity_score          DOUBLE PRECISION,

    -- Root Cause: Service Contribution
    top_service             TEXT,               -- e.g. 'AmazonApiGateway'
    top_service_cost        DOUBLE PRECISION,
    top_service_pct         DOUBLE PRECISION,   -- % of window cost from top service

    -- Root Cause: Lag-Aware Metric Correlation
    top_correlated_metric   TEXT,               -- e.g. 'cpu_usage_rate_1m'
    top_lag_minutes         INT,                -- negative = metric leads cost
    top_correlation         DOUBLE PRECISION,   -- Pearson r at top_lag

    -- Root Cause: Granger Causality
    granger_metric          TEXT,               -- metric that Granger-causes cost
    granger_p_value         DOUBLE PRECISION,   -- p < 0.05 is significant
    granger_max_lag         INT,                -- max lag used in test

    -- Human feedback (for future false-positive reduction)
    is_false_positive       BOOLEAN             DEFAULT FALSE,
    reviewed_by             TEXT,
    notes                   TEXT,

    -- Deduplication
    CONSTRAINT uq_anomaly_window UNIQUE (window_start, run_id)
);

-- Convert to hypertable with 1-day chunks (anomaly data volume is low)
SELECT create_hypertable(
    'detected_anomalies',
    'window_start',
    chunk_time_interval => INTERVAL '1 day',
    if_not_exists       => TRUE
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_anomaly_severity
    ON detected_anomalies (severity_score DESC, window_start DESC);

CREATE INDEX IF NOT EXISTS idx_anomaly_service
    ON detected_anomalies (top_service, window_start DESC);

CREATE INDEX IF NOT EXISTS idx_anomaly_run
    ON detected_anomalies (run_id, window_start);

COMMENT ON TABLE detected_anomalies IS
    'Persistent store for cost anomalies detected by anomaly_detection_v2.py. '
    'Includes ensemble vote counts, severity scores, and RCA findings.';

-- ---------------------------------------------------------------------------
-- 2. View: unreviewed high-severity anomalies
-- ---------------------------------------------------------------------------
CREATE OR REPLACE VIEW anomalies_to_review AS
SELECT
    window_start,
    window_end,
    ROUND(actual_cost::NUMERIC, 6)      AS actual_usd,
    ROUND(cost_delta::NUMERIC, 6)       AS delta_usd,
    ROUND(pct_over_bound::NUMERIC, 2)   AS pct_over,
    ROUND(severity_score::NUMERIC, 3)   AS severity,
    vote_count,
    methods_flagged,
    top_service,
    top_correlated_metric,
    top_lag_minutes,
    granger_metric,
    ROUND(granger_p_value::NUMERIC, 4)  AS granger_p,
    is_false_positive,
    detected_at
FROM detected_anomalies
WHERE is_false_positive = FALSE
ORDER BY severity_score DESC, window_start;

COMMENT ON VIEW anomalies_to_review IS
    'All non-dismissed anomalies, sorted by severity. Used for triage dashboard.';

-- ---------------------------------------------------------------------------
-- 3. Confirm setup
-- ---------------------------------------------------------------------------
DO $$
BEGIN
    RAISE NOTICE 'detected_anomalies hypertable created successfully.';
    RAISE NOTICE 'View anomalies_to_review is ready.';
END $$;
