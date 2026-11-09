-- =============================================================================
-- timescaledb/init.sql
-- CRIP Phase 2 — TimescaleDB Schema Initialisation
-- =============================================================================
-- This script runs automatically when the timescaledb container starts with
-- an empty data volume (via /docker-entrypoint-initdb.d/).
--
-- Objects created
-- ---------------
--  Extension : timescaledb
--  Tables    : aws_cur_records   (hypertable, 1-day chunks)
--              system_telemetry  (hypertable, 1-hour chunks)
--  Indexes   : covering indexes for analytical queries
--  View      : cur_cost_by_service  (convenience rollup)
-- =============================================================================

-- ---------------------------------------------------------------------------
-- 0. TimescaleDB extension
-- ---------------------------------------------------------------------------
CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;

-- ---------------------------------------------------------------------------
-- 1. aws_cur_records
--    Stores every line item from the synthetic AWS Cost & Usage Report.
--    Partitioned by `usage_start` using TimescaleDB hypertable chunking.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS aws_cur_records (
    -- Time dimension (required for hypertable)
    usage_start             TIMESTAMPTZ         NOT NULL,
    usage_end               TIMESTAMPTZ         NOT NULL,

    -- AWS CUR identity fields
    line_item_id            TEXT                NOT NULL,
    invoice_id              TEXT,
    bill_type               TEXT,
    payer_account_id        TEXT,

    -- Resource classification
    product_code            TEXT                NOT NULL,  -- e.g. AmazonApiGateway
    usage_type              TEXT,                          -- e.g. us-east-1-ApiGatewayRequest
    operation               TEXT,
    availability_zone       TEXT,
    resource_id             TEXT,

    -- Measurement
    usage_amount            DOUBLE PRECISION    NOT NULL DEFAULT 0,
    unblended_rate          DOUBLE PRECISION,
    unblended_cost          DOUBLE PRECISION    NOT NULL DEFAULT 0,
    blended_rate            DOUBLE PRECISION,
    blended_cost            DOUBLE PRECISION,
    currency_code           TEXT                NOT NULL DEFAULT 'USD',

    -- Product metadata
    product_name            TEXT,
    service_code            TEXT,
    region                  TEXT,

    -- Resource tags (flattened)
    tag_project             TEXT,
    tag_environment         TEXT,
    tag_team                TEXT,

    -- Ingestion metadata
    ingested_at             TIMESTAMPTZ         NOT NULL DEFAULT NOW(),

    -- Deduplication key
    CONSTRAINT uq_cur_line_item UNIQUE (usage_start, line_item_id)
);

-- Convert to hypertable: 1-day chunks align with CUR billing granularity.
-- Each chunk covers one calendar day of billing data.
SELECT create_hypertable(
    'aws_cur_records',
    'usage_start',
    chunk_time_interval => INTERVAL '1 day',
    if_not_exists       => TRUE
);

-- Analytical indexes
-- (a) Cost queries grouped by service over a time range
CREATE INDEX IF NOT EXISTS idx_cur_product_time
    ON aws_cur_records (product_code, usage_start DESC);

-- (b) Cost queries grouped by resource
CREATE INDEX IF NOT EXISTS idx_cur_resource_time
    ON aws_cur_records (resource_id, usage_start DESC);

-- (c) Full-text on usage_type for ad-hoc filtering
CREATE INDEX IF NOT EXISTS idx_cur_usage_type
    ON aws_cur_records (usage_type, usage_start DESC);

-- TimescaleDB compression policy (optional, enable after data is loaded)
-- ALTER TABLE aws_cur_records SET (
--     timescaledb.compress,
--     timescaledb.compress_segmentby = 'product_code'
-- );
-- SELECT add_compression_policy('aws_cur_records', INTERVAL '7 days');

COMMENT ON TABLE aws_cur_records IS
    'AWS Cost & Usage Report line items ingested from synthetic_cur.csv. '
    'Hypertable partitioned by usage_start in 1-day chunks.';

-- ---------------------------------------------------------------------------
-- 2. system_telemetry
--    Stores Prometheus metric samples pulled by ingest_to_timescale.py.
--    Partitioned by `ts` using 1-hour chunks — high-frequency data.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS system_telemetry (
    -- Time dimension (required for hypertable)
    ts                      TIMESTAMPTZ         NOT NULL,

    -- Prometheus identity
    metric_name             TEXT                NOT NULL,  -- e.g. node_cpu_seconds_total
    job                     TEXT,                          -- Prometheus job label
    instance                TEXT,                          -- target host:port
    service                 TEXT,                          -- custom crip label

    -- Dimension labels (serialised as key=value pairs for flexibility)
    labels                  JSONB,

    -- Measurement
    value                   DOUBLE PRECISION    NOT NULL,

    -- Ingestion metadata
    ingested_at             TIMESTAMPTZ         NOT NULL DEFAULT NOW()
);

-- Convert to hypertable: 1-hour chunks for high-cardinality metric data.
SELECT create_hypertable(
    'system_telemetry',
    'ts',
    chunk_time_interval => INTERVAL '1 hour',
    if_not_exists       => TRUE
);

-- Analytical indexes
-- (a) Primary query pattern: metric over time for a specific instance
CREATE INDEX IF NOT EXISTS idx_telemetry_metric_ts
    ON system_telemetry (metric_name, ts DESC);

-- (b) Instance / service drill-down
CREATE INDEX IF NOT EXISTS idx_telemetry_instance_ts
    ON system_telemetry (instance, ts DESC);

-- (c) JSONB labels for ad-hoc label filtering
CREATE INDEX IF NOT EXISTS idx_telemetry_labels
    ON system_telemetry USING GIN (labels);

-- TimescaleDB compression policy for telemetry (higher retention, smaller footprint)
-- ALTER TABLE system_telemetry SET (
--     timescaledb.compress,
--     timescaledb.compress_segmentby = 'metric_name, instance'
-- );
-- SELECT add_compression_policy('system_telemetry', INTERVAL '24 hours');

COMMENT ON TABLE system_telemetry IS
    'Prometheus metric samples collected by ingest_to_timescale.py. '
    'Hypertable partitioned by ts in 1-hour chunks.';

-- ---------------------------------------------------------------------------
-- 3. Convenience view: per-service cost rollup over the full CUR dataset
-- ---------------------------------------------------------------------------
CREATE OR REPLACE VIEW cur_cost_by_service AS
SELECT
    product_code,
    product_name,
    DATE_TRUNC('hour', usage_start)     AS hour_bucket,
    COUNT(*)                            AS line_item_count,
    SUM(usage_amount)                   AS total_usage_amount,
    SUM(unblended_cost)                 AS total_unblended_cost,
    AVG(unblended_cost)                 AS avg_cost_per_item,
    MAX(unblended_cost)                 AS max_cost_per_item,
    currency_code
FROM aws_cur_records
GROUP BY
    product_code,
    product_name,
    DATE_TRUNC('hour', usage_start),
    currency_code
ORDER BY
    hour_bucket,
    total_unblended_cost DESC;

COMMENT ON VIEW cur_cost_by_service IS
    'Hourly cost rollup per AWS service. Useful for cost trend analysis.';

-- ---------------------------------------------------------------------------
-- 4. Convenience view: recent system telemetry (last 1 hour)
-- ---------------------------------------------------------------------------
CREATE OR REPLACE VIEW recent_telemetry AS
SELECT
    metric_name,
    instance,
    service,
    ts,
    value,
    labels
FROM system_telemetry
WHERE ts >= NOW() - INTERVAL '1 hour'
ORDER BY ts DESC;

COMMENT ON VIEW recent_telemetry IS
    'Last 60 minutes of Prometheus metric samples.';

-- ---------------------------------------------------------------------------
-- 5. Verification query (runs at init time, output visible in container logs)
-- ---------------------------------------------------------------------------
DO $$
DECLARE
    v_cur_chunks  INT;
    v_tel_chunks  INT;
BEGIN
    SELECT COUNT(*) INTO v_cur_chunks
    FROM timescaledb_information.chunks
    WHERE hypertable_name = 'aws_cur_records';

    SELECT COUNT(*) INTO v_tel_chunks
    FROM timescaledb_information.chunks
    WHERE hypertable_name = 'system_telemetry';

    RAISE NOTICE '----------------------------------------------------';
    RAISE NOTICE 'CRIP Phase 2 schema initialised successfully.';
    RAISE NOTICE '  aws_cur_records  hypertable chunks : %', v_cur_chunks;
    RAISE NOTICE '  system_telemetry hypertable chunks : %', v_tel_chunks;
    RAISE NOTICE '----------------------------------------------------';
END $$;
