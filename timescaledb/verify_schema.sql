-- =============================================================================
-- verify_schema.sql
-- CRIP Phase 2 — Ad-hoc verification queries
-- Run via: docker exec -it timescaledb psql -U crip_user -d crip -f /verify_schema.sql
-- Or pipe: cat timescaledb/verify_schema.sql | docker exec -i timescaledb psql -U crip_user -d crip
-- =============================================================================

\echo ''
\echo '============================================================'
\echo ' CRIP Phase 2 — Schema & Data Verification'
\echo '============================================================'

-- 1. Confirm TimescaleDB extension is loaded
\echo ''
\echo '--- 1. Extensions ---'
SELECT name, default_version, installed_version
FROM pg_available_extensions
WHERE name = 'timescaledb';

-- 2. List hypertables
\echo ''
\echo '--- 2. Hypertables ---'
SELECT
    hypertable_name,
    num_chunks,
    compression_enabled,
    tablespaces
FROM timescaledb_information.hypertables
ORDER BY hypertable_name;

-- 3. Hypertable chunks detail
\echo ''
\echo '--- 3. Chunks (aws_cur_records) ---'
SELECT
    chunk_name,
    range_start::TEXT,
    range_end::TEXT,
    is_compressed
FROM timescaledb_information.chunks
WHERE hypertable_name = 'aws_cur_records'
ORDER BY range_start
LIMIT 10;

\echo ''
\echo '--- 3b. Chunks (system_telemetry) ---'
SELECT
    chunk_name,
    range_start::TEXT,
    range_end::TEXT,
    is_compressed
FROM timescaledb_information.chunks
WHERE hypertable_name = 'system_telemetry'
ORDER BY range_start
LIMIT 10;

-- 4. CUR row counts and cost by service
\echo ''
\echo '--- 4. CUR Records: cost by service ---'
SELECT
    product_code,
    COUNT(*)                          AS row_count,
    ROUND(SUM(usage_amount)::NUMERIC, 4) AS total_usage,
    ROUND(SUM(unblended_cost)::NUMERIC, 6) AS total_cost_usd
FROM aws_cur_records
GROUP BY product_code
ORDER BY total_cost_usd DESC;

-- 5. CUR time range
\echo ''
\echo '--- 5. CUR Time Range ---'
SELECT
    MIN(usage_start) AS earliest,
    MAX(usage_end)   AS latest,
    COUNT(*)         AS total_rows
FROM aws_cur_records;

-- 6. Top 5 most expensive intervals (spike detection)
\echo ''
\echo '--- 6. Top 5 Most Expensive 5-Minute Intervals ---'
SELECT
    usage_start,
    usage_end,
    ROUND(SUM(unblended_cost)::NUMERIC, 6) AS interval_cost_usd
FROM aws_cur_records
GROUP BY usage_start, usage_end
ORDER BY interval_cost_usd DESC
LIMIT 5;

-- 7. Hourly cost rollup via view
\echo ''
\echo '--- 7. Hourly Cost Rollup (cur_cost_by_service view, first 5 hours) ---'
SELECT
    hour_bucket,
    product_code,
    ROUND(total_unblended_cost::NUMERIC, 6) AS cost_usd,
    line_item_count
FROM cur_cost_by_service
ORDER BY hour_bucket, cost_usd DESC
LIMIT 20;

-- 8. System telemetry
\echo ''
\echo '--- 8. System Telemetry: rows by metric ---'
SELECT
    metric_name,
    COUNT(*)            AS sample_count,
    MIN(value)          AS min_val,
    ROUND(AVG(value)::NUMERIC, 6) AS avg_val,
    MAX(value)          AS max_val,
    MIN(ts)::TEXT       AS earliest,
    MAX(ts)::TEXT       AS latest
FROM system_telemetry
GROUP BY metric_name
ORDER BY metric_name;

-- 9. Index summary
\echo ''
\echo '--- 9. Indexes on CRIP tables ---'
SELECT
    tablename,
    indexname,
    indexdef
FROM pg_indexes
WHERE tablename IN ('aws_cur_records', 'system_telemetry')
  AND schemaname = 'public'
ORDER BY tablename, indexname;

\echo ''
\echo '============================================================'
\echo ' Verification complete.'
\echo '============================================================'
