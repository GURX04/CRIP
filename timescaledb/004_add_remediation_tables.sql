-- =============================================================================
-- timescaledb/004_add_remediation_tables.sql
-- CRIP Live Automated Remediation Actions & Counterfactual Impact Results
-- =============================================================================

CREATE TABLE IF NOT EXISTS remediation_actions (
    created_at          TIMESTAMPTZ         NOT NULL DEFAULT NOW(),
    action_id           TEXT                NOT NULL,
    proposal_id         TEXT                NOT NULL,
    detection_id        TEXT                NOT NULL,
    run_id              TEXT                NOT NULL,
    service             TEXT                NOT NULL,
    action_type         TEXT                NOT NULL,
    mode                TEXT                NOT NULL, -- 'DRY_RUN' | 'ENFORCE'
    params              JSONB               NOT NULL DEFAULT '{}'::jsonb,
    status              TEXT                NOT NULL, -- 'proposed' | 'pending_approval' | 'applied' | 'blocked_by_guardrail' | 'rolled_back' | 'rejected'
    reason              TEXT                NOT NULL,
    guardrail_log       JSONB               NOT NULL DEFAULT '{}'::jsonb,
    sim_applied_ts      TIMESTAMPTZ,
    sim_rollback_ts     TIMESTAMPTZ,
    wall_applied_ms     BIGINT,
    wall_rollback_ms    BIGINT,
    CONSTRAINT pk_remediation_actions PRIMARY KEY (created_at, action_id)
);

SELECT create_hypertable(
    'remediation_actions',
    'created_at',
    chunk_time_interval => INTERVAL '1 day',
    if_not_exists       => TRUE
);

CREATE INDEX IF NOT EXISTS idx_remediation_actions_service
    ON remediation_actions (service, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_remediation_actions_proposal
    ON remediation_actions (proposal_id, created_at DESC);


CREATE TABLE IF NOT EXISTS remediation_results (
    calculated_at               TIMESTAMPTZ         NOT NULL DEFAULT NOW(),
    result_id                   TEXT                NOT NULL,
    action_id                   TEXT                NOT NULL,
    proposal_id                 TEXT                NOT NULL,
    service                     TEXT                NOT NULL,
    actual_cost_usd             DOUBLE PRECISION    NOT NULL,
    counterfactual_cost_usd     DOUBLE PRECISION    NOT NULL,
    estimated_savings_usd       DOUBLE PRECISION    NOT NULL,
    legitimate_requests_dropped DOUBLE PRECISION    NOT NULL DEFAULT 0,
    collateral_penalty_usd      DOUBLE PRECISION    NOT NULL DEFAULT 0,
    net_benefit_usd             DOUBLE PRECISION    NOT NULL,
    estimation_method           TEXT                NOT NULL,
    CONSTRAINT pk_remediation_results PRIMARY KEY (calculated_at, result_id)
);

SELECT create_hypertable(
    'remediation_results',
    'calculated_at',
    chunk_time_interval => INTERVAL '1 day',
    if_not_exists       => TRUE
);

CREATE INDEX IF NOT EXISTS idx_remediation_results_action
    ON remediation_results (action_id, calculated_at DESC);

