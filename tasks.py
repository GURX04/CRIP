"""
tasks.py
========
CRIP Project Task Runner (Invoke)

Provides custom CLI commands via `crip <command>`:
  crip up                 - Start Docker Compose stack in detached mode
  crip down               - Tear down Docker Compose containers
  crip cur                - Generate 24h synthetic AWS CUR CSV
  crip ingest             - Bulk ingest CUR and Prometheus metrics to TimescaleDB
  crip detect             - Run Prophet ML anomaly detection
  crip detect-v2          - Run upgraded ensemble anomaly detection (Z-score + IF + Prophet)
  crip deploy-remediation - Package and deploy Lambda remediation handler to LocalStack
  crip pipeline           - Run complete end-to-end lifecycle (up -> ingest -> deploy -> detect)
  crip status             - Check Docker services health status
  crip gen-start          - Start the continuous event generator (foreground)
  crip gen-inject         - Inject an anomaly via the generator API
  crip gen-reset          - Reset the generator run (new run_id, clear injections)
  crip detect-live        - Start the streaming detector worker
  crip eval-live          - Start the live detection evaluator
  crip dash               - Start the live demo dashboard (port 8501)
  crip dash-record        - Record stream session to JSONL
  crip dash-replay        - Open recorded session in replay mode
"""

import os
import sys
import time
import zipfile
from pathlib import Path
from invoke import task

ROOT_DIR = Path(__file__).parent.resolve()
PYTHON = sys.executable
LOCALSTACK_URL = os.getenv("AWS_ENDPOINT_URL", "http://localhost:4566")
FUNCTION_NAME = "crip-cost-remediator"


@task(help={"build": "Rebuild container images before launching (--build)"})
def up(c, build=False):
    """Start Docker Compose stack in detached mode."""
    b_flag = " --build" if build else ""
    print(f">>> [crip] Starting Docker services (LocalStack, Target App, Prometheus, Node Exporter, TimescaleDB, Generator, Detector, Dashboard)...{b_flag}")
    c.run(f"docker compose up -d{b_flag}", pty=False, in_stream=False)
    print("\n[OK] Docker Compose containers launched.")
    print("     * LocalStack  : http://localhost:4566")
    print("     * Target App  : http://localhost:8080/health")
    print("     * Prometheus  : http://localhost:9090")
    print("     * TimescaleDB : localhost:5432 (db=crip, user=crip_user)")


@task(help={"volumes": "Also delete named data volumes (-v)"})
def down(c, volumes=False):
    """Tear down Docker Compose containers."""
    v_flag = " -v" if volumes else ""
    print(f">>> [crip] Stopping Docker Compose stack{v_flag}...")
    c.run(f"docker compose down{v_flag}", pty=False, in_stream=False)
    print("[OK] Containers stopped.")


@task
def status(c):
    """Check health and container status of CRIP services."""
    c.run("docker compose ps", pty=False, in_stream=False)


@task(help={"time": "Duration of test (e.g. 1m, 2m, 10m)", "host": "Target app URL (default http://localhost:8080)"})
def load_test(c, time="2m", host="http://localhost:8080"):
    """Run headless Locust load test and generate reports/report.html."""
    out_dir = ROOT_DIR / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    html_report = out_dir / "report.html"
    print(f">>> [crip] Running Locust load test against {host} for {time}...")
    print(f"[*] Generating HTML report -> {html_report}")
    c.run(
        f'"{PYTHON}" -m locust -f locustfile.py --headless '
        f'--host {host} --run-time {time} '
        f'--html "{html_report}"',
        pty=False,
    )
    print(f"\n[OK] Load test complete! Open report in browser:")
    print(f"     Invoke-Item \"{html_report}\"")


# ---------------------------------------------------------------------------
# Phase 4 — Continuous Event Generator tasks
# ---------------------------------------------------------------------------

@task(help={
    "sink":  "Output sink: redis | stdout | csv (default redis)",
    "seed":  "Random seed (default 42)",
    "speed": "Sim minutes per real second (default 5)",
    "start_ts": "Sim start ISO-8601 UTC timestamp",
    "no_api":   "Disable the FastAPI control server",
})
def gen_start(c, sink="redis", seed=42, speed=5, start_ts="", no_api=False):
    """Start the CRIP continuous event generator (foreground; Ctrl-C to stop)."""
    print(f">>> [crip] Starting continuous event generator (sink={sink}, seed={seed}, speed={speed})...")
    flags = f"--sink {sink} --seed {seed} --speed {speed}"
    if start_ts:
        flags += f" --start-ts {start_ts}"
    if no_api:
        flags += " --no-api"
    c.run(f'"{PYTHON}" -m generator.main {flags}', pty=False, in_stream=False)


@task(help={
    "type":        "Injection type: spike | step_change | slow_drift | cost_only | seasonality_break",
    "service":     "Target service name or 'all'",
    "magnitude":   "Magnitude multiplier (>0)",
    "duration":    "Duration in simulated minutes (default 30)",
    "ramp":        "Ramp-in/out in sim minutes (default 0)",
    "api_url":     "Generator control API URL (default http://localhost:8000)",
})
def gen_inject(c, type, service, magnitude, duration=30, ramp=0, api_url="http://localhost:8000"):
    """Inject an anomaly into the running generator via its control API."""
    import json as _json
    import urllib.request as _urllib

    payload = _json.dumps({
        "type": type,
        "service": service,
        "magnitude": float(magnitude),
        "duration_min": float(duration),
        "ramp_min": float(ramp),
    }).encode()

    req = _urllib.Request(
        f"{api_url}/inject",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with _urllib.urlopen(req, timeout=5) as resp:
            result = _json.loads(resp.read())
        print(f"[OK] Injection scheduled: {_json.dumps(result, indent=2)}")
    except Exception as exc:
        print(f"[ERROR] Could not reach generator API at {api_url}: {exc}")
        print("       Is the generator running?  Try: crip gen-start")


@task(help={
    "seed":    "New seed for the reset run (optional)",
    "api_url": "Generator control API URL (default http://localhost:8000)",
})
def gen_reset(c, seed=None, api_url="http://localhost:8000"):
    """Reset the generator: new run_id, clear all injections."""
    import json as _json
    import urllib.request as _urllib

    payload = _json.dumps({"seed": int(seed) if seed is not None else None}).encode()
    req = _urllib.Request(
        f"{api_url}/control/reset",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with _urllib.urlopen(req, timeout=5) as resp:
            result = _json.loads(resp.read())
        print(f"[OK] Generator reset: {_json.dumps(result, indent=2)}")
    except Exception as exc:
        print(f"[ERROR] Could not reach generator API at {api_url}: {exc}")


# ---------------------------------------------------------------------------
# Phase 5 — Live Dashboard tasks
# ---------------------------------------------------------------------------

@task(help={
    "host":    "Dashboard host (default 0.0.0.0)",
    "port":    "Dashboard port (default 8501)",
    "redis":   "Redis URL (overrides DASH_REDIS_URL)",
    "gen_url": "Generator API URL (overrides DASH_GENERATOR_URL)",
    "replay":  "Path to a .jsonl replay file (activates replay mode)",
    "log_level": "Log level (default INFO)",
})
def dash(c, host="0.0.0.0", port=8501, redis="", gen_url="", replay="", log_level="INFO"):
    """Start the CRIP live demo dashboard (port 8501)."""
    import os as _os
    env_overrides = []
    if redis:    env_overrides.append(f"DASH_REDIS_URL={redis}")
    if gen_url:  env_overrides.append(f"DASH_GENERATOR_URL={gen_url}")
    if replay:   env_overrides.append(f"DASH_REPLAY_FILE={replay}")
    env_str = " ".join(f'$env:{e}' for e in env_overrides)
    print(f">>> [crip] Starting dashboard on http://{host}:{port}/")
    if env_overrides:
        for e in env_overrides:
            k, v = e.split("=", 1)
            _os.environ[k] = v
    c.run(
        f'"{PYTHON}" -m dashboard.main',
        pty=False,
        in_stream=False,
        env={"DASH_HOST": host, "DASH_PORT": str(port), "DASH_LOG_LEVEL": log_level},
    )


@task(help={
    "output":   "Output .jsonl file path (default reports/session.jsonl)",
    "duration": "Recording duration in seconds (default: until Ctrl-C)",
    "redis":    "Redis URL (overrides DASH_REDIS_URL)",
})
def dash_record(c, output="reports/session.jsonl", duration=None, redis=""):
    """Record live Redis streams to a JSONL file for dashboard replay."""
    import os as _os
    if redis:
        _os.environ["DASH_REDIS_URL"] = redis
    dur_flag = f"--duration {duration}" if duration else ""
    print(f">>> [crip] Recording streams to {output} {f'for {duration}s' if duration else '(Ctrl-C to stop)'}...")
    c.run(f'"{PYTHON}" -m dashboard.recorder --output "{output}" {dur_flag}', pty=False, in_stream=False)


@task(help={
    "api_url": "Dashboard backend URL (default http://localhost:8501)",
})
def dash_status(c, api_url="http://localhost:8501"):
    """Show dashboard and pipeline health status."""
    import json as _json, urllib.request as _urllib
    try:
        with _urllib.urlopen(f"{api_url}/health", timeout=5) as resp:
            h = _json.loads(resp.read())
        print(f"[Dashboard Health @ {api_url}]")
        for k, v in h.items():
            if isinstance(v, dict):
                status = "✓" if v.get("ok") else "✗"
                detail = f" ({v.get('detail','')})" if v.get("detail") else ""
                print(f"  {status} {k}{detail}")
            else:
                print(f"  overall: {'✓' if v else '✗'}")
    except Exception as exc:
        print(f"[ERROR] Could not reach dashboard at {api_url}: {exc}")


@task(help={
    "output":   "Output .jsonl replay file",
    "api_url":  "Dashboard API URL",
})
def dash_replay(c, output="reports/session.jsonl", api_url="http://localhost:8501"):
    """Open the dashboard in replay mode using a recorded JSONL session file."""
    import os as _os, shutil as _shutil
    if not _os.path.exists(output):
        print(f"[ERROR] Replay file not found: {output}")
        return
    abs_path = _os.path.abspath(output)
    url = f"{api_url}?replay={abs_path}"
    print(f">>> [crip] Replay mode URL: {url}")
    if _shutil.which("start"):
        c.run(f"start {url}", pty=False, warn=True)
    else:
        print(f"Open in browser: {url}")


@task(help={"output": "Path to .jsonl file", "redis": "Redis URL"})
def dash_record_and_open(c, output="reports/demo.jsonl", redis="redis://localhost:6379"):
    """Record a 60-second session then immediately open it in replay mode."""
    dash_record(c, output=output, duration=60, redis=redis)
    dash_replay(c, output=output)


# ---------------------------------------------------------------------------
# Streaming Detection Pipeline tasks
# ---------------------------------------------------------------------------

@task(help={
    "redis": "Redis URL (overrides PIP_REDIS_URL)",
    "metrics_port": "Prometheus metrics port (default 9200)",
    "log_level": "Log level (default INFO)",
})
def detect_live(c, redis="", metrics_port=9200, log_level="INFO"):
    """Start the real-time streaming detector worker."""
    import os as _os
    if redis:
        _os.environ["PIP_REDIS_URL"] = redis
    if metrics_port:
        _os.environ["PIP_METRICS_PORT"] = str(metrics_port)
    print(f">>> [crip] Starting streaming detector worker (metrics on :{metrics_port})...")
    c.run(f'"{PYTHON}" -m pipeline.main --log-level {log_level}', pty=False, in_stream=False)


@task(help={
    "redis": "Redis URL (overrides PIP_EVAL_REDIS_URL)",
})
def eval_live(c, redis=""):
    """Start the live evaluator process tracking precision, recall, and latency."""
    flags = f' --redis-url "{redis}"' if redis else ''
    print(">>> [crip] Starting live detection evaluator...")
    c.run(f'"{PYTHON}" -m pipeline.evaluator{flags}', pty=False, in_stream=False)


@task(help={"date": "Report date YYYY-MM-DD", "interval": "Minutes per interval (default 5)"})
def cur(c, date="2026-10-26", interval=5):
    """Generate synthetic AWS CUR CSV (reports/synthetic_cur.csv)."""
    print(f">>> [crip] Generating synthetic AWS CUR for date={date}, interval={interval}min...")
    c.run(f'"{PYTHON}" generate_cur.py --date {date} --interval-min {interval}', pty=False, in_stream=False)


@task(help={"skip_prom": "Skip querying Prometheus if it is not running"})
def ingest(c, skip_prom=False):
    """Bulk-insert CUR CSV and active Prometheus telemetry into TimescaleDB."""
    cur_file = ROOT_DIR / "reports" / "synthetic_cur.csv"
    if not cur_file.exists():
        print(f"[*] CUR file not found at {cur_file}. Generating first...")
        cur(c)

    prom_flag = " --skip-prom" if skip_prom else ""
    print(">>> [crip] Running ETL pipeline (ingest_to_timescale.py)...")
    c.run(f'"{PYTHON}" ingest_to_timescale.py{prom_flag}', pty=False, in_stream=False)


@task
def deploy_remediation(c):
    """Zip remediation_handler.py and deploy Lambda + Function URL to LocalStack."""
    print(">>> [crip] Packaging and deploying remediation Lambda to LocalStack...")

    remediation_dir = ROOT_DIR / "remediation"
    remediation_dir.mkdir(parents=True, exist_ok=True)
    handler_file = remediation_dir / "remediation_handler.py"
    zip_path = remediation_dir / "function.zip"

    if not handler_file.exists():
        print(f"[ERROR] Handler not found: {handler_file}")
        return

    # Create zip archive
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(handler_file, arcname="remediation_handler.py")
    print(f"[*] Packaged {zip_path.name} ({zip_path.stat().st_size} bytes)")

    # Prefer python deployer (boto3) if available for cross-platform reliability
    deploy_script = remediation_dir / "deploy_lambda.py"
    if deploy_script.exists():
        c.run(f'"{PYTHON}" "{deploy_script}"', pty=False)
        return

    # Fallback to aws CLI commands
    print("[*] Deploying via aws CLI...")
    role_arn = "arn:aws:iam::000000000000:role/crip-lambda-role"
    c.run(
        f'aws --endpoint-url={LOCALSTACK_URL} iam create-role '
        f'--role-name crip-lambda-role '
        f'--assume-role-policy-document "{{\\"Version\\":\\"2012-10-17\\",\\"Statement\\":[{{\\"Effect\\":\\"Allow\\",\\"Principal\\":{{\\"Service\\":\\"lambda.amazonaws.com\\"}},\\"Action\\":\\"sts:AssumeRole\\"}}]}}" ',
        warn=True,
        pty=False,
    )

    c.run(
        f'aws --endpoint-url={LOCALSTACK_URL} lambda create-function '
        f'--function-name {FUNCTION_NAME} '
        f'--runtime python3.11 '
        f'--role {role_arn} '
        f'--handler remediation_handler.lambda_handler '
        f'--zip-file fileb://"{zip_path}" '
        f'--timeout 15',
        warn=True,
        pty=False,
    )

    c.run(
        f'aws --endpoint-url={LOCALSTACK_URL} lambda create-function-url-config '
        f'--function-name {FUNCTION_NAME} '
        f'--auth-type NONE',
        warn=True,
        pty=False,
    )

    c.run(
        f'aws --endpoint-url={LOCALSTACK_URL} lambda add-permission '
        f'--function-name {FUNCTION_NAME} '
        f'--statement-id FunctionURLAllowPublicAccess '
        f'--action lambda:InvokeFunctionUrl '
        f'--principal "*" '
        f'--function-url-auth-type NONE',
        warn=True,
        pty=False,
    )
    print(f"[OK] Lambda '{FUNCTION_NAME}' deployed to LocalStack.")


@task(help={
    "from_csv": "Run offline directly on CSV without TimescaleDB",
    "webhook_url": "Send dry-run remediation webhooks to this Lambda Function URL",
    "ci": "Confidence interval (default 0.95)"
})
def detect(c, from_csv=False, webhook_url="", ci=0.95):
    """Run Phase 3 Prophet ML anomaly detection."""
    csv_flag = ' --from-csv reports/synthetic_cur.csv' if from_csv else ''
    wh_flag = f' --webhook-url "{webhook_url}"' if webhook_url else ''
    print(f">>> [crip] Running Prophet anomaly detection (ci={ci})...")
    c.run(f'"{PYTHON}" anomaly_detection.py --ci {ci}{csv_flag}{wh_flag}', pty=False, in_stream=False)


@task(help={"from_csv": "Run offline directly on CSV without TimescaleDB"})
def detect_v2(c, from_csv=False):
    """Run Phase 3 v2 Ensemble anomaly detection (Z-score + IF + Prophet)."""
    csv_flag = ' --from-csv reports/synthetic_cur.csv' if from_csv else ''
    print(">>> [crip] Running Ensemble anomaly detection v2...")
    c.run(f'"{PYTHON}" anomaly_detection_v2.py{csv_flag}', pty=False, in_stream=False)


@task(help={"wait_seconds": "Seconds to pause for database/service readiness (default 20)"})
def pipeline(c, wait_seconds=20):
    """Execute end-to-end CRIP lifecycle: up -> wait -> ingest -> deploy_remediation -> detect."""
    print("=" * 70)
    print("  CRIP AUTOMATED END-TO-END PIPELINE")
    print("=" * 70)

    # 1. Start Docker stack
    print("\n--- STEP 1: Launching Docker Stack ---")
    up(c)

    # 2. Wait for DB readiness
    print(f"\n--- STEP 2: Waiting {wait_seconds}s for TimescaleDB and LocalStack to initialize ---")
    for sec in range(wait_seconds, 0, -5):
        print(f"     ... waiting ({sec}s remaining)")
        time.sleep(5)
    print("[*] Services ready.")

    # 3. Ingest CUR & metrics
    print("\n--- STEP 3: Ingesting Data into TimescaleDB ---")
    ingest(c)

    # 4. Deploy remediation Lambda to LocalStack
    print("\n--- STEP 4: Deploying Remediation Lambda to LocalStack ---")
    deploy_remediation(c)

    # 5. Run Anomaly Detection
    print("\n--- STEP 5: Running ML Anomaly Detection & Dispatching Remediation ---")
    detect(c)

    print("\n" + "=" * 70)
    print("  [SUCCESS] CRIP Pipeline completed successfully!")
    print("  * Report Plot : reports/forecast_plot.png")
    print("=" * 70)
