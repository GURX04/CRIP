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


@task
def up(c):
    """Start Docker Compose stack in detached mode."""
    print(">>> [crip] Starting Docker services (LocalStack, Target App, Prometheus, Node Exporter, TimescaleDB)...")
    c.run("docker compose up -d", pty=False)
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
    c.run(f"docker compose down{v_flag}", pty=False)
    print("[OK] Containers stopped.")


@task
def status(c):
    """Check health and container status of CRIP services."""
    c.run("docker compose ps", pty=False)


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


@task(help={"date": "Report date YYYY-MM-DD", "interval": "Minutes per interval (default 5)"})
def cur(c, date="2026-10-26", interval=5):
    """Generate synthetic AWS CUR CSV (reports/synthetic_cur.csv)."""
    print(f">>> [crip] Generating synthetic AWS CUR for date={date}, interval={interval}min...")
    c.run(f'"{PYTHON}" generate_cur.py --date {date} --interval-min {interval}', pty=False)


@task(help={"skip_prom": "Skip querying Prometheus if it is not running"})
def ingest(c, skip_prom=False):
    """Bulk-insert CUR CSV and active Prometheus telemetry into TimescaleDB."""
    cur_file = ROOT_DIR / "reports" / "synthetic_cur.csv"
    if not cur_file.exists():
        print(f"[*] CUR file not found at {cur_file}. Generating first...")
        cur(c)

    prom_flag = " --skip-prom" if skip_prom else ""
    print(">>> [crip] Running ETL pipeline (ingest_to_timescale.py)...")
    c.run(f'"{PYTHON}" ingest_to_timescale.py{prom_flag}', pty=False)


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
    c.run(f'"{PYTHON}" anomaly_detection.py --ci {ci}{csv_flag}{wh_flag}', pty=False)


@task(help={"from_csv": "Run offline directly on CSV without TimescaleDB"})
def detect_v2(c, from_csv=False):
    """Run Phase 3 v2 Ensemble anomaly detection (Z-score + IF + Prophet)."""
    csv_flag = ' --from-csv reports/synthetic_cur.csv' if from_csv else ''
    print(">>> [crip] Running Ensemble anomaly detection v2...")
    c.run(f'"{PYTHON}" anomaly_detection_v2.py{csv_flag}', pty=False)


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
