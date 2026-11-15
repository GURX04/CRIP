"""
tasks.py
========
CRIP Project Task Runner (Invoke)

Provides custom CLI commands via `crip <command>`:
  crip up                 - Start Docker Compose stack in detached mode
  crip down               - Tear down Docker Compose containers
  crip status             - Check Docker services health status
  crip gen-start          - Start the continuous event generator (foreground)
  crip gen-inject         - Inject an anomaly via the generator API
  crip gen-reset          - Reset the generator run (new run_id, clear injections)
  crip detect-live        - Start the streaming detector worker
  crip eval-live          - Start the live detection evaluator
  crip eval-offline       - Run offline evaluation benchmark suite
  crip eval-tune          - Run hyperparameter tuning on TUNE seeds
  crip eval-report        - Show latest offline evaluation report
  crip remediate-live     - Start live streaming remediation worker
  crip remediate-mode     - Switch remediation mode (dry-run / enforce)
  crip kill-switch        - Engage or disengage emergency kill switch
  crip dash               - Start the live demo dashboard (port 8501)
  crip dash-status        - Show dashboard and pipeline health status
  crip load-test          - Run headless Locust load test
"""

import os
import sys
import time
from pathlib import Path
from invoke import task

ROOT_DIR = Path(__file__).parent.resolve()
PYTHON = sys.executable


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


@task(help={
    "suite": "Evaluation suite to run: quick (2 seeds, 1 day) | full (30 seeds, 7 days)",
    "output": "Optional custom output directory path",
})
def eval_offline(c, suite="quick", output=""):
    """Run the offline evaluation benchmark suite across seeds and scenarios."""
    out_flag = f' --output "{output}"' if output else ""
    print(f">>> [crip] Running offline evaluation benchmark (suite={suite})...")
    c.run(f'"{PYTHON}" -m evaluation.cli benchmark --suite {suite}{out_flag}', pty=False, in_stream=False)


@task(help={"quick": "Quick mode tuning on fewer seeds and iterations"})
def eval_tune(c, quick=False):
    """Run hyperparameter search and calibration strictly on TUNE seeds."""
    quick_flag = " --quick" if quick else ""
    print(">>> [crip] Running hyperparameter tuning on TUNE seeds...")
    c.run(f'"{PYTHON}" -m evaluation.cli tune{quick_flag}', pty=False, in_stream=False)


@task
def eval_report(c):
    """Display the latest offline evaluation report path and summary."""
    c.run(f'"{PYTHON}" -m evaluation.cli report', pty=False, in_stream=False)


# ---------------------------------------------------------------------------
# Phase 6 — Live Remediation Tasks
# ---------------------------------------------------------------------------

@task(help={"mode": "Operating mode: dry | enforce (default dry)"})
def remediate_live(c, mode="dry"):
    """Start the live streaming remediation worker."""
    m_val = mode.upper()
    if m_val not in ("DRY", "DRY_RUN", "ENFORCE"):
        print(f"[ERROR] Invalid mode '{mode}'. Choose 'dry' or 'enforce'.")
        return
    mode_str = "ENFORCE" if m_val == "ENFORCE" else "DRY_RUN"
    print(f">>> [crip] Starting remediation worker (mode={mode_str})...")
    c.run(
        f'"{PYTHON}" -m remediation.worker',
        pty=False,
        in_stream=False,
        env={"REM_MODE": mode_str},
    )


@task(help={"mode": "Operating mode to set: dry | enforce"})
def remediate_mode(c, mode="dry"):
    """Switch remediation engine mode between dry-run and enforce."""
    m_val = mode.upper()
    mode_str = "ENFORCE" if m_val in ("ENFORCE", "REAL") else "DRY_RUN"
    try:
        import redis as _redis
        r = _redis.from_url(os.getenv("REM_REDIS_URL", "redis://localhost:6379"), decode_responses=True)
        r.set("crip:remediation:mode", mode_str)
        print(f"[OK] Remediation mode updated in Redis: {mode_str}")
    except Exception as exc:
        print(f"[ERROR] Could not connect to Redis: {exc}")


@task(help={"state": "Kill switch state: on | off (default on)"})
def kill_switch(c, state="on"):
    """Engage or disengage emergency kill switch (blocks all actions)."""
    is_on = state.lower() in ("on", "true", "1", "enable")
    try:
        import redis as _redis
        r = _redis.from_url(os.getenv("REM_REDIS_URL", "redis://localhost:6379"), decode_responses=True)
        r.set("crip:remediation:kill_switch", "1" if is_on else "0")
        status_str = "ENGAGED (all actions blocked)" if is_on else "DISENGAGED (normal operation)"
        print(f"[OK] Remediation kill switch is now: {status_str}")
    except Exception as exc:
        print(f"[ERROR] Could not connect to Redis: {exc}")
