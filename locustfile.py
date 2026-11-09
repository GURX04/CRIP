"""
locustfile.py
=============
Locust load-testing script for Phase 1 — CRIP target application.

Traffic Shape (LoadTestShape)
-----------------------------
The test runs a 10-minute (600 s) cycle that models:

  ┌─ Baseline ──────────────────────────────────────────────────────────────┐
  │  10 users  │ Warm-up / steady-state floor                               │
  ├─ Ramp-up ───────────────────────────────────────────────────────────────┤
  │  10 → 50   │ Gradual ramp over the first 2 minutes                      │
  ├─ Spike 1 ───────────────────────────────────────────────────────────────┤
  │  50 → 200  │ Sudden spike at t=120 s, held for 30 s                     │
  ├─ Recovery ──────────────────────────────────────────────────────────────┤
  │  200 → 50  │ Rapid drop-back to normal                                  │
  ├─ Spike 2 ───────────────────────────────────────────────────────────────┤
  │  50 → 200  │ Second spike at t=300 s (every 3 min thereafter)           │
  └─────────────────────────────────────────────────────────────────────────┘

Run command (from project root):
    locust -f locustfile.py --headless \
           --host http://localhost:8080 \
           --run-time 10m \
           --html reports/load_test_report.html

Or with the Locust Web UI:
    locust -f locustfile.py --host http://localhost:8080
"""

import math
import random
import time

from locust import FastHttpUser, LoadTestShape, between, task


# ---------------------------------------------------------------------------
# User behaviour definition
# ---------------------------------------------------------------------------

class CRIPUser(FastHttpUser):
    """
    Simulates a single virtual user interacting with the CRIP target app.
    Task weights reflect realistic API traffic distributions.
    """

    wait_time = between(0.5, 2.0)   # Think-time between requests (seconds)

    # ------------------------------------------------------------------
    # Tasks (weighted)
    # ------------------------------------------------------------------

    @task(5)
    def health_check(self):
        """Most frequent call — health probes from load-balancers / uptime monitors."""
        with self.client.get("/health", catch_response=True, name="GET /health") as resp:
            if resp.status_code == 200:
                resp.success()
            else:
                resp.failure(f"Unexpected status {resp.status_code}")

    @task(3)
    def fetch_data(self):
        """Read endpoint — simulates DB look-ups."""
        item_id = f"item-{random.randint(1, 500)}"
        with self.client.get(
            f"/api/data?item_id={item_id}",
            catch_response=True,
            name="GET /api/data",
        ) as resp:
            if resp.status_code in (200, 500):   # 500 is a simulated error
                resp.success()
            else:
                resp.failure(f"Unexpected status {resp.status_code}")

    @task(2)
    def ingest_data(self):
        """Write endpoint — simulates S3 PUT / DynamoDB PutItem."""
        payload = {
            "record_id": f"rec-{random.randint(10_000, 99_999)}",
            "payload": {
                "user": f"user-{random.randint(1, 200)}",
                "event": random.choice(["click", "view", "purchase", "scroll"]),
                "value": round(random.uniform(0.01, 999.99), 2),
                "ts": time.time(),
            },
        }
        with self.client.post(
            "/api/ingest",
            json=payload,
            catch_response=True,
            name="POST /api/ingest",
        ) as resp:
            if resp.status_code == 200:
                resp.success()
            else:
                resp.failure(f"Unexpected status {resp.status_code}")


# ---------------------------------------------------------------------------
# Custom traffic shape
# ---------------------------------------------------------------------------

class SpikyLoadShape(LoadTestShape):
    """
    Programmatic traffic profile that combines:
      - A warm-up ramp (0-120 s)
      - A steady baseline
      - Recurring sharp spikes every 3 minutes, lasting 30 s each
      - A cool-down phase in the final 60 s

    tick() is called every second by Locust.
    Returns: (user_count, spawn_rate) or None to stop the test.
    """

    # --- Configuration knobs (seconds) ---
    TOTAL_DURATION    = 600    # Total test run time
    RAMP_DURATION     = 120    # Time to ramp from baseline to normal level
    COOL_DOWN_START   = 540    # Start cooling down at t=540 s
    SPIKE_INTERVAL    = 180    # One spike every 3 minutes
    SPIKE_DURATION    = 30     # Each spike lasts 30 s
    SPIKE_HOLD        = 10     # Flat-top hold at peak before dropping (within spike window)

    # --- User counts ---
    BASELINE_USERS    = 10     # Absolute floor (warm-up start)
    NORMAL_USERS      = 50     # Post-ramp steady state
    SPIKE_USERS       = 200    # Peak during a spike
    COOLDOWN_USERS    = 5      # Wind-down floor

    # --- Spawn rates (users/second) ---
    RAMP_SPAWN_RATE   = 1      # Gradual ramp
    SPIKE_SPAWN_RATE  = 50     # Aggressive spike ramp
    DROP_SPAWN_RATE   = 30     # Rapid drop-back after spike

    def tick(self):
        run_time = self.get_run_time()

        # --- Stop after total duration ---
        if run_time >= self.TOTAL_DURATION:
            return None

        # --- Cool-down phase ---
        if run_time >= self.COOL_DOWN_START:
            return (self.COOLDOWN_USERS, self.DROP_SPAWN_RATE)

        # --- Warm-up ramp (0 → RAMP_DURATION) ---
        if run_time < self.RAMP_DURATION:
            ramp_progress = run_time / self.RAMP_DURATION           # 0.0 → 1.0
            users = int(
                self.BASELINE_USERS
                + ramp_progress * (self.NORMAL_USERS - self.BASELINE_USERS)
            )
            return (users, self.RAMP_SPAWN_RATE)

        # --- Steady state + recurring spikes ---
        # Time since the end of the ramp
        t = run_time - self.RAMP_DURATION

        # Position within the current spike cycle
        cycle_position = t % self.SPIKE_INTERVAL

        if cycle_position < self.SPIKE_HOLD:
            # Rising edge / flat-top of the spike
            return (self.SPIKE_USERS, self.SPIKE_SPAWN_RATE)

        if cycle_position < self.SPIKE_DURATION:
            # Falling edge of the spike
            drop_progress = (cycle_position - self.SPIKE_HOLD) / (
                self.SPIKE_DURATION - self.SPIKE_HOLD
            )
            users = int(
                self.SPIKE_USERS
                - drop_progress * (self.SPIKE_USERS - self.NORMAL_USERS)
            )
            return (users, self.DROP_SPAWN_RATE)

        # Normal steady state between spikes
        # Add a subtle sine wave (~±5 users) for natural variance
        sine_offset = int(5 * math.sin(2 * math.pi * t / 60))
        users = max(self.BASELINE_USERS, self.NORMAL_USERS + sine_offset)
        return (users, self.RAMP_SPAWN_RATE)
