"""
dashboard/recorder.py
=====================
Records live Redis stream events to a JSONL file for replay mode.

Usage
-----
    python -m dashboard.recorder --output reports/session.jsonl --duration 300
    # or via CLI:
    crip dash-record --output reports/session.jsonl --duration 300
"""
from __future__ import annotations

import argparse
import json
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import redis as redis_lib

from dashboard.config import DashboardSettings, get_settings


def _now_ms() -> int:
    return int(time.time() * 1000)


STREAM_TO_EVENT = {
    "crip:scores":     "tick",
    "crip:detections": "detection",
    "crip:injections": "injection",
    "crip:eval":       "eval",
}


def record(settings: DashboardSettings, output_path: Path, duration_s: Optional[float]) -> None:
    """Read Redis streams and append JSON lines to *output_path*."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    r = redis_lib.from_url(settings.redis_url, decode_responses=True)
    r.ping()
    print(f"[recorder] Connected to Redis. Writing to {output_path}", flush=True)

    streams = list(STREAM_TO_EVENT.keys())
    ids = {s: "$" for s in streams}

    start = time.monotonic()
    total = 0
    _stop = False

    def _sighandler(sig, frame):
        nonlocal _stop
        _stop = True

    signal.signal(signal.SIGINT,  _sighandler)
    signal.signal(signal.SIGTERM, _sighandler)

    with open(output_path, "a", encoding="utf-8") as f:
        while not _stop:
            if duration_s and (time.monotonic() - start) >= duration_s:
                break
            try:
                results = r.xread(ids, count=100, block=200)
            except Exception as exc:
                print(f"[recorder] xread error: {exc}", flush=True)
                time.sleep(1)
                continue

            if not results:
                continue

            for stream_name, messages in results:
                event_type = STREAM_TO_EVENT.get(stream_name, "unknown")
                for msg_id, fields in messages:
                    ids[stream_name] = msg_id
                    payload_str = fields.get("payload", "")
                    if not payload_str:
                        continue
                    try:
                        data = json.loads(payload_str)
                        data["_sse_event"] = event_type
                        data["_recorded_ms"] = _now_ms()
                        f.write(json.dumps(data) + "\n")
                        total += 1
                        if total % 100 == 0:
                            f.flush()
                            print(f"[recorder] {total} records written", flush=True)
                    except Exception as exc:
                        print(f"[recorder] parse error: {exc}", flush=True)

    print(f"[recorder] Done. {total} records written to {output_path}", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Record CRIP streams for replay")
    parser.add_argument("--output", default="reports/session.jsonl")
    parser.add_argument("--duration", type=float, default=None,
                        help="Recording duration in seconds (default: until Ctrl-C)")
    args = parser.parse_args(argv)

    settings = get_settings()
    record(settings, Path(args.output), args.duration)


if __name__ == "__main__":
    main()

