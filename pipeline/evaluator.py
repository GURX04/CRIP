"""
pipeline/evaluator.py
=====================
Standalone evaluation process that joins `crip:detections` with `crip:injections`
to compute real-time detection latency, precision, recall, and F1.

NEVER reads `crip:labels` or `crip:events`.
Publishes evaluation results to Redis Stream `crip:eval`.

Usage:
    python -m pipeline.evaluator
    # or via CLI:
    crip eval-live
"""

from __future__ import annotations

import argparse
import json
import logging
import signal
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

import redis

from pipeline.config import PipelineSettings, get_settings

try:
    import structlog

    logger = structlog.get_logger(__name__)
    _use_structlog = True
except ImportError:
    logger = logging.getLogger(__name__)
    _use_structlog = False


def _parse_ts(raw: Optional[str]) -> Optional[datetime]:
    if not raw:
        return None
    dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


@dataclass
class InjectionRecord:
    injection_id: str
    run_id: str
    type: str
    service: str
    magnitude: float
    duration_min: float
    sim_start_ts: Optional[datetime]
    sim_end_ts: Optional[datetime]
    wall_start_ms: int
    wall_end_ms: int
    detected: bool = False
    detection_latency_wall_s: Optional[float] = None
    detection_latency_sim_min: Optional[float] = None
    first_detector: Optional[str] = None


@dataclass
class DetectionRecord:
    detection_id: str
    run_id: str
    service: str
    sim_ts: datetime
    detect_wall_ts_ms: int
    first_event_wall_ts_ms: int
    severity: float
    tick_index: int
    matched_injection_id: Optional[str] = None


class Evaluator:
    def __init__(self, settings: PipelineSettings) -> None:
        self._settings = settings
        self._redis: Optional[redis.Redis] = None
        self._injections: Dict[str, InjectionRecord] = {}
        self._detections: List[DetectionRecord] = []
        self._first_event_ms: Optional[int] = None
        self._lock = threading.Lock()
        self._last_inj_id = "$"
        self._consumer_group = "evaluators"
        self._consumer_name = "eval-1"
        self._connect_redis()

    def _connect_redis(self) -> None:
        url = self._settings.eval_redis_url or self._settings.redis_url
        backoff = 1.0
        while True:
            try:
                r = redis.from_url(
                    url, decode_responses=True, socket_connect_timeout=5
                )
                r.ping()
                self._redis = r
                self._setup_consumer_group()
                return
            except Exception as exc:
                time.sleep(backoff)
                backoff = min(backoff * 2, 30.0)

    def _setup_consumer_group(self) -> None:
        try:
            self._redis.xgroup_create(
                self._settings.redis_stream_detections,
                self._consumer_group,
                id="0",
                mkstream=True,
            )
        except redis.exceptions.ResponseError as exc:
            if "BUSYGROUP" not in str(exc):
                raise

    def run(self, stop_event: threading.Event) -> None:
        last_print = time.monotonic()
        while not stop_event.is_set():
            try:
                self._poll_injections()
                self._poll_detections()
            except Exception as exc:
                time.sleep(1)
                continue

            if time.monotonic() - last_print >= 5.0:
                self._print_and_publish_stats()
                last_print = time.monotonic()

            time.sleep(0.1)

    def _poll_injections(self) -> None:
        if not self._redis:
            return
        stream = "crip:injections"
        results = self._redis.xread({stream: self._last_inj_id}, count=50, block=100)
        if not results:
            return
        for _stream_name, messages in results:
            for msg_id, fields in messages:
                self._last_inj_id = msg_id
                payload_str = fields.get("payload", "")
                if payload_str:
                    try:
                        self._ingest_injection(json.loads(payload_str))
                    except Exception:
                        pass

    def _ingest_injection(self, p: dict) -> None:
        inj_id = p.get("injection_id")
        if not inj_id:
            return
        with self._lock:
            sim_start = _parse_ts(p.get("sim_start_ts"))
            sim_end = _parse_ts(p.get("sim_end_ts"))
            if inj_id in self._injections:
                rec = self._injections[inj_id]
                if sim_end:
                    rec.sim_end_ts = sim_end
                if p.get("wall_end_ms"):
                    rec.wall_end_ms = int(p["wall_end_ms"])
            else:
                rec = InjectionRecord(
                    injection_id=inj_id,
                    run_id=p.get("run_id", ""),
                    type=p.get("type", "unknown"),
                    service=p.get("service", ""),
                    magnitude=float(p.get("magnitude", 1.0)),
                    duration_min=float(p.get("duration_min", 30.0)),
                    sim_start_ts=sim_start,
                    sim_end_ts=sim_end,
                    wall_start_ms=int(p.get("wall_start_ms", 0)),
                    wall_end_ms=int(p.get("wall_end_ms", 0)),
                )
                self._injections[inj_id] = rec

            # Retroactive match against already received detections
            for det in self._detections:
                if det.matched_injection_id is None:
                    self._check_match(rec, det)

    def _poll_detections(self) -> None:
        if not self._redis:
            return
        stream = self._settings.redis_stream_detections
        results = self._redis.xreadgroup(
            self._consumer_group,
            self._consumer_name,
            {stream: ">"},
            count=50,
            block=100,
        )
        if not results:
            return
        for _stream_name, messages in results:
            for msg_id, fields in messages:
                payload_str = fields.get("payload", "")
                if payload_str:
                    try:
                        self._ingest_detection(json.loads(payload_str))
                    except Exception:
                        pass
                self._redis.xack(stream, self._consumer_group, msg_id)

    def _ingest_detection(self, p: dict) -> None:
        det_id = p.get("detection_id", "")
        sim_ts = _parse_ts(p.get("sim_ts"))
        if not sim_ts:
            return
        first_wall = int(p.get("first_event_wall_ts_ms", 0))
        if self._first_event_ms is None and first_wall:
            self._first_event_ms = first_wall

        det = DetectionRecord(
            detection_id=det_id,
            run_id=p.get("run_id", ""),
            service=p.get("service", ""),
            sim_ts=sim_ts,
            detect_wall_ts_ms=int(p.get("detect_wall_ts_ms", 0)),
            first_event_wall_ts_ms=first_wall,
            severity=float(p.get("severity", 0.0)),
            tick_index=int(p.get("tick_index", 0)),
        )

        with self._lock:
            # Check match against all known injections
            for inj in self._injections.values():
                if self._check_match(inj, det, votes=p.get("detector_votes", [])):
                    break
            self._detections.append(det)

    def _check_match(
        self,
        inj: InjectionRecord,
        det: DetectionRecord,
        votes: Optional[List[dict]] = None,
    ) -> bool:
        if inj.service != "all" and inj.service != det.service:
            return False
        if not inj.sim_start_ts:
            return False

        tol_min = self._settings.eval_fp_tolerance_ticks * 5.0
        start = inj.sim_start_ts
        if inj.sim_end_ts:
            end = inj.sim_end_ts + timedelta(minutes=tol_min)
        else:
            end = start + timedelta(minutes=inj.duration_min + tol_min)

        if start <= det.sim_ts <= end:
            det.matched_injection_id = inj.injection_id
            if not inj.detected:
                inj.detected = True
                if inj.wall_start_ms and det.detect_wall_ts_ms:
                    inj.detection_latency_wall_s = (
                        det.detect_wall_ts_ms - inj.wall_start_ms
                    ) / 1000.0
                if inj.sim_start_ts:
                    inj.detection_latency_sim_min = (
                        det.sim_ts - inj.sim_start_ts
                    ).total_seconds() / 60.0
                if votes:
                    for v in votes:
                        if v.get("is_anomaly"):
                            inj.first_detector = v.get("detector_name")
                            break
            return True
        return False

    def compute_metrics(self) -> dict:
        with self._lock:
            tp = sum(1 for inj in self._injections.values() if inj.detected)
            fn = len(self._injections) - tp
            fp = sum(1 for d in self._detections if d.matched_injection_id is None)

            precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0
            f1 = (
                2 * precision * recall / (precision + recall)
                if (precision + recall) > 0
                else 0.0
            )

            lats = [
                inj.detection_latency_wall_s
                for inj in self._injections.values()
                if inj.detection_latency_wall_s is not None
            ]
            mean_lat = sum(lats) / len(lats) if lats else 0.0

            # Estimate simulated days
            if self._first_event_ms:
                elapsed_real_s = (int(time.time() * 1000) - self._first_event_ms) / 1000.0
                # sim minutes elapsed ~ real_s * 5 / 60
                sim_days = max(elapsed_real_s * 5.0 / 1440.0, 1.0 / 288.0)
            else:
                sim_days = 1.0 / 288.0

            fa_per_day = fp / sim_days

            entries = [
                {
                    "injection_id": inj.injection_id,
                    "type": inj.type,
                    "service": inj.service,
                    "detected": inj.detected,
                    "latency_wall_s": inj.detection_latency_wall_s,
                    "latency_sim_min": inj.detection_latency_sim_min,
                    "first_detector": inj.first_detector,
                }
                for inj in list(self._injections.values())[-10:]
            ]

            return {
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(f1, 4),
                "mean_latency_s": round(mean_lat, 2),
                "false_alarms_per_day": round(fa_per_day, 2),
                "entries": entries,
            }

    def _print_and_publish_stats(self) -> None:
        stats = self.compute_metrics()

        # Print running table
        print("\n" + "=" * 70)
        print(
            f" CRIP EVALUATION: Precision={stats['precision']*100:.1f}% | "
            f"Recall={stats['recall']*100:.1f}% | F1={stats['f1']:.3f} | "
            f"FA/day={stats['false_alarms_per_day']:.1f}"
        )
        print("-" * 70)
        print(f"{'ID':<10} {'Type':<12} {'Service':<18} {'Detected':<10} {'Lat(s)':<8} {'Detector'}")
        for e in stats["entries"]:
            lat = f"{e['latency_wall_s']:.1f}s" if e["latency_wall_s"] is not None else "—"
            det_mark = "✓" if e["detected"] else "✗"
            print(
                f"{e['injection_id'][:8]:<10} {e['type']:<12} {e['service']:<18} "
                f"{det_mark:<10} {lat:<8} {e['first_detector'] or '—'}"
            )
        print("=" * 70, flush=True)

        if self._redis:
            try:
                self._redis.xadd("crip:eval", {"payload": json.dumps(stats)}, maxlen=1000)
            except Exception:
                pass


def main(argv=None):
    parser = argparse.ArgumentParser(description="CRIP evaluation pipeline")
    parser.add_argument("--redis-url", default=None)
    args = parser.parse_args(argv)

    settings = get_settings()
    if args.redis_url:
        settings.eval_redis_url = args.redis_url

    stop_event = threading.Event()

    def _handle_signal(sig, frame):
        stop_event.set()

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    evaluator = Evaluator(settings)
    print("CRIP evaluator started. Listening to crip:detections and crip:injections...")
    evaluator.run(stop_event)


if __name__ == "__main__":
    main()

