"""
dashboard/streams.py
====================
Async Redis stream reader that fans out SSE messages to all connected clients.

Design
------
- A single background asyncio task (AsyncRedisReader) reads all relevant
  streams with XREAD (NOT consumer groups — we are read-only observers).
- Each connected SSE client gets a bounded asyncio.Queue.  When the queue is
  full, the OLDEST item is dropped (never the writer or reader).
- State (last-seen IDs, last score timestamps) is kept in-process so a page
  refresh can re-read recent data from /history instead of replaying the queue.
- Replay mode: if settings.replay_file is set, reads a JSONL file line-by-line
  at real-time cadence instead of connecting to Redis.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Set

from dashboard.config import DashboardSettings
from dashboard.models import SSEMessage

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Client registry
# ---------------------------------------------------------------------------

class ClientRegistry:
    """Thread-safe registry of active SSE client queues."""

    def __init__(self, maxsize: int) -> None:
        self._maxsize = maxsize
        self._clients: Set[asyncio.Queue] = set()
        self._lock = asyncio.Lock()

    async def add(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=self._maxsize)
        async with self._lock:
            self._clients.add(q)
        return q

    async def remove(self, q: asyncio.Queue) -> None:
        async with self._lock:
            self._clients.discard(q)

    async def broadcast(self, msg: SSEMessage) -> None:
        """Push msg to every client queue, dropping the oldest if full."""
        async with self._lock:
            clients = list(self._clients)
        for q in clients:
            if q.full():
                try:
                    q.get_nowait()   # drop oldest
                except asyncio.QueueEmpty:
                    pass
            try:
                q.put_nowait(msg)
            except asyncio.QueueFull:
                pass  # shouldn't happen after the get_nowait, but be safe

    @property
    def count(self) -> int:
        return len(self._clients)


# ---------------------------------------------------------------------------
# Shared reader state (last seen stream IDs + detector liveness)
# ---------------------------------------------------------------------------

class ReaderState:
    """Mutable state shared between the reader and /health endpoint."""

    def __init__(self) -> None:
        self.last_score_wall_ms: int = 0
        self.last_run_id: str = ""
        self.stream_ids: Dict[str, str] = {}  # stream_name -> last "$" id


# ---------------------------------------------------------------------------
# Main reader coroutine
# ---------------------------------------------------------------------------

class AsyncRedisReader:
    """
    Single asyncio task that reads all CRIP streams and fans out to clients.

    Call ``start(registry, state)`` to launch as a background task.
    Cancel the returned task to stop.
    """

    def __init__(self, settings: DashboardSettings) -> None:
        self._settings = settings
        self._registry: Optional[ClientRegistry] = None
        self._state: Optional[ReaderState] = None
        self._redis = None

    async def start(
        self,
        registry: ClientRegistry,
        state: ReaderState,
        stop_event: asyncio.Event,
    ) -> None:
        """Run forever until stop_event is set."""
        self._registry = registry
        self._state = state
        s = self._settings

        if s.replay_file:
            await self._run_replay(stop_event)
            return

        await self._connect_redis(stop_event)

        streams = [
            s.redis_stream_scores,
            s.redis_stream_detections,
            s.redis_stream_injections,
            s.redis_stream_eval,
            s.redis_stream_proposals,
            s.redis_stream_mitigations,
            s.redis_stream_results,
        ]
        # Start from "now" ($ = only new messages)
        ids = {stream: "$" for stream in streams}
        state.stream_ids = dict(ids)

        heartbeat_task = asyncio.create_task(self._heartbeat_loop(stop_event))

        try:
            while not stop_event.is_set():
                if not self._redis:
                    await asyncio.sleep(1)
                    await self._connect_redis(stop_event)
                    continue
                try:
                    results = await asyncio.get_event_loop().run_in_executor(
                        None,
                        lambda: self._redis.xread(  # type: ignore[union-attr]
                            ids,
                            count=s.redis_xread_count,
                            block=s.redis_block_ms,
                        ),
                    )
                except Exception as exc:
                    log.warning("redis_xread_error: %s", exc)
                    await asyncio.sleep(1)
                    await self._connect_redis(stop_event)
                    continue

                if not results:
                    continue

                for stream_name, messages in results:
                    # Update last-seen ID
                    if messages:
                        ids[stream_name] = messages[-1][0]
                        state.stream_ids[stream_name] = messages[-1][0]

                    for _msg_id, fields in messages:
                        payload_str = fields.get("payload", "")
                        if not payload_str:
                            continue
                        try:
                            sse = self._route(stream_name, payload_str)
                            if sse:
                                await self._registry.broadcast(sse)
                        except Exception as exc:
                            log.debug("stream routing error: %s", exc)
        finally:
            heartbeat_task.cancel()

    async def _heartbeat_loop(self, stop_event: asyncio.Event) -> None:
        while not stop_event.is_set():
            await asyncio.sleep(self._settings.heartbeat_interval_s)
            msg = SSEMessage(event="heartbeat", data={"ts_ms": _now_ms()})
            await self._registry.broadcast(msg)

    def _route(self, stream: str, payload_str: str) -> Optional[SSEMessage]:
        """Parse a stream message and route to the correct SSE event type."""
        s = self._settings
        data = json.loads(payload_str)

        if stream == s.redis_stream_scores:
            self._state.last_score_wall_ms = _now_ms()
            run_id = data.get("run_id", "")
            if run_id and run_id != self._state.last_run_id:
                self._state.last_run_id = run_id
                # Broadcast run_id change as a state event
                asyncio.ensure_future(
                    self._registry.broadcast(
                        SSEMessage(event="state", data={"run_id": run_id, "new_run": True})
                    )
                )
            return SSEMessage(event="tick", data=data)

        if stream == s.redis_stream_detections:
            return SSEMessage(event="detection", data=data)

        if stream == s.redis_stream_injections:
            return SSEMessage(event="injection", data=data)

        if stream == s.redis_stream_eval:
            return SSEMessage(event="eval", data=data)

        if stream == s.redis_stream_proposals:
            return SSEMessage(event="remediation_proposal", data=data)

        if stream == s.redis_stream_mitigations:
            return SSEMessage(event="mitigation", data=data)

        if stream == s.redis_stream_results:
            return SSEMessage(event="remediation_result", data=data)

        return None

    async def _connect_redis(self, stop_event: Optional[asyncio.Event] = None) -> bool:
        """Connect (or reconnect) to Redis with backoff."""
        import redis as redis_lib  # noqa: PLC0415
        try:
            r = redis_lib.from_url(
                self._settings.redis_url,
                decode_responses=True,
                socket_connect_timeout=0.5,
            )
            r.ping()
            self._redis = r
            log.info("dashboard redis connected: %s", self._settings.redis_url)
            return True
        except Exception as exc:
            log.debug("dashboard redis connect failed (%s)", exc)
            self._redis = None
            return False

    # ------------------------------------------------------------------
    # Replay mode
    # ------------------------------------------------------------------

    async def _run_replay(self, stop_event: asyncio.Event) -> None:
        """Read a JSONL replay file and fan out events at their original cadence."""
        path = Path(self._settings.replay_file)  # type: ignore[arg-type]
        if not path.exists():
            log.error("Replay file not found: %s", path)
            return

        log.info("Replay mode: reading %s", path)
        # First, broadcast a banner
        await self._registry.broadcast(
            SSEMessage(event="state", data={"replay": True, "file": str(path)})
        )

        lines = path.read_text(encoding="utf-8").splitlines()
        prev_wall: Optional[int] = None

        for line in lines:
            if stop_event.is_set():
                break
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                event_type = record.pop("_sse_event", "tick")
                wall_ts = record.get("wall_ts_ms") or record.get("detect_wall_ts_ms") or 0
                if prev_wall and wall_ts and wall_ts > prev_wall:
                    delay = (wall_ts - prev_wall) / 1000.0
                    await asyncio.sleep(min(delay, 2.0))  # cap replay delays
                prev_wall = wall_ts or prev_wall
                await self._registry.broadcast(SSEMessage(event=event_type, data=record))
            except Exception as exc:
                log.debug("replay line error: %s", exc)


def _now_ms() -> int:
    return int(time.time() * 1000)

