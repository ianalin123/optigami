"""
TrainingBroadcastServer — fire-and-forget broadcast hub for live training viewer.

The RL training process calls publish() after each env.step().
Spectator browsers connect via /ws/training WebSocket.
Broadcast is async and non-blocking: if no viewers are connected, observations are dropped.
"""
from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Optional

from fastapi import WebSocket, WebSocketDisconnect

logger = logging.getLogger(__name__)


@dataclass
class EpisodeInfo:
    episode_id: str
    task_name: str
    status: str = "running"       # "running" | "done" | "timeout" | "error"
    step: int = 0
    observation: dict = field(default_factory=dict)
    metrics: dict = field(default_factory=dict)
    fold_history: list = field(default_factory=list)
    score: Optional[float] = None
    final_metrics: Optional[dict] = None


class TrainingBroadcastServer:
    """Central hub for broadcasting RL training observations to spectator WebSockets.

    Thread-safe: publish() can be called from training threads (ThreadPoolExecutor).
    WebSocket handlers run in the asyncio event loop.
    """

    def __init__(self) -> None:
        self._spectators: list[WebSocket] = []
        self._registry: dict[str, EpisodeInfo] = {}
        self._batch_id: int = 0
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._lock = asyncio.Lock()

    # ── Episode publishing (called from training thread / async context) ──

    def publish(self, episode_id: str, data: dict) -> None:
        """Fire-and-forget: push an update from the training process.

        Safe to call from any thread. If no event loop is running, logs and returns.
        """
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.ensure_future(self._async_publish(episode_id, data), loop=loop)
            else:
                loop.run_until_complete(self._async_publish(episode_id, data))
        except RuntimeError:
            # No event loop — training without server
            pass

    async def _async_publish(self, episode_id: str, data: dict) -> None:
        msg_type = data.get("type", "episode_update")

        async with self._lock:
            if msg_type == "batch_start":
                self._batch_id = data.get("batch_id", self._batch_id + 1)
                self._registry.clear()
                await self._broadcast(data)
                return

            if msg_type == "batch_done":
                await self._broadcast(data)
                return

            if msg_type == "training_done":
                await self._broadcast(data)
                return

            # episode_update or episode_done
            ep = self._registry.setdefault(
                episode_id,
                EpisodeInfo(episode_id=episode_id, task_name=data.get("task_name", "")),
            )

            if msg_type == "episode_done":
                ep.status = data.get("status", "done")
                ep.score = data.get("score")
                ep.final_metrics = data.get("final_metrics")
            else:
                ep.step = data.get("step", ep.step)
                ep.status = "running"
                obs = data.get("observation", {})
                ep.observation = obs
                ep.metrics = obs.get("metrics", {})
                ep.fold_history = obs.get("fold_history", ep.fold_history)

        await self._broadcast({"episode_id": episode_id, **data})

    # ── Spectator management ──

    async def connect_spectator(self, websocket: WebSocket) -> None:
        """Accept a new viewer WebSocket and serve it until disconnect."""
        await websocket.accept()

        async with self._lock:
            self._spectators.append(websocket)

        # Send current registry snapshot immediately
        await self._send_registry(websocket)

        try:
            while True:
                # Viewers are read-only; drain any incoming messages (pings etc)
                await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
        except (WebSocketDisconnect, asyncio.TimeoutError, Exception):
            pass
        finally:
            await self.disconnect_spectator(websocket)

    async def disconnect_spectator(self, websocket: WebSocket) -> None:
        async with self._lock:
            self._spectators = [s for s in self._spectators if s is not websocket]

    # ── Batch control ──

    async def start_batch(self, batch_id: int, num_episodes: int, prompt_index: int = 0) -> None:
        """Call before starting a new training batch."""
        data = {
            "type": "batch_start",
            "batch_id": batch_id,
            "num_episodes": num_episodes,
            "prompt_index": prompt_index,
        }
        await self._async_publish("__batch__", data)

    async def finish_batch(
        self,
        batch_id: int,
        scores: list[float],
        best_episode_id: str = "",
    ) -> None:
        """Call after all episodes in a batch complete."""
        data = {
            "type": "batch_done",
            "batch_id": batch_id,
            "scores": scores,
            "best_episode_id": best_episode_id,
            "avg_score": sum(scores) / len(scores) if scores else 0.0,
        }
        await self._async_publish("__batch__", data)

    async def clear_batch(self) -> None:
        """Reset episode registry for next batch."""
        async with self._lock:
            self._registry.clear()

    # ── Internals ──

    async def _broadcast(self, message: dict) -> None:
        """Send message to all spectators, removing dead connections."""
        if not self._spectators:
            return
        payload = json.dumps(message, default=str)
        dead: list[WebSocket] = []
        for ws in list(self._spectators):
            try:
                await ws.send_text(payload)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self._spectators = [s for s in self._spectators if s is not ws]

    async def _send_registry(self, websocket: WebSocket) -> None:
        """Send the full episode registry to a newly connected viewer."""
        async with self._lock:
            episodes = {
                ep_id: {
                    "status": ep.status,
                    "task": ep.task_name,
                    "step": ep.step,
                    "observation": ep.observation,
                    "metrics": ep.metrics,
                    "score": ep.score,
                }
                for ep_id, ep in self._registry.items()
            }
            payload = {
                "type": "registry",
                "batch_id": self._batch_id,
                "episodes": episodes,
            }
        try:
            await websocket.send_text(json.dumps(payload, default=str))
        except Exception:
            pass

    @property
    def spectator_count(self) -> int:
        return len(self._spectators)

    @property
    def active_episodes(self) -> int:
        return sum(1 for ep in self._registry.values() if ep.status == "running")
