# redis_saver.py
import os
import json
import redis
from typing import Any, Optional


class RedisCheckpoint:
    """
    key = session:{thread_id} -> value = checkpoint(json string)
    """

    def __init__(self) -> None:
        self.r = redis.Redis(
            host=os.getenv("REDIS_HOST", "localhost"),
            port=int(os.getenv("REDIS_PORT", "6379")),
            decode_responses=True,
        )

    def _key(self, thread_id: str) -> str:
        return f"session:{thread_id}"

    def save(self, thread_id: str, checkpoint: Any) -> None:
        self.r.set(self._key(thread_id), json.dumps(checkpoint))

    def load(self, thread_id: str) -> Optional[Any]:
        raw = self.r.get(self._key(thread_id))
        return json.loads(raw) if raw else None