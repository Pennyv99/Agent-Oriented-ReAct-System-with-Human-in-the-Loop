# hil_store.py
import os
import json
import hashlib
import uuid
import redis
from typing import Any, Dict, Optional


class HILStore:
    def __init__(self) -> None:
        self.r = redis.Redis(
            host=os.getenv("REDIS_HOST", "localhost"),
            port=int(os.getenv("REDIS_PORT", "6379")),
            decode_responses=True,
        )

    def _args_hash(self, args: Dict[str, Any]) -> str:
        s = json.dumps(args, ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]

    def _allow_key(self, session_id: str, tool_name: str, args_hash: str) -> str:
        return f"hil:allow:{session_id}:{tool_name}:{args_hash}"

    def _pending_key(self, pending_id: str) -> str:
        return f"hil:pending:{pending_id}"

    def is_allowed(self, session_id: str, tool_name: str, args: Dict[str, Any]) -> bool:
        h = self._args_hash(args)
        return self.r.get(self._allow_key(session_id, tool_name, h)) == "1"

    def create_pending(self, session_id: str, tool_name: str, args: Dict[str, Any]) -> str:
        pending_id = str(uuid.uuid4())
        h = self._args_hash(args)
        payload = {
            "pending_id": pending_id,
            "session_id": session_id,
            "tool_name": tool_name,
            "args": args,
            "args_hash": h,
            "status": "PENDING",
        }
        self.r.set(self._pending_key(pending_id), json.dumps(payload, ensure_ascii=False))
        self.r.expire(self._pending_key(pending_id), 3600)
        return pending_id

    def get_pending(self, pending_id: str) -> Optional[Dict[str, Any]]:
        raw = self.r.get(self._pending_key(pending_id))
        return json.loads(raw) if raw else None

    def approve(self, pending_id: str, ttl_seconds: int = 3600) -> bool:
        p = self.get_pending(pending_id)
        if not p:
            return False
        allow_key = self._allow_key(p["session_id"], p["tool_name"], p["args_hash"])
        self.r.set(allow_key, "1")
        self.r.expire(allow_key, ttl_seconds)

        # update pending status
        p["status"] = "APPROVED"
        self.r.set(self._pending_key(pending_id), json.dumps(p, ensure_ascii=False))
        self.r.expire(self._pending_key(pending_id), 3600)
        return True