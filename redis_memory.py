import json
import redis
from langgraph.checkpoint.base import BaseCheckpointSaver

class RedisSaver(BaseCheckpointSaver):
    def __init__(self, host="localhost", port=6379):
        self.r = redis.Redis(host=host, port=port, decode_responses=True)

    def _key(self, thread_id):
        return f"session:{thread_id}"

    def put(self, config, checkpoint):
        thread_id = config["configurable"]["thread_id"]
        serialized = json.dumps(checkpoint)
        self.r.set(self._key(thread_id), serialized)

    def get(self, config):
        thread_id = config["configurable"]["thread_id"]
        data = self.r.get(self._key(thread_id))
        if data:
            return json.loads(data)
        return None