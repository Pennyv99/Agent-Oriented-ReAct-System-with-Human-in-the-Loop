import json
from typing import Any, Dict
from langchain_core.tools import BaseTool
from hil_store import HILStore

# Stable prefix so app.py can detect pending from tool outputs reliably
HIL_PENDING_PREFIX = "HIL_PENDING_JSON:"


class HILToolWrapper(BaseTool):
    """
    Tool gate:
    - Not approved -> create (deduped) pending_id, return a machine-readable marker
    - Approved -> execute the inner tool
    """
    hil: HILStore
    session_id: str
    inner_tool: Any
    controlled: bool = True

    def __init__(self, inner_tool: Any, hil: HILStore, session_id: str, controlled: bool = True):
        super().__init__(
            name=getattr(inner_tool, "name", "unknown_tool"),
            description=getattr(inner_tool, "description", ""),
            args_schema=getattr(inner_tool, "args_schema", None),
        )
        self.inner_tool = inner_tool
        self.hil = hil
        self.session_id = session_id
        self.controlled = controlled

    def _run(self, *args: Any, **kwargs: Any) -> str:
        raise NotImplementedError("Use async")

    async def _arun(self, **kwargs: Any) -> Any:
        tool_name = self.name
        tool_args: Dict[str, Any] = kwargs or {}

        # Gate
        if self.controlled and not self.hil.is_allowed(self.session_id, tool_name, tool_args):
            pending_id = self.hil.create_pending(self.session_id, tool_name, tool_args)

            payload = {
                "status": "PENDING",
                "pending_id": pending_id,
                "session_id": self.session_id,
                "tool_name": tool_name,
                "args": tool_args,
                "hint": "Call POST /hil/resume with this pending_id after approval (or /hil/approve then /hil/resume).",
            }

            # Return a tool output that is easy for backend to detect & parse.
            # This reduces dependence on the LLM's behavior.
            return HIL_PENDING_PREFIX + json.dumps(payload, ensure_ascii=False)

        # Approved -> execute real tool
        if hasattr(self.inner_tool, "ainvoke"):
            return await self.inner_tool.ainvoke(tool_args)
        if hasattr(self.inner_tool, "invoke"):
            return self.inner_tool.invoke(tool_args)

        return f"Tool '{tool_name}' has no invoker."