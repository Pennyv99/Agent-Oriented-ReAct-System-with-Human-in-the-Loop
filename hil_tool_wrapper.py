# hil_tool_wrapper.py
from typing import Any, Dict, Optional
from langchain_core.tools import BaseTool
from hil_store import HILStore


class HILToolWrapper(BaseTool):
    """
    wrap a tool：require approval -> return PENDING invoke；approval -> invoke
    """
    hil: HILStore
    session_id: str
    inner_tool: Any
    controlled: bool = True  # part of tools use HIL

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

        if self.controlled and not self.hil.is_allowed(self.session_id, tool_name, tool_args):
            pending_id = self.hil.create_pending(self.session_id, tool_name, tool_args)
            return (
                f"[HIL] Tool '{tool_name}' requires approval.\n"
                f"pending_id={pending_id}\n"
                f"Call POST /hil/approve with this pending_id, then ask the same question again."
            )

        # approval
        if hasattr(self.inner_tool, "ainvoke"):
            return await self.inner_tool.ainvoke(tool_args)
        # no invoker
        if hasattr(self.inner_tool, "invoke"):
            return self.inner_tool.invoke(tool_args)
        return f"Tool '{tool_name}' has no invoker."