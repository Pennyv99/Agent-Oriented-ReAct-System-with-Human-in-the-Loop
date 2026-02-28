from dotenv import load_dotenv
load_dotenv()

import os
import asyncio
import json
from fastapi import FastAPI
from pydantic import BaseModel

from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.prebuilt import create_react_agent
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.checkpoint.redis import RedisSaver

from pg_logger import PGLogger
from hil_store import HILStore
from hil_tool_wrapper import HILToolWrapper, HIL_PENDING_PREFIX


app = FastAPI(title="Agent-Oriented ReAct Service")

# LLM Initialization
llm = init_chat_model(
    model=os.getenv("LLM_MODEL", "openai:deepseek-v3"),
    temperature=float(os.getenv("LLM_TEMPERATURE", "0")),
    base_url=os.getenv("LLM_BASE_URL"),
    api_key=os.getenv("LLM_API_KEY"),
)

# MCP Tool Loading (once)
mcp_client = MultiServerMCPClient({
    "amap": {
        "url": os.getenv("AMAP_MCP_URL"),
        "transport": "sse",
    }
})

BASE_TOOLS = asyncio.get_event_loop().run_until_complete(
    mcp_client.get_tools()
)

# Infrastructure
postgres_logger = PGLogger()
hil_store = HILStore()
redis_saver = RedisSaver.from_conn_string(
    os.getenv("REDIS_URL", "redis://localhost:6379")
)

# Strengthen system prompt so the model won't "try another args"
SYSTEM_PROMPT = SystemMessage(
    content=(
        "You are an AI assistant capable of using external tools.\n"
        "IMPORTANT HIL RULE:\n"
        "If any tool returns a message starting with 'HIL_PENDING_JSON:',\n"
        "you must STOP immediately and reply to the user with the pending_id exactly.\n"
        "Do NOT call any other tools and do NOT retry with different arguments.\n"
    )
)

# Schemas
class ChatRequest(BaseModel):
    session_id: str
    message: str


class ApprovalRequest(BaseModel):
    pending_id: str


def _extract_hil_pending_from_messages(messages) -> dict | None:
    """
    Find the most recent tool output containing our pending marker, and parse it.
    We do NOT rely on the model to behave perfectly; we detect it ourselves.
    """
    for m in reversed(messages):
        content = getattr(m, "content", None)
        if isinstance(content, str) and content.startswith(HIL_PENDING_PREFIX):
            raw = content[len(HIL_PENDING_PREFIX):]
            try:
                return json.loads(raw)
            except Exception:
                # If parsing fails, still return something usable
                return {"pending_id": None, "status": "PENDING", "raw": content}
    return None


# Agent Factory (per session)
def build_agent_for_session(session_id: str):
    wrapped_tools = [
        HILToolWrapper(
            inner_tool=tool,
            hil=hil_store,
            session_id=session_id,
            controlled=True
        )
        for tool in BASE_TOOLS
    ]

    agent = create_react_agent(
        model=llm,
        tools=wrapped_tools,
        prompt=SYSTEM_PROMPT,
        checkpointer=redis_saver,
    )
    return agent


# Chat Endpoint
@app.post("/chat")
async def chat(request: ChatRequest):
    agent = build_agent_for_session(request.session_id)

    config = {
        "configurable": {
            "thread_id": request.session_id
        }
    }

    result = await agent.ainvoke(
        {"messages": [HumanMessage(content=request.message)]},
        config=config,
    )

    # 1) If any tool emitted a pending marker, return a structured response
    pending = _extract_hil_pending_from_messages(result.get("messages", []))
    if pending:
        # Optional: log pending response as well (helps audit)
        postgres_logger.log_chat(
            session_id=request.session_id,
            user_message=request.message,
            agent_response=f"[HIL_PENDING] pending_id={pending.get('pending_id')}",
            meta={"source": "fastapi_endpoint", "hil": pending},
        )
        return {
            "session_id": request.session_id,
            "status": "PENDING",
            **pending,
        }

    # 2) Otherwise normal final response
    final_response = result["messages"][-1].content

    postgres_logger.log_chat(
        session_id=request.session_id,
        user_message=request.message,
        agent_response=final_response,
        meta={"source": "fastapi_endpoint"}
    )

    return {
        "session_id": request.session_id,
        "status": "OK",
        "response": final_response
    }


# HIL Approval Endpoint (approve only)
@app.post("/hil/approve")
async def approve_tool(request: ApprovalRequest):
    approved = hil_store.approve(
        request.pending_id,
        ttl_seconds=3600
    )
    return {
        "pending_id": request.pending_id,
        "approved": approved
    }


# HIL Resume Endpoint (approve + resume)
@app.post("/hil/resume")
async def resume_after_approval(request: ApprovalRequest):
    """
    Approve the pending tool call, then resume agent execution from checkpoint.
    No need to ask the same question again.
    """
    approved = hil_store.approve(request.pending_id, ttl_seconds=3600)
    if not approved:
        return {"status": "ERROR", "error": "Invalid pending_id"}

    pending = hil_store.get_pending(request.pending_id)
    if not pending:
        return {"status": "ERROR", "error": "Pending payload missing/expired"}

    session_id = pending["session_id"]
    agent = build_agent_for_session(session_id)

    config = {
        "configurable": {
            "thread_id": session_id
        }
    }

    # Key idea: do NOT add a new HumanMessage; let LangGraph resume from checkpoint
    result = await agent.ainvoke(
        {"messages": []},
        config=config,
    )

    # If it still hits another pending (e.g., another tool in the plan), return it
    next_pending = _extract_hil_pending_from_messages(result.get("messages", []))
    if next_pending:
        return {
            "session_id": session_id,
            "status": "PENDING",
            **next_pending,
        }

    final_response = result["messages"][-1].content

    postgres_logger.log_chat(
        session_id=session_id,
        user_message=f"[HIL_RESUME] pending_id={request.pending_id}",
        agent_response=final_response,
        meta={"source": "hil_resume_endpoint", "approved_pending": pending}
    )

    return {
        "session_id": session_id,
        "status": "OK",
        "response": final_response
    }

# Retrieve Pending HIL State
@app.get("/hil/pending/{pending_id}")
async def get_pending_state(pending_id: str):
    pending_data = hil_store.get_pending(pending_id)
    return {
        "pending": pending_data
    }