# app.py
from dotenv import load_dotenv
load_dotenv()

import os
import asyncio
from fastapi import FastAPI
from pydantic import BaseModel

from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.prebuilt import create_react_agent
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.checkpoint.redis import RedisSaver

from pg_logger import PGLogger
from hil_store import HILStore
from hil_tool_wrapper import HILToolWrapper

# FastAPI Application Initialization

app = FastAPI(title="Agent-Oriented ReAct Service")

# LLM Initialization

llm = init_chat_model(
    model=os.getenv("LLM_MODEL", "openai:deepseek-v3"),
    temperature=float(os.getenv("LLM_TEMPERATURE", "0")),
    base_url=os.getenv("LLM_BASE_URL"),
    api_key=os.getenv("LLM_API_KEY"),
)

# MCP Tool Loading (Executed Once at Startup)

mcp_client = MultiServerMCPClient({
    "amap": {
        "url": os.getenv("AMAP_MCP_URL"),  # Example: https://mcp.amap.com/sse?key=xxxx
        "transport": "sse",
    }
})

BASE_TOOLS = asyncio.get_event_loop().run_until_complete(
    mcp_client.get_tools()
)

# Infrastructure Components

postgres_logger = PGLogger()
hil_store = HILStore()
redis_saver = RedisSaver.from_conn_string(
    os.getenv("REDIS_URL", "redis://localhost:6379")
)

SYSTEM_PROMPT = SystemMessage(
    content="You are an AI assistant capable of using external tools such as maps and weather services when necessary."
)

# Request Schemas

class ChatRequest(BaseModel):
    session_id: str
    message: str


class ApprovalRequest(BaseModel):
    pending_id: str


# Agent Factory (Per Session)

def build_agent_for_session(session_id: str):
    """
    Builds a session-scoped agent instance.
    Each session applies Human-in-the-Loop (HIL)
    control over tool execution.
    """

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
        # To enable persistent checkpointing,
        checkpointer=redis_saver
    )

    return agent


# Chat Endpoint

@app.post("/chat")
async def chat(request: ChatRequest):
    """
    Main conversational endpoint.
    Handles:
    - ReAct agent execution
    - HIL-controlled tool orchestration
    - PostgreSQL logging
    """

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

    final_response = result["messages"][-1].content

    # Persist conversation log into PostgreSQL
    postgres_logger.log_chat(
        session_id=request.session_id,
        user_message=request.message,
        agent_response=final_response,
        meta={"source": "fastapi_endpoint"}
    )

    return {
        "session_id": request.session_id,
        "response": final_response
    }


# HIL Approval Endpoint

@app.post("/hil/approve")
async def approve_tool(request: ApprovalRequest):
    """
    Approves a pending tool execution.
    Once approved, future identical tool calls
    within the session will execute automatically.
    """

    approved = hil_store.approve(
        request.pending_id,
        ttl_seconds=3600
    )

    return {
        "pending_id": request.pending_id,
        "approved": approved
    }

# Retrieve Pending HIL State

@app.get("/hil/pending/{pending_id}")
async def get_pending_state(pending_id: str):
    """
    Returns pending tool call information.
    Useful for monitoring and audit.
    """

    pending_data = hil_store.get_pending(pending_id)

    return {
        "pending": pending_data
    }