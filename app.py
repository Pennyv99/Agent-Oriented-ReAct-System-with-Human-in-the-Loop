import asyncio
from fastapi import FastAPI
from pydantic import BaseModel
from langgraph.prebuilt import create_react_agent
from langchain.chat_models import init_chat_model
from langchain_mcp_adapters.client import MultiServerMCPClient
from redis_memory import RedisSaver

app = FastAPI()

llm = init_chat_model(
    model="openai:deepseek-v3",
    temperature=0
)

# MCP client
client = MultiServerMCPClient({
    "amap": {
        "url": "https://mcp.amap.com/sse?key=你的key",
        "transport": "sse"
    }
})

tools = asyncio.run(client.get_tools())

checkpointer = RedisSaver()

agent = create_react_agent(
    model=llm,
    tools=tools,
    checkpointer=checkpointer
)

class ChatRequest(BaseModel):
    session_id: str
    message: str

@app.post("/chat")
async def chat(req: ChatRequest):
    config = {"configurable": {"thread_id": req.session_id}}

    result = await agent.ainvoke(
        {"messages": [("user", req.message)]},
        config=config
    )

    return {"response": result["messages"][-1].content}