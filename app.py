import os
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
load_dotenv()
from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_mcp_adapters.client import MultiServerMCPClient

# 🚨 引入官方最核心的原语
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langgraph.types import Command
from langgraph.checkpoint.redis import AsyncRedisSaver

from pg_logger import PGLogger

app = FastAPI(title="Agent-Oriented ReAct Service (Official HIL)")
print("API KEY:", os.getenv("LLM_API_KEY"))
# 初始化基础设施
llm = init_chat_model(
    model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
    temperature=float(os.getenv("LLM_TEMPERATURE", "0")),
    api_key=os.getenv("LLM_API_KEY"),
)
postgres_logger = PGLogger()

mcp_client = MultiServerMCPClient({
    "amap": {"url": os.getenv("AMAP_MCP_URL"), "transport": "sse"}
})

SYSTEM_PROMPT = SystemMessage(
    content="You are an AI assistant capable of using external tools. Answer concisely."
)

GLOBAL_AGENT = None


# ==========================================
# 1. 组装全局单例 Agent (纯官方中间件)
# ==========================================
@app.on_event("startup")
async def load_tools():
    global GLOBAL_AGENT
    base_tools = await mcp_client.get_tools()

    # 动态为所有 MCP 工具生成官方要求的 interrupt_on 拦截配置
    interrupt_on = {}
    for tool in base_tools:
        interrupt_on[tool.name] = {
            "allowed_decisions": ["approve", "reject", "edit"],
            "description": f"Calling {tool.name} requires human approval."
        }

    # 初始化官方的 HIL 中间件
    official_hitl_middleware = HumanInTheLoopMiddleware(
        interrupt_on=interrupt_on,
        description_prefix="[HIL INTERCEPT]"
    )

    redis_saver_cm = AsyncRedisSaver.from_conn_string(os.getenv("REDIS_URL", "redis://localhost:6379"))
    redis_saver = await redis_saver_cm.__aenter__()

    # 注入官方中间件构建 Agent
    GLOBAL_AGENT = create_agent(
        model=llm,
        tools=base_tools,  # 原生工具，不套任何马甲
        system_prompt=SYSTEM_PROMPT,
        middleware=[official_hitl_middleware],  # 纯官方配置
        checkpointer=redis_saver
    )


# ==========================================
# 2. FastAPI 极简路由接口
# ==========================================
class ChatRequest(BaseModel):
    session_id: str
    message: str


# 恢复请求：不再需要 pending_id，只要 session_id 和 决策(approve/reject)
class ResumeRequest(BaseModel):
    session_id: str
    decision: str = "approve"


@app.post("/chat")
async def chat(request: ChatRequest):
    config = {"configurable": {"thread_id": request.session_id}}

    # 让引擎去跑，官方中间件会在后台自动拦截
    result = await GLOBAL_AGENT.ainvoke(
        {"messages": [HumanMessage(content=request.message)]},
        config=config
    )

    # 官方中间件挂起时，会把拦截信息放在 "__interrupt__" 数组里
    if "__interrupt__" in result:
        # 提取官方的数据结构并直接返回
        hitl_req = result["__interrupt__"][0]
        action_requests = hitl_req.value.get("action_requests", [])

        return {
            "status": "PENDING",
            "session_id": request.session_id,
            "pending_actions": action_requests  # 里面包含了要调用的 tool_name 和 args
        }

    # 如果没被拦截，说明可以直接返回回答
    final_response = result["messages"][-1].content
    postgres_logger.log_chat(request.session_id, request.message, final_response, meta={"source": "chat"})

    return {"status": "OK", "response": final_response}


@app.post("/hil/resume")
async def resume_after_approval(request: ResumeRequest):
    config = {"configurable": {"thread_id": request.session_id}}

    # 1. 探查当前 Checkpoint，看大模型挂起了几个工具请求
    state = await GLOBAL_AGENT.aget_state(config)

    if not (state.tasks and state.tasks[0].interrupts):
        return {"status": "ERROR", "error": "当前 Session 没有正在等待审批的工具。"}

    # 取出官方中间件需要的结构
    interrupt_val = state.tasks[0].interrupts[0].value
    action_requests = interrupt_val.get("action_requests", [])

    # 2. 构造官方中间件要求的 decisions 列表 (如果并发调了多个工具，要全部同意)
    decisions = [{"type": request.decision}] * len(action_requests)

    # 3. 🚨 核心魔法：使用原生的 Command(resume=...) 注入决策，唤醒图引擎！
    result = await GLOBAL_AGENT.ainvoke(
        Command(resume={"decisions": decisions}),
        config=config
    )

    # 4. 再次防备连续调用下一个工具被挂起
    if "__interrupt__" in result:
        hitl_req = result["__interrupt__"][0]
        return {
            "status": "PENDING",
            "session_id": request.session_id,
            "pending_actions": hitl_req.value.get("action_requests", [])
        }

    # 正常输出大模型的最终总结
    final_response = result["messages"][-1].content
    postgres_logger.log_chat(request.session_id, f"[HIL_RESUME] {request.decision}", final_response,
                             meta={"source": "resume"})

    return {"status": "OK", "response": final_response}