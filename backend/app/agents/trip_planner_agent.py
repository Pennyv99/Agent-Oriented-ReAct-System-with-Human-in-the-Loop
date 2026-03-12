"""多智能体旅行规划系统"""

import json
from typing import Dict, Any, List
from hello_agents import SimpleAgent
from hello_agents.tools import MCPTool
from ..services.llm_service import get_llm
from ..models.schemas import TripRequest, TripPlan, DayPlan, Attraction, Meal, WeatherInfo, Location, Hotel
from ..config import get_settings
import asyncio
# ============ Agent提示词 ============

ATTRACTION_AGENT_PROMPT = """You are an expert in scenic spot search. Your task is to search for suitable scenic spots based on the city and user preferences.

**Important:**
You must use tools to search for attractions! Do not fabricate attraction information!

**Tool Call Format:**
When using the maps_mcp_text_search tool, you must strictly follow the format below:
`[TOOL_CALL:amap_mcp_maps_text_search:keywords=attraction_keyword,city=city_name]`

**Examples:**
User: "Search historical and cultural attractions in Beijing"
Your response: [TOOL_CALL:amap_mcp_maps_text_search:keywords=history culture,city=Beijing]

User: "Search parks in Shanghai"
Your response: [TOOL_CALL:amap_mcp_maps_text_search:keywords=park,city=Shanghai]

**Notes:**
1. You must use the tool, do not answer directly
2. The format must be completely correct, including brackets and colons
3. Parameters must be separated by commas
"""
WEATHER_AGENT_PROMPT = """
You are an expert in weather inquiries. Your task is to check the weather information of the specified city.

You MUST use the tool `amap_mcp_maps_weather` to get weather information.

Tool call format:

[TOOL_CALL:amap_mcp_maps_weather:city=city_name]

Rules:
- Always call the tool amap_mcp_maps_weather
- Do not generate fake weather data
- Do not answer directly
- Always return the tool call

Example:

User: What is the weather in Shanghai?
Assistant:
[TOOL_CALL:amap_mcp_maps_weather:city=Shanghai]
"""
# WEATHER_AGENT_PROMPT = """You are an expert in weather inquiries. Your task is to check the weather information of the specified city.
#
# **Important:**
# You must use tools to query weather! Do not fabricate weather information!
#
# **Tool Call Format:**
# When using the maps_weather tool, you must strictly follow the format below:
# `[TOOL_CALL:amap_maps_weather:city=city_name]`
#
# **Examples:**
# User: "Check the weather in Beijing"
# Your response: [TOOL_CALL:amap_maps_weather:city=Beijing]
#
# User: "What is the weather in Shanghai?"
# Your response: [TOOL_CALL:amap_maps_weather:city=Shanghai]
#
# **Notes:**
# 1. You must use the tool, do not answer directly
# 2. The format must be completely correct, including brackets and colons
# """

HOTEL_AGENT_PROMPT = """You are a hotel recommendation expert. Your task is to recommend suitable hotels based on the location of the city and scenic spots.

**Important:**
You must use tools to search for hotels! Do not fabricate hotel information!

**Tool Call Format:**
When using the maps_text_search tool to search hotels, you must strictly follow the format below:
`[TOOL_CALL:amap_mcp_maps_text_search:keywords=hotel,city=city_name]`

**Example:**
User: "Search hotels in Beijing"
Your response: [TOOL_CALL:amap_mcp_maps_text_search:keywords=hotel,city=Beijing]

**Notes:**
1. You must use the tool, do not answer directly
2. The format must be completely correct, including brackets and colons
3. Use the keyword "hotel" or "inn"
"""

PLANNER_AGENT_PROMPT = """You are an expert in itinerary planning. Your task is to generate a detailed travel plan based on the information of scenic spots and weather conditions.

Please strictly return the travel plan in the following JSON format:
```json
{
  "city": "City Name",
  "start_date": "YYYY-MM-DD",
  "end_date": "YYYY-MM-DD",
  "days": [
    {
      "date": "YYYY-MM-DD",
      "day_index": 0,
      "description": "Day 1 itinerary overview",
      "transportation": "Transportation method",
      "accommodation": "Accommodation type",
      "hotel": {
        "name": "Hotel Name",
        "address": "Hotel Address",
        "location": {"longitude": 116.397128, "latitude": 39.916527},
        "price_range": "300-500 RMB",
        "rating": "4.5",
        "distance": "2 km from attractions",
        "type": "Budget hotel",
        "estimated_cost": 400
      },
      "attractions": [
        {
          "name": "Attraction Name",
          "address": "Detailed Address",
          "location": {"longitude": 116.397128, "latitude": 39.916527},
          "visit_duration": 120,
          "description": "Detailed description of the attraction",
          "category": "Attraction category",
          "ticket_price": 60
        }
      ],
      "meals": [
        {"type": "breakfast", "name": "Breakfast recommendation", "description": "Breakfast description", "estimated_cost": 30},
        {"type": "lunch", "name": "Lunch recommendation", "description": "Lunch description", "estimated_cost": 50},
        {"type": "dinner", "name": "Dinner recommendation", "description": "Dinner description", "estimated_cost": 80}
      ]
    }
  ],
  "weather_info": [
    {
      "date": "YYYY-MM-DD",
      "day_weather": "Sunny",
      "night_weather": "Cloudy",
      "day_temp": 25,
      "night_temp": 15,
      "wind_direction": "South wind",
      "wind_power": "Level 1-3"
    }
  ],
  "overall_suggestions": "Overall travel suggestions",
  "budget": {
    "total_attractions": 180,
    "total_hotels": 1200,
    "total_meals": 480,
    "total_transportation": 200,
    "total": 2060
  }
}
```

**Important:**
1. The weather_info array must contain weather information for each day
2. Temperature must be numeric only (no °C units)
3. Arrange 2–3 attractions per day
4. Consider distances and visit durations between attractions
5. Each day must include breakfast, lunch, and dinner
6. Provide practical travel suggestions
7. **Budget information must be included**:
   - Attraction ticket price (ticket_price)
   - Estimated meal cost (estimated_cost)
   - Estimated hotel cost (estimated_cost)
   - Budget summary (budget) including total costs
8. ALL text must be in English.
"""


class MultiAgentTripPlanner:
    """多智能体旅行规划系统"""

    def __init__(self):
        """初始化多智能体系统"""
        print("🔄 Start initializing the multi-agent travel planning system...")

        try:
            settings = get_settings()
            self.llm = get_llm()

            # 创建共享的MCP工具(只创建一次)
            print("  - Create a shared MCP tool...")
            # self.amap_tool = MCPTool(
            #     name="amap_maps",
            #     description="amap maps serice",
            #     server_command=["uvx", "amap-mcp-server"],
            #     env={"AMAP_MAPS_API_KEY": settings.amap_api_key},
            # )
            self.amap_tool = MCPTool(
                name="amap_mcp",
                server_command=["uvx", "amap-mcp-server"],
                env={"AMAP_MAPS_API_KEY": settings.amap_api_key},
                auto_expand=True
            )
            self.amap_tool.expandable = True
            # asyncio.create_task(self.amap_tool._discover_tools())
            #
            # print("Available MCP tools:", self.amap_tool._available_tools)
            # print("Available MCP tools:", self.amap_tool.available_tools)
            # 创建景点搜索Agent
            print("  - Create a scenic spot search Agent...")
            self.attraction_agent = SimpleAgent(
                name="Attraction Search Expert",
                llm=self.llm,
                system_prompt=ATTRACTION_AGENT_PROMPT
            )
            self.attraction_agent.add_tool(self.amap_tool)

            # 创建天气查询Agent
            print("  - 创建天气查询Agent...")
            self.weather_agent = SimpleAgent(
                name="Weather Query Expert",
                llm=self.llm,
                system_prompt=WEATHER_AGENT_PROMPT
            )
            self.weather_agent.add_tool(self.amap_tool)
            print("Weather tools:", self.weather_agent.list_tools())

            # 创建酒店推荐Agent
            print("  - 创建酒店推荐Agent...")
            self.hotel_agent = SimpleAgent(
                name="Hotel Recommendation Expert",
                llm=self.llm,
                system_prompt=HOTEL_AGENT_PROMPT
            )
            self.hotel_agent.add_tool(self.amap_tool)

            # 创建行程规划Agent(不需要工具)
            print("  - 创建行程规划Agent...")
            self.planner_agent = SimpleAgent(
                name="Itinerary Planning Expert",
                llm=self.llm,
                system_prompt=PLANNER_AGENT_PROMPT
            )

            print(f"✅ 多智能体系统初始化成功")
            print(f"   景点搜索Agent: {len(self.attraction_agent.list_tools())} 个工具")
            print(f"   天气查询Agent: {len(self.weather_agent.list_tools())} 个工具")
            print(f"   酒店推荐Agent: {len(self.hotel_agent.list_tools())} 个工具")

        except Exception as e:
            print(f"❌ 多智能体系统初始化失败: {str(e)}")
            import traceback
            traceback.print_exc()
            raise

    def plan_trip(self, request: TripRequest) -> TripPlan:
        """
        使用多智能体协作生成旅行计划

        Args:
            request: 旅行请求

        Returns:
            旅行计划
        """
        try:
            print(f"\n{'='*60}")
            print(f"🚀 开始多智能体协作规划旅行...")
            print(f"目的地: {request.city}")
            print(f"日期: {request.start_date} 至 {request.end_date}")
            print(f"天数: {request.travel_days}天")
            print(f"偏好: {', '.join(request.preferences) if request.preferences else '无'}")
            print(f"{'='*60}\n")

            # 步骤1: 景点搜索Agent搜索景点
            print("📍 步骤1: 搜索景点...")
            attraction_query = self._build_attraction_query(request)
            attraction_response = self.attraction_agent.run(attraction_query)
            print(f"景点搜索结果: {attraction_response[:200]}...\n")

            # 步骤2: 天气查询Agent查询天气
            print("🌤️  步骤2: 查询天气...")
            weather_query = f"请查询{request.city}的天气信息"
            weather_response = self.weather_agent.run(weather_query)
            print(f"天气查询结果: {weather_response[:200]}...\n")

            # 步骤3: 酒店推荐Agent搜索酒店
            print("🏨 步骤3: 搜索酒店...")
            hotel_query = f"请搜索{request.city}的{request.accommodation}酒店"
            hotel_response = self.hotel_agent.run(hotel_query)
            print(f"酒店搜索结果: {hotel_response[:200]}...\n")

            # 步骤4: 行程规划Agent整合信息生成计划
            print("📋 步骤4: 生成行程计划...")
            planner_query = self._build_planner_query(request, attraction_response, weather_response, hotel_response)
            planner_response = self.planner_agent.run(planner_query)
            print(f"行程规划结果: {planner_response[:300]}...\n")

            # 解析最终计划
            trip_plan = self._parse_response(planner_response, request)

            print(f"{'='*60}")
            print(f"✅ 旅行计划生成完成!")
            print(f"{'='*60}\n")

            return trip_plan

        except Exception as e:
            print(f"❌ 生成旅行计划失败: {str(e)}")
            import traceback
            traceback.print_exc()
            return self._create_fallback_plan(request)
    
    def _build_attraction_query(self, request: TripRequest) -> str:
        """构建景点搜索查询 - 直接包含工具调用"""
        keywords = []
        if request.preferences:
            # 只取第一个偏好作为关键词
            keywords = request.preferences[0]
        else:
            keywords = "景点"

        # 直接返回工具调用格式
        query = f"Please use the amap_maps_text_search tool to search for {keywords} attractions in {request.city}.\n[TOOL_CALL:amap_maps_text_search:keywords={keywords},city={request.city}]"
        return query

    def _build_planner_query(self, request: TripRequest, attractions: str, weather: str, hotels: str = "") -> str:
        """构建行程规划查询"""
        query = f"""PPlease generate a {request.travel_days}-day travel itinerary for {request.city} based on the following information:

**Basic Information:**
- City: {request.city}
- Dates: {request.start_date} to {request.end_date}
- Days: {request.travel_days}
- Transportation: {request.transportation}
- Accommodation: {request.accommodation}
- Preferences: {', '.join(request.preferences) if request.preferences else 'None'}

**Attraction Information:**
{attractions}

**Weather Information:**
{weather}

**Hotel Information:**
{hotels}

**Requirements:**
1. Arrange 2–3 attractions per day
2. Each day must include breakfast, lunch, and dinner
3. Recommend a specific hotel each day (selected from hotel information)
4. Consider distances and transportation between attractions
5. Return the complete JSON format data
6. Attraction coordinates must be accurate
"""
        if request.free_text_input:
            query += f"\n**extra requirenments:** {request.free_text_input}"

        return query
    
    def _parse_response(self, response: str, request: TripRequest) -> TripPlan:
        """
        解析Agent响应
        
        Args:
            response: Agent响应文本
            request: 原始请求
            
        Returns:
            旅行计划
        """
        try:
            print(response)
            # 尝试从响应中提取JSON
            # 查找JSON代码块
            if "```json" in response:
                json_start = response.find("```json") + 7
                json_end = response.find("```", json_start)
                json_str = response[json_start:json_end].strip()
            elif "```" in response:
                json_start = response.find("```") + 3
                json_end = response.find("```", json_start)
                json_str = response[json_start:json_end].strip()
            elif "{" in response and "}" in response:
                # 直接查找JSON对象
                json_start = response.find("{")
                json_end = response.rfind("}") + 1
                json_str = response[json_start:json_end]
            else:
                raise ValueError("响应中未找到JSON数据")
            
            # 解析JSON
            data = json.loads(json_str)
            
            # 转换为TripPlan对象
            trip_plan = TripPlan(**data)
            
            return trip_plan
            
        except Exception as e:
            print(f"⚠️  解析响应失败: {str(e)}")
            print(f"   将使用备用方案生成计划")
            return self._create_fallback_plan(request)
    
    def _create_fallback_plan(self, request: TripRequest) -> TripPlan:
        """创建备用计划(当Agent失败时)"""
        from datetime import datetime, timedelta
        
        # 解析日期
        start_date = datetime.strptime(request.start_date, "%Y-%m-%d")
        
        # 创建每日行程
        days = []
        for i in range(request.travel_days):
            current_date = start_date + timedelta(days=i)
            
            day_plan = DayPlan(
                date=current_date.strftime("%Y-%m-%d"),
                day_index=i,
                description=f"Day {i+1} itinerary",
                transportation=request.transportation,
                accommodation=request.accommodation,
                attractions=[
                    Attraction(
                        name=f"{request.city} attraction {j+1}",
                        address=f"{request.city} city",
                        location=Location(longitude=116.4 + i*0.01 + j*0.005, latitude=39.9 + i*0.01 + j*0.005),
                        visit_duration=120,
                        description=f"This is a famous attraction in {request.city}",
                        category="attraction",
                    )
                    for j in range(2)
                ],
                meals=[
                    Meal(type="breakfast", name=f"Day {i + 1} breakfast", description="Local breakfast"),
                    Meal(type="lunch", name=f"Day {i + 1} lunch", description="Lunch recommendation"),
                    Meal(type="dinner", name=f"Day {i + 1} dinner", description="Dinner recommendation")
                ]
            )
            days.append(day_plan)
        
        return TripPlan(
            city=request.city,
            start_date=request.start_date,
            end_date=request.end_date,
            days=days,
            weather_info=[],
            overall_suggestions=f"This is a {request.travel_days}-day itinerary for {request.city}. It is recommended to check attraction opening hours in advance."
        )


# 全局多智能体系统实例
_multi_agent_planner = None


def get_trip_planner_agent() -> MultiAgentTripPlanner:
    """获取多智能体旅行规划系统实例(单例模式)"""
    global _multi_agent_planner

    if _multi_agent_planner is None:
        _multi_agent_planner = MultiAgentTripPlanner()

    return _multi_agent_planner

