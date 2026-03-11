"""数据模型定义"""

from typing import List, Optional, Union
from pydantic import BaseModel, Field, field_validator
from datetime import date


# ============ 请求模型 ============

class TripRequest(BaseModel):
    """旅行规划请求"""
    """Travel planning request"""
    city: str = Field(..., description="Destination city", example="Beijing")
    start_date: str = Field(..., description="Start date YYYY-MM-DD", example="2025-06-01")
    end_date: str = Field(..., description="End date YYYY-MM-DD", example="2025-06-03")
    travel_days: int = Field(..., description="Number of travel days", ge=1, le=30, example=3)
    transportation: str = Field(..., description="Transportation method", example="Public transportation")
    accommodation: str = Field(..., description="Accommodation preference", example="Budget hotel")
    preferences: List[str] = Field(default=[], description="Travel preference tags",
                                   example=["History & Culture", "Food"])
    free_text_input: Optional[str] = Field(default="", description="Additional requirements",
                                           example="Prefer more museums")

    class Config:
        json_schema_extra = {
            "example": {
                "city": "Beijing",
                "transportation": "Public transportation",
                "accommodation": "Budget hotel",
                "preferences": ["History & Culture", "Food"],
                "free_text_input": "Prefer more museums"
            }
        }


class POISearchRequest(BaseModel):
    """POI search request"""
    keywords: str = Field(..., description="Search keywords", example="Forbidden City")
    city: str = Field(..., description="City", example="Beijing")
    citylimit: bool = Field(default=True, description="Limit search within the city")


class RouteRequest(BaseModel):
    """Route planning request"""
    origin_address: str = Field(..., description="Origin address")
    destination_address: str = Field(..., description="Destination address")
    origin_city: Optional[str] = Field(default=None, description="Origin city")
    destination_city: Optional[str] = Field(default=None, description="Destination city")
    route_type: str = Field(default="walking", description="Route type: walking/driving/transit")


# ============ 响应模型 ============

class Location(BaseModel):
    """Geographic location"""
    longitude: float = Field(..., description="Longitude")
    latitude: float = Field(..., description="Latitude")


class Attraction(BaseModel):
    """Attraction information"""
    name: str = Field(..., description="Attraction name")
    address: str = Field(..., description="Address")
    location: Location = Field(..., description="Coordinates")
    visit_duration: int = Field(..., description="Suggested visit duration (minutes)")
    description: str = Field(..., description="Attraction description")
    category: Optional[str] = Field(default="Attraction", description="Attraction category")
    rating: Optional[float] = Field(default=None, description="Rating")
    photos: Optional[List[str]] = Field(default_factory=list, description="List of attraction image URLs")
    poi_id: Optional[str] = Field(default="", description="POI ID")
    image_url: Optional[str] = Field(default=None, description="Image URL")
    ticket_price: int = Field(default=0, description="Ticket price (RMB)")


class Meal(BaseModel):
    """Meal information"""
    type: str = Field(..., description="Meal type: breakfast/lunch/dinner/snack")
    name: str = Field(..., description="Meal name")
    address: Optional[str] = Field(default=None, description="Address")
    location: Optional[Location] = Field(default=None, description="Coordinates")
    description: Optional[str] = Field(default=None, description="Description")
    estimated_cost: int = Field(default=0, description="Estimated cost (RMB)")

class Hotel(BaseModel):
    """Hotel information"""
    name: str = Field(..., description="Hotel name")
    address: str = Field(default="", description="Hotel address")
    location: Optional[Location] = Field(default=None, description="Hotel location")
    price_range: str = Field(default="", description="Price range")
    rating: str = Field(default="", description="Rating")
    distance: str = Field(default="", description="Distance from attractions")
    type: str = Field(default="", description="Hotel type")
    estimated_cost: int = Field(default=0, description="Estimated cost (RMB/night)")


class DayPlan(BaseModel):
    """Daily itinerary"""
    date: str = Field(..., description="Date YYYY-MM-DD")
    day_index: int = Field(..., description="Day index (starting from 0)")
    description: str = Field(..., description="Daily itinerary description")
    transportation: str = Field(..., description="Transportation method")
    accommodation: str = Field(..., description="Accommodation")
    hotel: Optional[Hotel] = Field(default=None, description="Recommended hotel")
    attractions: List[Attraction] = Field(default=[], description="Attractions list")
    meals: List[Meal] = Field(default=[], description="Meals list")


class WeatherInfo(BaseModel):
    """Weather information"""
    date: str = Field(..., description="Date YYYY-MM-DD")
    day_weather: str = Field(default="", description="Daytime weather")
    night_weather: str = Field(default="", description="Night weather")
    day_temp: Union[int, str] = Field(default=0, description="Day temperature")
    night_temp: Union[int, str] = Field(default=0, description="Night temperature")
    wind_direction: str = Field(default="", description="Wind direction")
    wind_power: str = Field(default="", description="Wind power")


    @field_validator('day_temp', 'night_temp', mode='before')
    @classmethod
    def parse_temperature(cls, v):
        """解析温度,移除°C等单位"""
        if isinstance(v, str):
            # 移除°C, ℃等单位符号
            v = v.replace('°C', '').replace('℃', '').replace('°', '').strip()
            try:
                return int(v)
            except ValueError:
                return 0
        return v


class Budget(BaseModel):
    """预算信息"""
    total_attractions: int = Field(default=0, description="Total attraction ticket cost")
    total_hotels: int = Field(default=0, description="Total hotel cost")
    total_meals: int = Field(default=0, description="Total meal cost")
    total_transportation: int = Field(default=0, description="Total transportation cost")
    total: int = Field(default=0, description="Total cost")


class TripPlan(BaseModel):
    """旅行计划"""
    city: str = Field(..., description="Destination city")
    start_date: str = Field(..., description="Start date")
    end_date: str = Field(..., description="End date")
    days: List[DayPlan] = Field(..., description="Daily itineraries")
    weather_info: List[WeatherInfo] = Field(default=[], description="Weather information")
    overall_suggestions: str = Field(..., description="Overall travel suggestions")
    budget: Optional[Budget] = Field(default=None, description="Budget information")


class TripPlanResponse(BaseModel):
    """旅行计划响应"""
    success: bool = Field(..., description="Whether the request was successful")
    message: str = Field(default="", description="Message")
    data: Optional[TripPlan] = Field(default=None, description="Travel plan data")


class POIInfo(BaseModel):
    """POI信息"""
    id: str = Field(..., description="POI ID")
    name: str = Field(..., description="Name")
    type: str = Field(..., description="Type")
    address: str = Field(..., description="Address")
    location: Location = Field(..., description="Coordinates")
    tel: Optional[str] = Field(default=None, description="Phone number")


class POISearchResponse(BaseModel):
    """POI搜索响应"""
    success: bool = Field(..., description="Whether the request was successful")
    message: str = Field(default="", description="Message")
    data: List[POIInfo] = Field(default=[], description="POI list")


class RouteInfo(BaseModel):
    """路线信息"""
    distance: float = Field(..., description="Distance (meters)")
    duration: int = Field(..., description="Duration (seconds)")
    route_type: str = Field(..., description="Route type")
    description: str = Field(..., description="Route description")


class RouteResponse(BaseModel):
    """路线规划响应"""
    success: bool = Field(..., description="Whether the request was successful")
    message: str = Field(default="", description="Message")
    data: Optional[RouteInfo] = Field(default=None, description="Route information")


class WeatherResponse(BaseModel):
    """天气查询响应"""
    success: bool = Field(..., description="Whether the request was successful")
    message: str = Field(default="", description="Message")
    data: List[WeatherInfo] = Field(default=[], description="Weather information")


# ============ 错误响应 ============

class ErrorResponse(BaseModel):
    """错误响应"""
    success: bool = Field(default=False, description="Whether the request was successful")
    message: str = Field(..., description="Error message")
    error_code: Optional[str] = Field(default=None, description="Error code")