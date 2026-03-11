export interface TripPlan {

  city: string

  start_date: string
  end_date: string

  overall_suggestions: string

  days: DayPlan[]

  budget?: Budget

  weather_info?: WeatherInfo[]
}

export interface DayPlan {

  day_index: number
  date: string

  description: string
  transportation: string
  accommodation: string

  attractions: Attraction[]

  hotel?: Hotel

  meals: Meal[]
}

export interface Attraction {

  name: string

  address: string

  description: string

  visit_duration: number

  ticket_price?: number

  rating?: number

  location?: {
    longitude: number
    latitude: number
  }
}

export interface Hotel {

  name: string

  address: string

  type: string

  price_range: string

  rating: number

  distance: string
}

export interface Meal {

  type: string

  name: string

  description?: string
}

export interface Budget {

  total_attractions: number
  total_hotels: number
  total_meals: number
  total_transportation: number
  total: number
}

export interface WeatherInfo {

  date: string

  day_weather: string
  day_temp: number

  night_weather: string
  night_temp: number

  wind_direction: string
  wind_power: string
}
export interface TripFormData {

  city: string

  start_date: string
  end_date: string

  transportation: string

  accommodation: string

  preferences: string[]

  extra_requirements?: string

}