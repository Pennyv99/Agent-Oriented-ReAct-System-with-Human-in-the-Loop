import axios from "axios"
import type { TripFormData, TripPlanResponse } from "../types"

const api = axios.create({
  baseURL: "/",
  timeout: 120000
})

export async function generateTripPlan(
  data: TripFormData
): Promise<TripPlanResponse> {
  const res = await api.post("/api/trip/plan", data)
  return res.data
}

export async function getPhoto(name: string) {
  const res = await api.get("/api/poi/photo", {
    params: { name }
  })
  return res.data
}

export default api