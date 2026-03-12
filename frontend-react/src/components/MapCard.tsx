import { Card } from "antd"
import { useEffect, useRef } from "react"
import AMapLoader from "@amap/amap-jsapi-loader"

export default function MapCard({ plan }: any) {

  const mapRef = useRef<any>(null)
  const mapInstance = useRef<any>(null)

  useEffect(() => {

    if (!plan) return

    initMap()

  }, [plan])

  const initMap = async () => {

    const AMap = await AMapLoader.load({
      key: import.meta.env.VITE_AMAP_WEB_JS_KEY,
      version: "2.0"
    })


    if (!mapInstance.current) {

      mapInstance.current = new AMap.Map(mapRef.current, {
        zoom: 12
      })

    }

    const map = mapInstance.current

    const markers: any[] = []

    plan.days.forEach((day: any) => {

      day.attractions.forEach((attr: any, index: number) => {

        if (attr.location) {

          const marker = new AMap.Marker({
            position: [
              attr.location.longitude,
              attr.location.latitude
            ],
            title: attr.name,
            label: {
              content: `${index + 1}`,
              direction: "top"
            }
          })

          map.add(marker)
          markers.push(marker)

        }

      })

    })

    if (markers.length > 0) {
      map.setFitView(markers)
    }

  }

  return (

    <Card className="section-card map-bg" title="📍 Map">

      <div
        ref={mapRef}
        style={{ height: 400, width: "100%" }}
      />

    </Card>

  )

}