import { Card } from "antd"
import { useEffect } from "react"
import AMapLoader from "@amap/amap-jsapi-loader"

export default function MapCard({plan}:any){

  useEffect(()=>{

    initMap()

  },[])

  const initMap = async()=>{

    const AMap = await AMapLoader.load({

      key: import.meta.env.VITE_AMAP_WEB_JS_KEY,
      version:"2.0"

    })

    const map = new AMap.Map("map-container",{
      zoom:12
    })

    plan.days.forEach((day:any)=>{

      day.attractions.forEach((attr:any)=>{

        if(attr.location){

          new AMap.Marker({

            map,
            position:[
              attr.location.longitude,
              attr.location.latitude
            ],
            title:attr.name

          })

        }

      })

    })

  }

  return(

    <Card title="📍 Map">

      <div
        id="map-container"
        style={{height:400}}
      />

    </Card>

  )

}