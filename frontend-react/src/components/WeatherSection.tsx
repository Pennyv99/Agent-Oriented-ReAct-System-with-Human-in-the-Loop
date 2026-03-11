import { Card, Row, Col } from "antd"

export default function WeatherSection({weather}:any){

  return(

    <Card title="Weather">

      <Row gutter={16}>

        {weather.map((w:any,index:number)=>(

          <Col span={8} key={index}>

            <Card size="small">

              <p>{w.date}</p>

              <p>☀ {w.day_weather} {w.day_temp}°C</p>

              <p>🌙 {w.night_weather} {w.night_temp}°C</p>

              <p>💨 {w.wind_direction} {w.wind_power}</p>

            </Card>

          </Col>

        ))}

      </Row>

    </Card>

  )

}