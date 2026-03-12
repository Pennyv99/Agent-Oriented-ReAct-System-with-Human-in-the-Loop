import { Card } from "antd"
import type { TripPlan } from "../types"

export default function OverviewCard({plan}:{plan:TripPlan}){

  return(

    <Card className="section-card overview-bg" title={`${plan.city} Travel Plan`}>

      <p>
        📅 {plan.start_date} - {plan.end_date}
      </p>

      <p>
        💡 {plan.overall_suggestions}
      </p>

    </Card>

  )

}