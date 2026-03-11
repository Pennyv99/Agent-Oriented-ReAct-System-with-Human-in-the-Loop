import { Card, Row, Col } from "antd"

export default function BudgetCard({budget}:any){

  return(

    <Card title="💰 Budget">

      <Row gutter={16}>

        <Col span={6}>Attractions ¥{budget.total_attractions}</Col>
        <Col span={6}>Hotels ¥{budget.total_hotels}</Col>
        <Col span={6}>Meals ¥{budget.total_meals}</Col>
        <Col span={6}>Transport ¥{budget.total_transportation}</Col>

      </Row>

      <h2 style={{marginTop:20}}>
        Total ¥{budget.total}
      </h2>

    </Card>

  )

}