import {Card, Row, Col} from "antd"

export default function BudgetCard({budget}: any) {

    return (

        <Card title="💰 Budget">

            <Row gutter={16} style={{marginTop: 8}}>

                <Col span={6}>
                    <div className="budget-card attractions">
                        <div className="budget-title">📸 Attractions</div>
                        <div className="budget-value">¥{budget.total_attractions}</div>
                    </div>
                </Col>

                <Col span={6}>
                    <div className="budget-card hotels">
                        <div className="budget-title">🏢 Hotels</div>
                        <div className="budget-value">¥{budget.total_hotels}</div>
                    </div>
                </Col>

                <Col span={6}>
                    <div className="budget-card meals">
                        <div className="budget-title">🥢 Meals</div>
                        <div className="budget-value">¥{budget.total_meals}</div>
                    </div>
                </Col>

                <Col span={6}>
                    <div className="budget-card transport">
                        <div className="budget-title">🚇 Transport</div>
                        <div className="budget-value">¥{budget.total_transportation}</div>
                    </div>
                </Col>

            </Row>

            <div style={{
                fontSize: 28,
                fontWeight: 600,
                marginTop: 20
            }}>
                Total ¥{budget.total}
            </div>
        </Card>

    )

}