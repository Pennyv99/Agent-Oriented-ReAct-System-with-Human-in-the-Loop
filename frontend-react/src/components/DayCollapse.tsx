import {Collapse, Card, Input} from "antd"

export default function DayCollapse({days, editMode}: any) {

    return (

        <Card title="📅 Daily Plan">

            <Collapse accordion >

                {(days || []).map((day: any, index: number) => (

                    <Collapse.Panel className="section-card plan-bg"
                        header={`Day ${index + 1} ${day.date}`}
                        key={index}
                    >

                        <p>{day.description}</p>

                        {day.attractions.map((attr: any, i: number) => (

                            <Card
                                key={i}
                                size="small"
                                title={attr.name}
                                style={{marginBottom: 10}}
                            >

                                {editMode ? (

                                    <>
                                        <Input defaultValue={attr.address}/>
                                        <Input defaultValue={attr.description}/>
                                    </>

                                ) : (
                                    <>
                                        <p>{attr.address}</p>
                                        <p>{attr.description}</p>
                                    </>
                                )}

                            </Card>

                        ))}

                    </Collapse.Panel>

                ))}

            </Collapse>

        </Card>

    )

}