import {useState} from "react"

import {
    Form,
    Input,
    DatePicker,
    Select,
    Checkbox,
    Button,
    Row,
    Col,
    Card,
    message
} from "antd"

import {useNavigate} from "react-router-dom"
import {generateTripPlan} from "../api/api"

const {TextArea} = Input

function Home() {

    const navigate = useNavigate()
    const [loading, setLoading] = useState(false)

    const onFinish = async (values: any) => {

        setLoading(true)

        try {

            const requestData = {

                city: values.city,

                start_date: values.start_date.format("YYYY-MM-DD"),
                end_date: values.end_date.format("YYYY-MM-DD"),

                travel_days:
                    values.end_date.diff(values.start_date, "day") + 1,

                transportation: values.transportation,
                accommodation: values.accommodation,

                preferences: values.preferences || [],

                free_text_input: values.free_text_input || ""
            }
            const res = await generateTripPlan(requestData)
            console.log("API response:", res)
            if (res.success) {
                sessionStorage.setItem(
                    "tripPlan",
                    JSON.stringify(res.data)
                )

                message.success("Trip generated successfully!")
                navigate("/result")
            } else {
                message.error(res.message || "Failed to generate trip plan")
            }

        } catch (e: any) {

            message.error(e.message)

        }

        setLoading(false)
    }

    return (

        <div>

            {/* NAVBAR */}

            <div
                style={{
                    height: 60,
                    background: "#021b34",
                    color: "white",
                    display: "flex",
                    alignItems: "center",
                    padding: "0 24px",
                    fontWeight: 600,
                    fontSize: 18
                }}
            >
                🌍 Multi-Agents AI Travel Planner
            </div>


            {/* HERO SECTION */}

            <div
                style={{
                    background: "linear-gradient(135deg,#6c7ae0,#7b4bb7)",
                    padding: "70px 20px",
                    textAlign: "center",
                    color: "white"
                }}
            >

                <div style={{fontSize: 80, animation: "float 20s ease-in-out infinite"}}>✈️</div>

                <h1 style={{
                    fontSize: 42,
                    marginTop: 20,
                    marginBottom: 10
                }}>
                    AI Travel Planner
                </h1>

                <p style={{
                    fontSize: 16,
                    opacity: 0.9
                }}>
                    Personalized AI-powered travel planning for every journey
                </p>

            </div>


            {/* FORM CARD */}

            <div
                style={{
                    maxWidth: 1100,
                    margin: "-80px auto 80px auto",
                    padding: "0 20px"
                }}
            >

                <Card
                    style={{
                        height: 600,
                        borderRadius: 20,
                        boxShadow: "0 10px 40px rgba(0,0,0,0.08)"
                    }}

                >

                    <Form
                        layout="vertical"
                        initialValues={{
                            transportation: "Public Transport",
                            accommodation: "Budget Hotel",
                            preferences: ["Nature"]
                        }}
                        onFinish={onFinish}
                    >

                        {/* DESTINATION SECTION */}

                        <h3 style={{fontWeight: 600}}>
                            📍 Destination & Dates
                        </h3>

                        <div style={{
                            height: 2,
                            background: "#6c7ae0",
                            marginBottom: 24
                        }}/>

                        <Row gutter={24,24}>

                            <Col span={8}>
                                <Form.Item
                                    label="Destination City"
                                    name="city"
                                    rules={[{required: true}]}
                                >
                                    <Input placeholder="e.g. Beijing"/>
                                </Form.Item>
                            </Col>

                            <Col span={8}>
                                <Form.Item label="Start Date" name="start_date">
                                    <DatePicker style={{width: "100%"}}/>
                                </Form.Item>
                            </Col>

                            <Col span={8}>
                                <Form.Item label="End Date" name="end_date">
                                    <DatePicker style={{width: "100%"}}/>
                                </Form.Item>
                            </Col>

                        </Row>


                        {/* PREFERENCES SECTION */}

                        <h3 style={{
                            fontWeight: 600,
                            marginTop: 40
                        }}>
                            ⚙️ Travel Preferences
                        </h3>

                        <div style={{
                            height: 2,
                            background: "#6c7ae0",
                            marginBottom: 24
                        }}/>

                        <Row gutter={16}>

                            <Col span={10}>
                                <Form.Item label="Transportation" name="transportation">

                                    <Select options={[

                                        {value: "Public Transport", label: "Public Transport"},
                                        {value: "Driving", label: "Driving"},
                                        {value: "Walking", label: "Walking"}

                                    ]}/>

                                </Form.Item>
                            </Col>

                            <Col span={10}>
                                <Form.Item label="Accommodation" name="accommodation">

                                    <Select options={[

                                        {value: "Budget Hotel", label: "Budget Hotel"},
                                        {value: "Luxury Hotel", label: "Luxury Hotel"},
                                        {value: "Airbnb", label: "Airbnb"}

                                    ]}/>

                                </Form.Item>
                            </Col>

                            <Col span={12}>
                                <Form.Item label="Interests" name="preferences">

                                    <Checkbox.Group
                                        options={[
                                            "History",
                                            "Nature",
                                            "Food",
                                            "Shopping",
                                            "Art",
                                            "Leisure"
                                        ]}
                                    />

                                </Form.Item>
                            </Col>

                        </Row>


                        {/* EXTRA REQUIREMENTS */}

                        <Form.Item
                            label="Additional Requirements"
                            name="free_text_input"
                        >
                            <TextArea
                                rows={4}
                                placeholder="Any special preferences? e.g. food, museums, relaxed itinerary..."
                            />
                        </Form.Item>


                        {/* SUBMIT BUTTON */}

                        <Button
                            type="primary"
                            htmlType="submit"
                            loading={loading}
                            block
                            size="large"
                            style={{
                                height: 50,
                                borderRadius: 10
                            }}
                        >
                            Generate Trip Plan
                        </Button>

                    </Form>

                </Card>

            </div>

        </div>

    )

}

export default Home