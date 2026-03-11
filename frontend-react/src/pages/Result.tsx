import {useEffect, useState} from "react"
import {Layout, Menu, Card, Empty, Button, Space, message} from "antd"
import {useNavigate} from "react-router-dom"
import html2canvas from "html2canvas"
import jsPDF from "jspdf"

import OverviewCard from "../components/OverviewCard"
import BudgetCard from "../components/BudgetCard"
import MapCard from "../components/MapCard"
import DayCollapse from "../components/DayCollapse"
import WeatherSection from "../components/WeatherSection"

import type {TripPlan} from "../types"

const {Sider, Content} = Layout

export default function Result() {

    const navigate = useNavigate()

    const [tripPlan, setTripPlan] = useState<TripPlan | null>(null)
    const [editMode, setEditMode] = useState(false)

    useEffect(() => {

        const data = sessionStorage.getItem("tripPlan")

        if (data) {
            setTripPlan(JSON.parse(data))
        }

    }, [])

    const goBack = () => {
        navigate("/")
    }

    const exportImage = async () => {

        const element = document.querySelector(".main-content") as HTMLElement

        const canvas = await html2canvas(element)

        const link = document.createElement("a")

        link.download = "trip-plan.png"

        link.href = canvas.toDataURL()

        link.click()

    }

    const exportPDF = async () => {

        const element = document.querySelector(".main-content") as HTMLElement

        const canvas = await html2canvas(element)

        const imgData = canvas.toDataURL("image/png")

        const pdf = new jsPDF()

        pdf.addImage(imgData, "PNG", 0, 0, 210, 0)

        pdf.save("trip-plan.pdf")

    }

    if (!tripPlan) {

        return (

            <Empty
                description="No travel plan found"
            >
                <Button type="primary" onClick={goBack}>
                    Back to Home
                </Button>
            </Empty>

        )

    }

    return (

        <Layout>

            <Sider
                width={220}
                style={{
                    background: "#ffffff",
                    borderRight: "1px solid #f0f0f0",
                    paddingTop: 20
                }}
            >

                <Menu
                    mode="inline"
                    style={{borderRight: "none"}}
                    items={[

                        {key: "overview", label: "📋 Overview"},
                        {key: "budget", label: "💰 Budget"},
                        {key: "map", label: "📍 Map"},
                        {key: "days", label: "📅 Daily Plan"},
                        {key: "weather", label: "🌤 Weather"}

                    ]}
                />

            </Sider>

            <Layout>

                <Content style={{
                    padding: 40,
                    background: "#f7f9fc",
                    minHeight: "100vh"
                }}>

                    <div style={{marginBottom: 20}}>

                        <Space>

                            <Button onClick={goBack}>
                                ← Back
                            </Button>

                            <Button onClick={() => setEditMode(!editMode)}>
                                {editMode ? "Save Changes" : "Edit Plan"}
                            </Button>

                            <Button onClick={exportImage}>
                                Export Image
                            </Button>

                            <Button onClick={exportPDF}>
                                Export PDF
                            </Button>

                        </Space>

                    </div>

                    <div className="main-content">

                        <OverviewCard plan={tripPlan}/>

                        {tripPlan.budget && (
                            <BudgetCard budget={tripPlan.budget}/>
                        )}

                        <MapCard plan={tripPlan}/>

                        <DayCollapse
                            days={tripPlan.days}
                            editMode={editMode}
                        />

                        {tripPlan.weather_info && (
                            <WeatherSection weather={tripPlan.weather_info}/>
                        )}

                    </div>

                </Content>

            </Layout>

        </Layout>

    )

}