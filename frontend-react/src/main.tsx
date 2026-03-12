import React from "react"
import ReactDOM from "react-dom/client"
import App from "./App"

import "antd/dist/reset.css"
import "./index.css"


window._AMapSecurityConfig = {
  securityJsCode: import.meta.env.VITE_AMAP_SECURITY_CODE
}

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
)