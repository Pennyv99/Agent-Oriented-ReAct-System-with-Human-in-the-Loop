import React, { useState } from "react";
import "./App.css";

const API = "http://127.0.0.1:8000";

const session_id = "test-session2";
const user_id = "test-user";

function App() {

  const [input, setInput] = useState("");
  const [messages, setMessages] = useState([]);
  const [interrupt, setInterrupt] = useState(null);

  const sendMessage = async () => {

    if (!input) return;

    const newMessages = [
      ...messages,
      { role: "user", content: input }
    ];

    setMessages(newMessages);
    setInput("");

    const res = await fetch(`${API}/chat`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({
        session_id,
        user_id,
        message: input
      })
    });

    const data = await res.json();

    if (data.status === "PENDING") {

      setInterrupt(data.interrupt);

      setMessages([
        ...newMessages,
        { role: "assistant", content: "⚠️ Tool requires approval" }
      ]);

    } else {

      setMessages([
        ...newMessages,
        { role: "assistant", content: data.response }
      ]);

    }
  };

  const resume = async (decision) => {

    const res = await fetch(`${API}/hil/resume`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({
        session_id,
        user_id,
        decisions: [
          { type: decision }
        ]
      })
    });

    const data = await res.json();

    setInterrupt(null);

    setMessages(prev => [
      ...prev,
      { role: "assistant", content: data.response }
    ]);
  };

  return (
    <div className="app">

      <div className="chat">

        {messages.map((m, i) => (
          <div key={i} className={m.role === "user" ? "user" : "assistant"}>
            {m.content}
          </div>
        ))}

        {interrupt && (
          <div className="hitl">
            <button onClick={() => resume("approve")}>Approve</button>
            <button onClick={() => resume("reject")}>Reject</button>
          </div>
        )}

      </div>

      <div className="input-area">

        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Send a message..."
        />

        <button onClick={sendMessage}>
          Send
        </button>

      </div>

    </div>
  );
}

export default App;