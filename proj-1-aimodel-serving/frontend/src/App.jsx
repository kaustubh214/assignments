import React, { useState } from "react";
import "./App.css";

function App() {
  const [activeTab, setActiveTab] = useState("sentiment");
  const [sentimentInput, setSentimentInput] = useState("");
  const [sentimentResult, setSentimentResult] = useState(null);
  const [genInput, setGenInput] = useState("");
  const [genResult, setGenResult] = useState("");
  const [loading, setLoading] = useState(false);
  const [selectedModel, setSelectedModel] = useState("zephyr");

  const API_URL = import.meta.env.VITE_API_BASE_URL;

  const MODELS = [
    {
      value: "zephyr",
      label: "Llama 3.3 70B",
      description: "Smart, accurate — via Groq API",
      badge: "Recommended",
    },
    {
      value: "gpt2",
      label: "GPT-2",
      description: "Runs locally — fast but limited",
      badge: "Local",
    },
  ];

  const handleSentimentSubmit = async () => {
    if (!sentimentInput.trim()) return;
    setLoading(true);
    setSentimentResult(null);
    try {
      const response = await fetch(`${API_URL}/api/sentiment`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: sentimentInput }),
      });
      const data = await response.json();
      if (response.ok) setSentimentResult(data);
      else alert(data.detail || "Error processing request");
    } catch {
      alert("Could not connect to backend server. Make sure it is running!");
    } finally {
      setLoading(false);
    }
  };

  const handleGenerationSubmit = async () => {
    if (!genInput.trim()) return;
    setLoading(true);
    setGenResult("");
    try {
      const response = await fetch(`${API_URL}/api/generate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          prompt: genInput,
          max_tokens: 512,
          model: selectedModel,
        }),
      });
      const data = await response.json();
      if (response.ok) setGenResult(data.generated_text);
      else alert(data.detail || "Error processing request");
    } catch {
      alert("Could not connect to backend server. Make sure it is running!");
    } finally {
      setLoading(false);
    }
  };

  const sentimentClass = sentimentResult?.label?.toLowerCase();
  const activeModel = MODELS.find((m) => m.value === selectedModel);

  return (
    <div className="hub">
      <header className="hub-header">
        <h1>AI Model Hub</h1>
        <p>FastAPI + Hugging Face + Groq Pipeline Service</p>
      </header>

      <div className="pill-tabs">
        <button
          className={`pill-tab ${activeTab === "sentiment" ? "active" : ""}`}
          onClick={() => setActiveTab("sentiment")}
        >
          <span className="tab-icon">🎭</span> Sentiment Analysis
        </button>
        <button
          className={`pill-tab ${activeTab === "generation" ? "active" : ""}`}
          onClick={() => setActiveTab("generation")}
        >
          <span className="tab-icon">✍️</span> Text Generation
        </button>
      </div>

      <div className="panel">
        {activeTab === "sentiment" ? (
          <>
            <h2>Analyze sentiment</h2>
            <textarea
              value={sentimentInput}
              onChange={(e) => setSentimentInput(e.target.value)}
              placeholder="Type any sentence — e.g. This project deployment goes incredibly smoothly!"
            />
            <button
              className="btn"
              onClick={handleSentimentSubmit}
              disabled={loading}
            >
              {loading ? "Analyzing…" : "Evaluate"}
            </button>

            {sentimentResult && (
              <div className="result-card">
                <p className="result-label">Result</p>
                <div className="sentiment-row">
                  <span className={`badge ${sentimentClass}`}>
                    {sentimentResult.label}
                  </span>
                  <div className="score-bar">
                    <div
                      className="score-fill"
                      style={{
                        width: `${(sentimentResult.score * 100).toFixed(1)}%`,
                      }}
                    />
                  </div>
                  <span className="score-text">
                    {(sentimentResult.score * 100).toFixed(1)}%
                  </span>
                </div>
              </div>
            )}
          </>
        ) : (
          <>
            <h2>Text generation engine</h2>

            <div className="model-selector">
              <p className="model-selector-label">Model</p>
              <div className="model-options">
                {MODELS.map((model) => (
                  <button
                    key={model.value}
                    className={`model-option ${
                      selectedModel === model.value ? "active" : ""
                    }`}
                    onClick={() => setSelectedModel(model.value)}
                  >
                    <div className="model-option-top">
                      <span className="model-option-name">{model.label}</span>
                      <span className={`model-badge ${model.value}`}>
                        {model.badge}
                      </span>
                    </div>
                    <span className="model-option-desc">
                      {model.description}
                    </span>
                  </button>
                ))}
              </div>
            </div>

            <textarea
              value={genInput}
              onChange={(e) => setGenInput(e.target.value)}
              placeholder={
                selectedModel === "zephyr"
                  ? "Ask anything — e.g. Explain quantum computing in simple terms"
                  : "Give GPT-2 a sentence to complete — e.g. The future of technology is"
              }
            />
            <button
              className="btn"
              onClick={handleGenerationSubmit}
              disabled={loading}
            >
              {loading
                ? selectedModel === "zephyr"
                  ? "Thinking…"
                  : "Generating…"
                : "Generate"}
            </button>

            {genResult && (
              <div className="result-card">
                <p className="result-label">Output · {activeModel?.label}</p>
                <p className="gen-output">{genResult}</p>
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
}

export default App;
