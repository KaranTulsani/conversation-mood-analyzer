import React, { useState } from 'react';
import './App.css';

const API_URL = process.env.REACT_APP_API_URL;

function App() {
  const [text, setText] = useState("");
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);

  const analyzeConversation = async () => {
    if (!text.trim()) return;
    
    setLoading(true);
    setResults([]);

    const sentences = text
      .split(/\n/)
      .map(s => s.trim())
      .filter(Boolean);

    try {
      const res = await fetch(`${API_URL}/predict`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({ conversation: sentences })
      });

      const data = await res.json();
      setResults(data.results);
    } catch (err) {
      alert("Backend not reachable");
    }
    setLoading(false);
  };

  return (
    <div className="app-container">
      <div className="ambient-bg ambient-bg-1" />
      <div className="ambient-bg ambient-bg-2" />

      <div className="content-wrapper">
        {/* Header Section */}
        <header className="header">
          <div className="header-title-row">
            <div className="status-indicator" />
            <h1 className="main-title">Mood Trajectory</h1>
          </div>
          
          <p className="project-description">
            Real-time sentiment analysis for conversational data. Track emotional patterns,
            identify mood shifts, and visualize the trajectory of sentiment across dialogue sequences.
            Powered by deep learning models trained on multi-domain conversation datasets.
          </p>
          
          <div className="stats-row">
            <div className="stat-item">
              <span className="stat-label">MODEL:</span>
              <span className="stat-value">Transformer-LSTM</span>
            </div>
            <div className="stat-item">
              <span className="stat-label">ACCURACY:</span>
              <span className="stat-value">94.2%</span>
            </div>
            <div className="stat-item">
              <span className="stat-label">STATUS:</span>
              <span className="stat-value stat-active">● ACTIVE</span>
            </div>
          </div>
        </header>

        {/* Input Section */}
        <div className="input-section">
          <label className="input-label">
            Conversation Input
          </label>
          
          <textarea
            value={text}
            onChange={(e) => setText(e.target.value)}
            placeholder="Enter conversation text... (one sentence per line)"
            className="conversation-input"
          />
          
          <button
            onClick={analyzeConversation}
            disabled={loading || !text.trim()}
            className={`analyze-btn ${loading || !text.trim() ? 'disabled' : ''}`}
          >
            {loading ? '◌ Analyzing...' : '→ Analyze Sentiment'}
          </button>
        </div>

        {/* Results Section */}
        {results.length > 0 && (
          <div className="results-section">
            <div className="results-header">
              <div className="header-accent-line" />
              <h2 className="results-title">Analysis Results</h2>
              <span className="results-count">
                {results.length} {results.length === 1 ? 'Entry' : 'Entries'}
              </span>
            </div>
            
            <div className="results-grid">
              {results.map((item, idx) => (
                <div
                  key={idx}
                  className={`sentiment-card sentiment-${item.sentiment}`}
                  style={{ animationDelay: `${idx * 0.1}s` }}
                >
                  <div className="card-content">
                    <div className="card-index">
                      {idx + 1}
                    </div>
                    
                    <div className="card-body">
                      <p className="card-text">
                        {item.sentence}
                      </p>
                      
                      <div className="sentiment-badge">
                        <div className="sentiment-dot" />
                        <span className="sentiment-label">
                          {item.sentiment}
                        </span>
                      </div>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Footer */}
        <footer className="footer">
          Conversation Mood Trajectory Analysis v1.0 • Built with React + FastAPI
        </footer>
      </div>
    </div>
  );
}

export default App;