import React, { useState, useEffect } from 'react';
import './App.css';

const API_URL = process.env.REACT_APP_API_URL;

function App() {
  const [text, setText] = useState("");
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [apiStatus, setApiStatus] = useState('checking');

  // Check API health on component mount
  useEffect(() => {
    checkApiHealth();
  }, []);

  const checkApiHealth = async () => {
    try {
      const res = await fetch(`${API_URL}/health`, {
        method: "GET",
        headers: {
          "Content-Type": "application/json"
        }
      });
      
      if (res.ok) {
        setApiStatus('connected');
        setError(null);
      } else {
        setApiStatus('error');
        setError('API is not responding correctly');
      }
    } catch (err) {
      setApiStatus('error');
      setError(`Cannot connect to API at ${API_URL}`);
      console.error('API Health Check Error:', err);
    }
  };

  const analyzeConversation = async () => {
    if (!text.trim()) return;
    
    setLoading(true);
    setResults([]);
    setError(null);

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

      if (!res.ok) {
        const errorData = await res.json().catch(() => ({}));
        throw new Error(errorData.detail || `Server error: ${res.status}`);
      }

      const data = await res.json();
      setResults(data.results);
      setApiStatus('connected');
    } catch (err) {
      console.error('Analysis Error:', err);
      setError(err.message || "Backend not reachable. Please check if the API is running.");
      setApiStatus('error');
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
            <div className={`status-indicator ${apiStatus === 'connected' ? 'active' : apiStatus === 'error' ? 'error' : 'checking'}`} />
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
              <span className="stat-value stat-active">
                ● {apiStatus === 'connected' ? 'ACTIVE' : apiStatus === 'error' ? 'OFFLINE' : 'CHECKING...'}
              </span>
            </div>
          </div>

          {/* API Connection Warning */}
          {apiStatus === 'error' && (
            <div className="api-warning">
              <strong>⚠ API Connection Issue:</strong> {error}
              <button onClick={checkApiHealth} className="retry-btn">
                Retry Connection
              </button>
            </div>
          )}
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
            disabled={loading || !text.trim() || apiStatus === 'error'}
            className={`analyze-btn ${loading || !text.trim() || apiStatus === 'error' ? 'disabled' : ''}`}
          >
            {loading ? '◌ Analyzing...' : '→ Analyze Sentiment'}
          </button>
          
          {error && !loading && (
            <div className="error-message">
              {error}
            </div>
          )}
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
                        {item.text}
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
          <br />
          <small>API: {API_URL || 'Not configured'}</small>
        </footer>
      </div>
    </div>
  );
}

export default App;