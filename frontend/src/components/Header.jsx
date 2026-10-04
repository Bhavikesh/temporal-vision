import React from 'react';

export default function Header({
  backendStatus = 'connected_mock',
  onRetry = null,
  onToggleMode = null
}) {
  const isOffline = backendStatus === 'offline';
  const isConnectedMock = backendStatus === 'connected_mock';
  const isConnectedLive = backendStatus === 'connected_live';

  return (
    <header className="app-header">
      <div className="header-left">
        <div className="logo-badge">
          <svg className="logo-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M2 12s3-7 10-7 10 7 10 7-3 7-10 7-10-7-10-7Z" />
            <circle cx="12" cy="12" r="3" />
            <path d="M12 2v3M12 19v3M2 12h3M19 12h3" strokeOpacity="0.4" />
          </svg>
          <div>
            <h1 className="app-title">TEMPORAL VISION</h1>
            <p className="app-subtitle">Visual Intelligence Dashboard</p>
          </div>
        </div>
      </div>

      <div className="pipeline-flow">
        <div className="flow-step">
          <span className="step-num">1</span>
          <span>Video</span>
        </div>
        <span className="flow-arrow">→</span>
        <div className="flow-step">
          <span className="step-num">2</span>
          <span>Object Detection</span>
        </div>
        <span className="flow-arrow">→</span>
        <div className="flow-step">
          <span className="step-num">3</span>
          <span>Temporal</span>
        </div>
        <span className="flow-arrow">→</span>
        <div className="flow-step">
          <span className="step-num">4</span>
          <span>Events</span>
        </div>
        <span className="flow-arrow">→</span>
        <div className="flow-step">
          <span className="step-num">5</span>
          <span>Evidence</span>
        </div>
        <span className="flow-arrow">→</span>
        <div className="flow-step active-flow">
          <span className="step-num">6</span>
          <span>Explanation</span>
        </div>
      </div>

      <div className="header-right">
        {isOffline ? (
          <div className="status-pill offline-pill">
            <span className="status-dot offline-dot"></span>
            <span className="status-text">Backend Offline</span>
            {onRetry && (
              <button className="header-retry-btn" onClick={onRetry} title="Retry backend connection">
                Retry
              </button>
            )}
          </div>
        ) : (
          <div className="status-pill">
            <span className="status-dot"></span>
            <span className="status-text">Backend Online</span>
          </div>
        )}

        <div
          className={`mode-badge ${isOffline ? 'offline-mode' : isConnectedMock ? 'mock-mode' : 'live-mode'}`}
          onClick={onToggleMode}
          title={onToggleMode ? "Click to toggle backend mock / live query" : undefined}
          style={onToggleMode ? { cursor: 'pointer' } : {}}
        >
          {isOffline 
            ? "Backend Offline — Local Fallback" 
            : isConnectedMock 
            ? "Backend Connected — Mock Results" 
            : "Backend Connected — Live Results"}
        </div>
      </div>
    </header>
  );
}
