import React from 'react';

export default function ExplanationPanel({ selectedEvent }) {
  const explanationText =
    selectedEvent?.explanation ||
    "Person #01 picked up Laptop #02 at 00:05 and carried it away from Table #01.";

  return (
    <div className="panel explanation-panel-card">
      <div className="panel-header">
        <div className="panel-title-group">
          <svg className="panel-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" />
            <line x1="9" y1="10" x2="15" y2="10" />
            <line x1="9" y1="14" x2="13" y2="14" />
          </svg>
          <div>
            <h2 className="panel-title">EXPLANATION</h2>
            <span className="panel-subtitle">Natural Language Reasoning Engine</span>
          </div>
        </div>
        <div className="reasoning-pill">
          <span className="sparkle-icon">✦</span>
          <span>Temporal Reasoning Output</span>
        </div>
      </div>

      <div className="explanation-content-box">
        <div className="quote-mark">“</div>
        <p className="explanation-text">{explanationText}</p>
        <div className="explanation-meta">
          <span className="meta-tag">Event Reference: <strong>{selectedEvent?.type || "PICK_UP"} ({selectedEvent?.timestamp || "00:05"})</strong></span>
          <span className="meta-divider">•</span>
          <span className="meta-tag">Grounding: <strong>Verified by 3 Evidence Anchors</strong></span>
        </div>
      </div>
    </div>
  );
}
