import React from 'react';

export default function EvidencePanel({ selectedEvent }) {
  if (!selectedEvent) return null;

  const confidencePercent = Math.round(selectedEvent.confidence * 100);
  const evidenceItems = selectedEvent.evidence?.items || [
    { label: "Distance decreased", verified: true },
    { label: "Object position changed", verified: true },
    { label: "Synchronized movement", verified: true }
  ];

  return (
    <div className="panel evidence-panel-card">
      <div className="panel-header">
        <div className="panel-title-group">
          <svg className="panel-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" />
            <path d="m9 12 2 2 4-4" />
          </svg>
          <div>
            <h2 className="panel-title">EVIDENCE</h2>
            <span className="panel-subtitle">Temporal Verification & Multimodal Proof</span>
          </div>
        </div>
        <span className="confidence-badge-lg">{confidencePercent}% Confidence</span>
      </div>

      <div className="evidence-grid">
        <div className="evidence-stat-card">
          <span className="stat-label">EVENT</span>
          <span className="stat-value highlight-event-name">
            {selectedEvent.type.replace('_', ' ')}
          </span>
        </div>

        <div className="evidence-stat-card">
          <span className="stat-label">TIMESTAMP</span>
          <span className="stat-value mono-val">{selectedEvent.timestamp}</span>
        </div>

        <div className="evidence-stat-card">
          <span className="stat-label">SUBJECT</span>
          <span className="stat-value">{selectedEvent.subject}</span>
        </div>

        <div className="evidence-stat-card">
          <span className="stat-label">OBJECT</span>
          <span className="stat-value">{selectedEvent.object}</span>
        </div>
      </div>

      <div className="confidence-meter-container">
        <div className="meter-label-row">
          <span className="meter-title">Classification Confidence</span>
          <span className="meter-percent">{confidencePercent}%</span>
        </div>
        <div className="meter-bar-track">
          <div
            className="meter-bar-fill"
            style={{ width: `${confidencePercent}%` }}
          ></div>
        </div>
      </div>

      <div className="evidence-checklist-section">
        <h3 className="checklist-heading">Evidence Proof Points</h3>
        <div className="checklist-items">
          {evidenceItems.map((item, idx) => (
            <div key={idx} className="evidence-check-item">
              <span className="check-icon-box">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3">
                  <polyline points="20 6 9 17 4 12" />
                </svg>
              </span>
              <span className="check-label">{item.label}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
