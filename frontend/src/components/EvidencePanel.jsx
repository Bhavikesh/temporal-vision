import React from 'react';

/**
 * Format raw backend evidence into displayable items
 */
function parseEvidenceItems(evidence) {
  if (!evidence) return [];

  // If already an array of items
  if (Array.isArray(evidence)) {
    return evidence.map((item) => {
      if (typeof item === 'string') return { label: item, verified: true };
      return { label: item.label || item.description || JSON.stringify(item), verified: item.verified !== false };
    });
  }

  // If object with items array
  if (Array.isArray(evidence.items)) {
    return evidence.items.map((item) => {
      if (typeof item === 'string') return { label: item, verified: true };
      return { label: item.label || item.description || String(item), verified: item.verified !== false };
    });
  }

  // If key-value map e.g. { distance: "0.42m", positionChange: true, synchronizedMotion: true }
  if (typeof evidence === 'object') {
    return Object.entries(evidence)
      .filter(([k]) => k !== 'items')
      .map(([key, val]) => {
        const formattedKey = key.replace(/([A-Z])/g, ' $1').replace(/_/g, ' ');
        const label = typeof val === 'boolean' 
          ? (val ? `${formattedKey.charAt(0).toUpperCase() + formattedKey.slice(1)} detected` : `${formattedKey} not observed`)
          : `${formattedKey.charAt(0).toUpperCase() + formattedKey.slice(1)}: ${val}`;
        return { label, verified: val !== false };
      });
  }

  return [{ label: String(evidence), verified: true }];
}

export default function EvidencePanel({ selectedEvent }) {
  if (!selectedEvent) {
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
        </div>
        <div className="empty-panel-msg">No event selected.</div>
      </div>
    );
  }

  const confidencePercent = typeof selectedEvent.confidence === 'number'
    ? Math.round(selectedEvent.confidence <= 1 ? selectedEvent.confidence * 100 : selectedEvent.confidence)
    : 94;

  const evidenceItems = parseEvidenceItems(selectedEvent.evidence);

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
            {(selectedEvent.type || selectedEvent.event || selectedEvent.name || 'ACTION').replace('_', ' ')}
          </span>
        </div>

        <div className="evidence-stat-card">
          <span className="stat-label">TIMESTAMP</span>
          <span className="stat-value mono-val">{selectedEvent.timestamp || selectedEvent.time || '00:00'}</span>
        </div>

        <div className="evidence-stat-card">
          <span className="stat-label">SUBJECT</span>
          <span className="stat-value">{selectedEvent.subject || selectedEvent.actor || 'Person #01'}</span>
        </div>

        <div className="evidence-stat-card">
          <span className="stat-label">OBJECT</span>
          <span className="stat-value">{selectedEvent.object || selectedEvent.target || 'Laptop #02'}</span>
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
            style={{ width: `${Math.min(100, Math.max(0, confidencePercent))}%` }}
          ></div>
        </div>
      </div>

      <div className="evidence-checklist-section">
        <h3 className="checklist-heading">Evidence Proof Points</h3>
        {evidenceItems.length > 0 ? (
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
        ) : (
          <div className="no-evidence-note">No specific evidence points attached to this event.</div>
        )}
      </div>
    </div>
  );
}
