import React from 'react';

export default function EventTimeline({ events = [], selectedEventId, onSelectEvent }) {
  return (
    <div className="panel event-timeline-card">
      <div className="panel-header">
        <div className="panel-title-group">
          <svg className="panel-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <circle cx="12" cy="12" r="10" />
            <polyline points="12 6 12 12 16 14" />
          </svg>
          <div>
            <h2 className="panel-title">EVENT TIMELINE</h2>
            <span className="panel-subtitle">Temporal Action Sequence</span>
          </div>
        </div>
        <span className="event-count-badge">{events.length} Events</span>
      </div>

      <div className="timeline-container">
        <div className="timeline-track-line"></div>
        {events.map((evt, idx) => {
          const isSelected = evt.id === selectedEventId || evt.type === selectedEventId;
          const confidencePercent = Math.round(evt.confidence * 100);

          return (
            <div
              key={evt.id || idx}
              className={`timeline-item ${isSelected ? 'selected' : ''}`}
              onClick={() => onSelectEvent(evt)}
              role="button"
              tabIndex={0}
              onKeyDown={(e) => {
                if (e.key === 'Enter' || e.key === ' ') {
                  onSelectEvent(evt);
                }
              }}
            >
              <div className="timeline-dot-wrapper">
                <span className={`timeline-dot ${isSelected ? 'active-dot' : ''}`}></span>
              </div>

              <div className="timeline-content-card">
                <div className="timeline-header-row">
                  <span className="timestamp-tag">{evt.timestamp}</span>
                  <span className={`event-type-badge ${evt.type.toLowerCase()}`}>
                    {evt.type.replace('_', ' ')}
                  </span>
                  <span className="confidence-pill">{confidencePercent}%</span>
                </div>

                <div className="timeline-relation-row">
                  <span className="subject-name">{evt.subject}</span>
                  <span className="relation-arrow">→</span>
                  <span className="object-name">{evt.object}</span>
                </div>

                {evt.description && (
                  <p className="timeline-desc">{evt.description}</p>
                )}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
