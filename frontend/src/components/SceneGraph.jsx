import React from 'react';

export default function SceneGraph({ selectedEvent }) {
  const subjectName = selectedEvent?.subject || selectedEvent?.actor || 'person_01';
  const objectName = selectedEvent?.object || selectedEvent?.target || 'laptop_01';
  const eventType = (selectedEvent?.type || selectedEvent?.event || 'APPROACH').toUpperCase();

  const isPickUp = eventType === 'PICK_UP';
  const isApproach = eventType === 'APPROACH';
  const isReach = eventType === 'REACH';
  const isCarry = eventType === 'CARRY';

  // Relation label
  const activeRelation = isPickUp
    ? 'holding'
    : isCarry
    ? 'carrying'
    : isApproach
    ? 'approaching'
    : isReach
    ? 'reaching'
    : eventType.toLowerCase();

  const isDetached = isPickUp || isCarry;
  const contextAnchor = objectName.toLowerCase().includes('table') ? 'environment' : 'table_01';

  return (
    <div className="panel scene-graph-card">
      <div className="panel-header">
        <div className="panel-title-group">
          <svg className="panel-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <circle cx="6" cy="6" r="3" />
            <circle cx="6" cy="18" r="3" />
            <circle cx="18" cy="12" r="3" />
            <line x1="8.5" y1="7.5" x2="15.5" y2="10.5" />
            <line x1="8.5" y1="16.5" x2="15.5" y2="13.5" />
            <line x1="6" y1="9" x2="6" y2="15" />
          </svg>
          <div>
            <h2 className="panel-title">SCENE GRAPH</h2>
            <span className="panel-subtitle">Spatio-Temporal Relationships</span>
          </div>
        </div>
        <span className="graph-state-tag">
          State: {activeRelation.charAt(0).toUpperCase() + activeRelation.slice(1)} [Active]
        </span>
      </div>

      <div className="scene-graph-canvas">
        {/* Subject Node */}
        <div className="graph-node node-person">
          <div className="node-icon-box">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M19 21v-2a4 4 0 0 0-4-4H9a4 4 0 0 0-4 4v2" />
              <circle cx="12" cy="7" r="4" />
            </svg>
          </div>
          <div className="node-info">
            <span className="node-id">{subjectName}</span>
            <span className="node-type">Subject</span>
          </div>
        </div>

        {/* Relational Edge Center Hub */}
        <div className="graph-edges-column">
          <div className={`relation-edge active-edge ${isPickUp ? 'highlight-pickup pulse' : 'pulse'}`}>
            <span className="edge-label highlight-label">
              ● {activeRelation}
            </span>
            <span className="edge-arrow">↓</span>
          </div>

          <div className={`relation-edge ${!isDetached ? 'active-edge' : 'lifted-edge'}`}>
            <span className="edge-label">
              {isDetached ? "detached from surface" : `on ${contextAnchor} (stationary)`}
            </span>
            <span className="edge-arrow">↓</span>
          </div>
        </div>

        {/* Target Nodes */}
        <div className="graph-targets-column">
          {/* Target Object Node */}
          <div className={`graph-node node-laptop ${isPickUp || isCarry || isReach ? 'focused-node' : ''}`}>
            <div className="node-icon-box cyan-icon">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <rect width="18" height="12" x="3" y="4" rx="2" />
                <line x1="2" x2="22" y1="20" y2="20" />
              </svg>
            </div>
            <div className="node-info">
              <span className="node-id">{objectName}</span>
              <span className="node-type">Target Object</span>
            </div>
          </div>

          {/* Anchor Surface Node */}
          <div className={`graph-node node-table ${isApproach ? 'focused-node' : ''}`}>
            <div className="node-icon-box slate-icon">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <line x1="3" x2="21" y1="8" y2="8" strokeWidth="3" />
                <line x1="6" x2="6" y1="8" y2="20" strokeWidth="2" />
                <line x1="18" x2="18" y1="8" y2="20" strokeWidth="2" />
              </svg>
            </div>
            <div className="node-info">
              <span className="node-id">{contextAnchor}</span>
              <span className="node-type">Context Anchor</span>
            </div>
          </div>
        </div>
      </div>

      {/* Triplet representation summary */}
      <div className="triplets-summary">
        <span className="triplet-tag">
          <strong>{subjectName}</strong> → <em>{activeRelation}</em> → <strong>{objectName}</strong>
        </span>
        {!isDetached && contextAnchor !== objectName && (
          <span className="triplet-tag">
            <strong>{objectName}</strong> → <em>on</em> → <strong>{contextAnchor}</strong>
          </span>
        )}
      </div>
    </div>
  );
}

