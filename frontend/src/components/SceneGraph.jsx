import React from 'react';

export default function SceneGraph({ selectedEvent }) {
  const isPickUp = selectedEvent?.type === 'PICK_UP';
  const isApproach = selectedEvent?.type === 'APPROACH';
  const isReach = selectedEvent?.type === 'REACH';
  const isCarry = selectedEvent?.type === 'CARRY';

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
          {isPickUp ? "State: Holding [Active]" : isApproach ? "State: Approaching" : isReach ? "State: Reaching" : "State: Dynamic"}
        </span>
      </div>

      <div className="scene-graph-canvas">
        {/* Node: Person #01 */}
        <div className="graph-node node-person">
          <div className="node-icon-box">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M19 21v-2a4 4 0 0 0-4-4H9a4 4 0 0 0-4 4v2" />
              <circle cx="12" cy="7" r="4" />
            </svg>
          </div>
          <div className="node-info">
            <span className="node-id">Person #01</span>
            <span className="node-type">Subject (Human)</span>
          </div>
        </div>

        {/* Relational Edge Center Hub */}
        <div className="graph-edges-column">
          {/* Edge 1: Person -> Table or Person -> Laptop */}
          <div className={`relation-edge ${isApproach ? 'active-edge pulse' : 'inactive-edge'}`}>
            <span className="edge-label">approaching</span>
            <span className="edge-arrow">↓</span>
          </div>

          <div className={`relation-edge ${isReach ? 'active-edge pulse' : 'inactive-edge'}`}>
            <span className="edge-label">reaching</span>
            <span className="edge-arrow">↓</span>
          </div>

          <div className={`relation-edge ${isPickUp || isCarry ? 'active-edge highlight-pickup pulse' : 'inactive-edge'}`}>
            <span className="edge-label highlight-label">
              {isPickUp ? "● holding (PICK UP)" : isCarry ? "● holding & carrying" : "holding"}
            </span>
            <span className="edge-arrow">↓</span>
          </div>

          <div className={`relation-edge ${!isPickUp && !isCarry ? 'active-edge' : 'lifted-edge'}`}>
            <span className="edge-label">
              {isPickUp || isCarry ? "lifted from (detached)" : "on (stationary)"}
            </span>
            <span className="edge-arrow">↓</span>
          </div>
        </div>

        {/* Target Nodes (Laptop #02 and Table #01) */}
        <div className="graph-targets-column">
          {/* Node: Laptop #02 */}
          <div className={`graph-node node-laptop ${isPickUp || isCarry ? 'focused-node' : ''}`}>
            <div className="node-icon-box cyan-icon">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <rect width="18" height="12" x="3" y="4" rx="2" />
                <line x1="2" x2="22" y1="20" y2="20" />
              </svg>
            </div>
            <div className="node-info">
              <span className="node-id">Laptop #02</span>
              <span className="node-type">Target Object</span>
            </div>
          </div>

          {/* Node: Table #01 */}
          <div className={`graph-node node-table ${isApproach ? 'focused-node' : ''}`}>
            <div className="node-icon-box slate-icon">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <line x1="3" x2="21" y1="8" y2="8" strokeWidth="3" />
                <line x1="6" x2="6" y1="8" y2="20" strokeWidth="2" />
                <line x1="18" x2="18" y1="8" y2="20" strokeWidth="2" />
              </svg>
            </div>
            <div className="node-info">
              <span className="node-id">Table #01</span>
              <span className="node-type">Surface / Context</span>
            </div>
          </div>
        </div>
      </div>

      {/* Triplet representation summary */}
      <div className="triplets-summary">
        <span className="triplet-tag">
          <strong>Person #01</strong> → <em>{isPickUp ? "holding" : isApproach ? "approaching" : isReach ? "reaching" : "holding"}</em> → <strong>{isApproach ? "Table #01" : "Laptop #02"}</strong>
        </span>
        {!isPickUp && !isCarry && (
          <span className="triplet-tag">
            <strong>Laptop #02</strong> → <em>on</em> → <strong>Table #01</strong>
          </span>
        )}
      </div>
    </div>
  );
}
