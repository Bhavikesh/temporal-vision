import React, { useState, useEffect, useCallback } from 'react';
import Header from './components/Header';
import VideoPanel from './components/VideoPanel';
import EventTimeline from './components/EventTimeline';
import SceneGraph from './components/SceneGraph';
import EvidencePanel from './components/EvidencePanel';
import ExplanationPanel from './components/ExplanationPanel';
import { getResults, getObjects, getEvents } from './services/api';
import { mockObjects, mockEvents } from './data/mockData';
import { normalizeObjectList } from './services/objectAdapter';

export default function App() {
  const [objects, setObjects] = useState([]);
  const [events, setEvents] = useState([]);
  const [explanation, setExplanation] = useState(null);
  const [selectedEventId, setSelectedEventId] = useState('evt-3');
  const [currentTime, setCurrentTime] = useState(5.0);
  const [videoSrc, setVideoSrc] = useState(null);

  // Backend connection state
  const [isLoading, setIsLoading] = useState(true);
  const [backendStatus, setBackendStatus] = useState('connecting'); // 'connecting' | 'connected_mock' | 'connected_live' | 'offline'
  const [apiError, setApiError] = useState(null);
  const [currentDataset, setCurrentDataset] = useState('real'); // 'real' | 'mock_demo'

  // Fetch results from backend API based on selected dataset
  const fetchBackendData = useCallback(async (datasetId = 'real') => {
    setIsLoading(true);
    setApiError(null);

    const isMock = datasetId !== 'real';

    try {
      // Call GET http://localhost:8000/results?mock=false (or true)
      const result = await getResults(isMock);

      // Successfully connected to backend
      const perceptionList = result.perception && result.perception.length > 0
        ? result.perception
        : normalizeObjectList(mockObjects, 800, 480, isMock);

      const eventList = result.events && result.events.length > 0
        ? result.events
        : mockEvents;

      setObjects(perceptionList);
      setEvents(eventList);
      setExplanation(result.explanation);
      setVideoSrc(result.videoSrc);
      setBackendStatus(isMock ? 'connected_mock' : 'connected_live');

      // Set initial selected event if available
      const pickUpEvt = eventList.find((e) => (e.type || e.event) === 'PICK_UP');
      if (pickUpEvt) {
        setSelectedEventId(pickUpEvt.id || pickUpEvt.type);
      } else if (eventList.length > 0) {
        setSelectedEventId(eventList[0].id || eventList[0].type);
      }
    } catch (err) {
      console.warn('[API Integration] Backend is unreachable:', err.message);
      setBackendStatus('offline');
      setApiError(err.message || 'Failed to connect to backend at http://localhost:8000');

      // Fallback gracefully to local mock dataset so UI remains interactive
      const fallbackObjects = normalizeObjectList(mockObjects, 800, 480, true);
      setObjects(fallbackObjects);
      setEvents(mockEvents);
      setExplanation(null);
      setVideoSrc(null);

      const pickUpEvt = mockEvents.find((e) => e.type === 'PICK_UP');
      if (pickUpEvt) {
        setSelectedEventId(pickUpEvt.id);
      }
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchBackendData(currentDataset);
  }, [fetchBackendData, currentDataset]);

  const selectedEvent = events.find(
    (e) => e.id === selectedEventId || e.type === selectedEventId
  ) || events[0] || null;

  const handleSelectEvent = (evt) => {
    setSelectedEventId(evt.id || evt.type);
    if (evt.timestampSeconds !== undefined) {
      setCurrentTime(evt.timestampSeconds);
    }
  };

  // Bidirectional sync: when video time advances or is scrubbed, activate the closest event
  const handleTimeUpdate = (newTime) => {
    setCurrentTime(newTime);
    if (events && events.length > 0) {
      let closestEvt = events[0];
      let minDiff = Math.abs((closestEvt.timestampSeconds ?? 0) - newTime);

      for (let i = 1; i < events.length; i++) {
        const evtTime = events[i].timestampSeconds ?? 0;
        const diff = Math.abs(evtTime - newTime);
        if (diff < minDiff) {
          minDiff = diff;
          closestEvt = events[i];
        }
      }

      if (closestEvt && (closestEvt.id || closestEvt.type) !== selectedEventId) {
        setSelectedEventId(closestEvt.id || closestEvt.type);
      }
    }
  };

  const handleRetry = () => {
    fetchBackendData(currentDataset);
  };

  const handleSelectDataset = (datasetId) => {
    setCurrentDataset(datasetId);
    setUseMockBackend(datasetId !== 'real');
  };

  const handleToggleMode = () => {
    const nextMode = currentDataset === 'real' ? 'mock_demo' : 'real';
    setCurrentDataset(nextMode);
    setUseMockBackend(nextMode !== 'real');
  };

  return (
    <div className="dashboard-wrapper">
      <Header
        backendStatus={backendStatus}
        onRetry={handleRetry}
        onToggleMode={handleToggleMode}
        currentDataset={currentDataset}
        onSelectDataset={handleSelectDataset}
      />

      {/* Backend Offline Notification Banner if API is down */}
      {backendStatus === 'offline' && (
        <div className="backend-offline-banner">
          <div className="offline-banner-content">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" width="18" height="18">
              <circle cx="12" cy="12" r="10" />
              <line x1="12" y1="8" x2="12" y2="12" />
              <line x1="12" y1="16" x2="12.01" y2="16" />
            </svg>
            <span>
              <strong>Backend Offline:</strong> Could not connect to <code>http://localhost:8000/results?mock=false</code>. Displaying local fallback data. Start the backend server and click <strong>Retry</strong> to connect.
            </span>
          </div>
          <button className="banner-retry-btn" onClick={handleRetry}>
            Retry Connection
          </button>
        </div>
      )}

      {/* Loading Overlay if fetching */}
      {isLoading ? (
        <div className="dashboard-loading-state">
          <div className="spinner large-spinner"></div>
          <p className="loading-text">Connecting to Temporal Vision Backend...</p>
          <span className="loading-subtext">GET http://localhost:8000/results?mock={currentDataset !== 'real'}</span>
        </div>
      ) : (
        <main className="dashboard-main">
          {/* Top Grid: Left Video Panel + Right Event Timeline */}
          <section className="top-section-grid">
            <div className="grid-left-video">
              <VideoPanel
                objects={objects}
                selectedEvent={selectedEvent}
                videoSrc={videoSrc}
                onTimeUpdate={handleTimeUpdate}
                isMockMode={backendStatus !== 'connected_live'}
                onChangeDataset={handleSelectDataset}
                currentDataset={currentDataset}
              />
            </div>
            <div className="grid-right-timeline">
              <EventTimeline
                events={events}
                selectedEventId={selectedEventId}
                onSelectEvent={handleSelectEvent}
              />
            </div>
          </section>

          {/* Middle Grid: Bottom Left Scene Graph + Bottom Right Evidence */}
          <section className="mid-section-grid">
            <div className="grid-mid-left">
              <SceneGraph selectedEvent={selectedEvent} />
            </div>
            <div className="grid-mid-right">
              <EvidencePanel selectedEvent={selectedEvent} />
            </div>
          </section>

          {/* Bottom Area: Explanation Panel */}
          <section className="bottom-section">
            <ExplanationPanel
              selectedEvent={selectedEvent}
              explanation={explanation}
            />
          </section>
        </main>
      )}
    </div>
  );
}

