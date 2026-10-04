import React, { useState, useEffect } from 'react';
import Header from './components/Header';
import VideoPanel from './components/VideoPanel';
import EventTimeline from './components/EventTimeline';
import SceneGraph from './components/SceneGraph';
import EvidencePanel from './components/EvidencePanel';
import ExplanationPanel from './components/ExplanationPanel';
import { getObjects, getEvents } from './services/api';

export default function App() {
  const [objects, setObjects] = useState([]);
  const [events, setEvents] = useState([]);
  const [selectedEventId, setSelectedEventId] = useState('evt-3'); // Default to PICK_UP
  const [currentTime, setCurrentTime] = useState(5.0); // Timestamp state in seconds
  const [isMockMode, setIsMockMode] = useState(true);
  const [videoSrc, setVideoSrc] = useState(null); // Can be configured with real local video path

  useEffect(() => {
    // Load normalized objects and events from service layer
    getObjects().then(setObjects);
    getEvents().then((evts) => {
      setEvents(evts);
      // Ensure PICK_UP is selected initially if available
      const pickUpEvt = evts.find((e) => e.type === 'PICK_UP');
      if (pickUpEvt) {
        setSelectedEventId(pickUpEvt.id);
      }
    });
  }, []);

  const selectedEvent = events.find((e) => e.id === selectedEventId) || events[0] || null;

  const handleSelectEvent = (evt) => {
    setSelectedEventId(evt.id);
  };

  const handleTimeUpdate = (newTime) => {
    setCurrentTime(newTime);
  };

  const handleToggleMode = () => {
    setIsMockMode((prev) => !prev);
  };

  return (
    <div className="dashboard-wrapper">
      <Header isMockMode={isMockMode} onToggleMode={handleToggleMode} />

      <main className="dashboard-main">
        {/* Top Grid: Left Video Panel + Right Event Timeline */}
        <section className="top-section-grid">
          <div className="grid-left-video">
            <VideoPanel
              objects={objects}
              selectedEvent={selectedEvent}
              videoSrc={videoSrc}
              onTimeUpdate={handleTimeUpdate}
              isMockMode={isMockMode}
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
          <ExplanationPanel selectedEvent={selectedEvent} />
        </section>
      </main>
    </div>
  );
}
