import React, { useState, useRef, useEffect } from 'react';
import { formatConfidence } from '../services/objectAdapter';

/**
 * Format seconds into mm:ss display
 */
function formatTime(seconds) {
  if (isNaN(seconds) || seconds < 0) return "00:00";
  const mins = Math.floor(seconds / 60);
  const secs = Math.floor(seconds % 60);
  return `${String(mins).padStart(2, '0')}:${String(secs).padStart(2, '0')}`;
}

/**
 * Convert mm:ss string to seconds
 */
function parseTimeToSeconds(timeStr) {
  if (!timeStr) return 0;
  if (typeof timeStr === 'number') return timeStr;
  const parts = timeStr.split(':').map(Number);
  if (parts.length === 2) return parts[0] * 60 + parts[1];
  return Number(timeStr) || 0;
}

export default function VideoPanel({
  objects = [],
  selectedEvent = null,
  videoSrc = null,
  onTimeUpdate = null,
  isMockMode = true
}) {
  const videoRef = useRef(null);
  const containerRef = useRef(null);
  const fileInputRef = useRef(null);

  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(5.0); // Default to 00:05 for initial demo state
  const [duration, setDuration] = useState(10.0);
  const [activeVideoSrc, setActiveVideoSrc] = useState(videoSrc);
  const [isVideoLoading, setIsVideoLoading] = useState(false);
  const [hasVideoError, setHasVideoError] = useState(false);

  // Sync to selected event timestamp if event changed
  useEffect(() => {
    if (selectedEvent?.timestamp) {
      const eventSecs = parseTimeToSeconds(selectedEvent.timestamp);
      setCurrentTime(eventSecs);
      if (videoRef.current && !isNaN(eventSecs)) {
        videoRef.current.currentTime = eventSecs;
      }
      if (onTimeUpdate) {
        onTimeUpdate(eventSecs);
      }
    }
  }, [selectedEvent]);

  // Sync with prop videoSrc
  useEffect(() => {
    if (videoSrc) {
      setActiveVideoSrc(videoSrc);
      setHasVideoError(false);
    }
  }, [videoSrc]);

  // Mock simulation playback timer when no real HTML video is playing
  useEffect(() => {
    let timer = null;
    if (isPlaying && (!activeVideoSrc || hasVideoError)) {
      timer = setInterval(() => {
        setCurrentTime((prev) => {
          const next = prev + 0.1;
          if (next >= duration) {
            setIsPlaying(false);
            return duration;
          }
          if (onTimeUpdate) onTimeUpdate(next);
          return next;
        });
      }, 100);
    }
    return () => {
      if (timer) clearInterval(timer);
    };
  }, [isPlaying, activeVideoSrc, hasVideoError, duration, onTimeUpdate]);

  // Real HTML5 Video handlers
  const handlePlayPause = () => {
    if (activeVideoSrc && videoRef.current && !hasVideoError) {
      if (isPlaying) {
        videoRef.current.pause();
      } else {
        videoRef.current.play().catch(() => {
          // If auto-play blocked or source missing, fallback to mock timer
          setIsPlaying(true);
        });
      }
    } else {
      // Toggle mock timer playback
      if (currentTime >= duration) {
        setCurrentTime(0);
      }
      setIsPlaying(!isPlaying);
    }
  };

  const handleVideoTimeUpdate = () => {
    if (videoRef.current) {
      const time = videoRef.current.currentTime;
      setCurrentTime(time);
      if (onTimeUpdate) {
        onTimeUpdate(time);
      }
    }
  };

  const handleVideoLoadedMetadata = () => {
    if (videoRef.current) {
      setDuration(videoRef.current.duration || 10.0);
      setIsVideoLoading(false);
      setHasVideoError(false);
    }
  };

  const handleScrubberChange = (e) => {
    const newTime = parseFloat(e.target.value);
    setCurrentTime(newTime);
    if (videoRef.current && activeVideoSrc && !hasVideoError) {
      videoRef.current.currentTime = newTime;
    }
    if (onTimeUpdate) {
      onTimeUpdate(newTime);
    }
  };

  const handleFileUpload = (e) => {
    const file = e.target.files?.[0];
    if (file) {
      const localUrl = URL.createObjectURL(file);
      setActiveVideoSrc(localUrl);
      setHasVideoError(false);
      setIsPlaying(false);
      setCurrentTime(0);
    }
  };

  const currentFrameEstimate = Math.round(currentTime * 30);
  const progressPercent = duration > 0 ? (currentTime / duration) * 100 : 0;
  const isPickUpActive = selectedEvent?.type === 'PICK_UP';

  return (
    <div className="panel video-panel-card">
      {/* Panel Header */}
      <div className="panel-header">
        <div className="panel-title-group">
          <svg className="panel-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <rect width="18" height="14" x="3" y="5" rx="2" />
            <path d="m10 10 5 3-5 3v-6Z" fill="currentColor" />
          </svg>
          <div>
            <h2 className="panel-title">VIDEO PANEL</h2>
            <span className="panel-subtitle">
              {activeVideoSrc && !hasVideoError ? "Local Video Stream" : "Demo Stream & Object Localization"}
            </span>
          </div>
        </div>

        <div className="video-header-actions">
          {/* Hidden local video file picker for live testing */}
          <input
            ref={fileInputRef}
            type="file"
            accept="video/*"
            style={{ display: 'none' }}
            onChange={handleFileUpload}
          />
          <button
            className="video-source-btn"
            onClick={() => fileInputRef.current?.click()}
            title="Load local video file for real video testing"
          >
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" width="14" height="14">
              <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
              <polyline points="17 8 12 3 7 8" />
              <line x1="12" y1="3" x2="12" y2="15" />
            </svg>
            <span>{activeVideoSrc ? "Change Video" : "Load Video"}</span>
          </button>

          <div className="timestamp-badge">
            <span className={`rec-dot ${isPlaying ? 'pulsing' : ''}`}></span>
            <span>TIME: {formatTime(currentTime)}</span>
          </div>
        </div>
      </div>

      {/* Video Viewport Container */}
      <div className="video-viewport" ref={containerRef}>
        {/* Real HTML5 Video element if video source provided */}
        {activeVideoSrc && !hasVideoError ? (
          <video
            ref={videoRef}
            src={activeVideoSrc}
            className="real-video-element"
            onTimeUpdate={handleVideoTimeUpdate}
            onLoadedMetadata={handleVideoLoadedMetadata}
            onPlay={() => setIsPlaying(true)}
            onPause={() => setIsPlaying(false)}
            onEnded={() => setIsPlaying(false)}
            onError={() => setHasVideoError(true)}
            onWaiting={() => setIsVideoLoading(true)}
            onCanPlay={() => setIsVideoLoading(false)}
            playsInline
          />
        ) : (
          /* High-Tech Demo Canvas Placeholder when no video source active */
          <div className="demo-canvas-backdrop">
            <div className="grid-overlay"></div>
            {/* Visual representation of demo scene objects */}
            <div className="demo-scene-art">
              <div className="scene-camera-tag">CAMERA-01 • 1080p • 30 FPS</div>
            </div>
          </div>
        )}

        {/* Video Loading State */}
        {isVideoLoading && (
          <div className="video-loading-overlay">
            <div className="spinner"></div>
            <span>Buffering stream...</span>
          </div>
        )}

        {/* Responsive Bounding Box Overlay Layer */}
        <div className="bbox-overlay-layer">
          {objects.map((obj) => {
            const isHighlighted =
              (isPickUpActive && obj.id.includes('Laptop')) ||
              (selectedEvent?.subject === obj.id) ||
              (selectedEvent?.object === obj.id);

            const norm = obj.normalized || {
              left: (obj.bbox?.[0] / 800) * 100 || 20,
              top: (obj.bbox?.[1] / 480) * 100 || 20,
              width: (obj.bbox?.[2] / 800) * 100 || 25,
              height: (obj.bbox?.[3] / 480) * 100 || 30
            };

            const boxColor = obj.color || '#2563eb';

            return (
              <div
                key={obj.id}
                className={`bounding-box-item ${isHighlighted ? 'highlighted' : ''}`}
                style={{
                  left: `${norm.left}%`,
                  top: `${norm.top}%`,
                  width: `${norm.width}%`,
                  height: `${norm.height}%`,
                  borderColor: boxColor,
                  '--box-color': boxColor
                }}
              >
                {/* Object Label Pill */}
                <div
                  className="bbox-label-tag"
                  style={{ backgroundColor: boxColor }}
                >
                  <span className="bbox-object-id">{obj.id}</span>
                  <span className="bbox-confidence">{formatConfidence(obj.confidence)}</span>
                </div>

                {/* Tracking ID badge at bottom corner */}
                <div className="bbox-track-badge">
                  {obj.trackId || obj.class}
                </div>
              </div>
            );
          })}
        </div>

        {/* Center overlay play button (when paused or demo active) */}
        {!isPlaying && (
          <div className="video-center-placeholder">
            <button
              className="play-toggle-btn"
              onClick={handlePlayPause}
              title="Play Video Stream"
              aria-label="Play Video Stream"
            >
              <svg viewBox="0 0 24 24" fill="currentColor">
                <polygon points="6 4 20 12 6 20 6 4" />
              </svg>
            </button>
            <div className="demo-video-tag">
              <span className="demo-title">
                {activeVideoSrc ? "Local Video Stream" : "Demo Video Stream"}
              </span>
              <span className="demo-subtitle">
                {isMockMode ? "Demo Objects (Mock Data)" : "Real-Time Detection Overlay"}
              </span>
            </div>
          </div>
        )}

        {/* Video Control Bar */}
        <div className="video-controls-bar">
          <div className="scrubber-wrapper">
            <input
              type="range"
              min="0"
              max={duration || 10}
              step="0.05"
              value={currentTime}
              onChange={handleScrubberChange}
              className="scrubber-slider"
              aria-label="Video scrubber"
            />
            <div
              className="scrubber-progress-fill"
              style={{ width: `${progressPercent}%` }}
            ></div>
          </div>

          <div className="controls-labels">
            <div className="controls-left-group">
              <button
                className="mini-play-btn"
                onClick={handlePlayPause}
                aria-label={isPlaying ? "Pause" : "Play"}
              >
                {isPlaying ? (
                  <svg viewBox="0 0 24 24" fill="currentColor" width="14" height="14">
                    <rect x="6" y="4" width="4" height="16" rx="1" />
                    <rect x="14" y="4" width="4" height="16" rx="1" />
                  </svg>
                ) : (
                  <svg viewBox="0 0 24 24" fill="currentColor" width="14" height="14">
                    <polygon points="6 4 20 12 6 20 6 4" />
                  </svg>
                )}
              </button>
              <span className="time-display">
                {formatTime(currentTime)} / {formatTime(duration)}
              </span>
            </div>

            <div className="controls-right-group">
              <span className="frame-counter">Frame {currentFrameEstimate}</span>
              <span className="fps-indicator">• 30 FPS • 1080p</span>
            </div>
          </div>
        </div>
      </div>

      {/* Object Legend below video */}
      <div className="detected-objects-summary">
        <div className="legend-title-badge">
          {isMockMode ? "Demo Objects" : "Tracked Objects"}:
        </div>
        {objects.map((obj) => (
          <div
            key={obj.id}
            className={`object-chip ${
              selectedEvent?.subject === obj.id || selectedEvent?.object === obj.id
                ? 'chip-active'
                : ''
            }`}
          >
            <span
              className="chip-dot"
              style={{ backgroundColor: obj.color || '#2563eb' }}
            ></span>
            <span className="chip-label">{obj.id}</span>
            <span className="chip-conf">{formatConfidence(obj.confidence)}</span>
          </div>
        ))}
      </div>
    </div>
  );
}
