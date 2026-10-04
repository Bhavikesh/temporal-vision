import React, { useState, useRef, useEffect, useMemo } from 'react';
import { formatConfidence, getObjectsAtTimestamp } from '../services/objectAdapter';

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
 * Convert mm:ss or numeric string to seconds
 */
function parseTimeToSeconds(timeStr) {
  if (timeStr === undefined || timeStr === null) return 0;
  if (typeof timeStr === 'number') return timeStr;
  const parts = String(timeStr).split(':').map(Number);
  if (parts.length === 2 && !isNaN(parts[0]) && !isNaN(parts[1])) {
    return parts[0] * 60 + parts[1];
  }
  return Number(timeStr) || 0;
}

export default function VideoPanel({
  objects = [],
  selectedEvent = null,
  videoSrc = null,
  onTimeUpdate = null,
  isMockMode = false,
  onChangeDataset = null,
  currentDataset = 'real'
}) {
  const videoRef = useRef(null);
  const containerRef = useRef(null);
  const fileInputRef = useRef(null);

  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0.0);
  const [duration, setDuration] = useState(24.0);
  const [activeVideoSrc, setActiveVideoSrc] = useState(videoSrc || 'http://localhost:8000/output/debug_video.mp4');
  const [isVideoLoading, setIsVideoLoading] = useState(false);
  const [hasVideoError, setHasVideoError] = useState(false);
  const [showVideoMenu, setShowVideoMenu] = useState(false);

  // Sync to selected event timestamp when event is selected
  useEffect(() => {
    if (selectedEvent) {
      const eventSecs = selectedEvent.timestampSeconds !== undefined
        ? selectedEvent.timestampSeconds
        : parseTimeToSeconds(selectedEvent.timestamp);

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
    if (videoSrc !== undefined) {
      setActiveVideoSrc(videoSrc || (isMockMode ? null : 'http://localhost:8000/output/debug_video.mp4'));
      setHasVideoError(false);
    }
  }, [videoSrc, isMockMode]);

  // Simulation timer if real HTML5 video fails to load or no video
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
          setIsPlaying(true);
        });
      }
    } else {
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
      setDuration(videoRef.current.duration || 24.0);
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
      setShowVideoMenu(false);
    }
  };

  // Dynamically filter active objects at the CURRENT playback time
  const currentFrameObjects = useMemo(() => {
    if (!objects || objects.length === 0) return [];
    return getObjectsAtTimestamp(objects, currentTime);
  }, [objects, currentTime]);

  // Deduplicated list of all unique tracked entities for bottom footer chip legend
  const uniqueTrackedEntities = useMemo(() => {
    if (!objects || objects.length === 0) return [];
    const map = new Map();
    for (const obj of objects) {
      if (!map.has(obj.id)) {
        map.set(obj.id, obj);
      }
    }
    return Array.from(map.values());
  }, [objects]);

  const currentFrameEstimate = Math.round(currentTime * 30);
  const progressPercent = duration > 0 ? (currentTime / duration) * 100 : 0;
  const isPickUpActive = selectedEvent?.type === 'PICK_UP' || selectedEvent?.event === 'PICK_UP';

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
              {currentDataset === 'real' ? "Real Perception Stream (Demo A)" : "Benchmark Lab Stream (Demo B)"}
            </span>
          </div>
        </div>

        <div className="video-header-actions">
          {/* Hidden local video file picker for custom video loading */}
          <input
            ref={fileInputRef}
            type="file"
            accept="video/*"
            style={{ display: 'none' }}
            onChange={handleFileUpload}
          />
          
          <div className="video-menu-dropdown-wrapper">
            <button
              className="video-source-btn"
              onClick={() => setShowVideoMenu(!showVideoMenu)}
              title="Change Video / Dataset Analysis"
            >
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" width="14" height="14">
                <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                <polyline points="17 8 12 3 7 8" />
                <line x1="12" y1="3" x2="12" y2="15" />
              </svg>
              <span>Change Video ▾</span>
            </button>

            {showVideoMenu && (
              <div className="video-options-menu">
                <button
                  className={`menu-option-btn ${currentDataset === 'real' ? 'active-opt' : ''}`}
                  onClick={() => {
                    if (onChangeDataset) onChangeDataset('real');
                    setShowVideoMenu(false);
                  }}
                >
                  <strong>Demo A:</strong> Real Pipeline Video
                </button>
                <button
                  className={`menu-option-btn ${currentDataset === 'mock_demo' ? 'active-opt' : ''}`}
                  onClick={() => {
                    if (onChangeDataset) onChangeDataset('mock_demo');
                    setShowVideoMenu(false);
                  }}
                >
                  <strong>Demo B:</strong> Lab Benchmark Video
                </button>
                <button
                  className="menu-option-btn upload-opt"
                  onClick={() => {
                    fileInputRef.current?.click();
                    setShowVideoMenu(false);
                  }}
                >
                  <strong>Custom:</strong> Load Video File...
                </button>
              </div>
            )}
          </div>

          <div className="timestamp-badge">
            <span className={`rec-dot ${isPlaying ? 'pulsing' : ''}`}></span>
            <span>TIME: {formatTime(currentTime)}</span>
          </div>
        </div>
      </div>


      {/* Video Viewport Container */}
      <div className="video-viewport" ref={containerRef}>
        {/* Real HTML5 Video element pointing to debug video or uploaded video */}
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
            muted
          />
        ) : (
          /* Live Frame Image Stream fallback from actual perception frames */
          <div className="demo-canvas-backdrop">
            <img
              src={`http://localhost:8000/output/debug_frames/frame_${String(Math.min(714, Math.max(0, Math.floor(currentFrameEstimate / 3) * 3))).padStart(5, '0')}.jpg`}
              alt={`Frame ${currentFrameEstimate}`}
              className="real-frame-image"
              onError={(e) => {
                // If frame not found, hide image to show dark canvas
                e.target.style.display = 'none';
              }}
            />
            <div className="grid-overlay"></div>
            <div className="demo-scene-art">
              <div className="scene-camera-tag">PERCEPTION STREAM • {currentFrameEstimate} FRAMES • 30 FPS</div>
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

        {/* Responsive Bounding Box Overlay Layer (Rendered only if video element not showing baked overlay) */}
        <div className="bbox-overlay-layer">
          {currentFrameObjects.map((obj, idx) => {
            const isHighlighted =
              (isPickUpActive && obj.id.includes('laptop')) ||
              (selectedEvent?.subject === obj.id) ||
              (selectedEvent?.object === obj.id);

            const norm = obj.normalized || {
              left: (obj.bbox?.[0] / 3840) * 100 || 20,
              top: (obj.bbox?.[1] / 2160) * 100 || 20,
              width: (obj.bbox?.[2] / 3840) * 100 || 25,
              height: (obj.bbox?.[3] / 2160) * 100 || 30
            };

            const boxColor = obj.color || '#2563eb';

            return (
              <div
                key={`${obj.id}-${idx}`}
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

        {/* Center overlay play button (when paused) */}
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
                {activeVideoSrc && !hasVideoError ? "Annotated Video Stream" : "Demo Video Stream"}
              </span>
              <span className="demo-subtitle">
                {isMockMode ? "Demo Mode (Mock Data)" : "Real-Time Detection & Tracking Active"}
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
              max={duration || 24}
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
              <span className="fps-indicator">• 30 FPS • 4K</span>
            </div>
          </div>
        </div>
      </div>

      {/* Object Legend below video (Deduplicated Unique Entities) */}
      <div className="detected-objects-summary">
        <div className="legend-title-badge">
          {isMockMode ? "Demo Entities" : "Tracked Entities"}:
        </div>
        {uniqueTrackedEntities.map((obj) => (
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

