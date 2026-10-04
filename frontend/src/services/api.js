/**
 * API Service for Temporal Vision
 * 
 * Part 5: Backend Integration
 * Connects frontend to backend endpoints at http://localhost:8000
 * (or configurable via VITE_API_BASE_URL)
 */

import { mockObjects, mockEvents, defaultSceneRelations, pickUpSceneRelations } from '../data/mockData';
import { normalizeObjectList } from './objectAdapter';

// Configurable API base URL with fallback to local development server
export const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

/**
 * Health check endpoint
 * @returns {Promise<Object>}
 */
export const checkHealth = async () => {
  const response = await fetch(`${API_BASE_URL}/`, {
    method: 'GET',
    headers: { 'Accept': 'application/json' }
  });
  if (!response.ok) {
    throw new Error(`Health check failed with status: ${response.status}`);
  }
  return response.json();
};

/**
 * Fetch unified pipeline results from backend
 * GET http://localhost:8000/results?mock=true / ?mock=false
 * 
 * @param {boolean|string} dataset 'real' | 'demo_a' | 'demo_b' | boolean
 * @returns {Promise<{ perception: Array, events: Array, explanation: any, source: string, videoSrc: string }>}
 */
export const getResults = async (dataset = false) => {
  const isMock = typeof dataset === 'boolean' ? dataset : dataset !== 'real';
  const url = `${API_BASE_URL}/results?mock=${isMock}`;
  
  const response = await fetch(url, {
    method: 'GET',
    headers: {
      'Accept': 'application/json'
    }
  });

  if (!response.ok) {
    throw new Error(`API error ${response.status}: ${response.statusText}`);
  }

  const data = await response.json();

  // Normalize perception objects if provided by backend (either list of objects or list of PerceptionFrames)
  let rawObjects = [];
  if (Array.isArray(data.perception)) {
    if (data.perception.length > 0 && Array.isArray(data.perception[0].objects)) {
      // It's a list of PerceptionFrames: flatten all objects with timestamp
      rawObjects = data.perception.flatMap(frame => 
        (frame.objects || []).map(obj => ({
          ...obj,
          timestamp: frame.timestamp,
          frame_index: frame.frame_index
        }))
      );
    } else {
      rawObjects = data.perception;
    }
  }

  const normalizedPerception = rawObjects.length > 0
    ? normalizeObjectList(rawObjects, 3840, 2160, isMock)
    : [];

  // Normalize events to ensure id, type, and clean standard mm:ss timestamp format
  const normalizedEvents = (data.events || []).map((evt, idx) => {
    const eventType = evt.event || evt.type || 'APPROACH';
    const timestampSec = typeof evt.timestamp === 'number' ? evt.timestamp : parseFloat(evt.timestamp) || 0;
    const mins = Math.floor(timestampSec / 60);
    const secs = Math.round(timestampSec % 60);
    const timeFormatted = `${String(mins).padStart(2, '0')}:${String(secs).padStart(2, '0')}`;


    return {
      id: evt.id || `evt-${idx + 1}`,
      type: eventType,
      event: eventType,
      timestamp: timeFormatted,
      timestampSeconds: timestampSec,
      subject: evt.subject,
      object: evt.object,

      confidence: evt.confidence ?? 0.9,
      evidence: evt.evidence || {},
      description: evt.description || `${evt.subject} ${eventType.toLowerCase()} ${evt.object || ''}`,
      explanation: typeof data.explanation === 'string' ? data.explanation : (data.explanation?.explanation || '')
    };
  });

  return {
    perception: normalizedPerception,
    events: normalizedEvents,
    explanation: data.explanation || null,
    source: isMock ? 'backend_mock' : 'backend_live',
    videoSrc: isMock ? null : `${API_BASE_URL}/output/debug_video.mp4`,
    raw: data
  };
};


/**
 * Fetch detected and tracked objects
 * Tries backend first; falls back to local normalized mock data.
 * @returns {Promise<Array>} List of normalized object detections
 */
export const getObjects = async () => {
  try {
    const res = await getResults(true);
    if (res.perception && res.perception.length > 0) {
      return res.perception;
    }
  } catch (err) {
    console.warn('[API] Backend getObjects fallback to local mock:', err.message);
  }
  return Promise.resolve(normalizeObjectList(mockObjects, 800, 480, true));
};

/**
 * Fetch temporal events detected in the video stream
 * Tries backend first; falls back to local mock data.
 * @returns {Promise<Array>} List of temporal events with timestamps & evidence
 */
export const getEvents = async () => {
  try {
    const res = await getResults(true);
    if (res.events && res.events.length > 0) {
      return res.events;
    }
  } catch (err) {
    console.warn('[API] Backend getEvents fallback to local mock:', err.message);
  }
  return Promise.resolve(mockEvents);
};

/**
 * Fetch scene graph relationships
 * @param {string} [eventId] Optional event ID
 * @returns {Promise<Array>} Active graph relationships
 */
export const getScene = async (eventId = "evt-3") => {
  const event = mockEvents.find((e) => e.id === eventId);
  if (event && event.sceneGraph) {
    return Promise.resolve(event.sceneGraph);
  }
  return Promise.resolve(eventId === "evt-3" ? pickUpSceneRelations : defaultSceneRelations);
};

/**
 * Placeholder for POST /run pipeline (Multipart file upload)
 * @param {File} videoFile Video file to process
 * @returns {Promise<Object>}
 */
export const runPipeline = async (videoFile) => {
  const formData = new FormData();
  formData.append('video', videoFile);

  const response = await fetch(`${API_BASE_URL}/run`, {
    method: 'POST',
    body: formData
  });

  if (!response.ok) {
    throw new Error(`Pipeline run failed with status: ${response.status}`);
  }

  return response.json();
};
