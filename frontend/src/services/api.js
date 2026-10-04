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
 * GET http://localhost:8000/results?mock=true
 * 
 * @param {boolean} mock Whether to request mock dataset from backend (default true)
 * @returns {Promise<{ perception: Array, events: Array, explanation: any, source: string }>}
 */
export const getResults = async (mock = true) => {
  const url = `${API_BASE_URL}/results?mock=${mock}`;
  
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

  // Normalize perception objects if provided by backend
  const normalizedPerception = data.perception 
    ? normalizeObjectList(data.perception, 800, 480, mock)
    : [];

  return {
    perception: normalizedPerception,
    events: data.events || [],
    explanation: data.explanation || null,
    source: mock ? 'backend_mock' : 'backend_live',
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
