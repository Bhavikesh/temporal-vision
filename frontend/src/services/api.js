/**
 * API Service Placeholder for Temporal Vision
 * 
 * NOTE FOR MEMBER 3 (Backend):
 * Replace these placeholder mock promises with your actual REST or WebSocket endpoints.
 * Example:
 * export const getEvents = async (videoId) => {
 *   const res = await fetch(`/api/v1/videos/${videoId}/events`);
 *   return res.json();
 * };
 */

import { mockObjects, mockEvents, defaultSceneRelations, pickUpSceneRelations } from '../data/mockData';
import { normalizeObjectList } from './objectAdapter';

/**
 * Fetch detected and tracked objects
 * Passed through the objectAdapter to ensure a normalized frontend structure.
 * @returns {Promise<Array>} List of normalized object detections
 */
export const getObjects = async () => {
  return Promise.resolve(normalizeObjectList(mockObjects, 800, 480, true));
};

/**
 * Fetch temporal events detected in the video stream
 * @returns {Promise<Array>} List of temporal events with timestamps & evidence
 */
export const getEvents = async () => {
  return Promise.resolve(mockEvents);
};

/**
 * Fetch scene graph relationships
 * @param {string} [eventId] Optional event ID to filter active scene graph relations
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
 * Trigger video analysis pipeline (Video -> Object Detection -> Temporal -> Events -> Evidence -> Explanation)
 * @param {File|string} [videoInput] Video file or stream reference
 * @returns {Promise<Object>} Aggregated pipeline output
 */
export const analyzeVideo = async (videoInput = null) => {
  return Promise.resolve({
    status: "completed",
    videoSource: "mock_demo_stream.mp4",
    duration: "00:10",
    objects: mockObjects,
    events: mockEvents,
    summary: "Person #01 picked up Laptop #02 at 00:05 and carried it away from Table #01."
  });
};
