/**
 * Object Data Adapter for Temporal Vision
 * 
 * Purpose:
 * Normalizes raw object detection and tracking inputs (from Member 1 or Mock Data)
 * into a standardized frontend structure with responsive percentage-based coordinates.
 * 
 * Standard Frontend Object Format:
 * {
 *   id: string,              // e.g. "Person #01"
 *   class: string,           // e.g. "person"
 *   confidence: number,      // e.g. 0.98 (0.0 to 1.0)
 *   timestamp: number,       // in seconds, e.g. 4.0
 *   trackId: string,         // e.g. "TRK-01"
 *   color: string,           // Hex color code for UI rendering
 *   bbox: [x, y, w, h],      // [x, y, width, height]
 *   normalized: {            // 0% - 100% relative coordinates for responsive overlay
 *     left: number,
 *     top: number,
 *     width: number,
 *     height: number
 *   },
 *   isMock: boolean          // true if from mock/demo source
 * }
 */

// Preset palette for object classes
const CLASS_COLORS = {
  person: '#2563eb', // Blue
  human: '#2563eb',
  laptop: '#0891b2', // Cyan
  computer: '#0891b2',
  table: '#64748b',  // Slate
  desk: '#64748b',
  chair: '#d97706',  // Amber
  phone: '#7c3aed',  // Violet
  bottle: '#059669', // Emerald
  default: '#0284c7'
};

/**
 * Get color associated with an object class or id
 */
export function getObjectColor(objectClass = '', customColor = null) {
  if (customColor) return customColor;
  const key = (objectClass || '').toLowerCase().trim();
  return CLASS_COLORS[key] || CLASS_COLORS.default;
}

/**
 * Parses any bounding box representation into standardized [x, y, width, height]
 * and percentage-based { left, top, width, height } relative to video dimensions.
 * 
 * Supports:
 * - [x1, y1, x2, y2] (e.g. [120, 80, 300, 420])
 * - [x, y, width, height]
 * - Normalized [0..1] coordinates
 * - Object with { x, y, width, height } or { xmin, ymin, xmax, ymax }
 * 
 * @param {Array|Object} rawBbox 
 * @param {number} refWidth  Reference video width (default 800)
 * @param {number} refHeight Reference video height (default 480)
 */
export function normalizeBoundingBox(rawBbox, refWidth = 800, refHeight = 480) {
  if (!rawBbox) {
    return {
      bbox: [0, 0, 0, 0],
      normalized: { left: 0, top: 0, width: 0, height: 0 }
    };
  }

  let x = 0, y = 0, width = 0, height = 0;

  if (Array.isArray(rawBbox)) {
    if (rawBbox.length >= 4) {
      const [v0, v1, v2, v3] = rawBbox.map(Number);

      // Check if coordinates are normalized [0..1]
      const isNormalizedUnit = v0 <= 1 && v1 <= 1 && v2 <= 1 && v3 <= 1;

      if (isNormalizedUnit) {
        // Assume [x, y, w, h] or [xmin, ymin, xmax, ymax]
        if (v2 > v0 && v3 > v1 && v2 <= 1 && v3 <= 1) {
          // [xmin, ymin, xmax, ymax] normalized
          x = v0 * refWidth;
          y = v1 * refHeight;
          width = (v2 - v0) * refWidth;
          height = (v3 - v1) * refHeight;
        } else {
          // [x, y, w, h] normalized
          x = v0 * refWidth;
          y = v1 * refHeight;
          width = v2 * refWidth;
          height = v3 * refHeight;
        }
      } else {
        // Pixel coordinates: determine if [x1, y1, x2, y2] or [x, y, w, h]
        if (v2 > v0 && v3 > v1 && v2 > 10 && v3 > 10) {
          // Typically [x1, y1, x2, y2]
          x = v0;
          y = v1;
          width = v2 - v0;
          height = v3 - v1;
        } else {
          // [x, y, w, h]
          x = v0;
          y = v1;
          width = v2;
          height = v3;
        }
      }
    }
  } else if (typeof rawBbox === 'object') {
    if ('xmin' in rawBbox && 'xmax' in rawBbox) {
      x = Number(rawBbox.xmin);
      y = Number(rawBbox.ymin);
      width = Number(rawBbox.xmax) - x;
      height = Number(rawBbox.ymax) - y;
    } else {
      x = Number(rawBbox.x || rawBbox.left || 0);
      y = Number(rawBbox.y || rawBbox.top || 0);
      width = Number(rawBbox.width || rawBbox.w || 0);
      height = Number(rawBbox.height || rawBbox.h || 0);
    }
  }

  // Calculate percentage relative to video canvas
  const leftPercent = Math.max(0, Math.min(100, (x / refWidth) * 100));
  const topPercent = Math.max(0, Math.min(100, (y / refHeight) * 100));
  const widthPercent = Math.max(0, Math.min(100 - leftPercent, (width / refWidth) * 100));
  const heightPercent = Math.max(0, Math.min(100 - topPercent, (height / refHeight) * 100));

  return {
    bbox: [Math.round(x), Math.round(y), Math.round(width), Math.round(height)],
    normalized: {
      left: Number(leftPercent.toFixed(2)),
      top: Number(topPercent.toFixed(2)),
      width: Number(widthPercent.toFixed(2)),
      height: Number(heightPercent.toFixed(2))
    }
  };
}

/**
 * Normalizes a single raw object detection item into standardized frontend format
 * @param {Object} rawObj Raw object data from backend/ML pipeline
 * @param {number} refWidth Reference video width
 * @param {number} refHeight Reference video height
 * @param {boolean} isMock Whether this object comes from mock source
 * @returns {Object} Standardized frontend object
 */
export function normalizeObject(rawObj = {}, refWidth = 800, refHeight = 480, isMock = true) {
  if (!rawObj) return null;

  const id = rawObj.id || rawObj.label || rawObj.name || `Object #${rawObj.track_id || rawObj.trackId || '01'}`;
  const objectClass = rawObj.class || rawObj.type || rawObj.category || 'object';
  const confidence = typeof rawObj.confidence === 'number' 
    ? rawObj.confidence 
    : typeof rawObj.score === 'number' 
    ? rawObj.score 
    : 0.95;

  const timestamp = typeof rawObj.timestamp === 'number' 
    ? rawObj.timestamp 
    : typeof rawObj.time === 'number' 
    ? rawObj.time 
    : 0;

  const trackId = rawObj.trackId || rawObj.track_id || rawObj.tracker_id || `TRK-${id.replace(/\D/g, '') || '01'}`;
  const color = getObjectColor(objectClass, rawObj.color);
  const { bbox, normalized } = normalizeBoundingBox(rawObj.bbox || rawObj.box, refWidth, refHeight);

  return {
    id,
    class: objectClass,
    confidence: Number(confidence.toFixed(2)),
    timestamp,
    trackId,
    color,
    bbox,
    normalized,
    isMock: isMock ?? true
  };
}

/**
 * Normalizes an array of raw objects
 * @param {Array} rawObjects List of raw object detections
 * @param {number} refWidth Reference video width
 * @param {number} refHeight Reference video height
 * @param {boolean} isMock
 * @returns {Array} List of normalized frontend objects
 */
export function normalizeObjectList(rawObjects = [], refWidth = 800, refHeight = 480, isMock = true) {
  if (!Array.isArray(rawObjects)) return [];
  return rawObjects
    .map((obj) => normalizeObject(obj, refWidth, refHeight, isMock))
    .filter(Boolean);
}

/**
 * Filter or interpolate objects for a specific timestamp (in seconds)
 * when time-series detection frames are available.
 * @param {Array} objectsOrFrames List of objects or timestamped detection frames
 * @param {number} currentTime Current video playback time in seconds
 * @returns {Array} Active objects for the current frame/timestamp
 */
export function getObjectsAtTimestamp(objectsOrFrames = [], currentTime = 0) {
  if (!Array.isArray(objectsOrFrames) || objectsOrFrames.length === 0) {
    return [];
  }

  // If items contain explicit frame timestamps, filter by closest frame
  const hasTimestamps = objectsOrFrames.some((item) => typeof item.timestamp === 'number');
  if (!hasTimestamps) {
    // Return all static objects if no temporal frames provided
    return objectsOrFrames;
  }

  // Filter objects within temporal window (e.g., +/- 1.0 second) or static objects (timestamp === 0)
  return objectsOrFrames.filter((obj) => {
    if (obj.timestamp === 0 || obj.timestamp === undefined) return true;
    return Math.abs(obj.timestamp - currentTime) <= 1.5;
  });
}

/**
 * Formats confidence score as percentage string (e.g. 0.98 -> "98%")
 * @param {number} confidence 
 * @returns {string} Formatted percentage
 */
export function formatConfidence(confidence = 0) {
  return `${Math.round(confidence * 100)}%`;
}
