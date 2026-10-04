/**
 * Mock data for Temporal Vision - Frontend Demo Visualization
 * Scenario: Person #01 approaches a table, reaches for Laptop #02, picks it up, and carries it away.
 */

export const mockObjects = [
  {
    id: "Person #01",
    type: "person",
    label: "Person #01",
    color: "#2563eb", // Blue
    // Bounding box [x1, y1, x2, y2] relative to 800x480 canvas
    bbox: [120, 80, 300, 420],
    confidence: 0.98,
    trackId: "TRK-01"
  },
  {
    id: "Laptop #02",
    type: "laptop",
    label: "Laptop #02",
    color: "#0891b2", // Cyan
    bbox: [430, 240, 560, 330],
    confidence: 0.95,
    trackId: "TRK-02"
  },
  {
    id: "Table #01",
    type: "table",
    label: "Table #01",
    color: "#64748b", // Slate
    bbox: [350, 220, 720, 430],
    confidence: 0.96,
    trackId: "TRK-03"
  }
];

export const mockEvents = [
  {
    id: "evt-1",
    type: "APPROACH",
    timestamp: "00:02",
    subject: "Person #01",
    object: "Table #01",
    confidence: 0.91,
    description: "Person #01 approaches Table #01",
    evidence: {
      distance: "1.85m",
      positionChange: false,
      synchronizedMotion: false,
      items: [
        { label: "Proximity vector decreasing toward Table #01", verified: true },
        { label: "Linear approach velocity: 1.1 m/s", verified: true },
        { label: "Spatial target: Table #01 anchor zone", verified: true }
      ]
    },
    sceneGraph: [
      { source: "Person #01", relation: "approaching", target: "Table #01", active: true },
      { source: "Laptop #02", relation: "on", target: "Table #01", active: true }
    ],
    explanation: "Person #01 approaches Table #01 at 00:02 with directed forward trajectory."
  },
  {
    id: "evt-2",
    type: "REACH",
    timestamp: "00:03",
    subject: "Person #01",
    object: "Laptop #02",
    confidence: 0.93,
    description: "Person #01 reaches toward Laptop #02",
    evidence: {
      distance: "0.58m",
      positionChange: false,
      synchronizedMotion: false,
      items: [
        { label: "Upper extremity reaching trajectory detected", verified: true },
        { label: "Distance to Laptop #02 decreased to 0.58m", verified: true },
        { label: "Hand keypoint alignment with object boundary", verified: true }
      ]
    },
    sceneGraph: [
      { source: "Person #01", relation: "reaching", target: "Laptop #02", active: true },
      { source: "Laptop #02", relation: "on", target: "Table #01", active: true }
    ],
    explanation: "Person #01 reaches hand toward Laptop #02 located on Table #01 at 00:03."
  },
  {
    id: "evt-3",
    type: "PICK_UP",
    timestamp: "00:05",
    subject: "Person #01",
    object: "Laptop #02",
    confidence: 0.94,
    description: "Person #01 picks up Laptop #02",
    evidence: {
      distance: "0.42m",
      positionChange: true,
      synchronizedMotion: true,
      items: [
        { label: "Distance decreased (< 0.5m)", verified: true },
        { label: "Object position changed (vertical lift Δz > 15cm)", verified: true },
        { label: "Synchronized movement between Person #01 and Laptop #02", verified: true }
      ]
    },
    sceneGraph: [
      { source: "Person #01", relation: "holding", target: "Laptop #02", active: true },
      { source: "Laptop #02", relation: "lifted from", target: "Table #01", active: false }
    ],
    explanation: "Person #01 picked up Laptop #02 at 00:05 and carried it away from Table #01."
  },
  {
    id: "evt-4",
    type: "CARRY",
    timestamp: "00:08",
    subject: "Person #01",
    object: "Laptop #02",
    confidence: 0.92,
    description: "Person #01 carries Laptop #02 away",
    evidence: {
      distance: "0.38m",
      positionChange: true,
      synchronizedMotion: true,
      items: [
        { label: "Persistent contact & distance invariance (0.38m)", verified: true },
        { label: "Joint displacement vector matches subject locomotion", verified: true },
        { label: "Moving away from Table #01 origin position", verified: true }
      ]
    },
    sceneGraph: [
      { source: "Person #01", relation: "holding & carrying", target: "Laptop #02", active: true },
      { source: "Person #01", relation: "moving away from", target: "Table #01", active: true }
    ],
    explanation: "Person #01 carries Laptop #02 away from Table #01 at 00:08."
  }
];

export const defaultSceneRelations = [
  { source: "Person #01", relation: "approaching", target: "Table #01" },
  { source: "Person #01", relation: "reaching", target: "Laptop #02" },
  { source: "Laptop #02", relation: "on", target: "Table #01" }
];

export const pickUpSceneRelations = [
  { source: "Person #01", relation: "holding", target: "Laptop #02" }
];
