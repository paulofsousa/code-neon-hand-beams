import {
  HandLandmarker,
  FilesetResolver,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.22/vision_bundle.mjs";

// ---------- Configuration ----------
const MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task";
const WASM_URL =
  "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.22/wasm";

// 21 MediaPipe hand landmarks — connections between joints (wireframe edges).
const HAND_CONNECTIONS = [
  [0, 1], [1, 2], [2, 3], [3, 4],          // thumb
  [0, 5], [5, 6], [6, 7], [7, 8],          // index
  [5, 9], [9, 10], [10, 11], [11, 12],     // middle
  [9, 13], [13, 14], [14, 15], [15, 16],   // ring
  [13, 17], [17, 18], [18, 19], [19, 20],  // pinky
  [0, 17],                                 // palm base
];

// Fingertip landmark indices: thumb, index, middle, ring, pinky.
const FINGERTIPS = [4, 8, 12, 16, 20];

// Distinct beam color per fingertip pair (hand 1 ↔ hand 2).
const BEAM_COLORS = [
  "#ff2bd6", // pink   — thumbs
  "#ffd400", // yellow — indices
  "#39ff14", // green  — middles
  "#1aa7ff", // blue   — rings
  "#ff3b30", // red    — pinkies
];

// Pinch threshold as a fraction of canvas diagonal — scales across devices.
const PINCH_THRESHOLD_RATIO = 0.05;

// ---------- DOM refs ----------
const video = document.getElementById("video");
const canvas = document.getElementById("overlay");
const ctx = canvas.getContext("2d");
const handsCountEl = document.getElementById("hands-count");
const gestureEl = document.getElementById("gesture");
const fpsEl = document.getElementById("fps");
const startScreen = document.getElementById("start-screen");
const startBtn = document.getElementById("start-btn");
const errorScreen = document.getElementById("error-screen");
const errorMsg = document.getElementById("error-message");

// ---------- State ----------
let handLandmarker = null;
let running = false;
let lastVideoTime = -1;
let fpsSamples = [];

// ---------- Bootstrapping ----------
async function loadModel() {
  const fileset = await FilesetResolver.forVisionTasks(WASM_URL);
  handLandmarker = await HandLandmarker.createFromOptions(fileset, {
    baseOptions: { modelAssetPath: MODEL_URL, delegate: "GPU" },
    runningMode: "VIDEO",
    numHands: 2,
    minHandDetectionConfidence: 0.5,
    minHandPresenceConfidence: 0.5,
    minTrackingConfidence: 0.5,
  });
}

async function startCamera() {
  const constraints = {
    audio: false,
    video: {
      facingMode: "user",
      width: { ideal: 1280 },
      height: { ideal: 720 },
    },
  };
  const stream = await navigator.mediaDevices.getUserMedia(constraints);
  video.srcObject = stream;
  await new Promise((res) => {
    if (video.readyState >= 2) return res();
    video.onloadedmetadata = () => res();
  });
  await video.play();
  resizeCanvas();
}

function resizeCanvas() {
  // Match canvas backing store to the video frame so landmark coords
  // (which are normalized to the frame) map 1:1 to canvas pixels.
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
}

// ---------- Drawing ----------
function drawGlowLine(x1, y1, x2, y2, color, thickness = 3) {
  // Two-pass glow: wide soft halo, then crisp bright core on top.
  ctx.save();
  ctx.lineCap = "round";
  ctx.shadowColor = color;
  ctx.shadowBlur = 16;
  ctx.strokeStyle = color;
  ctx.globalAlpha = 0.55;
  ctx.lineWidth = thickness * 2.5;
  ctx.beginPath();
  ctx.moveTo(x1, y1);
  ctx.lineTo(x2, y2);
  ctx.stroke();

  ctx.shadowBlur = 8;
  ctx.globalAlpha = 1.0;
  ctx.lineWidth = thickness;
  ctx.strokeStyle = "#eafffb";
  ctx.beginPath();
  ctx.moveTo(x1, y1);
  ctx.lineTo(x2, y2);
  ctx.stroke();
  ctx.restore();
}

function drawGlowDot(x, y, color, radius = 4) {
  ctx.save();
  ctx.shadowColor = color;
  ctx.shadowBlur = 14;
  ctx.fillStyle = color;
  ctx.beginPath();
  ctx.arc(x, y, radius * 1.6, 0, Math.PI * 2);
  ctx.fill();

  ctx.shadowBlur = 6;
  ctx.fillStyle = "#ffffff";
  ctx.beginPath();
  ctx.arc(x, y, radius * 0.6, 0, Math.PI * 2);
  ctx.fill();
  ctx.restore();
}

function drawHandWireframe(landmarks, w, h) {
  const NEON = "#00ffe7";
  for (const [a, b] of HAND_CONNECTIONS) {
    const p1 = landmarks[a];
    const p2 = landmarks[b];
    drawGlowLine(p1.x * w, p1.y * h, p2.x * w, p2.y * h, NEON, 3);
  }
  for (const lm of landmarks) {
    drawGlowDot(lm.x * w, lm.y * h, NEON, 3);
  }
}

function drawBeams(hand1, hand2, w, h) {
  for (let i = 0; i < FINGERTIPS.length; i++) {
    const idx = FINGERTIPS[i];
    const p1 = hand1[idx];
    const p2 = hand2[idx];
    drawGlowLine(p1.x * w, p1.y * h, p2.x * w, p2.y * h, BEAM_COLORS[i], 4);
  }
}

// ---------- Gesture detection ----------
function detectPinch(landmarks, w, h) {
  const thumb = landmarks[4];
  const index = landmarks[8];
  const dx = (thumb.x - index.x) * w;
  const dy = (thumb.y - index.y) * h;
  const dist = Math.hypot(dx, dy);
  const diagonal = Math.hypot(w, h);
  return dist < diagonal * PINCH_THRESHOLD_RATIO;
}

// ---------- Main loop ----------
function updateFps(now) {
  fpsSamples.push(now);
  // Keep only samples from the last second.
  while (fpsSamples.length && now - fpsSamples[0] > 1000) fpsSamples.shift();
  fpsEl.textContent = fpsSamples.length;
}

function renderLoop() {
  if (!running) return;

  if (video.videoWidth && video.videoHeight) {
    if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
      resizeCanvas();
    }

    const w = canvas.width;
    const h = canvas.height;
    ctx.clearRect(0, 0, w, h);

    const now = performance.now();
    let result = null;
    if (video.currentTime !== lastVideoTime) {
      lastVideoTime = video.currentTime;
      result = handLandmarker.detectForVideo(video, now);
    }

    if (result && result.landmarks && result.landmarks.length > 0) {
      const hands = result.landmarks;

      // Beams render BEHIND the wireframes for cleaner look.
      if (hands.length === 2) {
        drawBeams(hands[0], hands[1], w, h);
      }

      for (const lm of hands) {
        drawHandWireframe(lm, w, h);
      }

      // HUD updates.
      handsCountEl.textContent = hands.length;

      // Pinch = any hand pinching.
      const pinching = hands.some((lm) => detectPinch(lm, w, h));
      if (pinching) {
        gestureEl.textContent = "PINCH!";
        gestureEl.classList.add("alert");
      } else {
        gestureEl.textContent = "None";
        gestureEl.classList.remove("alert");
      }
    } else {
      handsCountEl.textContent = "0";
      gestureEl.textContent = "None";
      gestureEl.classList.remove("alert");
    }

    updateFps(now);
  }

  requestAnimationFrame(renderLoop);
}

// ---------- Startup ----------
async function start() {
  startBtn.disabled = true;
  startBtn.textContent = "LOADING...";
  try {
    await loadModel();
    await startCamera();
    startScreen.hidden = true;
    running = true;
    requestAnimationFrame(renderLoop);
  } catch (err) {
    console.error(err);
    showError(err);
  }
}

function showError(err) {
  startScreen.hidden = true;
  errorScreen.hidden = false;
  errorMsg.textContent = (err && err.message) || String(err);
}

startBtn.addEventListener("click", start);

// Recompute canvas size if device orientation changes mid-session.
window.addEventListener("resize", () => {
  if (running) resizeCanvas();
});
