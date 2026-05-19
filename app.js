// ---------- Debug log panel ----------
// Visible on screen so errors are readable without DevTools on mobile/deployed.
const _logEl = (() => {
  const el = document.createElement("pre");
  el.id = "debug-log";
  el.style.cssText = [
    "position:fixed", "bottom:0", "left:0", "right:0",
    "max-height:40vh", "overflow-y:auto",
    "background:rgba(0,0,0,0.85)", "color:#0f0",
    "font:11px/1.4 monospace", "padding:8px",
    "white-space:pre-wrap", "word-break:break-all",
    "z-index:9999", "border-top:1px solid #0f0",
    "pointer-events:none",
  ].join(";");
  // Module scripts run after DOMContentLoaded, so listening for that event is
  // a no-op. Append immediately if body exists, otherwise fall back to the event.
  if (document.body) {
    document.body.appendChild(el);
  } else {
    document.addEventListener("DOMContentLoaded", () => document.body.appendChild(el));
  }
  return el;
})();

function dbg(level, ...args) {
  const ts = (performance.now() / 1000).toFixed(3);
  const prefix = `[${ts}] [${level}]`;
  const text = args.map((a) => {
    if (a instanceof Error) return `${a.name}: ${a.message}\n${a.stack || ""}`;
    if (typeof a === "object") { try { return JSON.stringify(a); } catch { return String(a); } }
    return String(a);
  }).join(" ");
  const line = `${prefix} ${text}`;
  if (level === "ERR") console.error(line);
  else if (level === "WARN") console.warn(line);
  else console.log(line);
  _logEl.textContent += line + "\n";
  _logEl.scrollTop = _logEl.scrollHeight;
}

const log  = (...a) => dbg("LOG ", ...a);
const warn = (...a) => dbg("WARN", ...a);
const err  = (...a) => dbg("ERR ", ...a);

// Catch unhandled rejections so they show in the panel too.
window.addEventListener("unhandledrejection", (e) => {
  err("unhandledrejection:", e.reason);
});
window.addEventListener("error", (e) => {
  err("globalerror:", e.message, "at", e.filename, e.lineno);
});

log("--- app module loading ---");
log("UA:", navigator.userAgent);
log("HTTPS:", location.protocol === "https:");
log("href:", location.href);

import {
  HandLandmarker,
  FilesetResolver,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.35/vision_bundle.mjs";

log("MediaPipe import OK");

// ---------- Configuration ----------
const MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task";
const WASM_URL =
  "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.35/wasm";

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
  log("loadModel: resolving WASM fileset from", WASM_URL);
  const fileset = await FilesetResolver.forVisionTasks(WASM_URL);
  log("loadModel: fileset resolved, creating HandLandmarker (GPU)");
  try {
    handLandmarker = await HandLandmarker.createFromOptions(fileset, {
      baseOptions: { modelAssetPath: MODEL_URL, delegate: "GPU" },
      runningMode: "VIDEO",
      numHands: 2,
      minHandDetectionConfidence: 0.5,
      minHandPresenceConfidence: 0.5,
      minTrackingConfidence: 0.5,
    });
    log("loadModel: HandLandmarker ready (GPU)");
  } catch (gpuErr) {
    warn("loadModel: GPU delegate failed, falling back to CPU:", gpuErr);
    handLandmarker = await HandLandmarker.createFromOptions(fileset, {
      baseOptions: { modelAssetPath: MODEL_URL, delegate: "CPU" },
      runningMode: "VIDEO",
      numHands: 2,
      minHandDetectionConfidence: 0.5,
      minHandPresenceConfidence: 0.5,
      minTrackingConfidence: 0.5,
    });
    log("loadModel: HandLandmarker ready (CPU fallback)");
  }
}

async function attachCameraStream(stream) {
  const tracks = stream.getVideoTracks();
  log("attachCameraStream: tracks:", tracks.length, tracks.map((t) => `${t.label} [${t.readyState}]`).join(", "));
  if (tracks.length) {
    const s = tracks[0].getSettings();
    log("attachCameraStream: track settings:", `${s.width}x${s.height} facing=${s.facingMode}`);
  }
  video.srcObject = stream;
  await new Promise((res) => {
    if (video.readyState >= 2) { log("attachCameraStream: metadata already ready"); return res(); }
    video.onloadedmetadata = () => { log("attachCameraStream: onloadedmetadata fired"); res(); };
  });
  log("attachCameraStream: calling video.play()");
  await video.play();
  log("attachCameraStream: play() resolved, video size:", video.videoWidth, "x", video.videoHeight);
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
// Progressive constraint fallback: start with the richest constraints, retry
// with looser ones if the browser/device rejects them (e.g. desktop webcams
// that don't advertise a facingMode, devices that can't hit 1280x720).
const CAMERA_CONSTRAINT_ATTEMPTS = [
  { audio: false, video: { facingMode: "user", width: { ideal: 1280 }, height: { ideal: 720 } } },
  { audio: false, video: { facingMode: "user" } },
  { audio: false, video: { width: { ideal: 1280 }, height: { ideal: 720 } } },
  { audio: false, video: true },
];

// iOS Safari requires getUserMedia to be called synchronously inside the
// user-gesture handler — any awaits before it cost the gesture context and
// the call gets rejected silently. So the click handler stays NON-async,
// fires off camera + model requests in parallel, and only then awaits.
function handleStart() {
  log("handleStart: user clicked START");
  log("handleStart: mediaDevices present:", !!navigator.mediaDevices);
  log("handleStart: getUserMedia present:", !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia));

  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    err("handleStart: getUserMedia missing");
    showError(
      "Browser unsupported",
      "navigator.mediaDevices.getUserMedia is missing — try the latest Safari/Chrome over HTTPS."
    );
    return;
  }

  startBtn.disabled = true;
  startBtn.textContent = "LOADING...";

  // Kick off the first getUserMedia call SYNCHRONOUSLY in the gesture handler
  // (iOS Safari requirement). The fallback chain runs only if this first one
  // fails — and at that point the user-gesture token is already spent, so the
  // browser must remember the permission decision for retries to work.
  let cameraPromise;
  try {
    log("handleStart: calling getUserMedia (primary constraints)");
    cameraPromise = navigator.mediaDevices
      .getUserMedia(CAMERA_CONSTRAINT_ATTEMPTS[0])
      .catch(async (firstErr) => {
        warn("primary getUserMedia failed:", firstErr.name, "-", firstErr.message);
        if (firstErr.name === "NotAllowedError" || firstErr.name === "SecurityError") {
          throw firstErr;
        }
        // Try remaining fallbacks (skip index 0 which we just tried).
        let lastErr = firstErr;
        for (let i = 1; i < CAMERA_CONSTRAINT_ATTEMPTS.length; i++) {
          const constraints = CAMERA_CONSTRAINT_ATTEMPTS[i];
          try {
            log(`getUserMedia fallback ${i}:`, JSON.stringify(constraints));
            return await navigator.mediaDevices.getUserMedia(constraints);
          } catch (e) {
            warn(`fallback ${i} failed:`, e.name, "-", e.message);
            lastErr = e;
            if (e.name === "NotAllowedError" || e.name === "SecurityError") throw e;
          }
        }
        throw lastErr;
      });
    log("handleStart: getUserMedia promise created (not yet resolved)");
  } catch (e) {
    err("handleStart: getUserMedia threw synchronously:", e);
    showError("Camera request threw", formatError(e));
    return;
  }

  log("handleStart: starting model load in parallel");
  const modelPromise = loadModel();

  Promise.resolve()
    .then(async () => {
      let stream;
      try {
        log("stage[camera]: awaiting getUserMedia");
        stream = await cameraPromise;
        log("stage[camera]: stream obtained");
      } catch (e) {
        err("stage[camera]: getUserMedia rejected:", e);
        throw { stage: "camera", err: e };
      }
      try {
        log("stage[video]: attaching stream");
        await attachCameraStream(stream);
        log("stage[video]: stream attached OK");
      } catch (e) {
        err("stage[video]: attachCameraStream threw:", e);
        throw { stage: "video", err: e };
      }
      try {
        log("stage[model]: awaiting HandLandmarker");
        await modelPromise;
        log("stage[model]: model ready");
      } catch (e) {
        err("stage[model]: model load failed:", e);
        throw { stage: "model", err: e };
      }
      log("startup complete — entering render loop");
      startScreen.hidden = true;
      running = true;
      requestAnimationFrame(renderLoop);
    })
    .catch((wrapped) => {
      const stage = wrapped && wrapped.stage ? wrapped.stage : "unknown";
      const e = wrapped && wrapped.err ? wrapped.err : wrapped;
      err(`startup failed at stage [${stage}]:`, e);
      const titles = {
        camera: "Camera access denied",
        video: "Video stream failed",
        model: "Hand tracking model failed to load",
        unknown: "Startup failed",
      };
      showError(titles[stage] || titles.unknown, formatError(e));
    });
}

function formatError(e) {
  if (!e) return "(no error details)";
  if (typeof e === "string") return e;
  const name = e.name || e.constructor?.name || "Error";
  const msg = e.message || String(e) || "(no message)";
  const stack = e.stack ? "\n\n" + e.stack : "";
  return `${name}: ${msg}${stack}`;
}

function showError(title, detail) {
  startScreen.hidden = true;
  errorScreen.hidden = false;
  const titleEl = errorScreen.querySelector("h2");
  if (titleEl) titleEl.textContent = title || "Something went wrong";
  errorMsg.textContent = detail || "(no details)";
  // Make debug log panel interactive so user can scroll/copy on error screens.
  _logEl.style.pointerEvents = "auto";
  // Re-enable retry from the error screen.
  let retryBtn = document.getElementById("retry-btn");
  if (!retryBtn) {
    retryBtn = document.createElement("button");
    retryBtn.id = "retry-btn";
    retryBtn.textContent = "RETRY";
    retryBtn.addEventListener("click", () => {
      errorScreen.hidden = true;
      startScreen.hidden = false;
      startBtn.disabled = false;
      startBtn.textContent = "START";
    });
    errorScreen.appendChild(retryBtn);
  }
}

if (!startBtn) {
  err("FATAL: start-btn element not found in DOM");
} else {
  startBtn.addEventListener("click", handleStart);
  // Some touchscreen kiosks/wrappers fire only touchend, not click — belt-and-suspenders.
  startBtn.addEventListener("touchend", (e) => {
    e.preventDefault();
    handleStart();
  }, { passive: false });
  log("start button listeners attached (click + touchend)");
}

// Probe device list early — doesn't require permission, just confirms a camera is present.
if (navigator.mediaDevices && navigator.mediaDevices.enumerateDevices) {
  navigator.mediaDevices.enumerateDevices()
    .then((devices) => {
      const cams = devices.filter((d) => d.kind === "videoinput");
      log(`enumerateDevices: ${cams.length} video input(s) found`);
      cams.forEach((c, i) => log(`  cam[${i}]: label="${c.label || "(hidden until permission granted)"}" id=${c.deviceId.slice(0, 8)}...`));
      if (cams.length === 0) {
        warn("No video input devices reported. Camera will fail to start.");
      }
    })
    .catch((e) => warn("enumerateDevices failed:", e));
}

// Recompute canvas size if device orientation changes mid-session.
window.addEventListener("resize", () => {
  if (running) resizeCanvas();
});
