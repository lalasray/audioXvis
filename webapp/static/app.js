const state = {
  config: null,
  meshMeta: null,
  runtime: null,
  renderers: {},
  events: null,
  viewMode: "single",
  audioPlayers: {
    primary: new Audio(),
    compare: new Audio(),
  },
  audioMuted: {
    primary: false,
    compare: false,
  },
  audioActive: {
    primary: false,
    compare: false,
  },
  overlayHistory: {
    primary: [],
    compare: [],
  },
};

const els = {
  micSelect: document.getElementById("micSelect"),
  sampleSelect: document.getElementById("sampleSelect"),
  fileInput: document.getElementById("fileInput"),
  compareSampleSelect: document.getElementById("compareSampleSelect"),
  compareFileInput: document.getElementById("compareFileInput"),
  startMicBtn: document.getElementById("startMicBtn"),
  playSampleBtn: document.getElementById("playSampleBtn"),
  playCompareSampleBtn: document.getElementById("playCompareSampleBtn"),
  stopBtn: document.getElementById("stopBtn"),
  stopCompareBtn: document.getElementById("stopCompareBtn"),
  stopBtnTop: document.getElementById("stopBtnTop"),
  transportStopBtn: document.getElementById("transportStopBtn"),
  transportPlayBtn: document.getElementById("transportPlayBtn"),
  mutePrimaryBtn: document.getElementById("mutePrimaryBtn"),
  muteCompareBtn: document.getElementById("muteCompareBtn"),
  muteBothBtn: document.getElementById("muteBothBtn"),
  refreshBtn: document.getElementById("refreshBtn"),
  resetViewBtn: document.getElementById("resetViewBtn"),
  frontViewBtn: document.getElementById("frontViewBtn"),
  backViewBtn: document.getElementById("backViewBtn"),
  sideViewBtn: document.getElementById("sideViewBtn"),
  topViewBtn: document.getElementById("topViewBtn"),
  zoomInBtn: document.getElementById("zoomInBtn"),
  zoomOutBtn: document.getElementById("zoomOutBtn"),
  angleA: document.getElementById("angleA"),
  angleB: document.getElementById("angleB"),
  angleC: document.getElementById("angleC"),
  deltaA: document.getElementById("deltaA"),
  deltaB: document.getElementById("deltaB"),
  deltaC: document.getElementById("deltaC"),
  deltaMean: document.getElementById("deltaMean"),
  statusLabel: document.getElementById("statusLabel"),
  sourceLabel: document.getElementById("sourceLabel"),
  primarySourceLabel: document.getElementById("primarySourceLabel"),
  compareSourceLabel: document.getElementById("compareSourceLabel"),
  comparisonSummary: document.getElementById("comparisonSummary"),
  timeLabel: document.getElementById("timeLabel"),
  progressFill: document.getElementById("progressFill"),
  waveCanvas: document.getElementById("waveCanvas"),
  pitchCanvas: document.getElementById("pitchCanvas"),
  glCanvas: document.getElementById("glCanvas"),
  glPrimaryCanvas: document.getElementById("glPrimaryCanvas"),
  glCompareCanvas: document.getElementById("glCompareCanvas"),
  glOverlayCanvas: document.getElementById("glOverlayCanvas"),
  singleStage: document.getElementById("singleStage"),
  splitStage: document.getElementById("splitStage"),
  overlayStage: document.getElementById("overlayStage"),
  modeButtons: Array.from(document.querySelectorAll("[data-view-mode]")),
};

async function fetchJson(url, options) {
  const response = await fetch(url, options);
  if (!response.ok) {
    throw new Error(`${response.status} ${response.statusText}`);
  }
  return response.json();
}

function formatTime(seconds) {
  const total = Math.max(0, Math.floor(seconds || 0));
  const mins = Math.floor(total / 60);
  const secs = total % 60;
  return `${mins}:${String(secs).padStart(2, "0")}`;
}

function mediaUrlForPath(path) {
  return `/api/media?path=${encodeURIComponent(path)}`;
}

function updateMuteButton() {
  Object.entries(state.audioPlayers).forEach(([track, audio]) => {
    audio.muted = Boolean(state.audioMuted[track]);
  });

  const primaryActive = state.audioActive.primary;
  const compareActive = state.audioActive.compare;
  els.mutePrimaryBtn.hidden = !primaryActive;
  els.muteCompareBtn.hidden = !compareActive;
  els.muteBothBtn.hidden = !(primaryActive && compareActive);

  els.mutePrimaryBtn.textContent = state.audioMuted.primary ? "Unmute Main" : "Mute Main";
  els.muteCompareBtn.textContent = state.audioMuted.compare ? "Unmute Compare" : "Mute Compare";
  els.muteBothBtn.textContent = state.audioMuted.primary && state.audioMuted.compare ? "Unmute Both" : "Mute Both";
  els.mutePrimaryBtn.classList.toggle("muted", state.audioMuted.primary);
  els.muteCompareBtn.classList.toggle("muted", state.audioMuted.compare);
  els.muteBothBtn.classList.toggle("muted", state.audioMuted.primary && state.audioMuted.compare);
}

function stopBrowserAudio(track = null) {
  const entries = track
    ? [[track, state.audioPlayers[track]]].filter(([, audio]) => Boolean(audio))
    : Object.entries(state.audioPlayers);
  entries.forEach(([trackKey, audio]) => {
    audio.pause();
    audio.removeAttribute("src");
    audio.load();
    state.audioActive[trackKey] = false;
  });
  updateMuteButton();
}

async function playBrowserAudio(track, path) {
  const audio = state.audioPlayers[track];
  if (!audio) return;
  stopBrowserAudio(track);
  audio.src = mediaUrlForPath(path);
  audio.currentTime = 0;
  audio.muted = Boolean(state.audioMuted[track]);
  try {
    await audio.play();
    state.audioActive[track] = true;
    updateMuteButton();
  } catch (error) {
    els.statusLabel.textContent = `Audio playback failed: ${error.message}`;
  }
}

function toggleTrackMute(track) {
  state.audioMuted[track] = !state.audioMuted[track];
  updateMuteButton();
}

function toggleBothMute() {
  const shouldMute = !(state.audioMuted.primary && state.audioMuted.compare);
  state.audioMuted.primary = shouldMute;
  state.audioMuted.compare = shouldMute;
  updateMuteButton();
}

function addMediaOption(select, path, label) {
  const existing = Array.from(select.options).find((option) => option.value === path);
  if (existing) {
    select.value = path;
    return;
  }
  const option = document.createElement("option");
  option.value = path;
  option.textContent = label;
  select.appendChild(option);
  select.value = path;
}

function drawWaveform(primary = [], compare = []) {
  const canvas = els.waveCanvas;
  const ctx = canvas.getContext("2d");
  const { width, height } = canvas;
  ctx.clearRect(0, 0, width, height);

  const gradient = ctx.createLinearGradient(0, 0, 0, height);
  gradient.addColorStop(0, "#7fe59d");
  gradient.addColorStop(1, "#3cbf78");
  ctx.fillStyle = "rgba(12, 16, 22, 0.95)";
  ctx.fillRect(0, 0, width, height);

  ctx.strokeStyle = "rgba(255,255,255,0.08)";
  ctx.beginPath();
  ctx.moveTo(0, height * 0.5);
  ctx.lineTo(width, height * 0.5);
  ctx.stroke();

  const mid = height * 0.5;
  const drawArea = (samples, fillStyle) => {
    if (!samples.length) return;
    ctx.beginPath();
    ctx.moveTo(0, mid);
    for (let i = 0; i < samples.length; i += 1) {
      const x = (i / (samples.length - 1)) * width;
      const y = mid - samples[i] * height * 0.38;
      ctx.lineTo(x, y);
    }
    ctx.lineTo(width, mid);
    ctx.lineTo(width, height);
    ctx.lineTo(0, height);
    ctx.closePath();
    ctx.fillStyle = fillStyle;
    ctx.fill();
  };

  drawArea(primary, gradient);
  drawArea(compare, "rgba(255, 167, 88, 0.5)");
}

function drawPitch(primary = [], compare = []) {
  const canvas = els.pitchCanvas;
  const ctx = canvas.getContext("2d");
  const { width, height } = canvas;
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "rgba(12, 16, 22, 0.95)";
  ctx.fillRect(0, 0, width, height);

  const minHz = 55;
  const maxHz = 500;
  const minLog = Math.log2(minHz);
  const maxLog = Math.log2(maxHz);
  const yForHz = (hz) => {
    if (!hz || hz <= 0) return null;
    const t = (Math.log2(Math.max(minHz, Math.min(maxHz, hz))) - minLog) / (maxLog - minLog);
    return height - 12 - t * (height - 24);
  };

  ctx.strokeStyle = "rgba(255,255,255,0.08)";
  ctx.lineWidth = 1;
  [100, 200, 300, 400].forEach((hz) => {
    const y = yForHz(hz);
    if (y === null) return;
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(width, y);
    ctx.stroke();
  });

  const drawLine = (samples, strokeStyle) => {
    if (!samples.length) return;
    ctx.beginPath();
    let started = false;
    for (let i = 0; i < samples.length; i += 1) {
      const y = yForHz(samples[i]);
      if (y === null) {
        started = false;
        continue;
      }
      const x = (i / Math.max(samples.length - 1, 1)) * width;
      if (!started) {
        ctx.moveTo(x, y);
        started = true;
      } else {
        ctx.lineTo(x, y);
      }
    }
    ctx.strokeStyle = strokeStyle;
    ctx.lineWidth = 2.5;
    ctx.stroke();
  };

  drawLine(primary, "#7fe59d");
  drawLine(compare, "rgba(255, 85, 70, 0.9)");
}

function makeMat4() {
  return new Float32Array(16);
}

function perspective(out, fovy, aspect, near, far) {
  const f = 1 / Math.tan(fovy / 2);
  out.fill(0);
  out[0] = f / aspect;
  out[5] = f;
  out[10] = (far + near) / (near - far);
  out[11] = -1;
  out[14] = (2 * far * near) / (near - far);
  return out;
}

function subtract(a, b) {
  return [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
}

function cross(a, b) {
  return [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ];
}

function dot(a, b) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

function normalize(v) {
  const len = Math.hypot(v[0], v[1], v[2]) || 1;
  return [v[0] / len, v[1] / len, v[2] / len];
}

function lookAt(out, eye, target, up) {
  const z = normalize(subtract(eye, target));
  const x = normalize(cross(up, z));
  const y = cross(z, x);
  out[0] = x[0];
  out[1] = y[0];
  out[2] = z[0];
  out[3] = 0;
  out[4] = x[1];
  out[5] = y[1];
  out[6] = z[1];
  out[7] = 0;
  out[8] = x[2];
  out[9] = y[2];
  out[10] = z[2];
  out[11] = 0;
  out[12] = -dot(x, eye);
  out[13] = -dot(y, eye);
  out[14] = -dot(z, eye);
  out[15] = 1;
  return out;
}

function createShader(gl, type, source) {
  const shader = gl.createShader(type);
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    throw new Error(gl.getShaderInfoLog(shader));
  }
  return shader;
}

class MeshRenderer {
  constructor(canvas, meshMeta, positions, colors, normals) {
    this.canvas = canvas;
    this.meshMeta = meshMeta;
    this.vertexCount = meshMeta.vertexCount;
    this.frameCount = meshMeta.frameCount;
    this.positions = new Float32Array(positions);
    this.colors = new Uint8Array(colors);
    this.normals = new Float32Array(normals);
    this.layers = [];
    this.visible = true;
    this.rafId = null;
    this.lastUploadedFrame = null;
    this.camera = {
      yaw: 0.5,
      pitch: -0.3,
      distance: meshMeta.bounds.scale * 1.8,
      target: meshMeta.bounds.center.slice(),
    };
    this.gl = canvas.getContext("webgl");
    if (!this.gl) {
      throw new Error("WebGL is not available in this browser.");
    }
    this._setup();
    this._bindInteractions();
    this.setPreset("front");
    this.requestRender();
  }

  _setup() {
    const gl = this.gl;
    const vertexShader = createShader(gl, gl.VERTEX_SHADER, `
      attribute vec3 aPosition;
      attribute vec3 aNormal;
      attribute vec3 aColor;
      uniform mat4 uProjection;
      uniform mat4 uView;
      uniform vec3 uTint;
      uniform vec3 uOffset;
      uniform float uScale;
      uniform float uTintMix;
      uniform float uAlpha;
      varying vec3 vColor;
      varying vec3 vNormal;
      varying float vAlpha;
      void main() {
        vec4 worldPos = vec4(aPosition * uScale + uOffset, 1.0);
        gl_Position = uProjection * uView * worldPos;
        vColor = mix(aColor, uTint, uTintMix);
        vNormal = aNormal;
        vAlpha = uAlpha;
      }
    `);
    const fragmentShader = createShader(gl, gl.FRAGMENT_SHADER, `
      precision mediump float;
      varying vec3 vColor;
      varying vec3 vNormal;
      varying float vAlpha;
      void main() {
        vec3 lightDir = normalize(vec3(0.4, 0.8, 0.7));
        vec3 normal = normalize(vNormal);
        float diffuse = max(dot(normal, lightDir), 0.0);
        float rim = pow(1.0 - max(dot(normal, normalize(vec3(0.0, 0.0, 1.0))), 0.0), 1.7);
        vec3 color = vColor * (0.22 + diffuse * 0.9) + vec3(0.18, 0.2, 0.24) * rim * 0.4;
        gl_FragColor = vec4(color, vAlpha);
      }
    `);
    const program = gl.createProgram();
    gl.attachShader(program, vertexShader);
    gl.attachShader(program, fragmentShader);
    gl.linkProgram(program);
    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
      throw new Error(gl.getProgramInfoLog(program));
    }
    gl.useProgram(program);
    this.program = program;

    this.positionBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, this.positionBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, this.vertexCount * 3 * 4, gl.DYNAMIC_DRAW);

    this.normalBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, this.normalBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, this.normals, gl.STATIC_DRAW);

    this.colorBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, this.colorBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, this.colors, gl.STATIC_DRAW);

    this.aPosition = gl.getAttribLocation(program, "aPosition");
    this.aNormal = gl.getAttribLocation(program, "aNormal");
    this.aColor = gl.getAttribLocation(program, "aColor");
    this.uProjection = gl.getUniformLocation(program, "uProjection");
    this.uView = gl.getUniformLocation(program, "uView");
    this.uTint = gl.getUniformLocation(program, "uTint");
    this.uOffset = gl.getUniformLocation(program, "uOffset");
    this.uScale = gl.getUniformLocation(program, "uScale");
    this.uTintMix = gl.getUniformLocation(program, "uTintMix");
    this.uAlpha = gl.getUniformLocation(program, "uAlpha");

    gl.enable(gl.DEPTH_TEST);
    gl.enable(gl.CULL_FACE);
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
    gl.clearColor(0.01, 0.02, 0.03, 1.0);
  }

  _bindInteractions() {
    let dragging = false;
    let lastX = 0;
    let lastY = 0;

    this.canvas.addEventListener("pointerdown", (event) => {
      dragging = true;
      lastX = event.clientX;
      lastY = event.clientY;
      this.canvas.setPointerCapture(event.pointerId);
    });

    this.canvas.addEventListener("pointermove", (event) => {
      if (!dragging) return;
      const dx = event.clientX - lastX;
      const dy = event.clientY - lastY;
      lastX = event.clientX;
      lastY = event.clientY;
      this.camera.yaw += dx * 0.01;
      this.camera.pitch = Math.max(-1.4, Math.min(1.4, this.camera.pitch + dy * 0.01));
      this.requestRender();
    });

    const endDrag = (event) => {
      dragging = false;
      try {
        this.canvas.releasePointerCapture(event.pointerId);
      } catch (_) {
        // ignore
      }
    };
    this.canvas.addEventListener("pointerup", endDrag);
    this.canvas.addEventListener("pointercancel", endDrag);

    this.canvas.addEventListener("wheel", (event) => {
      event.preventDefault();
      this.camera.distance = Math.max(
        this.meshMeta.bounds.scale * 0.6,
        Math.min(this.meshMeta.bounds.scale * 4.2, this.camera.distance * (1 + event.deltaY * 0.001)),
      );
      this.requestRender();
    }, { passive: false });
  }

  setPreset(name) {
    if (name === "front") {
      this.camera.yaw = 0.0;
      this.camera.pitch = 0.0;
    } else if (name === "back") {
      this.camera.yaw = Math.PI;
      this.camera.pitch = 0.0;
    } else if (name === "side") {
      this.camera.yaw = Math.PI / 2;
      this.camera.pitch = 0.0;
    } else if (name === "top") {
      this.camera.yaw = Math.PI;
      this.camera.pitch = Math.PI / 2;
    }
    this.requestRender();
  }

  zoomBy(factor) {
    this.camera.distance = Math.max(
      this.meshMeta.bounds.scale * 0.6,
      Math.min(this.meshMeta.bounds.scale * 4.2, this.camera.distance * factor),
    );
    this.requestRender();
  }

  setLayers(layers) {
    this.layers = layers;
    this.requestRender();
  }

  setVisible(visible) {
    this.visible = visible;
    if (visible) this.requestRender();
  }

  requestRender() {
    if (!this.visible || this.rafId !== null) return;
    this.rafId = requestAnimationFrame(() => {
      this.rafId = null;
      this.render();
    });
  }

  _uploadFrame(frameValue) {
    const safeFrame = Math.round(Math.max(0, Math.min(this.frameCount - 1, Number.isFinite(frameValue) ? frameValue : 0)));
    if (this.lastUploadedFrame === safeFrame) {
      return;
    }
    this.lastUploadedFrame = safeFrame;
    const start = safeFrame * this.vertexCount * 3;
    const end = start + this.vertexCount * 3;
    const framePositions = this.positions.subarray(start, end);

    const gl = this.gl;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.positionBuffer);
    gl.bufferSubData(gl.ARRAY_BUFFER, 0, framePositions);
  }

  render() {
    if (!this.visible) return;
    const gl = this.gl;
    const dpr = Math.min(window.devicePixelRatio || 1, 1.5);
    const width = Math.floor(this.canvas.clientWidth * dpr);
    const height = Math.floor(this.canvas.clientHeight * dpr);
    if (this.canvas.width !== width || this.canvas.height !== height) {
      this.canvas.width = width;
      this.canvas.height = height;
    }
    gl.viewport(0, 0, width, height);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

    const projection = makeMat4();
    perspective(projection, Math.PI / 5.2, width / Math.max(height, 1), 0.1, this.meshMeta.bounds.scale * 10);

    const eye = [
      this.camera.target[0] + Math.sin(this.camera.yaw) * Math.cos(this.camera.pitch) * this.camera.distance,
      this.camera.target[1] + Math.sin(this.camera.pitch) * this.camera.distance,
      this.camera.target[2] + Math.cos(this.camera.yaw) * Math.cos(this.camera.pitch) * this.camera.distance,
    ];
    const view = makeMat4();
    lookAt(view, eye, this.camera.target, [0, 1, 0]);

    gl.useProgram(this.program);
    gl.uniformMatrix4fv(this.uProjection, false, projection);
    gl.uniformMatrix4fv(this.uView, false, view);

    gl.bindBuffer(gl.ARRAY_BUFFER, this.positionBuffer);
    gl.enableVertexAttribArray(this.aPosition);
    gl.vertexAttribPointer(this.aPosition, 3, gl.FLOAT, false, 0, 0);

    gl.bindBuffer(gl.ARRAY_BUFFER, this.normalBuffer);
    gl.enableVertexAttribArray(this.aNormal);
    gl.vertexAttribPointer(this.aNormal, 3, gl.FLOAT, false, 0, 0);

    gl.bindBuffer(gl.ARRAY_BUFFER, this.colorBuffer);
    gl.enableVertexAttribArray(this.aColor);
    gl.vertexAttribPointer(this.aColor, 3, gl.UNSIGNED_BYTE, true, 0, 0);

    const layers = this.layers.length ? this.layers : [{ frameIndex: 0, alpha: 1, tintMix: 0, tint: [1, 1, 1] }];
    for (const layer of layers) {
      if ((layer.alpha ?? 1) <= 0) continue;
      const useDepth = layer.depthTest ?? true;
      if (useDepth) {
        gl.enable(gl.DEPTH_TEST);
        gl.depthMask(true);
      } else {
        gl.disable(gl.DEPTH_TEST);
        gl.depthMask(false);
      }
      this._uploadFrame(layer.frameIndex);
      gl.uniform3fv(this.uTint, layer.tint || [1, 1, 1]);
      gl.uniform3fv(this.uOffset, layer.offset || [0, 0, 0]);
      gl.uniform1f(this.uScale, layer.scale ?? 1);
      gl.uniform1f(this.uTintMix, layer.tintMix ?? 0);
      gl.uniform1f(this.uAlpha, layer.alpha ?? 1);
      gl.drawArrays(gl.TRIANGLES, 0, this.vertexCount);
    }
    gl.enable(gl.DEPTH_TEST);
    gl.depthMask(true);
  }
}

function setViewMode(mode) {
  state.viewMode = mode;
  [["single", els.singleStage], ["split", els.splitStage], ["overlay", els.overlayStage]].forEach(([name, element]) => {
    element.classList.toggle("active", name === mode);
  });
  state.renderers.single?.setVisible(mode === "single");
  state.renderers.splitPrimary?.setVisible(mode === "split");
  state.renderers.splitCompare?.setVisible(mode === "split");
  state.renderers.overlay?.setVisible(mode === "overlay");
  els.modeButtons.forEach((button) => {
    button.classList.toggle("active", button.dataset.viewMode === mode);
  });
}

function pushOverlayFrame(track, frameIndex) {
  const history = state.overlayHistory[track];
  if (!history) return;
  const last = history[history.length - 1];
  if (last !== frameIndex) {
    history.push(frameIndex);
    if (history.length > 12) {
      history.shift();
    }
  }
}

function resetOverlayHistory() {
  state.overlayHistory.primary = [];
  state.overlayHistory.compare = [];
}

function buildDifferenceOverlayLayers(primary, compare, comparison) {
  const overlayOffset = (state.meshMeta?.bounds?.scale || 80) * 0.0000035;
  if (!comparison.active) {
    resetOverlayHistory();
    return [{
      frameIndex: primary.frameIndex,
      alpha: 0.42,
      tintMix: 0.75,
      tint: [0.68, 0.7, 0.74],
      offset: [0, 0, 0],
      depthTest: false,
    }];
  }

  pushOverlayFrame("primary", primary.frameIndex);
  pushOverlayFrame("compare", compare.frameIndex);
  const deltaStrength = Math.min(Math.max(comparison.meanAbsAngleDelta / 28, 0), 1);
  const frameStrength = Math.min(Math.abs(comparison.frameDelta || 0) / 18, 1);
  const strength = Math.max(deltaStrength, frameStrength);
  const compareTint = [
    0.95 + 0.05 * strength,
    0.48 - 0.22 * strength,
    0.18 - 0.08 * strength,
  ];
  const primaryTrailFrames = state.overlayHistory.primary.slice(0, -1).slice(-9);
  const compareTrailFrames = state.overlayHistory.compare.slice(0, -1).slice(-9);
  const primaryTrailLayers = primaryTrailFrames.map((frameIndex, index) => {
    const age = primaryTrailFrames.length - index;
    return {
      frameIndex,
      alpha: 0.025 + (index / Math.max(primaryTrailFrames.length, 1)) * 0.055,
      tintMix: 1,
      tint: [0.52, 0.58, 0.68],
      offset: [-overlayOffset * (1 + age * 0.08), 0, 0],
      depthTest: false,
    };
  });
  const compareTrailLayers = compareTrailFrames.map((frameIndex, index) => {
    const age = compareTrailFrames.length - index;
    return {
      frameIndex,
      alpha: 0.04 + (index / Math.max(compareTrailFrames.length, 1)) * 0.1,
      tintMix: 1,
      tint: [0.95, 0.32, 0.12],
      offset: [overlayOffset * (1 + age * 0.16), 0, 0],
      scale: 1 + strength * 0.01 * age,
      depthTest: false,
    };
  });

  return [
    ...primaryTrailLayers,
    {
      frameIndex: primary.frameIndex,
      alpha: 0.34,
      tintMix: 1,
      tint: [0.58, 0.6, 0.64],
      offset: [-overlayOffset, 0, 0],
      depthTest: false,
    },
    ...compareTrailLayers,
    {
      frameIndex: compare.frameIndex,
      alpha: 0.28 + strength * 0.42,
      tintMix: 1,
      tint: compareTint,
      offset: [overlayOffset, 0, 0],
      scale: 1 + strength * 0.035,
      depthTest: false,
    },
  ];
}

function updateUi(runtime) {
  if (!runtime) return;
  const primary = runtime.tracks.primary;
  const compare = runtime.tracks.compare;
  const comparison = runtime.comparison;

  els.angleA.textContent = `${primary.angles.a.toFixed(1)}°`;
  els.angleB.textContent = `${primary.angles.b.toFixed(1)}°`;
  els.angleC.textContent = `${primary.angles.c.toFixed(1)}°`;
  els.deltaA.textContent = `${comparison.angleDelta.a >= 0 ? "+" : ""}${comparison.angleDelta.a.toFixed(1)}°`;
  els.deltaB.textContent = `${comparison.angleDelta.b >= 0 ? "+" : ""}${comparison.angleDelta.b.toFixed(1)}°`;
  els.deltaC.textContent = `${comparison.angleDelta.c >= 0 ? "+" : ""}${comparison.angleDelta.c.toFixed(1)}°`;
  els.deltaMean.textContent = `${comparison.meanAbsAngleDelta.toFixed(1)}°`;
  els.statusLabel.textContent = primary.error ? `Error: ${primary.error}` : primary.status;
  els.sourceLabel.textContent = comparison.active ? "Main + Compare" : primary.sourceLabel;
  els.primarySourceLabel.textContent = primary.sourceLabel;
  els.compareSourceLabel.textContent = compare.sourceLabel;
  els.comparisonSummary.textContent = comparison.summary;
  els.timeLabel.textContent = `${formatTime(primary.elapsedSec)} / ${formatTime(primary.durationSec)}`;
  els.progressFill.style.width = `${(primary.progress || 0) * 100}%`;

  drawWaveform(primary.waveform || [], compare.waveform || []);
  drawPitch(primary.pitchContour || [], compare.pitchContour || []);

  state.renderers.single?.setLayers([{ frameIndex: primary.frameIndex, alpha: 0.5, tintMix: 0, tint: [1, 1, 1] }]);
  state.renderers.splitPrimary?.setLayers([{ frameIndex: primary.frameIndex, alpha: 0.5, tintMix: 0, tint: [1, 1, 1] }]);
  state.renderers.splitCompare?.setLayers([{
    frameIndex: compare.frameIndex,
    alpha: 0.5,
    tintMix: 0,
    tint: [1, 1, 1],
  }]);
  state.renderers.overlay?.setLayers(buildDifferenceOverlayLayers(primary, compare, comparison));
}

async function loadRenderers() {
  const [meta, positions, colors, normals] = await Promise.all([
    fetchJson("/mesh/meta"),
    fetch("/mesh/positions.bin").then((res) => res.arrayBuffer()),
    fetch("/mesh/colors.bin").then((res) => res.arrayBuffer()),
    fetch("/mesh/normals.bin").then((res) => res.arrayBuffer()),
  ]);

  state.meshMeta = meta;
  const neutral = Math.round(meta.sequence.neutralFrame);
  state.renderers.single = new MeshRenderer(els.glCanvas, meta, positions, colors, normals);
  state.renderers.splitPrimary = new MeshRenderer(els.glPrimaryCanvas, meta, positions, colors, normals);
  state.renderers.splitCompare = new MeshRenderer(els.glCompareCanvas, meta, positions, colors, normals);
  state.renderers.overlay = new MeshRenderer(els.glOverlayCanvas, meta, positions, colors, normals);
  Object.values(state.renderers).forEach((renderer) => {
    renderer.setLayers([{ frameIndex: neutral, alpha: 0.5, tintMix: 0, tint: [1, 1, 1] }]);
  });
}

async function refreshState() {
  const runtime = await fetchJson("/api/state");
  state.runtime = runtime;
  updateUi(runtime);
}

function connectEvents() {
  if (state.events) {
    state.events.close();
  }
  state.events = new EventSource("/api/events");
  state.events.onmessage = (event) => {
    const runtime = JSON.parse(event.data);
    state.runtime = runtime;
    updateUi(runtime);
  };
  state.events.onerror = () => {
    els.statusLabel.textContent = "Event stream reconnecting";
  };
}

async function startMic() {
  stopBrowserAudio("primary");
  resetOverlayHistory();
  const device = Number(els.micSelect.value);
  await fetchJson("/api/start-mic", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ track: "primary", device: Number.isFinite(device) ? device : null }),
  });
  await refreshState();
}

async function stopTrack(track = null) {
  stopBrowserAudio(track);
  if (track === null) {
    resetOverlayHistory();
  } else if (state.overlayHistory[track]) {
    state.overlayHistory[track] = [];
  }
  await fetchJson("/api/stop", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ track }),
  });
  await refreshState();
}

async function startTrackFile(track, path, options = {}) {
  if (!path) return;
  if (state.overlayHistory[track]) {
    state.overlayHistory[track] = [];
  }
  await fetchJson("/api/start-file", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ track, path }),
  });
  if (options.playAudio) {
    await playBrowserAudio(track, path);
  }
  await refreshState();
}

async function uploadFile(file, track) {
  const response = await fetchJson("/api/upload-audio", {
    method: "POST",
    headers: { "Content-Type": file.type || "application/octet-stream", "X-Filename": file.name },
    body: file,
  });
  const select = track === "compare" ? els.compareSampleSelect : els.sampleSelect;
  addMediaOption(select, response.path, `Uploaded: ${file.name}`);
  els.statusLabel.textContent = `${file.name} uploaded. Click ${track === "compare" ? "Play Compare" : "Play Sample"} to start.`;
}

function bindControls() {
  els.startMicBtn.addEventListener("click", startMic);
  els.playSampleBtn.addEventListener("click", () => startTrackFile("primary", els.sampleSelect.value, { playAudio: true }));
  els.playCompareSampleBtn.addEventListener("click", () => startTrackFile("compare", els.compareSampleSelect.value, { playAudio: true }));
  els.transportPlayBtn.addEventListener("click", () => startTrackFile("primary", els.sampleSelect.value, { playAudio: true }));
  els.mutePrimaryBtn.addEventListener("click", () => toggleTrackMute("primary"));
  els.muteCompareBtn.addEventListener("click", () => toggleTrackMute("compare"));
  els.muteBothBtn.addEventListener("click", toggleBothMute);
  [els.stopBtn, els.stopBtnTop, els.transportStopBtn].forEach((button) => {
    button.addEventListener("click", () => stopTrack(null));
  });
  els.stopCompareBtn.addEventListener("click", () => stopTrack("compare"));

  els.refreshBtn.addEventListener("click", () => window.location.reload());
  const applyToAllRenderers = (method, ...args) => {
    Object.values(state.renderers).forEach((renderer) => renderer?.[method]?.(...args));
  };
  els.resetViewBtn.addEventListener("click", () => applyToAllRenderers("setPreset", "front"));
  els.frontViewBtn.addEventListener("click", () => applyToAllRenderers("setPreset", "front"));
  els.backViewBtn.addEventListener("click", () => applyToAllRenderers("setPreset", "back"));
  els.sideViewBtn.addEventListener("click", () => applyToAllRenderers("setPreset", "side"));
  els.topViewBtn.addEventListener("click", () => applyToAllRenderers("setPreset", "top"));
  els.zoomInBtn.addEventListener("click", () => applyToAllRenderers("zoomBy", 0.85));
  els.zoomOutBtn.addEventListener("click", () => applyToAllRenderers("zoomBy", 1.15));
  els.modeButtons.forEach((button) => {
    button.addEventListener("click", () => setViewMode(button.dataset.viewMode));
  });
  window.addEventListener("resize", () => {
    Object.values(state.renderers).forEach((renderer) => renderer?.requestRender());
  });

  els.fileInput.addEventListener("change", async (event) => {
    const [file] = event.target.files || [];
    if (file) {
      await uploadFile(file, "primary");
      event.target.value = "";
    }
  });
  els.compareFileInput.addEventListener("change", async (event) => {
    const [file] = event.target.files || [];
    if (file) {
      await uploadFile(file, "compare");
      event.target.value = "";
    }
  });
}

async function init() {
  state.config = await fetchJson("/api/config");
  els.micSelect.innerHTML = state.config.devices
    .map((device) => `<option value="${device.id}">${device.name}</option>`)
    .join("");
  const sampleOptions = state.config.samples
    .map((sample) => `<option value="${sample.path}">${sample.label}</option>`)
    .join("");
  els.sampleSelect.innerHTML = sampleOptions;
  els.compareSampleSelect.innerHTML = `<option value="">Upload a compare file first</option>`;
  Object.entries(state.audioPlayers).forEach(([track, audio]) => {
    audio.addEventListener("ended", () => {
      state.audioActive[track] = false;
      updateMuteButton();
    });
    audio.addEventListener("error", () => {
      state.audioActive[track] = false;
      updateMuteButton();
    });
  });

  await loadRenderers();
  bindControls();
  updateMuteButton();
  setViewMode("single");
  await refreshState();
  connectEvents();
}

init().catch((error) => {
  els.statusLabel.textContent = `Startup failed: ${error.message}`;
});
