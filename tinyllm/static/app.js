import * as THREE from "three";
import { OrbitControls } from "./vendor/OrbitControls.js";
import { BrowserModel } from "./browser-model.js";

const canvas = document.querySelector("#brain");
const space = document.querySelector("#space");
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setClearColor(0x05070c, 0);

const scene = new THREE.Scene();
scene.fog = new THREE.FogExp2(0x05070c, 0.006);
const camera = new THREE.PerspectiveCamera(48, 1, 0.1, 500);
camera.position.set(0, 15, 125);
const controls = new OrbitControls(camera, canvas);
controls.enableDamping = true;
controls.dampingFactor = 0.06;
controls.autoRotate = true;
controls.autoRotateSpeed = 0.16;

const promptColor = new THREE.Color(0x48dcff);
const answerColor = new THREE.Color(0xffb648);
const attentionColor = new THREE.Color(0xd878ff);
const baseColor = new THREE.Color(0x17243a);
let layout;
let points;
let colors;
let colorAttribute;
let socket;
let currentAssistant;
let promptLine;
let answerLine;
let siteConfig = { mode: "live" };
let browserModel;
let browserStopped = false;
const attentionLines = [];
const paths = [];
const activeNodes = new Set();
const conversation = [];

const layerGroup = new THREE.Group();
const layerSpheres = [];
for (let index = 0; index < 8; index += 1) {
  const material = new THREE.MeshBasicMaterial({
    color: new THREE.Color().setHSL(0.69 + index * 0.012, 0.78, 0.52),
    transparent: true,
    opacity: 0.25,
    blending: THREE.AdditiveBlending,
  });
  const sphere = new THREE.Mesh(new THREE.SphereGeometry(0.7, 12, 8), material);
  sphere.position.y = (index - 3.5) * 2.25;
  layerSpheres.push(sphere);
  layerGroup.add(sphere);
}
scene.add(layerGroup);

function resize() {
  const width = space.clientWidth;
  const height = space.clientHeight;
  renderer.setSize(width, height, false);
  camera.aspect = width / height;
  camera.updateProjectionMatrix();
}
window.addEventListener("resize", resize);
resize();

function createPath(color) {
  const geometry = new THREE.BufferGeometry().setFromPoints([]);
  geometry.userData.points = [];
  const material = new THREE.LineBasicMaterial({
    color,
    transparent: true,
    opacity: 0.78,
    blending: THREE.AdditiveBlending,
  });
  const line = new THREE.Line(geometry, material);
  scene.add(line);
  paths.push(line);
  return line;
}

function appendPath(line, tokenId) {
  if (!layout || !line) return;
  line.geometry.userData.points.push(new THREE.Vector3(...layout.positions[tokenId]));
  line.geometry.setFromPoints(line.geometry.userData.points);
}

function lightNode(tokenId, color) {
  if (!colorAttribute) return;
  color.toArray(colors, tokenId * 3);
  colorAttribute.needsUpdate = true;
  activeNodes.add(tokenId);
}

function resetCurrentTrace() {
  if (colorAttribute) {
    for (const tokenId of activeNodes) {
      baseColor.toArray(colors, tokenId * 3);
    }
    colorAttribute.needsUpdate = true;
  }
  activeNodes.clear();
  for (const line of attentionLines.splice(0)) {
    scene.remove(line);
    line.geometry.dispose();
    line.material.dispose();
  }
}

function updateAttention(sourceId, layers) {
  if (!layout) return;
  const source = layout.positions[sourceId];
  const vertices = [];
  const seen = new Set();
  for (const layer of layers) {
    for (const target of layer) {
      const key = `${sourceId}:${target.token_id}`;
      if (seen.has(key)) continue;
      seen.add(key);
      vertices.push(...source, ...layout.positions[target.token_id]);
    }
  }
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.Float32BufferAttribute(vertices, 3));
  const lines = new THREE.LineSegments(
    geometry,
    new THREE.LineBasicMaterial({
      color: attentionColor,
      transparent: true,
      opacity: 0.32,
      blending: THREE.AdditiveBlending,
    }),
  );
  scene.add(lines);
  attentionLines.push(lines);
}

function updateSignals(event) {
  document.querySelector("#token-label").textContent = visibleToken(event.token_text);
  document.querySelector("#confidence").textContent = `${(event.probability * 100).toFixed(2)}%`;
  document.querySelector("#entropy").textContent = `${(event.entropy * 100).toFixed(1)}%`;
  const min = Math.min(...event.layer_activity);
  const max = Math.max(...event.layer_activity);
  const range = Math.max(1e-8, max - min);
  document.querySelector("#layers").replaceChildren(
    ...event.layer_activity.map((activity, index) => {
      const amount = (activity - min) / range;
      const bar = document.createElement("div");
      bar.className = "layer";
      bar.title = `Layer ${index + 1}: ${activity.toFixed(4)}`;
      bar.style.height = `${8 + amount * 47}px`;
      layerSpheres[index].scale.setScalar(0.65 + amount * 1.7);
      layerSpheres[index].material.opacity = 0.22 + amount * 0.62;
      return bar;
    }),
  );
  document.querySelector("#alternatives").replaceChildren(
    ...event.alternatives.map((alternative) => {
      const item = document.createElement("span");
      item.className = "alternative";
      item.textContent = `${visibleToken(alternative.token_text)} ${(alternative.probability * 100).toFixed(1)}%`;
      return item;
    }),
  );
}

function visibleToken(text) {
  if (!text) return "(empty)";
  return text.replaceAll("\n", "\\n").replaceAll("\t", "\\t").replaceAll(" ", "·");
}

function addMessage(role, content) {
  const element = document.createElement("div");
  element.className = `message ${role}`;
  element.textContent = content;
  document.querySelector("#conversation").append(element);
  element.scrollIntoView({ block: "nearest" });
  return element;
}

function setBusy(busy) {
  document.querySelector("#send").disabled = busy;
  document.querySelector("#stop").disabled = !busy;
  document.querySelector("#prompt").disabled = busy;
}

function beginGeneration(userText) {
  conversation.push({ role: "user", content: userText });
  addMessage("user", userText);
  currentAssistant = addMessage("assistant", "");
  for (const path of paths) path.material.opacity = 0.2;
  resetCurrentTrace();
  promptLine = createPath(promptColor);
  answerLine = createPath(answerColor);
  setBusy(true);

  if (siteConfig.mode === "browser") {
    runBrowserGeneration();
    return;
  }

  const backend = siteConfig.backend_url || location.origin;
  const socketUrl = new URL("/ws/visualize", backend);
  socketUrl.protocol = socketUrl.protocol === "https:" ? "wss:" : "ws:";
  socket = new WebSocket(socketUrl);
  socket.addEventListener("open", () => {
    socket.send(JSON.stringify({
      messages: conversation,
      temperature: Number(document.querySelector("#temperature").value),
      top_p: Number(document.querySelector("#top-p").value),
      max_tokens: Number(document.querySelector("#max-tokens").value),
      retrieval: document.querySelector("#retrieval").checked,
    }));
  });
  socket.addEventListener("message", ({ data }) => handleEvent(JSON.parse(data)));
  socket.addEventListener("close", () => setBusy(false));
  socket.addEventListener("error", () => {
    currentAssistant.textContent = "Visualizer connection failed.";
    setBusy(false);
  });
}

async function runBrowserGeneration() {
  browserStopped = false;
  try {
    const options = {
      temperature: Number(document.querySelector("#temperature").value),
      topP: Number(document.querySelector("#top-p").value),
      maxTokens: Number(document.querySelector("#max-tokens").value),
    };
    for await (const event of browserModel.generate(conversation, options, () => browserStopped)) {
      handleEvent(event);
    }
  } catch (error) {
    handleEvent({ type: "error", message: error.message });
  }
}

function handleEvent(event) {
  if (event.type === "ready") {
    document.querySelector("#checkpoint").textContent = `${event.checkpoint.phase} checkpoint / step ${event.checkpoint.step}`;
  } else if (event.type === "prompt_token") {
    appendPath(promptLine, event.token_id);
    lightNode(event.token_id, promptColor);
  } else if (event.type === "token") {
    appendPath(answerLine, event.token_id);
    lightNode(event.token_id, answerColor);
    updateAttention(event.token_id, event.attention_targets);
    updateSignals(event);
    currentAssistant.textContent = event.text;
    currentAssistant.scrollIntoView({ block: "nearest" });
  } else if (event.type === "done") {
    currentAssistant.textContent = event.text || "(no output)";
    conversation.push({ role: "assistant", content: event.text || "(no output)" });
    setBusy(false);
  } else if (event.type === "error") {
    currentAssistant.textContent = `Error: ${event.message}`;
    setBusy(false);
  }
}

document.querySelector("#chat-form").addEventListener("submit", (event) => {
  event.preventDefault();
  const input = document.querySelector("#prompt");
  const text = input.value.trim();
  if (!text) return;
  input.value = "";
  beginGeneration(text);
});

document.querySelector("#stop").addEventListener("click", () => {
  browserStopped = true;
  if (socket) socket.close();
  if (siteConfig.mode === "browser") {
    document.querySelector("#stop").disabled = true;
    return;
  }
  if (currentAssistant && currentAssistant.textContent) {
    conversation.push({ role: "assistant", content: currentAssistant.textContent });
  }
  setBusy(false);
});

document.querySelector("#clear").addEventListener("click", () => {
  for (const path of paths.splice(0)) {
    scene.remove(path);
    path.geometry.dispose();
    path.material.dispose();
  }
  resetCurrentTrace();
  if (!layout || !colorAttribute) return;
  for (let tokenId = 0; tokenId < layout.count; tokenId += 1) {
    baseColor.toArray(colors, tokenId * 3);
  }
  colorAttribute.needsUpdate = true;
});

for (const [inputId, outputId] of [["temperature", "temperature-value"], ["top-p", "top-p-value"]]) {
  document.querySelector(`#${inputId}`).addEventListener("input", (event) => {
    document.querySelector(`#${outputId}`).textContent = event.target.value;
  });
}

const raycaster = new THREE.Raycaster();
raycaster.params.Points.threshold = 1.1;
const mouse = new THREE.Vector2();
const tooltip = document.querySelector("#tooltip");
canvas.addEventListener("pointermove", (event) => {
  if (!points) return;
  const rect = canvas.getBoundingClientRect();
  mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
  mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
  raycaster.setFromCamera(mouse, camera);
  const hit = raycaster.intersectObject(points)[0];
  if (!hit) {
    tooltip.style.display = "none";
    return;
  }
  tooltip.textContent = visibleToken(layout.labels[hit.index]);
  tooltip.style.display = "block";
  tooltip.style.left = `${event.clientX - rect.left + 12}px`;
  tooltip.style.top = `${event.clientY - rect.top + 12}px`;
});
canvas.addEventListener("pointerleave", () => { tooltip.style.display = "none"; });

function animate() {
  requestAnimationFrame(animate);
  controls.update();
  renderer.render(scene, camera);
}
requestAnimationFrame(animate);

async function loadLayout() {
  const layoutUrl = siteConfig.mode === "browser"
    ? siteConfig.layout_url
    : new URL("/visualizer/layout", siteConfig.backend_url || location.origin);
  const response = await fetch(layoutUrl);
  if (!response.ok) throw new Error(await response.text());
  layout = await response.json();
  const positions = new Float32Array(layout.positions.flat());
  colors = new Float32Array(layout.count * 3);
  for (let tokenId = 0; tokenId < layout.count; tokenId += 1) {
    baseColor.toArray(colors, tokenId * 3);
  }
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  colorAttribute = new THREE.BufferAttribute(colors, 3);
  geometry.setAttribute("color", colorAttribute);
  const material = new THREE.PointsMaterial({
    size: 0.42,
    vertexColors: true,
    transparent: true,
    opacity: 0.7,
    depthWrite: false,
    blending: THREE.AdditiveBlending,
  });
  points = new THREE.Points(geometry, material);
  scene.add(points);
  document.querySelector("#checkpoint").textContent = `${layout.checkpoint.phase} checkpoint / step ${layout.checkpoint.step}`;
  document.querySelector("#loading")?.remove();
}

function modeNote(text) {
  const note = document.querySelector("#mode-note");
  note.style.display = "block";
  note.textContent = text;
}

async function start() {
  try {
    const response = await fetch("./site-config.json");
    if (response.ok) siteConfig = await response.json();
  } catch {
    siteConfig = { mode: "live" };
  }
  if (siteConfig.mode === "browser") {
    setBusy(true);
    document.querySelector("#retrieval").disabled = true;
    modeNote("Loading the trained model in your browser. The first visit downloads about 110 MB.");
    const [, model] = await Promise.all([
      loadLayout(),
      BrowserModel.load(siteConfig, modeNote),
    ]);
    browserModel = model;
    document.querySelector("#checkpoint").textContent =
      `${model.config.checkpoint.phase} checkpoint / step ${model.config.checkpoint.step}`;
    modeNote(`${model.config.execution_provider.toUpperCase()} browser inference ready. Wikipedia retrieval requires the local backend.`);
    setBusy(false);
    return;
  }
  await loadLayout();
}

start().catch((error) => {
  const loading = document.querySelector("#loading");
  if (loading) loading.textContent = `Could not start visualizer: ${error.message}`;
  else modeNote(`Could not start visualizer: ${error.message}`);
  setBusy(false);
});
