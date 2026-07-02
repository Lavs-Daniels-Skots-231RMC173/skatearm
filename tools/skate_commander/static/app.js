/* Skate Commander — frontend.
 * Live mode: talks to the FastAPI server over WebSocket.
 * Preview mode: if window.PREVIEW_DATA is defined (model + recorded frames),
 * runs a playback loop with no backend; meshes degrade to a stick figure. */

import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { STLLoader } from "three/addons/loaders/STLLoader.js";
import { TransformControls } from "three/addons/controls/TransformControls.js";

const PREVIEW = typeof window.PREVIEW_DATA !== "undefined";
const $ = (id) => document.getElementById(id);
const CLEAN = !PREVIEW && new URLSearchParams(location.search).has("clean");

const GROUPS = {
  left:  { label: "LEFT ARM",  idx: [8, 9, 10, 11, 12, 13, 14, 15] },
  right: { label: "RIGHT ARM", idx: [16, 17, 18, 19, 20, 21, 22, 23] },
  head:  { label: "HEAD",      idx: [24, 25] },
  legs:  { label: "LEGS",      idx: [0, 1, 2, 3, 4, 5, 6, 7] },
};
const ROLE = { 9: "abduction", 11: "elbow", 15: "gripper",
               17: "abduction", 19: "elbow", 23: "gripper" };

let model, ws = null, state = null, curGroup = "left";
let jointGroups = {};       // protocol index -> THREE.Group + meta
let limits = {};            // idx -> [lo, hi]
let rows = {};              // idx -> row elements
let eeObjs = {};            // "left"/"right" -> THREE.Object3D (wrist link)
const markers = {};         // "left"/"right" -> gizmo sphere
let draggingArm = null;
let measureOn = false;          // I2: measure tool active (click two points)
let traceOn = true;
let wsOnline = false;       // browser <-> server WebSocket up?
let lastMsg = 0;            // perf-time (ms) of the last telemetry frame
let reconnectMs = 1000;     // current reconnect backoff

// ---------------------------------------------------------------- three.js
const scene = new THREE.Scene();          // transparent: CSS gradient shows
const FLOOR_Z = -0.95;                    // wheel contact plane of skt_v3
const camera = new THREE.PerspectiveCamera(50, 1, 0.01, 50);
camera.up.set(0, 0, 1);
camera.position.set(1.65, -1.65, 0.46);
const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
$("viewport").appendChild(renderer.domElement);
const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(0, 0, -0.15);         // frame the whole robot
if (CLEAN) {
    const _cq = new URLSearchParams(location.search).get("cam");
    if (_cq) { const n = _cq.split(",").map(Number); camera.position.set(n[0], n[1], n[2]); controls.target.set(n[3] || 0, n[4] || 0, n[5] || 0); }
    else { camera.position.set(-0.36, 1.5, 0.64); controls.target.set(0, 0, 0.2); }
  }

scene.add(new THREE.HemisphereLight(0xffffff, 0x223344, 1.1));
const dir = new THREE.DirectionalLight(0xffffff, 1.4);
dir.position.set(2, -3, 4);
scene.add(dir);
const grid = new THREE.GridHelper(4, 24, 0x2a3240, 0x1b212b);
grid.rotation.x = Math.PI / 2;
grid.position.z = FLOOR_Z;                 // floor under the wheels
if (!CLEAN) scene.add(grid);
const axes = new THREE.AxesHelper(0.22);   // world frame triad on the floor
if (axes.setColors) axes.setColors(new THREE.Color(0xFF6981), new THREE.Color(0x3FB950), new THREE.Color(0x7A95FF)); // X/Y/Z tokens
axes.material.transparent = true;
axes.material.opacity = 0.55;
axes.position.z = FLOOR_Z;
if (!CLEAN) scene.add(axes);

function resize() {
  const vp = $("viewport");
  let w = vp.clientWidth, h = vp.clientHeight;
  if (vp.classList.contains("split")) {            // ego+exo: the twin shares width with the work-cam pane
    const cam = $("cam-pip");
    if (cam && cam.classList.contains("show")) w = Math.max(120, w - cam.offsetWidth);
  }
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
  renderer.setSize(w, h);
}
window.addEventListener("resize", resize);

function rpyToQuat(rpy) {  // URDF fixed-axis rpy -> quaternion
  return new THREE.Quaternion().setFromEuler(
    new THREE.Euler(rpy[0], rpy[1], rpy[2], "ZYX"));
}

let robotRoots = [];
function buildRobot() {
  const linkGroup = {};
  const g = (name) => linkGroup[name] ||
    (linkGroup[name] = Object.assign(new THREE.Group(), { name }));
  const children = new Set();
  for (const j of model.joints) {
    const jg = new THREE.Group();
    jg.position.set(...j.xyz);
    const baseQuat = rpyToQuat(j.rpy);
    jg.quaternion.copy(baseQuat);
    jg.add(g(j.child));
    g(j.parent).add(jg);
    children.add(j.child);
    if (j.index !== null && j.index !== undefined) {
      jointGroups[j.index] = { grp: jg, baseQuat,
        axis: new THREE.Vector3(...j.axis).normalize() };
      if (j.lower !== null) limits[j.index] = [j.lower, j.upper];
    }
  }
  robotRoots = [];
  for (const name of Object.keys(model.links))
    if (!children.has(name)) { const root = g(name); scene.add(root); robotRoots.push(root); }   // root link(s)

  // visuals
  const loader = new STLLoader();
  let meshFails = 0;
  for (const [name, link] of Object.entries(model.links)) {
    let fallback = !PREVIEW ? 0 : 1;
    for (const v of link.visuals) {
      if (PREVIEW) continue;
      loader.load(`/meshes/${v.mesh}`, (geo) => {
        const mat = new THREE.MeshStandardMaterial({
          color: v.color ? new THREE.Color(...v.color.slice(0, 3)) : 0x888888,
          metalness: 0.25, roughness: 0.65 });
        const mesh = new THREE.Mesh(geo, mat);
        mesh.position.set(...v.xyz);
        mesh.quaternion.copy(rpyToQuat(v.rpy));
        mesh.scale.set(...v.scale);
        g(name).add(mesh);
      }, undefined, () => {
        meshFails++;
        console.error(`mesh failed: /meshes/${v.mesh}`);
        const ov = $("overlay");
        ov.style.color = "var(--warn)";
        ov.textContent = `⚠ ${meshFails} mesh(es) failed to load — ` +
          "check --model-dir points at skt_v3 with skt_v3_meshes/ " +
          "(stick figure shown instead)";
        if (!fallback++) stickFigure(name, g);
      });
    }
    if (PREVIEW) stickFigure(name, g);
  }

  // wrist (end-effector) objects = child links of the a6 joints
  for (const [arm, lastIdx] of [["left", 14], ["right", 22]]) {
    const j = model.joints.find((j) => j.index === lastIdx);
    if (j) eeObjs[arm] = g(j.child);
  }
  setupGizmo();
  setupTraces();
  resize();
}

// ---- drag-gizmo (cartesian IK teleop) --------------------------------------
let tc = null;
function setupGizmo() {
  if (PREVIEW) return;
  const colors = { left: 0x58a6ff, right: 0xffa657 };
  for (const arm of ["left", "right"]) {
    const m = new THREE.Mesh(
      new THREE.SphereGeometry(0.028, 16, 16),
      new THREE.MeshBasicMaterial({ color: colors[arm], transparent: true,
                                    opacity: 0.65, depthTest: false }));
    m.userData.arm = arm;
    scene.add(m);
    markers[arm] = m;
  }
  tc = new TransformControls(camera, renderer.domElement);
  tc.setMode("translate");
  tc.setSize(0.55);
  tc.setSpace("world");
  scene.add(tc);
  tc.addEventListener("dragging-changed", (e) => {
    controls.enabled = !e.value;
      { const _gh = document.getElementById("giz-hud"); if (_gh) _gh.style.display = e.value ? "block" : "none"; }
    if (e.value) {
      draggingArm = tc.object ? tc.object.userData.arm : null;
    } else {
      if (draggingArm) send({ type: "ik_clear", arm: draggingArm });
      draggingArm = null;
    }
  });
  let lastSend = 0;
  tc.addEventListener("objectChange", () => {
    { const _p2 = tc.object && tc.object.position, _gh = document.getElementById("giz-hud");
      if (_p2 && _gh) { _gh.style.display = "block"; _gh.textContent =
        (tc.getMode ? tc.getMode().toUpperCase() : "MOVE") + "   " +
        (_p2.x * 1000).toFixed(0) + " · " + (_p2.y * 1000).toFixed(0) + " · " + (_p2.z * 1000).toFixed(0) + " mm"; } }
    if (!draggingArm || !state || !state.live) return;
    const now = performance.now();
    if (now - lastSend < 50) return;          // 20 Hz target stream
    lastSend = now;
    const p = tc.object.position;
    const msg = { type: "ik_target", arm: draggingArm, pos: [p.x, p.y, p.z] };
    if (tc.getMode && tc.getMode() === "rotate") {   // 6-DoF: hold pos, command orientation
      const q = tc.object.quaternion;
      msg.rot = [q.x, q.y, q.z, q.w];
    }
    send(msg);
  });
  // click a sphere to attach the gizmo; click elsewhere to detach
  const ray = new THREE.Raycaster();
  renderer.domElement.addEventListener("pointerdown", (e) => {
    if (tc.dragging || overlaysOn() || window.__obstacleActive || window.__markerActive) return;   // no IK gizmo while overlays declutter / placing obstacles
    const r = renderer.domElement.getBoundingClientRect();
    const ndc = new THREE.Vector2(
      ((e.clientX - r.left) / r.width) * 2 - 1,
      -((e.clientY - r.top) / r.height) * 2 + 1);
    ray.setFromCamera(ndc, camera);
    const hit = ray.intersectObjects(Object.values(markers), false)[0];
    if (hit) tc.attach(hit.object);
  });
}

// ---- TCP traces -------------------------------------------------------------
const traces = {};
function setupTraces() {
  const colors = { left: 0x58a6ff, right: 0xffa657 };
  for (const arm of ["left", "right"]) {
    const N = 800;
    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position",
      new THREE.BufferAttribute(new Float32Array(N * 3), 3));
    geo.setDrawRange(0, 0);
    const line = new THREE.Line(geo,
      new THREE.LineBasicMaterial({ color: colors[arm], transparent: true,
                                    opacity: 0.8 }));
    line.frustumCulled = false;
    scene.add(line);
    traces[arm] = { line, n: 0, N, last: new THREE.Vector3(1e9, 0, 0) };
  }
}

function clearTraces() {
  for (const t of Object.values(traces)) {
    t.n = 0;
    t.line.geometry.setDrawRange(0, 0);
    t.last.set(1e9, 0, 0);
  }
}

function tcpWorld(arm, out) {
  // world position of the TCP: wrist link origin + active tool offset
  const t = state && state.tools && state.tools[arm];
  const o = t ? t.offset_mm : null;
  out.set(o ? o[0] / 1000 : 0, o ? o[1] / 1000 : 0, o ? o[2] / 1000 : 0);
  return eeObjs[arm].localToWorld(out);
}

const _wp = new THREE.Vector3();
function updateEE() {
  for (const arm of ["left", "right"]) {
    if (!eeObjs[arm]) continue;
    tcpWorld(arm, _wp);
    if (markers[arm] && draggingArm !== arm) {
      markers[arm].position.copy(_wp);
      eeObjs[arm].getWorldQuaternion(markers[arm].quaternion);   // keep rotate-ring aligned to the wrist
    }
    const t = traces[arm];
    if (t && traceOn && _wp.distanceTo(t.last) > 0.004) {
      const a = t.line.geometry.attributes.position;
      if (t.n < t.N) {
        a.setXYZ(t.n++, _wp.x, _wp.y, _wp.z);
      } else {                                   // ring: shift left
        a.array.copyWithin(0, 3);
        a.setXYZ(t.N - 1, _wp.x, _wp.y, _wp.z);
      }
      a.needsUpdate = true;
      t.line.geometry.setDrawRange(0, t.n);
      t.last.copy(_wp);
    }
  }
}

function stickFigure(name, g) {   // mesh-less degradation (preview mode)
  const mat = new THREE.MeshStandardMaterial({ color: 0x58a6ff,
    metalness: 0.1, roughness: 0.6 });
  g(name).add(new THREE.Mesh(new THREE.SphereGeometry(0.022, 12, 12), mat));
  for (const j of model.joints.filter((j) => j.parent === name)) {
    const v = new THREE.Vector3(...j.xyz);
    if (v.length() < 1e-6) continue;
    const cyl = new THREE.Mesh(
      new THREE.CylinderGeometry(0.012, 0.012, v.length(), 8), mat);
    cyl.position.copy(v.clone().multiplyScalar(0.5));
    cyl.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0),
                                      v.clone().normalize());
    g(name).add(cyl);
  }
}

function setAngles(q) {
  if (!q) return;
  for (const [idx, j] of Object.entries(jointGroups)) {
    const quat = new THREE.Quaternion()
      .setFromAxisAngle(j.axis, q[idx] || 0);
    j.grp.quaternion.copy(j.baseQuat).multiply(quat);
  }
}

(function animate() {
  requestAnimationFrame(animate);
  controls.update();
  updateEE();
  renderer.render(scene, camera);
  if (window.__statsTick) window.__statsTick();
})();

// ---------------------------------------------------------------- panel
function send(obj) {
  if (PREVIEW) { pvHandle(obj); return; }      // interactive preview: drive locally, no socket
  if (ws && ws.readyState === 1) ws.send(JSON.stringify(obj));
}

let seqLoopWanted = false;
let lastSeqSig = "";

function buildSeqPanel() {
  const wrap = $("joints");
  rows = {};
  cartEls = null;
  wrap.innerHTML = `
    <div class="panel-head"><span>SEQUENCE</span>
      <small>record poses · play them back</small></div>
    <div class="seq-controls">
      <button id="sq-add" title="Record the current pose">● ADD POSE</button>
      <button id="sq-play">▶ PLAY</button>
      <button id="sq-loop">LOOP: OFF</button>
      <button id="sq-stop">■ STOP</button>
    </div>
    <div class="seq-controls">
      <button id="sq-save">SAVE…</button>
      <select id="sq-files"></select>
      <button id="sq-load">LOAD</button>
      <button id="sq-clear">CLEAR ALL</button>
    </div>
    <div id="seq-timeline" class="seq-tl"></div>
    <div id="seq-list"></div>
    <div class="seq-hint">Jog or drag the robot into a pose, press
    <b>● ADD POSE</b>, repeat. <b>▶ PLAY</b> glides through the list
    (any manual input or E-STOP interrupts playback). Sequences are saved
    on the server in <code>sequences/</code>.</div>`;
  $("sq-add").onclick = () => send({ type: "wp_add" });
  $("sq-play").onclick = () => send({ type: "wp_play", loop: seqLoopWanted });
  $("sq-stop").onclick = () => send({ type: "wp_stop" });
  $("sq-loop").onclick = () => {
    seqLoopWanted = !seqLoopWanted;
    $("sq-loop").textContent = `LOOP: ${seqLoopWanted ? "ON" : "OFF"}`;
  };
  $("sq-clear").onclick = () => send({ type: "wp_clear" });
  $("sq-save").onclick = () => {
    const name = prompt("Sequence name (letters/digits/_-):");
    if (name) { send({ type: "wp_save", name }); setTimeout(refreshSeqFiles, 400); }
  };
  $("sq-load").onclick = () => {
    const sel = $("sq-files");
    if (sel.value) send({ type: "wp_load", name: sel.value });
  };
  refreshSeqFiles();
  lastSeqSig = "";
}

async function refreshSeqFiles() {
  if (PREVIEW) return;
  try {
    const names = await (await fetch("/api/sequences")).json();
    const sel = $("sq-files");
    if (sel) sel.innerHTML =
      names.map((n) => `<option value="${n}">${n}</option>`).join("");
  } catch (e) { /* server down; banner already shows it */ }
}

function updateSeqPanel() {
  const seq = state && state.seq;
  if (!seq) return;
  const sig = JSON.stringify([seq.names, seq.idx, seq.playing, seq.active]);
  if (sig === lastSeqSig) return;
  lastSeqSig = sig;
  const play = $("sq-play");
  if (play) play.className = seq.playing ? "playing" : "";
  const list = $("seq-list");
  if (!list) return;
  list.innerHTML = "";
  if (!seq.names.length) {
    list.innerHTML = `<div class="seq-empty">No poses recorded yet.<br>
      Jog or drag the robot into a pose, then press <b>● ADD POSE</b>.</div>`;
    return;
  }
  seq.names.forEach((nm, i) => {
    const row = document.createElement("div");
    row.className = "seqrow" + (seq.active && seq.idx === i ? " active" : "");
    row.innerHTML = `<span class="nm">${i + 1}. ${nm}</span>
      <button data-a="goto" title="Glide to this pose">▶</button>
      <button data-a="del" title="Delete">&#10005;</button>`;
    row.querySelector('[data-a="goto"]').onclick =
      () => previewServer("waypoint", i, "Waypoint " + (i + 1) + " — glide to this pose",
        () => send({ type: "wp_goto", idx: i }));
    row.querySelector('[data-a="del"]').onclick =
      () => send({ type: "wp_delete", idx: i });
    list.appendChild(row);
  });
}

// ---- program tab (python over the bridge) -----------------------------------
const DEMO_PROGRAM = `# Skate program — drives the SAME safe bridge as the panel.
# rbt.movej(joint, deg)   joint = "L4" (left J4) / "R2" / "H1" / index
# rbt.movel(arm, dx=, dy=, dz=)   nudge the TCP in mm, world axes
# rbt.home() · rbt.gripper(arm, deg) · rbt.waypoint(1) · rbt.wait(s)
# rbt.tcp(arm) · rbt.q() · rbt.status() · print(...)
rbt.home()
rbt.movej("L4", 60)              # left elbow up
for d in (40, 80, 40):           # wave the right forearm
    rbt.movej("R4", d)
rbt.movel("right", dz=60)        # square with the right TCP
rbt.movel("right", dy=-80)
rbt.movel("right", dz=-60)
rbt.movel("right", dy=80)
print("tcp right:", rbt.tcp("right"), "mm")
rbt.home()
`;

const EXAMPLES = {
  "demo": DEMO_PROGRAM,
  "wave right arm": `rbt.home()
for d in (30, 80, 30, 80, 30):
    rbt.movej("R4", d)
rbt.home()
`,
  "raise both arms": `# one pose moves several joints together (no mirror needed)
rbt.home()
rbt.pose({"L2": 35, "R2": 35})     # shoulders out
rbt.pose({"L4": 80, "R4": 80})     # elbows up
rbt.wait(0.5)
rbt.home()
`,
  "pick & place (right)": `# absolute world points in mm — same IK + guard as the gizmo
rbt.home()
rbt.gripper("right", 30)             # open
rbt.moveto("right", 130, 350, 35)    # above the part
rbt.moveto("right", 130, 350, -43)   # down to it
rbt.gripper("right", 0)              # close
rbt.moveto("right", 130, 350, 75)    # lift
rbt.moveto("right", 60, 320, 75)     # over the bin
rbt.gripper("right", 30)             # release
rbt.home()
`,
  "gripper cycle": `for _ in range(3):
    rbt.gripper("right", 30)
    rbt.wait(0.4)
    rbt.gripper("right", 0)
    rbt.wait(0.4)
`,
  "control flow (loop · if · wait)": `# loop, branch on robot state, dwell
rbt.home()
for i in range(3):                       # repeat N times
    rbt.movej("R4", 70)
    if rbt.blocked() or rbt.contact():   # react to the guard / a touch
        print("stopped early on pass", i)
        break
    rbt.wait(0.3)                        # dwell
    rbt.movej("R4", 20)
rbt.home()
`,
};

// [label shown, text inserted after "rbt.", first arg to auto-select]
const RBT_API = [
  ["movej(joint, deg)", 'movej("L4", 0)', '"L4"'],
  ["pose({joint: deg, ...})", 'pose({"L4": 0})', '"L4"'],
  ["movel(arm, dx=, dy=, dz=)", 'movel("right", dz=0)', '"right"'],
  ["moveto(arm, x, y, z)", 'moveto("right", 0, 0, 0)', '"right"'],
  ["home()", 'home()', null],
  ["gripper(arm, deg)", 'gripper("right", 0)', '"right"'],
  ["waypoint(i_or_name)", 'waypoint(1)', '1'],
  ["wait(seconds)", 'wait(1.0)', '1.0'],
  ["tcp(arm)", 'tcp("right")', '"right"'],
  ["q()", 'q()', null],
  ["status()", 'status()', null],
  ["ok()  — safe to keep going?", 'ok()', null],
  ["blocked()  — guard blocking?", 'blocked()', null],
  ["contact()  — reflex tripped?", 'contact()', null],
  ["near(arm, x, y, z)", 'near("right", 0, 0, 0)', '"right"'],
];

let progCode = DEMO_PROGRAM;
let progSig = "";

let acHide = () => {};        // closes the autocomplete dropdown (set on build)
let progLastErr = null;       // last error log line we highlighted
let progLastStep = null;      // last paused step line we highlighted

function caretCoords(ta) {     // pixel position of the caret (mirror-div trick)
  const div = document.createElement("div");
  const cs = getComputedStyle(ta);
  for (const p of ["fontFamily", "fontSize", "fontWeight", "lineHeight",
      "letterSpacing", "paddingTop", "paddingLeft", "paddingRight",
      "paddingBottom", "borderWidth", "boxSizing", "tabSize"]) div.style[p] = cs[p];
  div.style.position = "absolute"; div.style.visibility = "hidden";
  div.style.whiteSpace = "pre-wrap"; div.style.wordWrap = "break-word";
  div.style.width = ta.clientWidth + "px";
  div.textContent = ta.value.slice(0, ta.selectionStart);
  const span = document.createElement("span");
  span.textContent = "."; div.appendChild(span);
  document.body.appendChild(div);
  const x = span.offsetLeft, y = span.offsetTop;
  document.body.removeChild(div);
  const r = ta.getBoundingClientRect();
  return { left: r.left + x - ta.scrollLeft, top: r.top + y - ta.scrollTop };
}

function setupProgAutocomplete(ta) {
  let items = [], sel = 0, open = false;
  const box = document.createElement("div");
  box.id = "pg-ac"; box.style.display = "none";
  document.body.appendChild(box);
  const close = () => { open = false; box.style.display = "none"; };
  acHide = close;
  const tokenAt = () => {
    const m = ta.value.slice(0, ta.selectionStart).match(/rbt\.(\w*)$/);
    return m ? m[1] : null;
  };
  const render = () => {
    box.innerHTML = "";
    items.forEach((it, i) => {
      const d = document.createElement("div");
      d.className = "ac-item" + (i === sel ? " sel" : "");
      d.textContent = it[0];
      d.onmousedown = (e) => { e.preventDefault(); accept(i); };
      box.appendChild(d);
    });
  };
  const show = (prefix) => {
    items = RBT_API.filter((a) => a[1].startsWith(prefix));
    if (!items.length) { close(); return; }
    sel = 0; render();
    const c = caretCoords(ta);
    box.style.left = c.left + "px"; box.style.top = (c.top + 18) + "px";
    box.style.display = "block"; open = true;
  };
  const accept = (i) => {
    const it = items[i]; if (!it) { close(); return; }
    const start = ta.selectionStart;
    const m = ta.value.slice(0, start).match(/rbt\.(\w*)$/);
    const from = start - (m ? m[1].length : 0);
    ta.value = ta.value.slice(0, from) + it[1] + ta.value.slice(start);
    progCode = ta.value;
    const rel = it[2] ? it[1].indexOf(it[2]) : -1;
    if (rel >= 0) ta.setSelectionRange(from + rel, from + rel + it[2].length);
    else { const c = from + it[1].length; ta.setSelectionRange(c, c); }
    ta.focus(); close();
  };
  ta.addEventListener("input", () => {
    const t = tokenAt(); if (t !== null) show(t); else close();
  });
  ta.addEventListener("keydown", (e) => {
    if (!open) {
      if ((e.ctrlKey || e.metaKey) && e.key === " ") {
        const t = tokenAt(); if (t !== null) { e.preventDefault(); show(t); }
      }
      return;
    }
    if (e.key === "ArrowDown") { e.preventDefault(); sel = (sel + 1) % items.length; render(); }
    else if (e.key === "ArrowUp") { e.preventDefault(); sel = (sel - 1 + items.length) % items.length; render(); }
    else if (e.key === "Enter" || e.key === "Tab") { e.preventDefault(); accept(sel); }
    else if (e.key === "Escape") { e.preventDefault(); close(); }
  });
  ta.addEventListener("blur", () => setTimeout(close, 120));
  ta.addEventListener("scroll", () => { if (open) close(); });
}

function highlightProgLine(ta, lineNo, focus) {
  if (!ta || !lineNo || lineNo < 1) return;
  const lines = ta.value.split("\n");
  if (lineNo > lines.length) return;
  let start = 0;
  for (let i = 0; i < lineNo - 1; i++) start += lines[i].length + 1;
  try {
    ta.setSelectionRange(start, start + lines[lineNo - 1].length);
    if (focus) ta.focus();
  } catch (_) {}
  const lh = parseFloat(getComputedStyle(ta).lineHeight) || 16;
  ta.scrollTop = Math.max(0, (lineNo - 3) * lh);
}

function buildProgPanel() {
  const wrap = $("joints");
  rows = {};
  cartEls = null;
  acHide(); document.getElementById("pg-ac")?.remove();
  wrap.innerHTML = `
    <div class="panel-head"><span>PROGRAM</span>
      <small>python · every move goes through the safe bridge</small></div>
    <div class="nl-row">
      <input id="pg-nl" type="text" autocomplete="off" spellcheck="false"
             placeholder="Describe a task — &ldquo;raise both arms, then home&rdquo;">
      <button id="pg-gen" title="Generate an rbt program from your description">✦ GEN</button>
    </div>
    <div class="prog-controls">
      <button id="pg-run" title="Run the program (releases a paused one)">▶ RUN</button>
      <button id="pg-step" title="Click to Step — execute exactly one motion command">⏭ STEP</button>
      <button id="pg-stop">■ STOP</button>
      <button id="pg-rec" title="Teach-in: move the robot (sliders / gizmo / cartesian), every settled pose becomes a line of code">● REC</button>
      <span id="pg-state" class="prog-state">idle</span>
    </div>
    <div class="prog-snips" title="Insert a control-flow block at the cursor">
      <span class="ps-lbl">+ flow</span>
      <button id="ps-repeat" title="repeat N times (for loop)">repeat</button>
      <button id="ps-while" title="loop while a condition holds">while</button>
      <button id="ps-if" title="branch on robot state">if</button>
      <button id="ps-wait" title="dwell for N seconds">wait</button>
    </div>
    <textarea id="pg-code" spellcheck="false"></textarea>
    <div class="prog-controls">
      <button id="pg-save">SAVE…</button>
      <select id="pg-files"></select>
      <button id="pg-load">LOAD</button>
      <select id="pg-examples" title="Load an example program"></select>
    </div>
    <pre id="pg-log"></pre>`;
  const ta = $("pg-code");
  ta.value = progCode;
  ta.oninput = () => (progCode = ta.value);
  if (PREVIEW) {
    for (const id of ["pg-run", "pg-step", "pg-stop", "pg-rec", "pg-save",
                      "pg-load", "pg-examples", "pg-nl", "pg-gen",
                      "ps-repeat", "ps-while", "ps-if", "ps-wait"]) {
      $(id).disabled = true;
      $(id).title = "preview is a recording — run the local server";
    }
    ta.readOnly = true;
    $("pg-log").textContent = "> programs need the local server";
  } else {
    $("pg-run").onclick = () => send({ type: "prog_run", code: progCode });
    $("pg-step").onclick = () => send({ type: "prog_step", code: progCode });
    $("pg-stop").onclick = () => send({ type: "prog_stop" });
    {                                              // control-flow snippet bar -> insert a skeleton at the caret
      const taSn = $("pg-code");
      const SNIPS = {
        "ps-repeat": "for i in range(3):\n    ",
        "ps-while": "while rbt.ok():\n    ",
        "ps-if": "if rbt.blocked():\n    rbt.home()\n",
        "ps-wait": "rbt.wait(1.0)\n",
      };
      const insertSnip = (snip) => {
        const s = taSn.selectionStart, e = taSn.selectionEnd;
        const lineStart = taSn.value.lastIndexOf("\n", s - 1) + 1;
        const pre = taSn.value.slice(lineStart, s);
        const indent = (pre.match(/^\s*/) || [""])[0];
        const onFresh = pre.trim() === "";
        let text = snip.split("\n").map((ln, i) => (i === 0 || ln === "" ? ln : indent + ln)).join("\n");
        if (!onFresh) text = "\n" + indent + text;          // start the block on its own line
        taSn.value = taSn.value.slice(0, s) + text + taSn.value.slice(e);
        progCode = taSn.value;
        const caret = s + text.length;
        taSn.focus(); taSn.setSelectionRange(caret, caret);
      };
      for (const id of Object.keys(SNIPS)) { const b = $(id); if (b) b.onclick = () => insertSnip(SNIPS[id]); }
    }
    {
      const nlIn = $("pg-nl"), gen = $("pg-gen");
      const genNL = async () => {
        const text = nlIn.value.trim();
        if (!text) return;
        gen.disabled = true;
        const label = gen.textContent;
        gen.textContent = "…";
        try {
          const r = await fetch("/api/nl", {
            method: "POST",
            headers: { "content-type": "application/json" },
            body: JSON.stringify({ text }),
          });
          const d = await r.json();
          if (d.code) {
            progCode = d.code;
            ta.value = d.code;
            $("pg-log").textContent =
              "> generated (" + d.engine + ") — review, then ▶ RUN or ⏭ STEP";
          } else {
            $("pg-log").textContent =
              "x " + (d.error || "could not parse") + (d.hint ? "\n" + d.hint : "");
          }
        } catch (e) {
          $("pg-log").textContent = "x generate failed: " + e;
        } finally {
          gen.disabled = false;
          gen.textContent = label;
        }
      };
      gen.onclick = genNL;
      nlIn.onkeydown = (e) => {
        if (e.key === "Enter") { e.preventDefault(); genNL(); }
      };
    }
    $("pg-rec").onclick = async () => {
      const rec = state && state.prog && state.prog.rec;
      if (rec && rec.on) {
        send({ type: "rec_stop" });
        setTimeout(async () => {
          try {
            const code = await (await fetch("/api/recording")).text();
            if (code.trim()) {
              progCode = (progCode.trim() ? progCode.replace(/\s+$/, "")
                + "\n\n" : "") + code;
              ta.value = progCode;
              ta.scrollTop = ta.scrollHeight;
            }
          } catch (e) { /* server down */ }
        }, 400);
      } else {
        send({ type: "rec_start" });
      }
    };
    setupProgAutocomplete(ta);
    const exSel = $("pg-examples");
    exSel.innerHTML = '<option value="">examples…</option>' +
      Object.keys(EXAMPLES).map((k) => `<option>${k}</option>`).join("");
    exSel.onchange = () => {
      if (EXAMPLES[exSel.value]) { progCode = EXAMPLES[exSel.value]; ta.value = progCode; }
      exSel.value = "";
    };
    $("pg-save").onclick = () => {
      const name = prompt("Program name (letters/digits/_-):");
      if (name) {
        send({ type: "prog_save", name, code: progCode });
        setTimeout(refreshProgFiles, 400);
      }
    };
    $("pg-load").onclick = async () => {
      const sel = $("pg-files");
      if (!sel.value) return;
      try {
        const code = await (await fetch(`/api/programs/${sel.value}`)).text();
        progCode = code;
        ta.value = code;
      } catch (e) { /* server down */ }
    };
    refreshProgFiles();
  }
  progSig = "";
}

async function refreshProgFiles() {
  try {
    const names = await (await fetch("/api/programs")).json();
    const sel = $("pg-files");
    if (sel) sel.innerHTML =
      names.map((n) => `<option value="${n}">${n}</option>`).join("");
  } catch (e) { /* server down */ }
}

function updateProgPanel() {
  const p = state && state.prog;
  if (!p) return;
  const sig = JSON.stringify([p.running, p.paused, p.line, p.n,
                              p.log && p.log.length,
                              p.log && p.log[p.log.length - 1],
                              p.rec && p.rec.on, p.rec && p.rec.n]);
  if (sig === progSig) return;
  progSig = sig;
  const rec = $("pg-rec");
  if (rec && p.rec) {
    rec.textContent = p.rec.on ? `■ REC · ${p.rec.n}` : "● REC";
    rec.className = p.rec.on ? "recording" : "";
  }
  const st = $("pg-state");
  if (st) {
    st.textContent = !p.running ? "idle"
      : p.paused ? `paused → line ${p.line ?? "?"}: ${p.current ?? ""}`
      : `running · cmd #${p.n}${p.line ? " · line " + p.line : ""}`;
    st.className = "prog-state" + (p.running ? (p.paused ? " warn" : " on") : "");
  }
  const run = $("pg-run");
  if (run) run.className = p.running && !p.paused ? "playing" : "";
  const ta = $("pg-code");
  if (ta && !PREVIEW) ta.readOnly = p.running;
  const log = $("pg-log");
  if (log && p.log) {
    log.textContent = p.log.join("\n");
    log.scrollTop = log.scrollHeight;
  }
  if (ta && !PREVIEW) {                       // mark the error / current step line
    const lastLog = (p.log && p.log[p.log.length - 1]) || "";
    const m = /line (\d+)/.exec(lastLog);
    if (/^x/.test(lastLog) && m && lastLog !== progLastErr) {
      progLastErr = lastLog; highlightProgLine(ta, +m[1], true);
    } else if (!/^x/.test(lastLog)) {
      progLastErr = null;
      if (p.paused && p.line && p.line !== progLastStep) {
        progLastStep = p.line; highlightProgLine(ta, p.line, false);
      }
      if (!p.paused) progLastStep = null;
    }
  }
}

function buildPanel() {
  if (curGroup === "seq") { buildSeqPanel(); return; }
  if (curGroup === "prog") { buildProgPanel(); return; }
  const wrap = $("joints");
  wrap.innerHTML = "";
  rows = {};
  cartEls = null;
  const head = document.createElement("div");
  head.className = "panel-head";
  head.innerHTML = `<span>${GROUPS[curGroup].label}</span>
    <small>${GROUPS[curGroup].idx.length} joints · hold ± or drag the thumb</small>`;
  wrap.appendChild(head);
  const legsLockedReal = curGroup === "legs" && state && state.mode === "real";
  for (const [k, idx] of GROUPS[curGroup].idx.entries()) {
    const jname = model.joint_names[idx];
    const [lo, hi] = limits[idx] || [-3.14, 3.14];
    const row = document.createElement("div");
    row.className = "jrow";
    const human = `J${k + 1}${ROLE[idx] ? " · " + ROLE[idx] : ""}`;
    row.innerHTML = `
      <div class="jname" title="protocol index ${idx}"><b>${human}</b>${jname}</div>
      <button class="jlim" data-l="lo" title="Jump to the lower limit (guard permitting)">⇤</button>
      <button class="jbtn" data-d="-1">−</button>
      <div class="jbar" role="slider" tabindex="0" aria-label="${human}" aria-valuemin="${Math.round(lo*180/Math.PI)}" aria-valuemax="${Math.round(hi*180/Math.PI)}" aria-valuenow="0"><div class="jfill"></div><div class="jthumb"></div></div>
      <button class="jbtn" data-d="1">+</button>
      <button class="jlim" data-l="hi" title="Jump to the upper limit (guard permitting)">⇥</button>
      <div class="jval"><span class="ang">—</span><small class="sub">—</small></div>`;
    wrap.appendChild(row);
    const locked = legsLockedReal;            // preview joints are drivable client-side
    for (const b of row.querySelectorAll(".jlim")) {
      if (locked) { b.disabled = true; continue; }
      b.onclick = () => send({ type: "set_joint", idx,
                               value: b.dataset.l === "lo" ? lo : hi });
    }
    for (const b of row.querySelectorAll(".jbtn")) {
      if (locked) {
        b.disabled = true;
        if (PREVIEW) b.title = "preview is a recording — run the local server";
        continue;
      }
      const d = parseInt(b.dataset.d);
      const stop = () => send({ type: "jog_stop", idx });
      b.addEventListener("pointerdown", (e) => {
        e.preventDefault(); send({ type: "jog_start", idx, dir: d }); });
      b.addEventListener("pointerup", stop);
      b.addEventListener("pointerleave", stop);
      b.addEventListener("pointercancel", stop);
    }
    // draggable slider: thumb = commanded target, fill = actual position
    const bar = row.querySelector(".jbar");
    const r = { fill: row.querySelector(".jfill"),
                thumb: row.querySelector(".jthumb"),
                ang: row.querySelector(".ang"),
                sub: row.querySelector(".sub"), bar, lo, hi, dragging: false };
    if (r.ang) { r.ang.dataset.idx = idx; r.ang.dataset.lo = lo; r.ang.dataset.hi = hi; if (!locked) r.ang.classList.add("editable"); }
    if (locked) {
      bar.classList.add("disabled");
      bar.tabIndex = -1; bar.setAttribute("aria-disabled", "true");
    } else {
      const valAt = (e) => {
        const rect = bar.getBoundingClientRect();
        const f = Math.min(1, Math.max(0, (e.clientX - rect.left) / rect.width));
        return lo + f * (hi - lo);
      };
      let lastSend = 0;
      const sendVal = (e, force) => {
        const now = performance.now();
        if (!force && now - lastSend < 40) return;
        lastSend = now;
        const v = valAt(e);
        send({ type: "set_joint", idx, value: v });
        r.thumb.style.left = `${(100 * (v - lo)) / (hi - lo)}%`;
      };
      bar.addEventListener("pointerdown", (e) => {
        e.preventDefault();
        r.dragging = true;
        bar.setPointerCapture(e.pointerId);
        sendVal(e, true);
      });
      bar.addEventListener("pointermove", (e) => {
        if (r.dragging) sendVal(e);
      });
      const end = (e) => {
        if (r.dragging) { r.dragging = false; sendVal(e, true); }
      };
      bar.addEventListener("pointerup", end);
      bar.addEventListener("pointercancel", end);
      bar.addEventListener("keydown", (e) => {
        const step = (e.shiftKey ? 8 : 2) * Math.PI / 180;
        const cur = (state && state.targ && state.targ[idx] != null)
          ? state.targ[idx] : ((state && state.q && state.q[idx]) ?? lo);
        let v = null;
        if (e.key === "ArrowRight" || e.key === "ArrowUp") v = cur + step;
        else if (e.key === "ArrowLeft" || e.key === "ArrowDown") v = cur - step;
        else if (e.key === "Home") v = lo;
        else if (e.key === "End") v = hi;
        else return;
        e.preventDefault();
        v = Math.min(hi, Math.max(lo, v));
        send({ type: "set_joint", idx, value: v });
        r.thumb.style.left = `${(100 * (v - lo)) / (hi - lo)}%`;
        bar.setAttribute("aria-valuenow", Math.round((v * 180) / Math.PI));
      });
    }
    rows[idx] = r;
  }
  if (legsLockedReal) {
    const note = document.createElement("div");
    note.className = "panel-note";
    note.textContent = "Lower chain locked in REAL mode — balance belongs to the firmware.";
    wrap.prepend(note);
  }
  if (curGroup === "left" || curGroup === "right") buildCartBlock(wrap);
}

// ---- cartesian jog + tool (TCP) block --------------------------------------
let cartEls = null;          // {x,y,z} readout spans for the open arm tab
let cartStepMm = 5;
let toolsCache = null;       // {name: [x,y,z] mm}
let cartTimers = [];

function stopCartTimers() {
  for (const t of cartTimers) clearInterval(t);
  cartTimers = [];
}

async function refreshTools(sel, arm) {
  if (PREVIEW) return;
  try {
    toolsCache = await (await fetch("/api/tools")).json();
  } catch (e) { return; }
  if (!sel) return;
  const cur = state && state.tools && state.tools[arm]
    ? state.tools[arm].name : "flange";
  sel.innerHTML = Object.keys(toolsCache).map((n) =>
    `<option value="${n}"${n === cur ? " selected" : ""}>${n}</option>`).join("");
}

function buildCartBlock(wrap) {
  const arm = curGroup;
  const blk = document.createElement("div");
  blk.innerHTML = `
    <div class="panel-head"><span>CARTESIAN · TCP</span>
      <small>step-jog the tool point · world axes</small></div>
    <div id="cart"></div>
    <div class="cart-foot">
      <label>STEP</label>
      <select id="cart-step">
        ${[1, 5, 20, 50].map((v) => `<option value="${v}"${v === cartStepMm
          ? " selected" : ""}>${v} mm</option>`).join("")}
      </select>
      <span class="vsep"></span>
      <label>TOOL</label>
      <select id="tool-sel"></select>
      <button id="tool-def" title="Define a named TCP offset (mm, wrist frame)">+</button>
      <button id="tool-del" title="Delete the selected tool">✕</button>
    </div>`;
  wrap.appendChild(blk);
  const cart = blk.querySelector("#cart");
  cartEls = {};
  const axes = [["X", 0], ["Y", 1], ["Z", 2]];
  for (const [nm, ax] of axes) {
    const row = document.createElement("div");
    row.className = "cart-row";
    row.innerHTML = `<b class="ax-${nm.toLowerCase()}">${nm}</b>
      <button class="jbtn" data-d="-1">−</button>
      <span class="cart-val">—</span>
      <button class="jbtn" data-d="1">+</button>`;
    cart.appendChild(row);
    cartEls[ax] = row.querySelector(".cart-val");
    for (const b of row.querySelectorAll(".jbtn")) {
      if (PREVIEW) { b.disabled = true; continue; }
      const dir = parseInt(b.dataset.d);
      const fire = () => {
        const delta = [0, 0, 0];
        delta[ax] = (dir * cartStepMm) / 1000;
        send({ type: "cart_step", arm, delta });
      };
      b.addEventListener("pointerdown", (e) => {
        e.preventDefault();
        fire();
        cartTimers.push(setInterval(fire, 150));
      });
      for (const ev of ["pointerup", "pointerleave", "pointercancel"])
        b.addEventListener(ev, stopCartTimers);
    }
  }
  blk.querySelector("#cart-step").onchange = (e) =>
    (cartStepMm = parseInt(e.target.value));
  const sel = blk.querySelector("#tool-sel");
  if (PREVIEW) {
    sel.disabled = true;
    blk.querySelector("#tool-def").disabled = true;
    blk.querySelector("#tool-del").disabled = true;
  } else {
    refreshTools(sel, arm);
    sel.onchange = () => send({ type: "tool_set", arm, name: sel.value });
    blk.querySelector("#tool-def").onclick = () => {
      const name = prompt("Tool name (letters/digits/_-):");
      if (!name) return;
      const xyz = prompt("TCP offset x,y,z in mm (wrist frame):", "0,0,120");
      if (!xyz) return;
      const v = xyz.split(",").map(Number);
      if (v.length !== 3 || v.some(isNaN)) { alert("need three numbers"); return; }
      send({ type: "tool_def", name, xyz_mm: v });
      send({ type: "tool_set", arm, name });
      setTimeout(() => refreshTools(sel, arm), 400);
    };
    blk.querySelector("#tool-del").onclick = () => {
      if (sel.value === "flange") return;
      send({ type: "tool_del", name: sel.value });
      setTimeout(() => refreshTools(sel, arm), 400);
    };
  }
}

const z = (x) => (Math.abs(x) < 0.005 ? 0 : x);   // kill -0.0 flicker
const deg = (r) => (z(r) * 180 / Math.PI).toFixed(1) + "°";

const _tcp = new THREE.Vector3();
function updateCartReadout() {
  if (!cartEls || !eeObjs[curGroup]) return;
  const p = tcpWorld(curGroup, _tcp);
  for (const [ax, el] of Object.entries(cartEls))
    el.textContent = `${(p.getComponent(+ax) * 1000).toFixed(0)} mm`;
}

function updatePanel() {
  if (!state) return;
  if (curGroup === "seq") { updateSeqPanel(); return; }
  if (curGroup === "prog") { updateProgPanel(); return; }
  updateCartReadout();
  const q = state.q || [], dq = state.dq || [], temps = state.temps || [];
  for (const [idx, r] of Object.entries(rows)) {
    const a = q[idx], t = temps[idx], targ = state.targ && state.targ[idx];
    if (a === undefined || a === null) continue;
    const span = r.hi - r.lo || 1;
    r.fill.style.width = `${(100 * (a - r.lo)) / span}%`;
    if (!r.dragging && targ !== null && targ !== undefined)
      r.thumb.style.left = `${(100 * (targ - r.lo)) / span}%`;
    r.ang.textContent = deg(a);
    if (r.bar) r.bar.setAttribute("aria-valuenow",
      Math.round(((targ != null && targ !== undefined ? targ : a) * 180) / Math.PI));
    const tcls = t > 50 ? "temp-bad" : t > 40 ? "temp-warn" : "temp-ok";
    r.sub.className = "sub " + tcls;
    r.sub.textContent =
      `${z(dq[idx] || 0).toFixed(2)} r/s · ${t ? t.toFixed(0) : "—"}°C`;
  }
}

// ---------------------------------------------------------------- top bar
function chip(id, on, txtOn, txtOff, offClass = "bad") {
  const el = $(id);
  el.textContent = on ? txtOn : txtOff;
  el.className = "chip " + (on ? "on" : offClass);
}

function updateTop() {
  if (!state) return;
  const vp = $("viewport");
  if (vp) {
    vp.classList.toggle("cue-estop", !!state.estop && !state.observe);
    vp.classList.toggle("cue-dampened", !state.estop && !state.live && !state.observe);
    vp.classList.toggle("cue-live", !state.estop && !!state.live);
    vp.classList.toggle("cue-observe", !!state.observe);   // calm azure, not a fault
  }
  chip("chip-link", state.connected, "LINK", "NO LINK");
  const lat = $("chip-lat");
  if (lat) {
    const ms = rttMs || frameMs;                   // prefer real RTT, fall back to freshness
    if (wsOnline && ms) {
      lat.style.display = "";
      lat.textContent = Math.round(ms) + "ms";
      lat.className = "chip " + (ms > 260 ? "bad" : ms > 120 ? "warn" : "");
    } else lat.style.display = "none";
  }
  const bw = $("chip-bw");
  if (bw) {
    if (wsOnline && kbps) {
      bw.style.display = "";
      bw.textContent = kbps >= 1000 ? (kbps / 1024).toFixed(1) + " MB/s"
                                    : Math.round(kbps) + " KB/s";
    } else bw.style.display = "none";
  }
  chip("chip-armed", state.armed, "ARMED", "ARMING…", "");
  chip("chip-live", state.live, "LIVE", "DAMPENED", "warn");
  const liveEl = $("chip-live");
  if (liveEl) {
    const moving = !!(state.homing || state.routing) || performance.now() < movingUntil;
    if (state.live) { liveEl.textContent = moving ? "MOVING" : "LIVE"; liveEl.classList.toggle("moving", moving); }
    else liveEl.classList.remove("moving");
    if (state.observe) { liveEl.textContent = "OBSERVE"; liveEl.className = "chip on"; }
  }
  const xc = $("chip-ext");
  if (xc) {
    xc.style.display = state.external ? "" : "none";  // external commander drives
    xc.className = "chip on moving";
  }
  const gc = $("chip-guard");
  if (gc && state.guard) {
    gc.style.display = state.guard.on ? "" : "none";
    gc.textContent = state.guard.blocking ? "LIMIT" : "GUARD";
    gc.className = "chip " + (state.guard.blocking ? "warn" : "on");
  }
  const sc = $("chip-sing");
  if (sc) {
    const mv = state.manip ? Object.values(state.manip).filter(v => v != null) : [];
    const lo = mv.length ? Math.min(...mv) : null;
    sc.style.display = (lo != null && lo < 0.06) ? "" : "none";
    sc.className = "chip warn";
  }
  const rc = $("chip-rec");
  if (rc) {
    const rec = state.prog && state.prog.rec;
    rc.style.display = rec && rec.on ? "" : "none";
    if (rec && rec.on) rc.textContent = `REC · ${rec.n}`;
    rc.className = "chip bad";
  }
  const hc = $("chip-home");
  if (hc) {
    const show = state.homing || state.routing;
    hc.style.display = show ? "" : "none";
    hc.textContent = state.routing ? "ROUTING" : "HOMING";
    hc.className = "chip on";
  }
  const cc = $("chip-contact");
  if (cc) {
    const ct = state.contact && state.contact.tripped;
    cc.style.display = ct ? "" : "none";
    cc.textContent = ct && state.contact.joint != null
      ? `CONTACT · J${state.contact.joint}` : "CONTACT";
    cc.className = "chip bad";
  }
  const tmax = state.temps ? Math.max(...state.temps) : 0;
  const tEl = $("chip-temp");
  tEl.textContent = `T ${tmax.toFixed(0)}°C`;
  tEl.className = "chip " + (state.overtemp ? "bad" : tmax > 45 ? "warn" : "");
  const mb = $("btn-mirror");
  if (mb) mb.classList.toggle("on", !!state.mirror);   // icon + label kept; .on = active
  const cb = $("btn-carry");
  if (cb) {
    cb.classList.toggle("on", !!state.carry);
    const cp = $("carry-pad"); if (cp) cp.style.display = state.carry ? "" : "none";
  }
  $("mode-sim").className = state.mode === "sim" ? "active sim" : "";
  $("mode-real").className = state.mode === "real" ? "active real" : "";
  const obt = $("observe-toggle");
  if (obt) {
    obsOn = !!state.observe;
    obt.classList.toggle("on", obsOn);
    obt.setAttribute("aria-pressed", obsOn ? "true" : "false");
  }
  $("foot-mode").textContent = `mode: ${state.mode}`;
  const es = $("btn-estop");
  es.textContent = state.estop ? "RESUME" : "E-STOP";
  es.className = state.estop ? "resume" : "";
  // v0.4 bug: a local `tc` (temp chip) shadowed the gizmo — detach never ran
  if (tc && tc.object && !state.live) tc.detach();   // no gizmo while dampened
  const banner = $("banner");
  if (PREVIEW) { banner.className = "preview";
    banner.textContent = pvDriven
      ? "PREVIEW — you're driving the twin (jog · sliders · Home). Advanced features need the local server"
      : "PREVIEW — recording plays · grab a joint slider, hold ± or press Home to drive it yourself";
  } else if (state.observe) { banner.className = "observe";
    banner.textContent = state.external
      ? "OBSERVE — an external commander is driving (e.g. MoveIt)"
      : "OBSERVE — watching only · toggle OBSERVE off to command";
  } else if (!state.live) { banner.className = "dampened";
    banner.textContent = state.estop
      ? "DAMPENED — press RESUME to enable motion"
      : state.overtemp ? "DAMPENED — overtemp latch"
      : (state.contact && state.contact.tripped)
          ? "DAMPENED — contact reflex: click CONTACT to reset"
      : !state.connected ? "waiting for telemetry…" : "DAMPENED";
  } else banner.className = "";
}

$("btn-estop").onclick = () =>
  send({ type: state && state.estop ? "resume" : "estop" });
let previewViz = null, previewPending = null, ghostGroup = null;
function clearGhost() {
  if (!ghostGroup) return;
  ghostGroup.traverse((o) => { if (o.isMesh && o.material) o.material.dispose(); });
  scene.remove(ghostGroup); ghostGroup = null;
}
function makeGhost(q) {          // translucent robot at the target joint pose
  clearGhost();
  if (!q || !robotRoots.length) return;
  const grp = new THREE.Group(), map = new Map();
  const walk = (a, b) => { map.set(a, b); for (let i = 0; i < a.children.length; i++) walk(a.children[i], b.children[i]); };
  for (const root of robotRoots) { const c = root.clone(true); walk(root, c); grp.add(c); }
  grp.traverse((o) => { if (o.isMesh) {
    o.material = new THREE.MeshBasicMaterial({ color: 0x3fc463, transparent: true, opacity: 0.2, depthWrite: false });
    o.renderOrder = 6;
  } });
  for (const [idx, j] of Object.entries(jointGroups)) {
    const cg = map.get(j.grp); if (!cg) continue;
    cg.quaternion.copy(j.baseQuat).multiply(new THREE.Quaternion().setFromAxisAngle(j.axis, q[idx] || 0));
  }
  scene.add(grp); ghostGroup = grp;
}
function clearPreview() {
  if (previewViz) { scene.remove(previewViz); previewViz = null; }
  clearGhost();
  previewPending = null;
  if ($("approve-bar")) $("approve-bar").style.display = "none";
}
function showPreview(text, opts) {          // opts: {q?, tcp?, onApprove}
  clearPreview();
  opts = opts || {};
  if (opts.q) makeGhost(opts.q);
  const grp = new THREE.Group();
  if (opts.route) {                          // planned collision-free route's TCP trail per arm
    for (const arm in opts.route) {
      const pts = (opts.route[arm] || []).map((p) => new THREE.Vector3(p[0], p[1], p[2]));
      if (pts.length < 2) continue;
      const line = new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts),
        new THREE.LineBasicMaterial({ color: 0x3B82F6, transparent: true, opacity: 0.95, depthTest: false }));
      line.renderOrder = 7; grp.add(line);
      for (const p of pts) {
        const dot = new THREE.Mesh(new THREE.SphereGeometry(0.0075, 8, 6),
          new THREE.MeshBasicMaterial({ color: 0x3B82F6, depthTest: false }));
        dot.position.copy(p); dot.renderOrder = 7; grp.add(dot);
      }
    }
  }
  if (opts.tcp) {
    for (const arm in opts.tcp) {
      const t = opts.tcp[arm];
      const ring = new THREE.Mesh(new THREE.RingGeometry(0.042, 0.052, 28),
        new THREE.MeshBasicMaterial({ color: 0x3fc463, transparent: true, opacity: 0.8,
          side: THREE.DoubleSide, depthTest: false }));
      ring.position.set(t[0], t[1], t[2]); ring.renderOrder = 7; grp.add(ring);
    }
  }
  scene.add(grp); previewViz = grp;
  previewPending = { onApprove: opts.onApprove };
  if ($("approve-text")) $("approve-text").textContent = text;
  if ($("approve-bar")) $("approve-bar").style.display = "flex";
}
async function previewServer(action, idx, text, onApprove) {
  if (PREVIEW || previewPending) return;
  try {
    const d = await fetch("/api/preview?action=" + action + (idx != null ? "&i=" + idx : ""))
      .then((r) => r.json());
    if (d && d.q) showPreview(text, { q: d.q, tcp: d.tcp, route: d.route, onApprove });
    else onApprove();
  } catch (_) { onApprove(); }
}
if ($("btn-home")) $("btn-home").onclick = () =>
  previewServer("home", null, "HOME — glide arms to the safe pose", () => send({ type: "home" }));
if ($("approve-go")) $("approve-go").onclick = () => {
  const p = previewPending; clearPreview();
  if (p && p.onApprove) p.onApprove();
};
if ($("approve-cancel")) $("approve-cancel").onclick = clearPreview;
document.addEventListener("keydown", (e) => {     // Escape cancels a pending move preview
  if (e.key === "Escape" && previewPending) clearPreview();
});
if ($("chip-contact"))
  $("chip-contact").onclick = () => send({ type: "reset_contact" });
$("btn-mirror").onclick = () =>
  send({ type: "mirror", on: !(state && state.mirror) });
if ($("btn-carry")) {
  $("btn-carry").onclick = () =>
    send({ type: state && state.carry ? "carry_release" : "carry_grab" });
  document.querySelectorAll("#carry-pad button").forEach(b => {
    b.onclick = () => {
      const d = b.dataset.cd.split(",").map(Number), s = 0.02;
      send({ type: "carry_step", delta: [d[0] * s, d[1] * s, d[2] * s] });
    };
  });
}
if ($("btn-trace")) {
  $("btn-trace").onclick = () => {
    traceOn = !traceOn;
    $("btn-trace").classList.toggle("on", traceOn);
    for (const t of Object.values(traces)) t.line.visible = traceOn;
  };
  $("btn-clear-trace").onclick = clearTraces;
}

// ---- rendered robot camera (MJPEG panel) --------------------------------
if (!PREVIEW && $("btn-cam")) {
  const pip = $("cam-pip"), img = $("cam-img"), sel = $("cam-sel");
  let camOn = false;
  const start = () =>
    (img.src = "/camstream?cam=" + encodeURIComponent(sel.value) + "&t=" + Date.now());
  const stop = () => img.removeAttribute("src");
  fetch("/api/cameras").then((r) => r.json()).then((d) => {
    sel.innerHTML = "";
    for (const c of d.cameras || []) {
      const o = document.createElement("option");
      o.value = c; o.textContent = c;
      if (c === d.current) o.selected = true;
      sel.appendChild(o);
    }
    if (!(d.cameras && d.cameras.length)) $("btn-cam").disabled = true;
    updateCamActions();
  }).catch(() => ($("btn-cam").disabled = true));
  $("btn-cam").onclick = () => {
    camOn = !camOn;
    $("btn-cam").classList.toggle("on", camOn);
    pip.classList.toggle("show", camOn);
    if (!camOn && $("viewport").classList.contains("split")) {   // turning the camera off exits split
      $("viewport").classList.remove("split");
      if ($("cam-expand")) $("cam-expand").classList.remove("on");
    }
    camOn ? start() : stop();
    resize();
  };
  const mark = $("cam-mark"), info = $("cam-info");
  const CAM_W = 640, CAM_H = 480;
  const showMark = (px) => {
    if (px) {
      mark.style.left = (px[0] / CAM_W * 100) + "%";
      mark.style.top = (px[1] / CAM_H * 100) + "%";
      mark.style.display = "block";
    } else mark.style.display = "none";
  };
  const updateCamActions = () => {                 // pick actions only on the work camera
    const work = sel.value === "cam_work";
    for (const id of ["cam-detect", "cam-pick", "cam-servo", "cam-smart"])
      if ($(id)) $(id).style.display = work ? "" : "none";
    if (!work && $("cam-obj")) $("cam-obj").style.display = "none";
  };
  const toWork = () => { sel.value = "cam_work"; updateCamActions(); if (camOn) start(); };
  sel.onchange = () => { showMark(null); updateCamActions(); if (camOn) start(); };
  if ($("cam-expand")) $("cam-expand").onclick = () => {
    const vp = $("viewport"), willSplit = !vp.classList.contains("split");
    if (willSplit && !camOn) $("btn-cam").click();              // split needs the camera on
    pip.style.left = pip.style.top = pip.style.right = pip.style.bottom = pip.style.width = "";  // drop any dragged position
    vp.classList.toggle("split", willSplit);
    $("cam-expand").classList.toggle("on", willSplit);
    $("cam-expand").title = willSplit ? "Collapse the split (floating camera)"
                                      : "Expand the camera (ego + exo, side by side)";
    $("cam-expand").setAttribute("aria-pressed", willSplit);
    $("cam-expand").setAttribute("aria-label", willSplit ? "Collapse split, float the camera"
                                                         : "Expand to ego + exo split view");
    resize();
  };
  $("cam-detect").onclick = async () => {
    toWork();
    info.textContent = "detecting…";
    try {
      const d = await (await fetch("/api/detect")).json();
      if (d.found) {
        showMark(d.pixel);
        info.textContent = "target  " + d.world_mm.map((v) => v.toFixed(0)).join("  ") + " mm";
      } else {
        showMark(null);
        info.textContent = "no target" + (d.error ? " (" + d.error + ")" : "");
      }
    } catch (e) { info.textContent = "detect failed"; }
  };
  $("cam-pick").onclick = async () => {
    toWork();
    info.textContent = "pick…";
    try {
      const d = await (await fetch("/api/pick", { method: "POST" })).json();
      if (d.found) {
        showMark(d.pixel);
        info.textContent = "picking  " + d.world_mm.map((v) => v.toFixed(0)).join("  ") +
          " mm" + (d.ran ? "" : " — press RESUME");
      } else {
        showMark(null);
        info.textContent = "no target" + (d.error ? " (" + d.error + ")" : "");
      }
    } catch (e) { info.textContent = "pick failed"; }
  };
  $("cam-servo").onclick = async () => {
    toWork();
    info.textContent = "visual servo…";
    try {
      const d = await (await fetch("/api/servo_pick", { method: "POST" })).json();
      if (d.found) {
        info.textContent = "servo pick  " +
          (d.world_mm ? d.world_mm.map((v) => v.toFixed(0)).join("  ") + " mm" : "") +
          " · img " + d.image_err_px + "px" + (d.ran ? "" : " — press RESUME");
      } else {
        showMark(null);
        info.textContent = "no target" + (d.error ? " (" + d.error + ")" : "");
      }
    } catch (e) { info.textContent = "servo failed"; }
  };
  const runSmartPick = async () => {
    info.textContent = "smart pick…";
    try {
      const sel = $("cam-obj");
      const q = sel && sel.value !== "" ? "?target=" + encodeURIComponent(sel.value) : "";
      const d = await (await fetch("/api/smart_pick" + q, { method: "POST" })).json();
      if (d.found && d.ran) {
        info.textContent = "smart pick  " + (d.label ? d.label + " · " : "") +
          (d.center_mm ? d.center_mm.map((v) => v.toFixed(0)).join(" ") + " mm" : "");
        graspOn = true; if ($("btn-grasp")) $("btn-grasp").classList.add("on");
        await buildGrasp();
      } else if (d.feasible === false) {
        info.textContent = d.error || "object too wide for the gripper";
        graspOn = true; await buildGrasp();
      } else {
        info.textContent = d.error ? "smart pick: " + d.error : "no object — armed?";
      }
    } catch (e) { info.textContent = "smart pick failed"; }
  };
  $("cam-smart").onclick = async () => {
    toWork();
    graspOn = true; if ($("btn-grasp")) $("btn-grasp").classList.add("on");
    const ok = await buildGrasp();                // show what we'd grasp first
    if (!ok) { info.textContent = "no object to pick"; return; }
    const sel = $("cam-obj");
    const label = sel && sel.selectedOptions[0] ? sel.selectedOptions[0].textContent : "object";
    showPreview("Smart pick: " + label + " ?", { onApprove: runSmartPick });
  };
} else if ($("btn-cam")) {
  $("btn-cam").disabled = true;
  $("btn-cam").title = "preview is a recording — run the local server";
}

let obsOn = false;
$("observe-toggle").onclick = () => send({ type: "observe", on: !obsOn });
$("observe-toggle").onkeydown = (e) => {
  if (e.key === "Enter") { e.preventDefault(); send({ type: "observe", on: !obsOn }); }
};
$("mode-sim").onclick = () => send({ type: "set_mode", mode: "sim" });
$("mode-real").onclick = () => {
  if (confirm("Switch to REAL robot? It will stay DAMPENED until you press RESUME."))
    send({ type: "set_mode", mode: "real" });
};
for (const tab of document.querySelectorAll("#tabs div")) {
  tab.onclick = () => {
    document.querySelectorAll("#tabs div").forEach((t) =>
      t.classList.remove("active"));
    tab.classList.add("active");
    curGroup = tab.dataset.group;
    buildPanel();
  };
}

// ---- keyboard shortcuts + legend (operator hotkeys) ----------------------
(function keyboardShortcuts() {
  const MAP = { " ": "btn-estop", h: "btn-home", c: "btn-cam", e: "cam-expand",
                t: "btn-trace", d: "btn-dex", p: "btn-pcl", g: "btn-grasp",
                l: "btn-layers", m: "btn-mirror", b: "btn-coll", f: "btn-force" };
  const legend = $("keys-legend");
  const setLegend = (show) => { if (legend) legend.style.display = show ? "flex" : "none"; };
  const legendOpen = () => legend && legend.style.display === "flex";
  const typing = (el) => el && (el.tagName === "INPUT" || el.tagName === "TEXTAREA" ||
                                el.tagName === "SELECT" || el.isContentEditable);
  document.addEventListener("keydown", (e) => {
    if (e.ctrlKey || e.metaKey || e.altKey) return;
    if (e.key === "Escape") { setLegend(false); return; }
    if (e.key === "?") { e.preventDefault(); setLegend(!legendOpen()); return; }
    if (typing(document.activeElement)) return;            // don't hijack text fields
    if (legendOpen()) return;                              // legend up: only Esc / ? act
    if (document.activeElement && document.activeElement.tagName === "BUTTON" &&
        (e.key === " " || e.key === "Enter")) return;      // let a focused button self-activate
    if (e.key >= "1" && e.key <= "6") {                    // arm-group tabs
      const t = document.querySelectorAll("#tabs div")[+e.key - 1];
      if (t) { t.click(); e.preventDefault(); }
      return;
    }
    const id = MAP[e.key.toLowerCase()];
    if (id) { const b = $(id); if (b && !b.disabled) { b.click(); e.preventDefault(); } }
  });
  if ($("btn-keys")) $("btn-keys").onclick = () => setLegend(!legendOpen());
  if ($("keys-close")) $("keys-close").onclick = () => setLegend(false);
  if (legend) legend.addEventListener("click", (e) => { if (e.target === legend) setLegend(false); });
})();

// ---------------------------------------------------------------- data in
let lastFrame = 0, frameMs = 0;        // telemetry freshness (inter-frame gap)
let rttMs = 0;                          // real round-trip (ping echo)
let bwBytes = 0, kbps = 0, bwT = 0;     // rolling downlink bandwidth
let prevQ = null, movingUntil = 0;      // idle-vs-moving cue (Mecademic solid/blink)
function onState(s) {
  const _n = performance.now();
  if (lastFrame) frameMs = frameMs ? frameMs * 0.8 + (_n - lastFrame) * 0.2 : (_n - lastFrame);
  lastFrame = _n;
  if (s.pong != null) {                 // real round-trip from the ping echo
    const rtt = _n - s.pong;
    rttMs = rttMs ? rttMs * 0.7 + rtt * 0.3 : rtt;
  }
  const _q = s.q || s.targ;
  if (_q && prevQ && _q.length === prevQ.length) {
    let d = 0; for (let i = 0; i < _q.length; i++) { const a = Math.abs(_q[i] - prevQ[i]); if (a > d) d = a; }
    if (d > 0.0015) movingUntil = _n + 450;   // sticky so the cue doesn't flicker
  }
  if (_q) prevQ = _q.slice();
  const modeChanged = state && state.mode !== s.mode;
  state = s;
  setAngles(s.q || s.targ);
  updatePanel();
  updateTop();
  if (window.__snapHooks) for (const h of window.__snapHooks) { try { h(s); } catch (_) {} }
  if (modeChanged) buildPanel();
}

function setOffline(off, msg) {
  document.body.classList.toggle("offline", off);   // dims/freezes stale chips
  if (off) {
    const b = $("banner");
    b.className = "dampened";
    b.textContent = msg || "no connection to the server — retrying…";
  }
}

function connectWS() {
  ws = new WebSocket(`ws://${location.host}/ws`);
  ws.onopen = () => { wsOnline = true; reconnectMs = 1000; lastMsg = performance.now(); };
  ws.onmessage = (e) => {
    lastMsg = performance.now();
    bwBytes += (e.data && e.data.length) || 0;     // rolling downlink bandwidth
    if (document.body.classList.contains("offline")) setOffline(false);
    onState(JSON.parse(e.data));
  };
  ws.onerror = () => { try { ws.close(); } catch (_) {} };   // -> onclose -> retry
  ws.onclose = () => {
    wsOnline = false;
    setOffline(true, "no connection to the server — retrying… " +
      "(server down, or missing: pip install websockets)");
    setTimeout(connectWS, reconnectMs);
    reconnectMs = Math.min(Math.round(reconnectMs * 1.6), 5000);   // gentle backoff
  };
}

// stale-telemetry watchdog: socket looks open but frames stopped -> force reconnect
setInterval(() => {
  if (!PREVIEW && wsOnline && performance.now() - lastMsg > 1500) {
    setOffline(true, "telemetry stalled — reconnecting…");
    try { ws.close(); } catch (_) {}
  }
}, 500);

// link probe: real round-trip ping + downlink bandwidth (1 Hz)
setInterval(() => {
  if (PREVIEW) return;
  if (wsOnline) send({ type: "ping", t: performance.now() });
  const now = performance.now();
  if (bwT) { const dt = (now - bwT) / 1000; if (dt > 0) kbps = (bwBytes / 1024) / dt; }
  bwBytes = 0; bwT = now;
}, 1000);

// ── Interactive preview: drive the twin's joints client-side (no server) ─────
// In preview there's no server, so jog / sliders / Home are routed here instead
// of over the socket: they edit a local 26-vec pose fed straight back through
// onState() (pure forward kinematics — the twin already renders from q).
let pvQ = null;                  // current driven pose (26-vec)
let pvDriven = false;            // has the user taken control of the recording?
let pvBase = null;               // a recorded frame, reused for non-joint telemetry
const pvJog = {};                // idx -> jog interval id

function pvInit() {
  const f = window.PREVIEW_DATA.frames;
  pvBase = f[Math.floor(f.length / 2)] || f[0];          // a live (resumed) frame
  pvQ = ((f[0] && f[0].q) || new Array(26).fill(0)).slice();
}
function pvClamp(idx, v) {
  const l = (typeof limits !== "undefined" && limits[idx]) || [-3.14, 3.14];
  return Math.min(l[1], Math.max(l[0], v));
}
function pvRender() {
  const z = new Array(pvQ.length).fill(0);
  onState({ ...pvBase, q: pvQ.slice(), targ: pvQ.slice(), dq: z,
            live: true, estop: false, connected: true });
}
function pvTake() {
  if (pvDriven) return;
  pvDriven = true;
  const b = $("pv-replay"); if (b) b.style.display = "block";
}
function pvSet(idx, v) { if (pvQ) { pvQ[idx] = pvClamp(idx, v); pvTake(); } }
function pvJogStart(idx, dir) {
  if (!pvQ) return;
  pvTake();
  if (pvJog[idx]) clearInterval(pvJog[idx]);
  pvJog[idx] = setInterval(() => { pvQ[idx] = pvClamp(idx, pvQ[idx] + dir * 0.6 * 0.033); }, 33);
}
function pvJogStop(idx) { if (pvJog[idx]) { clearInterval(pvJog[idx]); pvJog[idx] = null; } }
function pvHome() {
  if (!pvQ) return;
  pvTake();
  const tgt = ((window.PREVIEW_DATA.frames[0] || {}).q || pvQ).slice();
  const from = pvQ.slice(); let k = 0; const N = 24;
  const iv = setInterval(() => {
    k++; const t = k / N;
    for (let i = 0; i < pvQ.length; i++) pvQ[i] = from[i] + (tgt[i] - from[i]) * t;
    if (k >= N) clearInterval(iv);
  }, 33);
}
function pvReplay() {
  for (const k in pvJog) pvJogStop(k);
  pvDriven = false;
  const b = $("pv-replay"); if (b) b.style.display = "none";
}
function pvHandle(o) {
  if (!o || !pvQ) return;
  if (o.type === "set_joint") pvSet(o.idx, o.value);
  else if (o.type === "jog_start") pvJogStart(o.idx, o.dir);
  else if (o.type === "jog_stop") pvJogStop(o.idx);
  else if (o.type === "home") pvHome();
  // ik / carry / mirror / wp_* / smart_pick / speed … are server-only -> ignored
}
function pvMountReplay() {
  if ($("pv-replay")) return;
  const b = document.createElement("button");
  b.id = "pv-replay"; b.textContent = "▶ Replay demo";
  b.title = "Hand control back to the recorded demo";
  b.style.cssText = "position:fixed;left:50%;bottom:18px;transform:translateX(-50%);" +
    "z-index:50;display:none;padding:7px 14px;border-radius:8px;cursor:pointer;" +
    "border:1px solid var(--line,#2A2F37);background:var(--panel,#16191E);" +
    "color:var(--fg,#D6DAE0);font:600 12px/1 var(--mono,monospace);";
  b.onclick = pvReplay;
  document.body.appendChild(b);
}

function startPlayback() {
  const frames = window.PREVIEW_DATA.frames;
  pvInit();
  const FRAME_MS = 100;                       // recorded at 10 Hz
  const t0 = performance.now();
  const lerp = (a, b, t) =>
    a && b ? a.map((v, i) => v + (b[i] - v) * t) : a || b;
  setInterval(() => {
    if (pvDriven) { pvRender(); return; }      // user has taken control of the joints
    const x = ((performance.now() - t0) / FRAME_MS) % frames.length;
    const i = Math.floor(x), t = x - i, j = (i + 1) % frames.length;
    const a = frames[i], b = frames[j];
    onState({ ...a, q: lerp(a.q, b.q, t), dq: lerp(a.dq, b.dq, t),
              temps: lerp(a.temps, b.temps, t),
              targ: lerp(a.targ, b.targ, t) });
  }, 33);                                     // smooth 30 fps interpolation
}

// ---------------------------------------------------------------- boot
(async function boot() {
  model = PREVIEW ? window.PREVIEW_DATA.model
        : await (await fetch("/api/model")).json();
  buildRobot();
  buildPanel();
  if (PREVIEW) {
    for (const id of ["btn-estop", "btn-mirror", "btn-carry"]) {
      $(id).disabled = true;
      $(id).title = "preview is a recording — run the local server to control";
    }
    const bh = $("btn-home"); if (bh) { bh.onclick = pvHome; bh.title = "Glide to the home pose"; }
    pvMountReplay();
    startPlayback();
  } else {
    connectWS();
  }
})();


// ---- manipulability heat-map (dexterity cloud) --------------------------
// A point cloud over each arm's reachable workspace, coloured by
// manipulability (reciprocal Jacobian condition number): blue = near-singular,
// warm = isotropic/dexterous. Fetched once from /api/reachmap, then toggled.
let dexCloud = null, dexOn = false, dexBusy = false;
function manipColor(t) {                    // t in [0,1] -> cold..hot
  t = Math.max(0, Math.min(1, t));
  const stops = [[0.13, 0.20, 0.60], [0.00, 0.62, 0.78], [0.22, 0.80, 0.35],
                 [0.96, 0.85, 0.22], [0.92, 0.28, 0.18]];
  const x = t * (stops.length - 1), i = Math.min(Math.floor(x), stops.length - 2),
        f = x - i, a = stops[i], b = stops[i + 1];
  return [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, a[2] + (b[2] - a[2]) * f];
}
async function buildDexCloud() {
  const pos = [], col = [];
  for (const arm of ["left", "right"]) {
    const r = await fetch(`/api/reachmap?arm=${arm}&n=2500`).then(x => x.json()).catch(() => null);
    if (!r || !r.points) continue;
    for (const p of r.points) {
      pos.push(p[0], p[1], p[2]);
      const c = manipColor(p[3] / 0.45);    // 0.45 ~ top of the manip range
      col.push(c[0], c[1], c[2]);
    }
  }
  const geo = new THREE.BufferGeometry();
  geo.setAttribute("position", new THREE.Float32BufferAttribute(pos, 3));
  geo.setAttribute("color", new THREE.Float32BufferAttribute(col, 3));
  dexCloud = new THREE.Points(geo, new THREE.PointsMaterial({
    size: 0.013, vertexColors: true, transparent: true, opacity: 0.85,
    sizeAttenuation: true, depthWrite: false }));
  dexCloud.frustumCulled = false;
  scene.add(dexCloud);
}

// ---- overlay coordination: declutter when heavy overlays are shown --------
// One heavy cloud at a time (DEX <-> PCL), and hide the drag-gizmo + wrist
// handles while any overlay is up so the twin stays readable.
function overlaysOn() { return pclOn || dexOn || graspOn; }
function setOverlayHint() {
  const el = $("overlay-hint"); if (!el) return;
  const parts = [];
  if (dexOn) parts.push("DEX — reach dexterity · warm = agile, blue = near-singular");
  if (pclOn) parts.push("PCL — work-camera depth → 3D points");
  if (graspOn) parts.push("GRASP — synthesised top-down grasps · azure = selected");
  el.textContent = parts.join("      ·      ");
  el.style.display = parts.length ? "block" : "none";
}
function syncGizmo() {
  const hide = overlaysOn();
  for (const m of Object.values(markers)) m.visible = !hide;
  if (hide && tc && tc.object) tc.detach();
  setOverlayHint();
}

if ($("btn-dex")) {
  $("btn-dex").onclick = async () => {
    if (PREVIEW || dexBusy) return;
    dexOn = !dexOn;
    $("btn-dex").classList.toggle("on", dexOn);
    if (dexOn) { const cb = $("btn-coll"); if (cb && cb.classList.contains("on")) cb.click(); }
    if (dexOn && pclOn) {                 // one heavy cloud at a time
      pclOn = false; if (pclCloud) pclCloud.visible = false;
      $("btn-pcl").classList.remove("on");
    }
    if (dexOn && !dexCloud) {
      dexBusy = true; $("btn-dex").classList.add("busy");
      await buildDexCloud();
      $("btn-dex").classList.remove("busy"); dexBusy = false;
    }
    if (dexCloud) dexCloud.visible = dexOn;
    syncGizmo();
  };
}


// ---- work-camera point cloud --------------------------------------------
// What the work camera sees, back-projected to a coloured 3D cloud in the twin
// (depth render -> world points, each coloured by its RGB pixel).
let pclCloud = null, pclOn = false, pclBusy = false;
async function buildPcl() {
  const r = await fetch("/api/pointcloud?stride=4").then(x => x.json()).catch(() => null);
  if (!r || !r.points || !r.points.length) return;
  const pos = [], col = [];
  for (const p of r.points) { pos.push(p[0], p[1], p[2]); col.push(p[3], p[4], p[5]); }
  const geo = new THREE.BufferGeometry();
  geo.setAttribute("position", new THREE.Float32BufferAttribute(pos, 3));
  geo.setAttribute("color", new THREE.Float32BufferAttribute(col, 3));
  pclCloud = new THREE.Points(geo, new THREE.PointsMaterial({
    size: 0.008, vertexColors: true, sizeAttenuation: true }));
  pclCloud.frustumCulled = false;
  scene.add(pclCloud);
}
if ($("btn-pcl")) {
  $("btn-pcl").onclick = async () => {
    if (PREVIEW || pclBusy) return;
    pclOn = !pclOn;
    $("btn-pcl").classList.toggle("on", pclOn);
    if (pclOn) { const cb = $("btn-coll"); if (cb && cb.classList.contains("on")) cb.click(); }
    if (pclOn && dexOn) {                 // one heavy cloud at a time
      dexOn = false; if (dexCloud) dexCloud.visible = false;
      $("btn-dex").classList.remove("on");
    }
    if (pclOn && !pclCloud) {
      pclBusy = true; $("btn-pcl").classList.add("busy");
      await buildPcl();
      $("btn-pcl").classList.remove("busy"); pclBusy = false;
    }
    if (pclCloud) pclCloud.visible = pclOn;
    syncGizmo();
  };
}



// ---- smart-pick grasp overlay -------------------------------------------
// A grasp synthesised on the work-camera cloud (table removed, object
// clustered): the object footprint, the parallel-jaw line (closes across the
// minor axis) and the top-down approach, drawn in the twin. /api/grasp returns
// mm; the twin is world metres. Azure = graspable, amber = too wide for the jaws.
let graspViz = null, graspOn = false, graspBusy = false, graspObjs = [];
function clearGrasp() {
  if (graspViz) { scene.remove(graspViz); graspViz = null; }
}
function graspLine(ptsMm, color, opacity) {
  const pos = [];
  for (const p of ptsMm) pos.push(p[0] / 1000, p[1] / 1000, p[2] / 1000);
  const geo = new THREE.BufferGeometry();
  geo.setAttribute("position", new THREE.Float32BufferAttribute(pos, 3));
  const ln = new THREE.Line(geo, new THREE.LineBasicMaterial({
    color, transparent: true, opacity: opacity == null ? 0.95 : opacity,
    depthTest: false }));
  ln.renderOrder = 6;
  return ln;
}
function selectedGraspId() {
  const sel = $("cam-obj");
  if (sel && sel.value !== "") return parseInt(sel.value);
  return graspObjs.length ? graspObjs[0].id : -1;
}
const GRASP_COLOUR = {
  magenta: 0xe23bd0, red: 0xe5534b, orange: 0xfb8c00, yellow: 0xf4c430,
  green: 0x43c463, cyan: 0x26c6da, blue: 0x4f86f7, purple: 0x9b59d0,
  white: 0xe8e8ec, grey: 0x9aa3b2, black: 0x5a606c,
};
function objColour(g) {
  return GRASP_COLOUR[g.colour] != null ? GRASP_COLOUR[g.colour] : 0x37c8ff;
}
function graspQuad(cornersMm, hex, opacity) {        // filled footprint (selected)
  const v = cornersMm.map((p) => [p[0] / 1000, p[1] / 1000, p[2] / 1000]);
  const pos = [...v[0], ...v[1], ...v[2], ...v[0], ...v[2], ...v[3]];
  const geo = new THREE.BufferGeometry();
  geo.setAttribute("position", new THREE.Float32BufferAttribute(pos, 3));
  const m = new THREE.Mesh(geo, new THREE.MeshBasicMaterial({
    color: hex, transparent: true, opacity, side: THREE.DoubleSide,
    depthWrite: false, depthTest: false }));
  m.renderOrder = 5;
  return m;
}
function makeLabel(text, hex, bright) {              // floating 3D text sprite
  const cv = document.createElement("canvas"), ctx = cv.getContext("2d");
  const font = "bold 42px Inter, system-ui, sans-serif";
  ctx.font = font;
  cv.width = Math.ceil(ctx.measureText(text).width) + 48; cv.height = 68;
  ctx.font = font; ctx.textBaseline = "middle";
  ctx.fillStyle = bright ? "rgba(10,12,16,0.88)" : "rgba(10,12,16,0.5)";
  ctx.fillRect(0, 0, cv.width, cv.height);
  ctx.fillStyle = "#" + (hex >>> 0).toString(16).padStart(6, "0");
  if (!bright) ctx.globalAlpha = 0.65;
  ctx.fillText(text, 24, cv.height / 2 + 2);
  const tex = new THREE.CanvasTexture(cv);
  tex.minFilter = THREE.LinearFilter;
  const spr = new THREE.Sprite(new THREE.SpriteMaterial({
    map: tex, transparent: true, depthTest: false, depthWrite: false }));
  spr.renderOrder = 8;
  const h = bright ? 0.05 : 0.042;
  spr.scale.set(h * cv.width / cv.height, h, 1);
  return spr;
}
function drawGraspObjs() {
  clearGrasp();
  if (!graspObjs.length) return;
  const selId = selectedGraspId();
  const grp = new THREE.Group();
  for (const g of graspObjs) {
    const on = g.id === selId;
    const col = objColour(g);
    const jaw = g.feasible ? col : 0xff8a3d;        // amber jaws if too wide
    const f = g.footprint, c = g.center_mm;
    if (on) grp.add(graspQuad(f, col, 0.16));
    grp.add(graspLine([f[0], f[1], f[2], f[3], f[0]], col, on ? 0.95 : 0.4));
    grp.add(graspLine([g.jaws[0], g.jaws[1]], jaw, on ? 1.0 : 0.5));
    grp.add(graspLine([c, [c[0], c[1], c[2] + 95]], col, on ? 0.7 : 0.3));
    const lab = makeLabel(g.label + (g.feasible ? "" : " · wide"), col, on);
    lab.position.set(c[0] / 1000, c[1] / 1000, (c[2] + 150) / 1000);
    grp.add(lab);
  }
  scene.add(grp);
  graspViz = grp;
}
async function buildGrasp() {
  const r = await fetch("/api/grasps?stride=4").then((x) => x.json()).catch(() => null);
  const info = $("cam-info"), sel = $("cam-obj");
  graspObjs = (r && r.found && r.objects) ? r.objects : [];
  if (sel) {
    const prev = sel.value;
    sel.innerHTML = "";
    for (const g of graspObjs) {
      const o = document.createElement("option");
      o.value = g.id;
      o.textContent = g.label + (g.feasible ? "" : " (wide)");
      sel.appendChild(o);
    }
    if (prev !== "" && graspObjs.some((g) => String(g.id) === prev)) sel.value = prev;
    sel.style.display = graspObjs.length ? "" : "none";
  }
  if (!graspObjs.length) {
    clearGrasp();
    if (info) info.textContent = "grasp: " + ((r && (r.reason || r.error)) || "no object");
    return false;
  }
  drawGraspObjs();
  if (info) info.textContent = graspObjs.length +
    (graspObjs.length > 1 ? " objects: " : " object: ") +
    graspObjs.map((g) => g.label).join(", ");
  return true;
}
if ($("btn-grasp")) {
  $("btn-grasp").onclick = async () => {
    if (PREVIEW || graspBusy) return;
    graspOn = !graspOn;
    $("btn-grasp").classList.toggle("on", graspOn);
    if (graspOn) {
      graspBusy = true; $("btn-grasp").classList.add("busy");
      const ok = await buildGrasp();
      $("btn-grasp").classList.remove("busy"); graspBusy = false;
      if (!ok) { graspOn = false; $("btn-grasp").classList.remove("on"); }
    } else clearGrasp();
    syncGizmo();
  };
}
if ($("cam-obj")) $("cam-obj").onchange = () => { if (graspOn) drawGraspObjs(); };


// ---- cam PiP: draggable by its bar so it doesn't cover the twin -----------
(function makePipDraggable() {
  const pip = $("cam-pip"), bar = $("cam-bar");
  if (!pip || !bar) return;
  let drag = null;
  bar.addEventListener("pointerdown", (e) => {
    if (e.target.closest("button, select, option")) return;   // leave controls alone
    if ($("viewport").classList.contains("split")) return;     // no dragging while split-docked
    const r = pip.getBoundingClientRect();
    const par = pip.offsetParent ? pip.offsetParent.getBoundingClientRect()
                                 : { left: 0, top: 0 };
    drag = { dx: e.clientX - r.left, dy: e.clientY - r.top, par };
    pip.style.right = "auto"; pip.style.bottom = "auto";
    pip.style.left = (r.left - par.left) + "px";
    pip.style.top = (r.top - par.top) + "px";
    pip.classList.add("dragging");
    try { bar.setPointerCapture(e.pointerId); } catch (_) {}
    e.preventDefault();
  });
  bar.addEventListener("pointermove", (e) => {
    if (!drag) return;
    const par = pip.offsetParent;
    const maxX = (par ? par.clientWidth : window.innerWidth) - pip.offsetWidth - 4;
    const maxY = (par ? par.clientHeight : window.innerHeight) - pip.offsetHeight - 4;
    const x = e.clientX - drag.dx - drag.par.left;
    const y = e.clientY - drag.dy - drag.par.top;
    pip.style.left = Math.max(4, Math.min(x, maxX)) + "px";
    pip.style.top = Math.max(4, Math.min(y, maxY)) + "px";
  });
  const end = (e) => {
    if (!drag) return;
    drag = null; pip.classList.remove("dragging");
    try { bar.releasePointerCapture(e.pointerId); } catch (_) {}
  };
  bar.addEventListener("pointerup", end);
  bar.addEventListener("pointercancel", end);
})();



// ---- overlay layer-tree: eye + opacity per overlay, in one place ---------
(function layersPanel() {
  const pop = $("layers-pop"), btn = $("btn-layers");
  if (!pop || !btn) return;
  const OVL = { trace: "btn-trace", dex: "btn-dex", pcl: "btn-pcl", grasp: "btn-grasp", coll: "btn-coll", force: "btn-force" };
  const refresh = () => {
    for (const r of pop.querySelectorAll(".lyr")) {
      const b = $(OVL[r.dataset.ov]);
      r.classList.toggle("on", !!(b && b.classList.contains("on")));
    }
  };
  const setOpacity = (ov, f) => {
    if (ov === "pcl" && pclCloud) { pclCloud.material.transparent = true; pclCloud.material.opacity = f; }
    if (ov === "dex" && dexCloud) { dexCloud.material.transparent = true; dexCloud.material.opacity = f; }
  };
  btn.onclick = (e) => {
    e.stopPropagation();
    const show = pop.style.display === "none" || !pop.style.display;
    pop.style.display = show ? "block" : "none";
    if (show) refresh();
  };
  for (const r of pop.querySelectorAll(".lyr")) {
    const ov = r.dataset.ov;
    const eye = r.querySelector(".eye");
    if (eye) eye.onclick = () => { const b = $(OVL[ov]); if (b) b.click(); setTimeout(refresh, 40); };
    const opa = r.querySelector(".opa");
    if (opa) opa.oninput = (e) => setOpacity(ov, e.target.value / 100);
  }
  document.addEventListener("click", (e) => {
    if (!e.target.closest("#layers-pop") && !btn.contains(e.target)) pop.style.display = "none";
  });
})();



// ---- view presets: one-click task layouts (overlay combinations) --------
(function viewPresets() {
  const setOv = (id, want) => {
    const b = $(id);
    if (b && b.classList.contains("on") !== want) b.click();
  };
  const P = {
    clean:   { "btn-trace": false, "btn-dex": false, "btn-pcl": false, "btn-grasp": false },
    pick:    { "btn-cam": true, "btn-trace": false, "btn-dex": false, "btn-pcl": false, "btn-grasp": true },
    inspect: { "btn-cam": true, "btn-trace": false, "btn-dex": false, "btn-pcl": true, "btn-grasp": false },
    reach:   { "btn-trace": false, "btn-dex": true, "btn-pcl": false, "btn-grasp": false },
  };
  for (const btn of document.querySelectorAll(".presets button[data-preset]")) {
    btn.onclick = () => { const p = P[btn.dataset.preset]; if (p) for (const id in p) setOv(id, p[id]); };
  }

  // --- saveable custom presets (localStorage) ---
  const TRACK = ["btn-cam", "btn-trace", "btn-dex", "btn-pcl", "btn-grasp"];
  const KEY = "skate.presets.v1";
  const wrap = $("saved-presets"), nameIn = $("preset-name"), saveBtn = $("preset-save");
  if (!wrap || !nameIn || !saveBtn) return;
  const opaInput = (ov) => document.querySelector('.lyr[data-ov="' + ov + '"] .opa');

  const capture = () => {
    const ov = {};
    for (const id of TRACK) { const b = $(id); if (b) ov[id] = b.classList.contains("on"); }
    const opa = {};
    for (const k of ["dex", "pcl"]) { const s = opaInput(k); if (s) opa[k] = +s.value; }
    return { ov, opa };
  };
  const apply = (s) => {
    if (s.ov) for (const id in s.ov) setOv(id, s.ov[id]);
    if (s.opa) for (const k in s.opa) {
      const inp = opaInput(k);
      if (inp) { inp.value = s.opa[k]; inp.dispatchEvent(new Event("input")); }
    }
  };
  const load = () => { try { return JSON.parse(localStorage.getItem(KEY)) || []; } catch (_) { return []; } };
  const store = (a) => { try { localStorage.setItem(KEY, JSON.stringify(a)); } catch (_) {} };

  const render = () => {
    const arr = load();
    wrap.innerHTML = "";
    arr.forEach((p, i) => {
      const chip = document.createElement("div");
      chip.className = "saved-chip";
      const go = document.createElement("button");
      go.className = "chip-apply"; go.textContent = p.name;
      go.title = "apply preset: " + p.name;
      go.setAttribute("aria-label", "Apply preset " + p.name);
      go.onclick = () => apply(p);
      const del = document.createElement("button");
      del.className = "del"; del.textContent = "×";
      del.title = "delete preset";
      del.setAttribute("aria-label", "Delete preset " + p.name);
      del.onclick = () => { const a = load(); a.splice(i, 1); store(a); render(); };
      chip.append(go, del);
      wrap.appendChild(chip);
    });
    wrap.style.display = arr.length ? "grid" : "none";
  };
  const doSave = () => {
    const name = (nameIn.value || "").trim().slice(0, 18);
    if (!name) { nameIn.focus(); return; }
    const arr = load(), st = capture(); st.name = name;
    const ex = arr.findIndex((p) => p.name === name);
    if (ex >= 0) arr[ex] = st; else arr.push(st);
    store(arr); nameIn.value = ""; render();
  };
  saveBtn.onclick = doSave;
  nameIn.onkeydown = (e) => { if (e.key === "Enter") { e.preventDefault(); doSave(); } };
  render();
})();


// ================= Isaac-style functional shell =================
// tabs that switch, a gizmo that drives the camera, Stage selection that
// drives Property, tool-rail transform modes, live RAW + CONSOLE + CONTENT.
(function shellFunctions() {
  const g = (id) => document.getElementById(id);

  // --- generic tab switching ---
  const wireTabs = (headSel) => document.querySelectorAll(headSel).forEach((hd) => {
    const scope = hd.parentElement;
    hd.querySelectorAll(".pt[data-tab]").forEach((pt) => pt.addEventListener("click", () => {
      hd.querySelectorAll(".pt").forEach((x) => x.classList.remove("act"));
      pt.classList.add("act");
      scope.querySelectorAll(".tabpane").forEach((p) => {
        p.style.display = (p.dataset.pane === pt.dataset.tab) ? "" : "none";
      });
    }));
  });
  wireTabs(".pane-hd.tabset"); wireTabs(".tl-hd");

  // --- viewport camera snap views (functional nav gizmo + view buttons) ---
  function setView(name) {
    try {
      const t = controls.target, D = 2.7;
      const p = { top: [t.x, t.y, t.z + D], front: [t.x, t.y - D, t.z],
                  side: [t.x + D, t.y, t.z], persp: [t.x + 1.9, t.y - 1.9, t.z + 0.7] }[name];
      if (!p) return;
      camera.position.set(p[0], p[1], p[2]); controls.update();
      const sub = g("view-sub"); if (sub) sub.textContent = name === "persp" ? "exo - orbit" : name + " - ortho";
      document.querySelectorAll(".hd-tools .vbtn").forEach((b) => b.classList.toggle("act", b.dataset.view === name));
    } catch (_) {}
  }
  document.querySelectorAll("[data-view]").forEach((b) => b.addEventListener("click", () => setView(b.dataset.view)));

  // --- Stage hierarchy: pick an arm node -> drive the Property arm tab ---
  document.querySelectorAll(".snode[data-group]").forEach((n) => n.addEventListener("click", () => {
    document.querySelectorAll(".snode").forEach((x) => x.classList.remove("sel"));
    n.classList.add("sel");
    const t = document.querySelector('#tabs div[data-group="' + n.dataset.group + '"]');
    if (t) t.click();
  }));
  document.querySelectorAll(".snode[data-exp] .tw").forEach((tw) => tw.addEventListener("click", (e) => {
    e.stopPropagation();
    const n = tw.closest(".snode"); n.classList.toggle("collapsed");
    tw.innerHTML = n.classList.contains("collapsed") ? "▸" : "▾";
  }));

  // --- tool rail: active tool + transform mode on the IK gizmo ---
  document.querySelectorAll("#toolrail .rail-tool[data-tool]").forEach((b) => b.addEventListener("click", () => {
    const _annot = ["measure", "marker", "obstacle"];          // mutually exclusive annotation tools
    if (_annot.includes(b.dataset.tool)) {
      document.querySelectorAll("#toolrail .rail-tool[data-tool]").forEach((x) => {
        if (x === b || !_annot.includes(x.dataset.tool) || !x.classList.contains("on")) return;
        x.classList.remove("on");
        if (x.dataset.tool === "measure") { measureOn = false; if (window.__measureClear) window.__measureClear(); }
        else if (x.dataset.tool === "marker" && window.__markerMode) window.__markerMode(false);
        else if (x.dataset.tool === "obstacle" && window.__obstacleMode) window.__obstacleMode(false);
      });
    }
    if (b.dataset.tool === "measure") {
      measureOn = !b.classList.contains("on"); b.classList.toggle("on", measureOn);
      if (measureOn) {
        const _mk = document.querySelector('#toolrail [data-tool="marker"]'); if (_mk && _mk.classList.contains("on")) { _mk.classList.remove("on"); if (window.__markerMode) window.__markerMode(false); }
        const _ob = document.querySelector('#toolrail [data-tool="obstacle"]'); if (_ob && _ob.classList.contains("on")) { _ob.classList.remove("on"); if (window.__obstacleMode) window.__obstacleMode(false); }
      }
      if (!measureOn && window.__measureClear) window.__measureClear();
      const _h = document.getElementById("overlay-hint");
      if (_h) { _h.style.display = measureOn ? "block" : "none"; if (measureOn) _h.textContent = "Measure: click two points on the robot"; }
      return;
    }
    if (b.dataset.tool === "marker") {
      const on = !b.classList.contains("on"); b.classList.toggle("on", on);
      if (on) {
        const _ob = document.querySelector('#toolrail [data-tool="obstacle"]'); if (_ob && _ob.classList.contains("on")) { _ob.classList.remove("on"); if (window.__obstacleMode) window.__obstacleMode(false); }
        const _ms = document.querySelector('#toolrail [data-tool="measure"]'); if (_ms && _ms.classList.contains("on")) { _ms.classList.remove("on"); measureOn = false; if (window.__measureClear) window.__measureClear(); }
      }
      if (window.__markerMode) window.__markerMode(on);
      const _h = document.getElementById("overlay-hint");
      if (_h) { _h.style.display = on ? "block" : "none"; if (on) _h.textContent = "Annotate: + point spawns a target, drag its gizmo to a reachable spot, then L/R sends an arm · Esc clears"; }
      return;
    }
    if (b.dataset.tool === "obstacle") {
      const on = !b.classList.contains("on"); b.classList.toggle("on", on);
      if (on) {
        const _mk = document.querySelector('#toolrail [data-tool="marker"]'); if (_mk && _mk.classList.contains("on")) { _mk.classList.remove("on"); if (window.__markerMode) window.__markerMode(false); }
        const _ms = document.querySelector('#toolrail [data-tool="measure"]'); if (_ms && _ms.classList.contains("on")) { _ms.classList.remove("on"); measureOn = false; if (window.__measureClear) window.__measureClear(); }
      }
      if (window.__obstacleMode) window.__obstacleMode(on);
      const _h = document.getElementById("overlay-hint");
      if (_h) { _h.style.display = on ? "block" : "none"; if (on) _h.textContent = "Obstacle: + box to add, then click a box and drag its gizmo to place it"; }
      return;
    }
    if (b.dataset.tool === "snap") {
      const on = !b.classList.contains("on"); b.classList.toggle("on", on);
      try { if (typeof tc !== "undefined" && tc.setTranslationSnap) {
        tc.setTranslationSnap(on ? 0.01 : null);
        tc.setRotationSnap(on ? THREE.MathUtils.degToRad(15) : null);
        tc.setScaleSnap(on ? 0.1 : null);
      } } catch (_) {}
      const _gh = document.getElementById("giz-hud");
      if (_gh) { _gh.style.display = "block"; _gh.textContent = on ? "SNAP ON · 10 mm / 15°" : "SNAP OFF";
        setTimeout(() => { if (!draggingArm) _gh.style.display = "none"; }, 1100); }
      return;
    }
    { // switching to a transform tool clears the annotation tools so they don't linger
      const _mk = document.querySelector('#toolrail [data-tool="marker"]');
      if (_mk && _mk.classList.contains("on")) { _mk.classList.remove("on"); if (window.__markerMode) window.__markerMode(false); }
      const _ms = document.querySelector('#toolrail [data-tool="measure"]');
      if (_ms && _ms.classList.contains("on")) { _ms.classList.remove("on"); measureOn = false; if (window.__measureClear) window.__measureClear(); }
      const _ob = document.querySelector('#toolrail [data-tool="obstacle"]');
      if (_ob && _ob.classList.contains("on")) { _ob.classList.remove("on"); if (window.__obstacleMode) window.__obstacleMode(false); }
      const _hh = document.getElementById("overlay-hint"); if (_hh) _hh.style.display = "none";
    }
    document.querySelectorAll("#toolrail .rail-tool[data-tool]").forEach((x) => x.classList.remove("act"));
    b.classList.add("act");
    const m = b.dataset.tool;
    try { if (typeof tc !== "undefined" && tc.setMode && (m === "translate" || m === "rotate" || m === "scale")) tc.setMode(m); } catch (_) {}
  }));
  if (g("btn-resume2")) g("btn-resume2").addEventListener("click", () => { if (state && state.estop) g("btn-estop").click(); });
  if (g("btn-estop2")) g("btn-estop2").addEventListener("click", () => { if (state && !state.estop) g("btn-estop").click(); });

  // --- CONTENT browser: preset cards apply on click ---
  const grid = g("content-grid");
  if (grid) {
    const card = (label, onClick) => {
      const c = document.createElement("button"); c.className = "asset";
      c.innerHTML = '<span class="thumb"></span><span class="al"></span>';
      c.querySelector(".al").textContent = label; c.onclick = onClick; return c;
    };
    [["clean", "Clean"], ["pick", "Pick"], ["inspect", "Inspect"], ["reach", "Reach"]].forEach(([p, l]) =>
      grid.appendChild(card(l, () => { const b = document.querySelector('.presets button[data-preset="' + p + '"]'); if (b) b.click(); })));
    try {
      (JSON.parse(localStorage.getItem("skate.presets.v1")) || []).forEach((pr) =>
        grid.appendChild(card(pr.name, () => {
          const chip = [...document.querySelectorAll("#saved-presets .saved-chip .chip-apply")].find((x) => x.textContent === pr.name);
          if (chip) chip.click();
        })));
    } catch (_) {}
  }

  // --- live RAW readout + CONSOLE event log, polled from telemetry ---
  const raw = g("raw-readout"), clog = g("console-log");
  let prev = {};
  const log = (msg, cls) => {
    if (!clog) return;
    const ln = document.createElement("div"); ln.className = "cl " + (cls || "");
    ln.textContent = "[" + new Date().toLocaleTimeString() + "] " + msg;
    clog.appendChild(ln);
    while (clog.children.length > 200) clog.removeChild(clog.firstChild);
    clog.scrollTop = clog.scrollHeight;
  };
  log("Skate Commander ready - shell initialised", "ok");
  setInterval(() => {
    if (typeof state === "undefined" || !state) return;
    if (raw && raw.parentElement && raw.parentElement.style.display !== "none") {
      const q = state.q || state.targ || [];
      let s = "JOINT ANGLES (deg)\n";
      for (let i = 0; i < q.length; i++) s += " q" + String(i).padStart(2, "0") + "  " + (q[i] * 180 / Math.PI).toFixed(2).padStart(9) + "\n";
      if (state.temps) s += "\nTEMPS  " + state.temps.map((t) => t.toFixed(0) + "C").join("  ") + "\n";
      s += "\nMODE " + state.mode + "   " + (state.estop ? "E-STOP" : state.live ? "LIVE" : "DAMPENED");
      raw.textContent = s;
    }
    const cur = { estop: state.estop, live: state.live, homing: !!(state.homing || state.routing),
      contact: !!(state.contact && state.contact.tripped), mode: state.mode, connected: state.connected };
    if (prev.estop !== undefined) {
      if (cur.estop !== prev.estop) log(cur.estop ? "E-STOP engaged - motion dampened" : "RESUME - motion enabled", cur.estop ? "bad" : "ok");
      else if (cur.live !== prev.live) log(cur.live ? "Motion enabled (LIVE)" : "Motion dampened", "");
      if (cur.homing !== prev.homing) log(cur.homing ? "HOME - gliding to safe pose" : "HOME - reached", "");
      if (cur.contact && !prev.contact) log("CONTACT reflex tripped - arm dampened", "bad");
      if (cur.mode !== prev.mode) log("Mode -> " + cur.mode, "");
      if (cur.connected !== prev.connected) log(cur.connected ? "UDP link up" : "UDP link lost", cur.connected ? "ok" : "bad");
    }
    prev = cur;
  }, 250);
})();


// ================= Isaac functional add-ons =================
// visibility eyes, click-to-select in 3D, editable joint values, resizable panels.
(function shellInteractions() {
  const g = (id) => document.getElementById(id);

  // --- 1) STAGE visibility eyes actually show/hide robot, grid, axes ---
  const setVis = (which, on) => {
    try {
      if ((which === "robot" || which === "world") && typeof robotRoots !== "undefined")
        robotRoots.forEach((r) => { r.visible = on; });
      if (which === "grid" || which === "world") {
        if (typeof grid !== "undefined") grid.visible = on;
        if (typeof axes !== "undefined") axes.visible = on;
      }
    } catch (_) {}
  };
  document.querySelectorAll(".stage-tree .snode .eye").forEach((eye) => {
    const node = eye.closest(".snode");
    const nm = (node.querySelector(".nm").textContent || "").toLowerCase();
    const which = nm.includes("world") ? "world" : nm.includes("grid") ? "grid" : "robot";
    eye.addEventListener("click", (e) => {
      e.stopPropagation();
      const on = !eye.classList.contains("on");
      eye.classList.toggle("on", on);
      setVis(which, on);
    });
  });

  // --- 2) click a robot part in the viewport -> select its arm in Stage + Property ---
  const armOf = (i) => i >= 8 && i <= 15 ? "left" : i >= 16 && i <= 23 ? "right" : i >= 24 ? "head" : "legs";
  const selectArm = (group) => { const sn = document.querySelector('.snode[data-group="' + group + '"]'); if (sn) sn.click(); };
  try {
    const ray2 = new THREE.Raycaster();
    let dn = null;
    renderer.domElement.addEventListener("pointerdown", (e) => { dn = [e.clientX, e.clientY]; });
    renderer.domElement.addEventListener("pointerup", (e) => {
      if (!dn) return;
      const moved = Math.hypot(e.clientX - dn[0], e.clientY - dn[1]); dn = null;
      if (moved > 5) return;                                  // was an orbit-drag, not a click
      const rect = renderer.domElement.getBoundingClientRect();
      const m = new THREE.Vector2(((e.clientX - rect.left) / rect.width) * 2 - 1,
                                  -((e.clientY - rect.top) / rect.height) * 2 + 1);
      ray2.setFromCamera(m, camera);
      const meshes = [];
      (typeof robotRoots !== "undefined" ? robotRoots : []).forEach((r) => r.traverse((o) => { if (o.isMesh) meshes.push(o); }));
      const hit = ray2.intersectObjects(meshes, false)[0];
      if (!hit) return;
      const anc = new Set(); let p = hit.object; while (p) { anc.add(p); p = p.parent; }
      let best = -1;
      for (const [i, j] of Object.entries(jointGroups)) if (anc.has(j.grp) && +i > best) best = +i;
      if (best >= 0) selectArm(armOf(best));
    });
  } catch (_) {}

  // --- 3) editable joint angles: click the value, type degrees, Enter to send ---
  let editor = null;
  const closeEditor = () => { if (editor) { editor.remove(); editor = null; } };
  document.addEventListener("click", (e) => {
    const ang = e.target.closest ? e.target.closest("#joints .ang.editable") : null;
    if (!ang) { if (editor && e.target !== editor) closeEditor(); return; }
    closeEditor();
    const idx = +ang.dataset.idx, lo = +ang.dataset.lo, hi = +ang.dataset.hi;
    const cur = parseFloat((ang.textContent || "0").replace(/[^0-9.\-]/g, "")) || 0;
    const r = ang.getBoundingClientRect();
    editor = document.createElement("input"); editor.className = "ang-edit"; editor.value = cur.toFixed(1);
    editor.style.left = r.left + "px"; editor.style.top = r.top + "px"; editor.style.width = Math.max(56, r.width + 8) + "px";
    document.body.appendChild(editor); editor.focus(); editor.select();
    const commit = (ok) => {
      if (ok) { const d = parseFloat(editor.value);
        if (!isNaN(d)) { let v = d * Math.PI / 180; v = Math.min(hi, Math.max(lo, v)); send({ type: "set_joint", idx, value: v }); } }
      closeEditor();
    };
    editor.addEventListener("keydown", (ev) => { if (ev.key === "Enter") commit(true); else if (ev.key === "Escape") commit(false); ev.stopPropagation(); });
    editor.addEventListener("blur", () => commit(true));
  });

  // --- 4) resizable panels (Isaac-style grab handles) ---
  const makeResizer = (handle, onDrag) => handle.addEventListener("pointerdown", (e) => {
    e.preventDefault(); handle.setPointerCapture(e.pointerId);
    const move = (ev) => onDrag(ev);
    const up = () => { try { handle.releasePointerCapture(e.pointerId); } catch (_) {}
      handle.removeEventListener("pointermove", move); handle.removeEventListener("pointerup", up); try { resize(); } catch (_) {} };
    handle.addEventListener("pointermove", move); handle.addEventListener("pointerup", up);
  });
  const rd = g("rightdock"), main = g("main");
  if (rd && main) {
    const h = document.createElement("div"); h.className = "rsz rsz-v"; rd.appendChild(h);
    makeResizer(h, (ev) => {
      const w = Math.min(540, Math.max(240, main.getBoundingClientRect().right - ev.clientX));
      main.style.gridTemplateColumns = "48px 1fr " + w + "px"; try { resize(); } catch (_) {}
    });
  }
  const sp = g("stage-pane");
  if (sp) {
    const h = document.createElement("div"); h.className = "rsz rsz-h"; sp.appendChild(h);
    makeResizer(h, (ev) => {
      const rect = g("rightdock").getBoundingClientRect();
      const pct = Math.min(75, Math.max(20, ((ev.clientY - rect.top) / rect.height) * 100));
      sp.style.flex = "0 0 " + pct + "%";
    });
  }
})();


// ================= Isaac refinements: menus, expandable joints, live gizmo, quality =================
(function shellRefinements() {
  const g = (id) => document.getElementById(id);
  const click = (id) => { const e = g(id); if (e) e.click(); };
  const showTab = (name) => { const pt = document.querySelector('.pt[data-tab="' + name + '"]'); if (pt) pt.click(); };
  function setLayout(name) {
    const m = g("main"), sp = g("stage-pane");
    if (name === "wide") m.style.gridTemplateColumns = "48px 1fr 276px";
    else if (name === "tallstage") { if (sp) sp.style.flex = "0 0 62%"; }
    else { m.style.gridTemplateColumns = "48px 1fr 330px"; if (sp) sp.style.flex = "0 0 40%"; }
    try { resize(); } catch (_) {}
  }

  // ---- 1) working menu-bar dropdowns ----
  const MENUS = {
    File: [["Save scene", () => window.__sceneSave && window.__sceneSave()], ["Load scene", () => window.__sceneLoad && window.__sceneLoad()], ["Reload cockpit", () => location.reload()], ["Keyboard shortcuts...", () => click("btn-keys")]],
    Edit: [["Home (safe pose)", () => click("btn-home")], ["Clear traces", () => click("btn-clear-trace")],
           ["Reset contact reflex", () => click("chip-contact")]],
    Create: [["Camera tools — under development", () => {}]],
    Window: ["__panels__"],
    Tools: [["Mirror arms", () => click("btn-mirror")], ["Dual-arm carry", () => click("btn-carry")],
            ["Dexterity map", () => click("btn-dex")]],
    Utilities: [["Console", () => showTab("console")], ["Raw values", () => showTab("raw")]],
    Layout: [["Default", () => setLayout("default")], ["Wide viewport", () => setLayout("wide")], ["Tall Stage", () => setLayout("tallstage")]],
    Help: [["Keyboard shortcuts", () => click("btn-keys")],
           ["About", () => alert("Skate Commander - web cockpit for the R.Botic Skate humanoid work-cell.")]],
  };
  let openMenu = null;
  const closeMenus = () => { document.querySelectorAll(".menu-dd").forEach((d) => d.remove());
    document.querySelectorAll("#menus .menu").forEach((m) => m.classList.remove("open")); openMenu = null; };
  document.querySelectorAll("#menus .menu").forEach((mEl) => mEl.addEventListener("click", (e) => {
    e.stopPropagation();
    const name = mEl.textContent.trim();
    if (openMenu === name) { closeMenus(); return; }
    closeMenus();
    const items = MENUS[name]; if (!items) return;
    openMenu = name; mEl.classList.add("open");
    const dd = document.createElement("div"); dd.className = "menu-dd";
    const r = mEl.getBoundingClientRect(); dd.style.left = r.left + "px"; dd.style.top = (r.bottom + 2) + "px";
    if (items[0] === "__panels__") {
      [["Tool rail", "toolrail"], ["Stage", "stage-pane"], ["Property", "panel"], ["Timeline", "timeline"]].forEach(([lab, id]) => {
        const el = g(id); const it = document.createElement("button"); it.className = "ddi";
        const vis = el && el.style.display !== "none";
        it.innerHTML = '<span class="ck">' + (vis ? "✓" : "") + "</span>" + lab;
        it.onclick = () => { if (el) el.style.display = el.style.display === "none" ? "" : "none"; try { resize(); } catch (_) {} closeMenus(); };
        dd.appendChild(it);
      });
    } else items.forEach(([lab, fn]) => {
      const it = document.createElement("button"); it.className = "ddi"; it.textContent = lab;
      it.onclick = () => { try { fn(); } catch (_) {} closeMenus(); }; dd.appendChild(it);
    });
    document.body.appendChild(dd);
  }));
  document.addEventListener("click", () => closeMenus());

  // ---- 2) expandable joints in the Stage tree ----
  const ARMN = { left: 8, right: 8, head: 2, legs: 8 };
  document.querySelectorAll(".snode[data-group]").forEach((node) => {
    const grp = node.dataset.group, n = ARMN[grp] || 0;
    const tw = node.querySelector(".tw"); if (!tw) return;
    tw.innerHTML = "&#9656;"; tw.style.cursor = "pointer";
    tw.addEventListener("click", (e) => {
      e.stopPropagation();
      let kids = node.nextElementSibling;
      if (kids && kids.classList.contains("jkids")) { kids.remove(); tw.innerHTML = "&#9656;"; return; }
      kids = document.createElement("div"); kids.className = "jkids";
      for (let i = 1; i <= n; i++) {
        const j = document.createElement("div"); j.className = "snode lv3 jchild";
        j.innerHTML = '<span class="tw"></span><span class="nm">J' + i + '</span><em class="ty">Revolute</em>';
        j.onclick = () => {
          node.click();
          const ji = i - 1;
          setTimeout(() => { const rows = g("joints").children; if (rows[ji]) { rows[ji].scrollIntoView({ block: "center" }); rows[ji].classList.add("flash"); setTimeout(() => rows[ji].classList.remove("flash"), 700); } }, 130);
        };
        kids.appendChild(j);
      }
      node.after(kids); tw.innerHTML = "&#9662;";
    });
  });

  // ---- 3) live-rotating nav gizmo (tracks camera orientation) ----
  const gz = { x: document.querySelector("#nav-gizmo .gz-x"), y: document.querySelector("#nav-gizmo .gz-y"), z: document.querySelector("#nav-gizmo .gz-z") };
  const av = { x: new THREE.Vector3(1, 0, 0), y: new THREE.Vector3(0, 1, 0), z: new THREE.Vector3(0, 0, 1) };
  function updateGizmo() {
    try {
      const q = camera.quaternion.clone().invert(); const C = 28, R = 17;
      for (const k of ["x", "y", "z"]) {
        const el = gz[k]; if (!el) continue;
        const d = av[k].clone().applyQuaternion(q);
        el.style.right = "auto";
        el.style.left = (C + d.x * R - 8.5) + "px";
        el.style.top = (C - d.y * R - 8.5) + "px";
        el.style.opacity = (0.5 + 0.5 * (d.z + 1) / 2).toFixed(2);
        el.style.zIndex = d.z > 0 ? 6 : 4;
      }
    } catch (_) {}
  }
  setInterval(updateGizmo, 50);
  updateGizmo();

  // ---- 4) render-quality toggle in the viewport toolbar ----
  const rtx = document.querySelector(".hd-render");
  if (rtx) {
    let hi = true; rtx.style.cursor = "pointer"; rtx.title = "Toggle render quality";
    rtx.addEventListener("click", () => {
      hi = !hi;
      try { renderer.setPixelRatio(hi ? Math.min(2, window.devicePixelRatio || 1) : 1); resize(); } catch (_) {}
      rtx.textContent = hi ? "RTX" : "LOW"; rtx.classList.toggle("lo", !hi);
    });
  }
})();


/* ============================================================
   LIVE TELEMETRY PLOTS  (Foxglove / PlotJuggler-style strip charts)
   Self-contained: samples the global `state` at 30 Hz into a ring
   buffer and draws scrolling line charts in the TIMELINE pane.
   setInterval (not rAF) so it keeps sampling in a backgrounded tab.
   ============================================================ */
(function setupPlots() {
  const cv = document.getElementById("plot-canvas");
  if (!cv || !cv.getContext) return;
  const ctx = cv.getContext("2d");
  const legendEl = document.getElementById("plot-legend");
  const metricEl = document.getElementById("plot-metric");
  const pauseBtn = document.getElementById("plot-pause");
  const winEl = document.getElementById("plot-win");

  const PAL = ["#3B82F6", "#F5A623", "#3FB950", "#A78BFA", "#22D3EE", "#FF6981", "#E3B341", "#7A95FF"];
  const AXC = { x: "#FF6981", y: "#3FB950", z: "#7A95FF" };
  const WINDOW_S = 15, HZ = 30, CAP = WINDOW_S * HZ + 90;
  const buf = [];
  const _v = new THREE.Vector3();
  let paused = false, pauseT = 0, metric = "angle";
  let hidden = {}, lastGroup = null, lastMetric = null, lastN = -1;

  function lineDefs() {
    if (metric === "tcp")
      return [{ k: "x", label: "X", c: AXC.x }, { k: "y", label: "Y", c: AXC.y }, { k: "z", label: "Z", c: AXC.z }];
    if (metric === "rtt")
      return [{ k: "rtt", label: "RTT", c: "#3B82F6" }, { k: "kbps", label: "KB/s", c: "#F5A623" }];
    const idx = (GROUPS[curGroup] && GROUPS[curGroup].idx) || [];
    return idx.map((ji, n) => ({ k: ji, label: "J" + (n + 1), c: PAL[n % PAL.length] }));
  }

  function valOf(s, d) {
    if (metric === "tcp") return s.tcp ? s.tcp[d.k] : null;
    if (metric === "rtt") return s[d.k];
    const arr = metric === "vel" ? s.dq : metric === "temp" ? s.temps : s.q;
    if (!arr) return null;
    let v = arr[d.k];
    if (v == null) return null;
    if (metric === "angle") v = v * 180 / Math.PI;
    return v;
  }

  function sample() {
    if (!state) return;
    const s = {
      t: performance.now(),
      q: state.q ? state.q.slice() : null,
      dq: state.dq ? state.dq.slice() : null,
      temps: state.temps ? state.temps.slice() : null,
      rtt: rttMs || 0, kbps: kbps || 0, tcp: null
    };
    try { const p = tcpWorld(curGroup, _v); if (p) s.tcp = { x: p.x * 1000, y: p.y * 1000, z: p.z * 1000 }; } catch (_) {}
    buf.push(s);
    while (buf.length > CAP) buf.shift();
  }

  function buildLegend(defs) {
    hidden = {};
    legendEl.innerHTML = "";
    defs.forEach((d, i) => {
      const chip = document.createElement("button");
      chip.className = "plg";
      chip.innerHTML = '<i style="background:' + d.c + '"></i><b>' + d.label + '</b><em></em>';
      chip.addEventListener("click", () => { hidden[i] = !hidden[i]; chip.classList.toggle("off", !!hidden[i]); });
      legendEl.appendChild(chip);
    });
  }

  function fit() {
    const w = cv.clientWidth, h = cv.clientHeight, dpr = window.devicePixelRatio || 1;
    if (cv.width !== Math.round(w * dpr) || cv.height !== Math.round(h * dpr)) {
      cv.width = Math.round(w * dpr); cv.height = Math.round(h * dpr);
    }
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return { w, h };
  }

  function draw() {
    if (cv.offsetParent === null) return;            // TIMELINE pane hidden
    const { w, h } = fit();
    const defs = lineDefs();
    if (curGroup !== lastGroup || metric !== lastMetric || defs.length !== lastN) {
      buildLegend(defs); lastGroup = curGroup; lastMetric = metric; lastN = defs.length;
    }
    ctx.clearRect(0, 0, w, h);
    ctx.fillStyle = "#08090B"; ctx.fillRect(0, 0, w, h);
    if (!defs.length) { ctx.fillStyle = "#5A616B"; ctx.font = "11px ui-monospace, monospace"; ctx.textAlign = "center"; ctx.fillText("no joint signals — pick an arm tab (LEFT / RIGHT / HEAD / LEGS)", w / 2, h / 2); ctx.textAlign = "left"; return; }
    { const U = { angle: "deg", vel: "rad/s", temp: "°C", tcp: "mm", rtt: "ms" }[metric]; if (U) { ctx.fillStyle = "#5A616B"; ctx.font = "9px ui-monospace, monospace"; ctx.textAlign = "right"; ctx.fillText(U, w - 6, 11); ctx.textAlign = "left"; } }
    const x0 = 40, x1 = w - 8, y0 = 8, y1 = h - 14;
    const now = paused ? pauseT : performance.now(), t0 = now - WINDOW_S * 1000;
    const xFor = t => x1 - (now - t) / 1000 / WINDOW_S * (x1 - x0);

    let mn = Infinity, mx = -Infinity;
    for (const s of buf) {
      if (s.t < t0) continue;
      defs.forEach((d, i) => {
        if (hidden[i]) return;
        const v = valOf(s, d);
        if (v == null || !isFinite(v)) return;
        if (v < mn) mn = v; if (v > mx) mx = v;
      });
    }
    if (!isFinite(mn)) { mn = -1; mx = 1; }
    if (mx - mn < 1e-6) { mn -= 1; mx += 1; }
    const pd = (mx - mn) * 0.12; mn -= pd; mx += pd;
    const yFor = v => y1 - (v - mn) / (mx - mn) * (y1 - y0);

    ctx.lineWidth = 1; ctx.font = "9px ui-monospace, monospace"; ctx.textBaseline = "middle";
    for (let g = 0; g <= 4; g++) {
      const yy = y0 + (y1 - y0) * g / 4, val = mx - (mx - mn) * g / 4;
      ctx.strokeStyle = "#2A2F37"; ctx.globalAlpha = .55; ctx.beginPath(); ctx.moveTo(x0, yy); ctx.lineTo(x1, yy); ctx.stroke();
      ctx.globalAlpha = 1; ctx.fillStyle = "#8B929C"; ctx.fillText(val.toFixed(1), 3, yy);
    }
    ctx.textBaseline = "alphabetic";
    for (let ts = 5; ts < WINDOW_S; ts += 5) {
      const xx = xFor(now - ts * 1000);
      ctx.strokeStyle = "#2A2F37"; ctx.globalAlpha = .35; ctx.beginPath(); ctx.moveTo(xx, y0); ctx.lineTo(xx, y1); ctx.stroke();
      ctx.globalAlpha = .8; ctx.fillStyle = "#5A616B"; ctx.fillText("-" + ts + "s", xx - 8, h - 3);
    }
    ctx.globalAlpha = 1;

    ctx.lineWidth = 1.5; ctx.lineJoin = "round";
    defs.forEach((d, i) => {
      if (hidden[i]) return;
      ctx.strokeStyle = d.c; ctx.beginPath();
      let started = false, lastV = null, lastX = 0, lastY = 0;
      for (const s of buf) {
        if (s.t < t0) continue;
        const v = valOf(s, d);
        if (v == null || !isFinite(v)) { started = false; continue; }
        const X = xFor(s.t), Y = yFor(v);
        if (!started) { ctx.moveTo(X, Y); started = true; } else ctx.lineTo(X, Y);
        lastV = v; lastX = X; lastY = Y;
      }
      ctx.stroke();
      if (started) { ctx.fillStyle = d.c; ctx.beginPath(); ctx.arc(lastX, lastY, 2.6, 0, 6.283); ctx.fill(); }
      const chip = legendEl.children[i];
      if (chip) { const em = chip.querySelector("em"); if (em) em.textContent = lastV == null ? "—" : lastV.toFixed(metric === "vel" ? 2 : 1); }
    });
    if (window.__trajScrub != null) {                  // trajectory-replay playhead at the scrubbed time
      const px = xFor(window.__trajScrub);
      if (px >= x0 && px <= x1) {
        ctx.strokeStyle = "#F5A623"; ctx.lineWidth = 1.5; ctx.globalAlpha = .9;
        ctx.beginPath(); ctx.moveTo(px, y0); ctx.lineTo(px, y1); ctx.stroke(); ctx.globalAlpha = 1;
      }
    }
  }

  setInterval(() => { if (!paused) sample(); draw(); }, Math.round(1000 / HZ));

  if (metricEl) metricEl.addEventListener("change", () => { metric = metricEl.value; lastN = -1; });
  if (pauseBtn) pauseBtn.addEventListener("click", () => {
    paused = !paused; if (paused) pauseT = performance.now();
    pauseBtn.classList.toggle("on", paused);
    pauseBtn.innerHTML = paused ? "▶ Resume" : "❚❚ Pause";
  });
  if (winEl) winEl.textContent = WINDOW_S + "s";
  { const xb = document.getElementById("plot-export");
    if (xb) xb.onclick = () => {                        // current plot signal → CSV
      const defs = lineDefs();
      if (!buf.length || !defs.length) return;
      const t0 = buf[0].t, NL = String.fromCharCode(10);
      const U = { angle: "deg", vel: "radps", temp: "C", tcp: "mm", rtt: "" }[metric] || "";
      const rows = ["t_s," + defs.map((d) => d.label + (U ? "_" + U : "")).join(",")];
      for (const s of buf) {
        const row = [((s.t - t0) / 1000).toFixed(3)];
        for (const d of defs) { const v = valOf(s, d); row.push(v == null ? "" : (+v).toFixed(3)); }
        rows.push(row.join(","));
      }
      window.csvDownload("skate_" + metric + ".csv", rows.join(NL) + NL);
    };
  }
})();


/* ============================================================
   F2: LIVE TF FRAME TREE  (RViz2-style)
   Parents token-coloured AxesHelper triads to base + each flange so
   they track the kinematics automatically; the FRAMES tab toggles
   them and shows each frame's live world position (mm).
   ============================================================ */
(function setupFrames() {
  const frEls = {
    base: document.querySelector('[data-xyz="base"]'),
    left: document.querySelector('[data-xyz="left"]'),
    right: document.querySelector('[data-xyz="right"]'),
  };
  if (!frEls.base) return;
  const triads = {};
  const _p = new THREE.Vector3();
  const TOK = [0xFF6981, 0x3FB950, 0x7A95FF];
  function mkTriad(parent, size, name) {
    const a = new THREE.AxesHelper(size);
    if (a.setColors) a.setColors(new THREE.Color(TOK[0]), new THREE.Color(TOK[1]), new THREE.Color(TOK[2]));
    a.material.transparent = true; a.material.opacity = 0.95;
    a.material.depthTest = false; a.renderOrder = 6;
    a.visible = false; parent.add(a);
    if (name && typeof makeLabel === "function") { const lab = makeLabel(name, 0xD6DAE0, true); lab.position.set(size * 0.6, size * 0.6, size * 0.8); a.add(lab); }
    return a;
  }
  function ensureTriads() {
    if (!triads.base && robotRoots[0]) triads.base = mkTriad(robotRoots[0], 0.14, "base_link");
    if (!triads.left && eeObjs.left) triads.left = mkTriad(eeObjs.left, 0.11, "armL_flange");
    if (!triads.right && eeObjs.right) triads.right = mkTriad(eeObjs.right, 0.11, "armR_flange");
  }
  document.querySelectorAll(".frames-list .fr").forEach((fr) => {
    const key = fr.dataset.frame, eye = fr.querySelector(".eye");
    if (!eye) return;
    eye.addEventListener("click", () => {
      ensureTriads();
      const on = !eye.classList.contains("on");
      eye.classList.toggle("on", on);
      if (key === "world") { try { axes.visible = on; } catch (_) {} }
      else if (triads[key]) triads[key].visible = on;
    });
  });
  const visible = () => { const p = document.querySelector('.tabpane[data-pane="frames"]'); return p && p.offsetParent !== null; };
  const objOf = { base: () => robotRoots[0], left: () => eeObjs.left, right: () => eeObjs.right };
  setInterval(() => {
    if (!visible()) return;
    ensureTriads();
    for (const key of ["base", "left", "right"]) {
      const o = objOf[key](), el = frEls[key];
      if (!o || !el) continue;
      o.getWorldPosition(_p);
      el.textContent = (_p.x * 1000).toFixed(0) + " · " + (_p.y * 1000).toFixed(0) + " · " + (_p.z * 1000).toFixed(0);
    }
  }, 120);
})();


/* ============================================================
   F4: GLOBAL SPEED OVERRIDE  (teach-pendant velocity scaling)
   Sends {type:"speed", scale} on slider input; the bridge scales jog
   + glide cruise speeds. Reflects state.speed_scale when not dragging
   (so multiple clients stay in sync).
   ============================================================ */
(function setupSpeed() {
  const sl = document.getElementById("speed-slider"), v = document.getElementById("speed-val");
  if (!sl) return;
  let dragging = false;
  const apply = () => { v.textContent = sl.value + "%"; send({ type: "speed", scale: (+sl.value) / 100 }); };
  sl.addEventListener("input", () => { dragging = true; apply(); });
  sl.addEventListener("change", () => { dragging = false; });
  setInterval(() => {
    if (dragging || !state || state.speed_scale == null) return;
    const pct = Math.round(state.speed_scale * 100);
    if (+sl.value !== pct) { sl.value = pct; v.textContent = pct + "%"; }
  }, 400);
})();


/* ============================================================
   I1: ISAAC SIM-TRANSPORT BAR
   Play/Pause toggle (resume + un/pause), Stop (E-stop), Step (one
   autonomous tick while paused), Reset (home). A run-time clock that
   counts only while playing. Wires to existing + new server commands.
   ============================================================ */
(function setupTransport() {
  const play = $("tp-play"), stop = $("tp-stop"), step = $("tp-step"),
        reset = $("tp-reset"), tEl = $("tp-time"), bar = $("transport");
  if (!play) return;
  let simMs = 0, lastT = performance.now();
  play.addEventListener("click", () => {
    const running = state && !state.estop && !state.paused;
    if (running) { send({ type: "pause", on: true }); }
    else { if (state && state.estop) send({ type: "resume" }); send({ type: "pause", on: false }); }
  });
  stop.addEventListener("click", () => send({ type: "estop" }));
  step.addEventListener("click", () => send({ type: "step", n: 1 }));
  reset.addEventListener("click", () => { const h = $("btn-home"); if (h) h.click(); });
  setInterval(() => {
    const now = performance.now(), dt = now - lastT; lastT = now;
    const running = state && !state.estop && !state.paused;
    if (running) simMs += dt;
    play.innerHTML = running ? "&#10074;&#10074;" : "&#9654;";
    play.title = running ? "Pause" : "Play / Resume";
    if (bar) bar.classList.toggle("paused", !!(state && state.paused));
    const s = Math.floor(simMs / 1000);
    tEl.textContent = Math.floor(s / 60) + ":" + String(s % 60).padStart(2, "0");
  }, 250);
})();


/* ============================================================
   I2: MEASURE TOOL  (Isaac Utilities) — click two points on the robot,
   draw an amber line + the distance (mm) at the midpoint. Capture-phase
   pointerdown so a hit overrides the gizmo, but empty-space drags still orbit.
   ============================================================ */
(function setupMeasure() {
  if (typeof renderer === "undefined" || !renderer || !renderer.domElement) return;
  const ray = new THREE.Raycaster();
  let pts = [], line = null, label = null, dots = [];
  function robotMeshes() { const m = []; (robotRoots || []).forEach(r => r.traverse(o => { if (o.isMesh) m.push(o); })); return m; }
  function clear() {
    if (line) { scene.remove(line); line.geometry.dispose(); line = null; }
    if (label) { scene.remove(label); label = null; }
    dots.forEach(d => scene.remove(d)); dots = [];
    pts = [];
  }
  window.__measureClear = clear;
  function dot(p) {
    const d = new THREE.Mesh(new THREE.SphereGeometry(0.012, 12, 12),
      new THREE.MeshBasicMaterial({ color: 0xF5A623, depthTest: false }));
    d.renderOrder = 8; d.position.copy(p); scene.add(d); dots.push(d);
  }
  let _mdn = null;                                                // click-vs-drag: a drag orbits, a clean click measures
  renderer.domElement.addEventListener("pointerdown", (e) => {
    if (!measureOn || e.button !== 0) { _mdn = null; return; }
    _mdn = { x: e.clientX, y: e.clientY };
  });
  renderer.domElement.addEventListener("pointerup", (e) => {
    if (!measureOn || e.button !== 0 || !_mdn) return;
    const moved = Math.hypot(e.clientX - _mdn.x, e.clientY - _mdn.y);
    _mdn = null;
    if (moved > 5) return;                                        // a drag -> orbited the view, don't place a point
    const r = renderer.domElement.getBoundingClientRect();
    const ndc = new THREE.Vector2(((e.clientX - r.left) / r.width) * 2 - 1,
                                  -((e.clientY - r.top) / r.height) * 2 + 1);
    ray.setFromCamera(ndc, camera);
    const hit = ray.intersectObjects(robotMeshes(), false)[0];
    if (!hit) return;                                             // click missed the robot -> nothing to measure
    if (pts.length >= 2) clear();                                 // start a fresh measurement
    pts.push(hit.point.clone()); dot(hit.point);
    if (pts.length === 2) {
      line = new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts),
        new THREE.LineBasicMaterial({ color: 0xF5A623, depthTest: false }));
      line.renderOrder = 7; scene.add(line);
      const dist = pts[0].distanceTo(pts[1]) * 1000;
      label = makeLabel(dist.toFixed(1) + " mm", 0xF5A623, true);
      label.position.copy(pts[0]).add(pts[1]).multiplyScalar(0.5);
      scene.add(label);
    }
  });
})();


/* ============================================================
   I3: VIEWPORT STATS HUD  (Isaac-style) — FPS / frame-ms / draw-calls /
   triangles from the Three.js renderer.info, toggled by the STATS button.
   ============================================================ */
(function setupStats() {
  const hud = $("stats-hud"), btn = $("btn-stats");
  if (!hud || !btn) return;
  let frames = 0, lastT = performance.now(), on = false;
  window.__statsTick = () => { frames++; };
  btn.addEventListener("click", () => { on = !on; btn.classList.toggle("act", on); hud.style.display = on ? "block" : "none"; });
  setInterval(() => {
    if (!on) return;
    const now = performance.now(), dt = now - lastT;
    const fps = dt > 0 ? Math.round(frames * 1000 / dt) : 0;
    frames = 0; lastT = now;
    const ri = (typeof renderer !== "undefined" && renderer.info) ? renderer.info.render : { calls: 0, triangles: 0 };
    const tris = ri.triangles > 999 ? (ri.triangles / 1000).toFixed(1) + "k" : ri.triangles;
    hud.innerHTML = "FPS <b>" + fps + "</b> &middot; <b>" + (fps ? (1000 / fps).toFixed(1) : "—") + "</b> ms<br>" +
      "calls <b>" + ri.calls + "</b> &middot; tris <b>" + tris + "</b>";
  }, 1000);
})();


/* ============================================================
   I4: STAGE SEARCH + SELECTION OUTLINE
   Search filters the Stage rows by text; clicking an arm node draws a
   blue BoxHelper around that arm in 3D, updated live as it moves.
   ============================================================ */
(function setupStageSearchOutline() {
  const inp = document.getElementById("stage-search");
  if (inp) inp.addEventListener("input", () => {
    const q = inp.value.trim().toLowerCase();
    document.querySelectorAll("#dock .snode, #dock .tbtn").forEach((el) => {
      const t = (el.textContent || "").toLowerCase();
      el.style.display = (!q || t.includes(q)) ? "" : "none";
    });
  });
  let selBox = null;
  function outline(group) {
    if (selBox) { scene.remove(selBox); if (selBox.geometry) selBox.geometry.dispose(); selBox = null; }
    const idx = GROUPS[group] && GROUPS[group].idx;
    if (!idx || !idx.length || !jointGroups[idx[0]]) return;
    selBox = new THREE.BoxHelper(jointGroups[idx[0]].grp, 0x3B82F6);
    if (selBox.material) { selBox.material.depthTest = false; selBox.material.transparent = true; }
    selBox.renderOrder = 7; scene.add(selBox);
  }
  document.querySelectorAll(".snode[data-group]").forEach((n) =>
    n.addEventListener("click", () => outline(n.dataset.group)));
  setInterval(() => { if (selBox) selBox.update(); }, 120);
})();



// ── Collision-mesh overlay — the capsule/box model the guard reasons over ──
(function setupCollision() {
  const btn = $("btn-coll");
  if (!btn) return;
  let group = null, meshes = [], on = false;
  const baseMat = new THREE.MeshBasicMaterial({ color: 0x3B82F6, transparent: true, opacity: 0.20, depthWrite: false });
  const hitMat  = new THREE.MeshBasicMaterial({ color: 0xE5484D, transparent: true, opacity: 0.44, depthWrite: false });
  function geomOf(g) {
    const s = (g.s && g.s.length === 3) ? g.s : [0.02, 0.02, 0.02];
    let geo;
    if (g.t === 6) geo = new THREE.BoxGeometry(2 * s[0], 2 * s[1], 2 * s[2]);            // box (half-extents)
    else if (g.t === 3) { geo = new THREE.CapsuleGeometry(s[0], 2 * s[1], 6, 14); geo.rotateX(Math.PI / 2); }  // capsule (MuJoCo Z -> three Y)
    else if (g.t === 2) geo = new THREE.SphereGeometry(s[0], 16, 12);                    // sphere
    else if (g.t === 5) { geo = new THREE.CylinderGeometry(s[0], s[0], 2 * s[1], 18); geo.rotateX(Math.PI / 2); }  // cylinder
    else geo = new THREE.BoxGeometry(2 * s[0], 2 * s[1], 2 * s[2]);
    return new THREE.Mesh(geo, baseMat);
  }
  function paint(list) {
    if (!on || !list || !list.length) { if (group) group.visible = false; return; }
    if (!group) { group = new THREE.Group(); group.renderOrder = 4; scene.add(group); }
    if (meshes.length !== list.length) {                       // (re)build mesh cache
      for (const m of meshes) { group.remove(m); m.geometry.dispose(); }
      meshes = list.map((g) => { const m = geomOf(g); group.add(m); return m; });
    }
    for (let i = 0; i < list.length; i++) {                    // pose + tint each geom
      const g = list[i], m = meshes[i];
      m.position.set(g.p[0], g.p[1], g.p[2]);
      m.quaternion.set(g.q[1], g.q[2], g.q[3], g.q[0]);        // server sends [w,x,y,z]; three is (x,y,z,w)
      m.material = g.h ? hitMat : baseMat;                     // redden geoms in a violating contact
    }
    group.visible = true;
  }
  (window.__snapHooks = window.__snapHooks || []).push((s) => paint(s && s.collision));
  btn.onclick = () => {
    on = !on;
    btn.classList.toggle("on", on);
    if (on) { ["btn-dex", "btn-pcl"].forEach((oid) => { const ob = $(oid); if (ob && ob.classList.contains("on")) ob.click(); }); }  // one heavy 3D layer at a time
    send({ type: "collision", on });                          // server only streams geoms when on
    if (!on && group) group.visible = false;
  };
})();



// ── Trajectory replay — scrub / replay the last 45 s of motion (rosbag-style)
(function setupTrajectory() {
  const scrub = $("traj-scrub"), playBtn = $("traj-play"),
        liveEl = $("traj-live"), tEl = $("traj-t");
  if (!scrub || typeof setAngles !== "function") return;
  const SPAN = 45000;                          // rolling record window (ms)
  const rec = [];                              // [{ t, q }]
  let live = true, playing = false, scrubT = 0, lastRec = 0;

  function record(s) {
    if (!s || !s.q) return;
    const now = performance.now();
    if (now - lastRec < 45) return;            // ~20 Hz
    lastRec = now;
    rec.push({ t: now, q: s.q.slice() });
    while (rec.length && rec[0].t < now - SPAN) rec.shift();
  }
  function sampleAt(t) {
    if (!rec.length) return null;
    let best = rec[0], bd = 1e15;
    for (const r of rec) { const d = Math.abs(r.t - t); if (d < bd) { bd = d; best = r; } }
    return best;
  }
  function applyScrub() {
    if (live) return;
    const now = performance.now();
    if (scrubT < now - SPAN) { goLive(); return; }     // aged out of the window
    const r = sampleAt(scrubT);
    if (r && r.q) setAngles(r.q);                       // freeze the twin at the recorded pose
    window.__trajScrub = scrubT;                        // share with the plots for a playhead line
    tEl.textContent = "−" + ((now - scrubT) / 1000).toFixed(1) + "s";
  }
  function goLive() {
    live = true; playing = false; scrub.value = 1000; window.__trajScrub = null;
    liveEl.classList.add("on"); liveEl.innerHTML = "&#9679; LIVE";
    playBtn.classList.remove("on"); tEl.textContent = "now";
  }
  function enterScrub() {
    live = false; liveEl.classList.remove("on"); liveEl.innerHTML = "&#9208; REPLAY";
  }
  function setFrac(f) { scrubT = performance.now() - (1 - f) * SPAN; }  // f: 0 oldest .. 1 live

  (window.__snapHooks = window.__snapHooks || []).push((s) => {
    record(s);
    if (!live) applyScrub();                            // hold the frozen pose against live updates
  });

  scrub.oninput = () => {
    const f = scrub.value / 1000;
    if (f >= 0.999) { goLive(); return; }
    enterScrub(); playing = false; playBtn.classList.remove("on");
    setFrac(f); applyScrub();
  };
  liveEl.onclick = () => goLive();
  { const xb = $("traj-export");
    if (xb) xb.onclick = () => {                        // recorded joint trajectory → CSV
      if (!rec.length) return;
      const n = rec[0].q.length, t0 = rec[0].t, NL = String.fromCharCode(10);
      const rows = ["t_s," + Array.from({ length: n }, (_, i) => "q" + i + "_deg").join(",")];
      for (const r of rec) {
        const row = [((r.t - t0) / 1000).toFixed(3)];
        for (let i = 0; i < n; i++) row.push((r.q[i] * 180 / Math.PI).toFixed(3));
        rows.push(row.join(","));
      }
      window.csvDownload("skate_trajectory.csv", rows.join(NL) + NL);
    };
  }
  playBtn.onclick = () => {
    if (!rec.length) return;
    if (live) { enterScrub(); setFrac(0); }             // start replay from the oldest sample
    playing = !playing; playBtn.classList.toggle("on", playing);
  };

  setInterval(() => {                                   // replay advance (setInterval survives bg tab)
    if (live || !playing) return;
    scrubT += 55;
    const now = performance.now();
    if (scrubT >= now - 120) { goLive(); return; }
    scrub.value = Math.max(0, Math.min(1000, (1 - (now - scrubT) / SPAN) * 1000));
    applyScrub();
  }, 50);

  goLive();
})();



// ── Diagnostics panel — aggregated joint + system health (RViz robot_monitor)
(function setupDiagnostics() {
  const tree = $("diag-tree");
  if (!tree) return;
  const dot = (lv) => '<i class="ddot ' + lv + '"></i>';
  const tLevel = (t) => (t == null ? "ok" : t >= 58 ? "bad" : t >= 50 ? "warn" : "ok");
  const tab = document.querySelector('.pt[data-tab="diag"]');
  const worstOf = (s) => {
    let w = "ok"; const up = (l) => { if (l === "bad") w = "bad"; else if (l === "warn" && w !== "bad") w = "warn"; };
    if (!s.connected) up("bad"); if (s.overtemp) up("bad"); if (s.estop) up("warn");
    if (s.guard && s.guard.blocking) up("warn"); if (s.contact && s.contact.tripped) up("warn");
    if (s.temps) for (const t of s.temps) up(tLevel(t));
    return w;
  };
  const row = (lv, k, v) => '<div class="drow">' + dot(lv) + '<span class="dk">' + k + '</span><span class="dv">' + v + '</span></div>';
  let last = "", lastT = 0;

  function render(s) {
    if (!s) return;
    if (tab) tab.dataset.health = worstOf(s);              // status dot on the tab even while hidden
    if (tree.offsetParent === null) return;                // pane hidden — dot only, skip the tree build
    const now = performance.now();
    if (now - lastT < 400) return;                          // ~2.5 Hz is plenty
    lastT = now;
    const o = [];
    o.push('<div class="dgrp dgsys">System</div>');
    o.push(row(s.connected ? "ok" : "bad", "Link", s.connected ? "connected · " + (s.mode || "sim") : "no link"));
    o.push(row(s.estop ? "warn" : "ok", "E-STOP", s.estop ? "STOPPED" : "armed"));
    o.push(row(s.overtemp ? "bad" : "ok", "Overtemp 58°C", s.overtemp ? "LATCHED" : "ok"));
    const g = s.guard || {};
    o.push(row(g.blocking ? "warn" : "ok", "Collision guard", g.on ? (g.blocking ? "blocking a move" : "on") : "off"));
    const c = s.contact || {};
    o.push(row(c.tripped ? "warn" : "ok", "Contact reflex", c.tripped ? ("tripped · J" + (c.joint != null ? c.joint + 1 : "?")) : (c.on ? "armed" : "off")));
    o.push(row("ok", "Link RTT", (typeof rttMs !== "undefined" && rttMs) ? rttMs.toFixed(0) + " ms" : "—"));

    const T = s.temps, V = s.dq, L = s.tau;
    for (const gn in GROUPS) {
      const G = GROUPS[gn], idx = G.idx || [];
      if (!idx.length) continue;
      let maxT = 0, worst = "ok";
      idx.forEach((ji) => {
        const t = T ? T[ji] : null;
        if (t != null && t > maxT) maxT = t;
        const lv = tLevel(t);
        if (lv === "bad") worst = "bad"; else if (lv === "warn" && worst !== "bad") worst = "warn";
      });
      o.push('<div class="dgrp">' + dot(worst) + G.label + '<em class="dgsum">max ' + (maxT ? maxT.toFixed(0) + "°C" : "—") + '</em></div>');
      idx.forEach((ji, n) => {
        const t = T ? T[ji] : null, v = V ? V[ji] : null, ld = L ? L[ji] : null;
        o.push('<div class="drow dj">' + dot(tLevel(t)) + '<span class="dk">J' + (n + 1) + '</span><span class="dv">' +
          (t != null ? t.toFixed(0) + "°C" : "—") + '<em>' + (v != null ? v.toFixed(2) + " r/s" : "—") +
          (ld != null ? " · " + ld.toFixed(1) + " Nm" : "") + '</em></span></div>');
      });
    }
    const html = o.join("");
    if (html !== last) { tree.innerHTML = html; last = html; }
  }
  (window.__snapHooks = window.__snapHooks || []).push(render);
})();



// ── Joint-limit meters — proximity edge on each slider + amber link box in 3D
(function setupJointLimits() {
  const level = (p) => (p < 0.04 ? "bad" : p < 0.13 ? "warn" : "ok");
  const limBoxes = {};                                   // idx -> THREE.BoxHelper (at a limit)
  function tick(s) {
    if (!s || typeof rows === "undefined") return;
    const q = s.q || s.targ;
    if (!q) return;
    const idxs = (GROUPS[curGroup] && GROUPS[curGroup].idx) || [];
    // drop 3D boxes for joints no longer in the active group
    for (const k in limBoxes) {
      if (idxs.indexOf(+k) === -1) {
        scene.remove(limBoxes[k]);
        if (limBoxes[k].geometry) limBoxes[k].geometry.dispose();
        delete limBoxes[k];
      }
    }
    for (const idx of idxs) {
      const r = rows[idx];
      if (!r || !r.bar || r.hi == null || r.hi <= r.lo) continue;
      const a = q[idx];
      if (a == null) continue;
      const p = Math.min(a - r.lo, r.hi - a) / (r.hi - r.lo);
      const lv = level(p), side = (a - r.lo) < (r.hi - a) ? "lo" : "hi";
      r.bar.classList.remove("lim-ok", "lim-warn", "lim-bad", "lim-lo", "lim-hi");
      r.bar.classList.add("lim-" + lv, "lim-" + side);
      if (r.ang) {
        r.ang.classList.remove("lim-warn", "lim-bad");
        if (lv !== "ok") r.ang.classList.add("lim-" + lv);
      }
      // 3D: outline the link whose joint is hard against its range end
      const grp = jointGroups[idx] && jointGroups[idx].grp;
      if (lv === "bad" && grp) {
        if (!limBoxes[idx]) {
          const bx = new THREE.BoxHelper(grp, 0xF5A623);
          if (bx.material) { bx.material.depthTest = false; bx.material.transparent = true; }
          bx.renderOrder = 8; scene.add(bx); limBoxes[idx] = bx;
        } else { limBoxes[idx].update(); }
      } else if (limBoxes[idx]) {
        scene.remove(limBoxes[idx]);
        if (limBoxes[idx].geometry) limBoxes[idx].geometry.dispose();
        delete limBoxes[idx];
      }
    }
  }
  (window.__snapHooks = window.__snapHooks || []).push(tick);
})();



// ── Stage properties-inspector — click any Stage node -> its properties ────
(function setupStageInspector() {
  const insp = $("stage-inspector"), nameEl = $("si-name"), typeEl = $("si-type"), rowsEl = $("si-rows");
  const tree = document.querySelector(".stage-tree");
  if (!insp || !tree) return;
  const V = new THREE.Vector3();
  let cur = null, lastT = 0;
  function objFor(node) {
    const g = node.dataset.group;
    if (g && GROUPS[g]) { const jg = jointGroups[GROUPS[g].idx[0]]; return jg && jg.grp; }
    const nm = (node.querySelector(".nm") || {}).textContent || "";
    if (/Skate/.test(nm)) return robotRoots[0] || null;
    if (/Grid/.test(nm)) return (typeof grid !== "undefined") ? grid : null;
    return null;                                       // World = scene origin
  }
  function rowsFor(node) {
    const nm = (node.querySelector(".nm") || {}).textContent || "";
    const eye = node.querySelector(".eye");
    const out = [["Visible", eye ? (eye.classList.contains("on") ? "shown" : "hidden") : "shown"]];
    const obj = objFor(node);
    if (obj) { obj.getWorldPosition(V); out.push(["World pos", (V.x * 1000).toFixed(0) + ", " + (V.y * 1000).toFixed(0) + ", " + (V.z * 1000).toFixed(0) + " mm"]); }
    else if (/World/.test(nm)) out.push(["World pos", "0, 0, 0 mm"]);
    const g = node.dataset.group;
    if (g && GROUPS[g]) out.push(["Joints", GROUPS[g].idx.length + " DoF"]);
    return out;
  }
  function paint(node) {
    nameEl.textContent = (node.querySelector(".nm") || {}).textContent || "—";
    typeEl.textContent = (node.querySelector(".ty") || {}).textContent || "";
    rowsEl.innerHTML = rowsFor(node).map(([k, v]) => '<div class="si-row"><span class="si-k">' + k + '</span><span class="si-v">' + v + '</span></div>').join("");
  }
  function show(node) {
    document.querySelectorAll(".stage-tree .snode").forEach((n) => n.classList.toggle("insp", n === node));
    cur = node; paint(node); insp.style.display = "block"; insp.scrollIntoView({ block: "nearest" });
  }
  tree.addEventListener("click", (e) => {
    if (e.target.closest(".eye") || e.target.closest(".tw")) return;   // visibility / expand keep their own actions
    const node = e.target.closest(".snode");
    if (node) show(node);
  });
  (window.__snapHooks = window.__snapHooks || []).push(() => {           // live world-pos refresh for moving links
    if (!cur || insp.style.display === "none") return;
    const now = performance.now(); if (now - lastT < 400) return; lastT = now;
    paint(cur);
  });
})();



// ── Viewport display settings — grid / axes / FOV / background / quality ───
(function setupViewportSettings() {
  const pop = $("vset-pop"), btn = $("btn-vset");
  if (!pop || !btn) return;
  btn.onclick = (e) => {
    e.stopPropagation();
    const show = (pop.style.display === "none" || !pop.style.display);
    if (show) {                                            // reflect the real scene state on open
      if (typeof grid !== "undefined" && $("vs-grid")) $("vs-grid").checked = grid.visible;
      if (typeof axes !== "undefined" && $("vs-axes")) $("vs-axes").checked = axes.visible;
      const rtx = document.querySelector(".hd-render");
      if (rtx && $("vs-hd")) $("vs-hd").checked = rtx.classList.contains("on");
      if ($("vs-fov")) { $("vs-fov").value = Math.round(camera.fov); if ($("vs-fov-v")) $("vs-fov-v").textContent = Math.round(camera.fov) + "°"; }
    }
    pop.style.display = show ? "block" : "none";
  };
  document.addEventListener("click", (e) => { if (!pop.contains(e.target) && e.target !== btn) pop.style.display = "none"; });

  const gridCb = $("vs-grid");
  if (gridCb) gridCb.onchange = () => { if (typeof grid !== "undefined") grid.visible = gridCb.checked; };
  const axesCb = $("vs-axes");
  if (axesCb) axesCb.onchange = () => { if (typeof axes !== "undefined") axes.visible = axesCb.checked; };

  const fov = $("vs-fov"), fovV = $("vs-fov-v");
  if (fov) fov.oninput = () => { camera.fov = +fov.value; camera.updateProjectionMatrix(); if (fovV) fovV.textContent = fov.value + "°"; };

  for (const sw of pop.querySelectorAll(".vs-sw")) {
    sw.onclick = () => {
      pop.querySelectorAll(".vs-sw").forEach((s) => s.classList.toggle("on", s === sw));
      scene.background = new THREE.Color(sw.dataset.bg);
    };
  }
  const hd = $("vs-hd");
  if (hd) hd.onchange = () => { const rtx = document.querySelector(".hd-render"); if (rtx) { rtx.click(); hd.checked = rtx.classList.contains("on"); } };
})();



// ── Waypoint timeline — visual marker strip over the SEQ list ──────────────
(function setupSeqTimeline() {
  let sig = "";
  function render(s) {
    const tl = document.getElementById("seq-timeline");
    if (!tl) return;                                        // #seq-timeline exists only on the SEQ tab
    const seq = s && s.seq;
    if (!seq) return;
    const k = JSON.stringify([seq.names, seq.idx, seq.active]);
    if (k === sig && tl.querySelector(".seq-tl-pt")) return; // unchanged + already drawn
    sig = k;
    if (!seq.names.length) { tl.innerHTML = ""; return; }
    tl.innerHTML = '<div class="seq-tl-line"></div>' + seq.names.map((nm, i) =>
      '<button class="seq-tl-pt' + (seq.active && seq.idx === i ? " active" : "") +
      '" data-i="' + i + '" title="' + (i + 1) + ". " + nm + '">' + (i + 1) + "</button>").join("");
    tl.querySelectorAll(".seq-tl-pt").forEach((b) => {
      b.onclick = () => {
        const i = +b.dataset.i;
        if (typeof previewServer === "function")
          previewServer("waypoint", i, "Waypoint " + (i + 1) + " — glide to this pose", () => send({ type: "wp_goto", idx: i }));
        else send({ type: "wp_goto", idx: i });
      };
    });
  }
  (window.__snapHooks = window.__snapHooks || []).push(render);
})();



// ── Scene markers / annotations — click to drop a labeled point in 3D ──────
(function setupMarkers() {
  const vp = document.getElementById("viewport"), listEl = document.getElementById("marker-list");
  if (!vp || typeof renderer === "undefined" || !renderer || !renderer.domElement) return;
  const ray = new THREE.Raycaster();
  let on = false, markers = [], selId = null, lastMove = 0;
  const SPAWN = [0.15, 0.15, 0.30];                // a reachable point (right-arm zone); drag the gizmo to place it

  const mtc = new TransformControls(camera, renderer.domElement);  // dedicated marker move-gizmo
  mtc.setMode("translate"); mtc.setSize(0.6); mtc.setSpace("world");
  scene.add(mtc);
  mtc.addEventListener("dragging-changed", (e) => { controls.enabled = !e.value; if (!e.value) { renderList(); if (selId != null && markers[selId]) checkReach(markers[selId]); } });
  mtc.addEventListener("objectChange", () => { const now = performance.now(); if (now - lastMove > 60) { lastMove = now; renderList(); } });

  function makeMarker(pos, n) {
    const grp = new THREE.Group();
    grp.add(new THREE.Mesh(new THREE.SphereGeometry(0.014, 14, 12), new THREE.MeshBasicMaterial({ color: 0xF5A623, depthTest: false })));
    if (typeof makeLabel === "function") { const lab = makeLabel("#" + n, 0xF5A623, true); lab.position.set(0, 0, 0.05); grp.add(lab); grp.userData.lab = lab; }
    grp.position.set(pos[0], pos[1], pos[2]); grp.renderOrder = 10; scene.add(grp);
    return grp;
  }
  function relabel() {
    markers.forEach((m, i) => { if (m.userData.lab) m.remove(m.userData.lab);
      if (typeof makeLabel === "function") { const lab = makeLabel("#" + (i + 1), 0xF5A623, true); lab.position.set(0, 0, 0.05); m.add(lab); m.userData.lab = lab; } });
  }
  function spawn() { const side = (markers.length % 2 === 0) ? 1 : -1; const grp = makeMarker([SPAWN[0] * side, SPAWN[1], SPAWN[2]], markers.length + 1); markers.push(grp); selId = markers.length - 1; mtc.attach(grp); renderList(); checkReach(grp); }
  function select(i) { selId = (i != null && markers[i]) ? i : null; if (selId != null) mtc.attach(markers[selId]); else mtc.detach(); renderList(); }
  function del(i) { if (!markers[i]) return; if (i === selId) { selId = null; mtc.detach(); } scene.remove(markers[i]); markers.splice(i, 1); if (selId != null && i < selId) selId--; relabel(); renderList(); }
  function clearAll() { for (const m of markers) scene.remove(m); markers = []; selId = null; mtc.detach(); renderList(); }
  function addToProgram(m, i) {                     // append a guarded rbt.moveto for this point into the PROG editor
    const line = 'rbt.moveto("right", ' + Math.round(m.position.x * 1000) + ', ' + Math.round(m.position.y * 1000) + ', ' + Math.round(m.position.z * 1000) + ')  # marker #' + (i + 1);
    progCode = (progCode && !progCode.endsWith("\n") ? progCode + "\n" : progCode) + line + "\n";
    const pt = document.querySelector('[data-group="prog"]'); if (pt) pt.click();   // open PROG so the appended line is visible
    const ta = document.getElementById("pg-code"); if (ta) ta.value = progCode;
  }
  function colorOf(grp) { const rl = grp.userData.reachL, rr = grp.userData.reachR; return (rl == null && rr == null) ? 0xF5A623 : ((rl || rr) ? 0x3FB950 : 0xE5484D); }
  function applyColor(grp) { const m = grp.children[0]; if (m && m.material) m.material.color.setHex(colorOf(grp)); }
  async function checkReach(grp) {                  // IK feasibility for both arms -> colour the point + dim the arm that can't reach
    const p = grp.position, q = "&x=" + p.x + "&y=" + p.y + "&z=" + p.z;
    try {
      const [l, r] = await Promise.all([
        fetch("/api/reachable?arm=left" + q).then((x) => x.json()),
        fetch("/api/reachable?arm=right" + q).then((x) => x.json()),
      ]);
      grp.userData.reachL = !!(l && l.reachable);
      grp.userData.reachR = !!(r && r.reachable);
    } catch (e) { grp.userData.reachL = null; grp.userData.reachR = null; }
    applyColor(grp); renderList();
  }
  function bimanual() {                             // send the first two points to BOTH arms at once
    if (markers.length < 2) return;
    const a = markers[0], b = markers[1];
    const reach = (m, arm) => (arm === "left" ? m.userData.reachL : m.userData.reachR);
    const opt1 = reach(a, "left") && reach(b, "right");     // a->left, b->right both reach
    const opt2 = reach(a, "right") && reach(b, "left");     // a->right, b->left both reach
    let aArm, bArm;
    if (opt1 && !opt2) { aArm = "left"; bArm = "right"; }
    else if (opt2 && !opt1) { aArm = "right"; bArm = "left"; }
    else if (a.position.x <= b.position.x) { aArm = "left"; bArm = "right"; }   // else by side (more -x = left)
    else { aArm = "right"; bArm = "left"; }
    send({ type: "ik_target", arm: aArm, pos: [a.position.x, a.position.y, a.position.z], auto: true });
    send({ type: "ik_target", arm: bArm, pos: [b.position.x, b.position.y, b.position.z], auto: true });
  }
  function renderList() {
    if (!listEl) return;
    if (!on) { listEl.style.display = "none"; listEl.innerHTML = ""; return; }
    listEl.style.display = "block";
    listEl.innerHTML = '<div class="ml-hd">MARKERS<span class="ml-acts"><button class="ml-add" title="Add a target point in front of the robot">+ point</button><button class="ml-both" title="Send the first two points to both arms at once (auto-assigns arm by reachability)"' + (markers.length < 2 ? ' disabled' : '') + '>⇄ both</button><button class="ml-clr">clear</button></span></div>' +
      (markers.length ? markers.map((m, i) => {
        const rl = m.userData.reachL, rr = m.userData.reachR;
        return '<div class="ml-row' + (i === selId ? ' sel' : '') + '" data-sel="' + i + '"><span>#' + (i + 1) + '</span><em>' +
          (m.position.x * 1000).toFixed(0) + ", " + (m.position.y * 1000).toFixed(0) + ", " + (m.position.z * 1000).toFixed(0) + ' mm</em>' +
          '<button class="mk-go' + (rl === false ? ' unreach' : '') + '" data-go="left" data-i="' + i + '" title="' + (rl === false ? 'Left arm cannot reach this point' : 'Glide the LEFT arm TCP here (needs RESUME)') + '">&#8594;L</button>' +
          '<button class="mk-go' + (rr === false ? ' unreach' : '') + '" data-go="right" data-i="' + i + '" title="' + (rr === false ? 'Right arm cannot reach this point' : 'Glide the RIGHT arm TCP here (needs RESUME)') + '">&#8594;R</button>' +
          '<button class="mk-prog" data-i="' + i + '" title="Append rbt.moveto(right) for this point to the PROG program">&#8594;P</button>' +
          '<button class="mk-del" data-i="' + i + '" title="Delete">&#10005;</button></div>';
      }).join("")
        : '<div class="ml-empty">click &ldquo;+ point&rdquo;, drag its gizmo to a reachable spot, then &#8594;L / &#8594;R</div>');
    const add = listEl.querySelector(".ml-add"); if (add) add.onclick = spawn;
    const both = listEl.querySelector(".ml-both"); if (both) both.onclick = bimanual;
    listEl.querySelector(".ml-clr").onclick = clearAll;
    listEl.querySelectorAll(".mk-del").forEach((b) => { b.onclick = (ev) => { ev.stopPropagation(); del(+b.dataset.i); }; });
    listEl.querySelectorAll(".mk-go").forEach((b) => { b.onclick = (ev) => { ev.stopPropagation(); const m = markers[+b.dataset.i]; if (!m) return; send({ type: "ik_target", arm: b.dataset.go, pos: [m.position.x, m.position.y, m.position.z], auto: true }); }; });
    listEl.querySelectorAll(".mk-prog").forEach((b) => { b.onclick = (ev) => { ev.stopPropagation(); const m = markers[+b.dataset.i]; if (!m) return; addToProgram(m, +b.dataset.i); }; });
    listEl.querySelectorAll("[data-sel]").forEach((rw) => { rw.onclick = (ev) => { if (ev.target && ev.target.dataset && ev.target.dataset.i != null) return; select(+rw.dataset.sel); }; });
  }
  renderer.domElement.addEventListener("pointerdown", (e) => {   // click a marker to select it (gizmo); click empty deselects (a drag still orbits)
    if (!on || e.button !== 0 || mtc.dragging) return;
    const r = renderer.domElement.getBoundingClientRect();
    const ndc = new THREE.Vector2(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1);
    ray.setFromCamera(ndc, camera);
    const spheres = markers.map((m) => m.children[0]);
    const hit = ray.intersectObjects(spheres, false)[0];
    if (hit) { const idx = markers.findIndex((m) => m.children[0] === hit.object); if (idx >= 0) select(idx); }
    else select(null);
  });
  document.addEventListener("keydown", (e) => { if (on && e.key === "Escape") { if (selId != null) select(null); else clearAll(); } });
  window.__markerActive = false;
  window.__markerMode = (v) => {
    on = v; window.__markerActive = v;
    if (typeof tc !== "undefined" && tc && tc.object) tc.detach();
    if (!v) { selId = null; mtc.detach(); if (listEl) listEl.style.display = "none"; }
    else renderList();
  };
  window.__getMarkers = () => markers.map((m) => [m.position.x, m.position.y, m.position.z]);
  window.__loadMarkers = (list) => {                // restore markers from a saved scene
    clearAll();
    for (const p of (list || [])) { if (Array.isArray(p) && p.length === 3) { const grp = makeMarker(p, markers.length + 1); markers.push(grp); checkReach(grp); } }
    selId = null; mtc.detach(); renderList();
  };
})();



// ── TCP force — per-arm end-effector force estimate (N) as a 3D arrow ───────
(function setupForce() {
  const btn = $("btn-force");
  if (!btn) return;
  let on = false;
  const COL = 0x6AA0F8, HI_COL = 0xF5A623;           // accent-lt · amber when straining
  const SCALE = 0.006, MIN_N = 0.5, MAXL = 0.45, MINL = 0.08, HI_N = 12;
  const arr = {}, labs = {}, lastTxt = {}, fsm = {};
  const ALPHA = 0.08;                                // EMA on the force vector — the raw torque estimate is jittery in motion (~0.4 s time constant)
  function hide() {
    for (const a in arr) if (arr[a]) arr[a].visible = false;
    for (const a in labs) if (labs[a]) labs[a].visible = false;
  }
  function paint(force) {
    if (!on || !force) { hide(); return; }
    for (const a in force) {
      const d = force[a];
      if (!d) {                                      // no data for this arm
        if (arr[a]) arr[a].visible = false;
        if (labs[a]) labs[a].visible = false;
        continue;
      }
      const raw = new THREE.Vector3(d.f[0], d.f[1], d.f[2]);
      if (!fsm[a]) fsm[a] = raw.clone();             // low-pass the noisy torque estimate (EMA)
      else fsm[a].lerp(raw, ALPHA);
      const mag = fsm[a].length();
      if (mag < MIN_N) {                             // below the noise floor → hide
        if (arr[a]) arr[a].visible = false;
        if (labs[a]) labs[a].visible = false;
        continue;
      }
      const p = new THREE.Vector3(d.p[0], d.p[1], d.p[2]);          // TCP (sim = twin frame)
      const dir = fsm[a].clone().normalize();
      const len = Math.min(MAXL, Math.max(MINL, mag * SCALE));      // N → metres, clamped
      const col = mag > HI_N ? HI_COL : COL;                       // amber when the TCP is straining
      if (!arr[a]) {
        arr[a] = new THREE.ArrowHelper(dir, p, len, col, len * 0.34, len * 0.22);
        arr[a].renderOrder = 7;
        scene.add(arr[a]);
      } else {
        arr[a].visible = true;
        arr[a].position.copy(p);
        arr[a].setDirection(dir);
        arr[a].setLength(len, len * 0.34, len * 0.22);
      }
      arr[a].setColor(col);
      if (typeof makeLabel === "function") {                        // magnitude label at the tip
        const key = Math.round(mag) + " N|" + col;
        if (key !== lastTxt[a]) {                                   // recreate only when value/colour changes
          if (labs[a]) scene.remove(labs[a]);
          labs[a] = makeLabel(Math.round(mag) + " N", col, true);
          labs[a].renderOrder = 8;
          scene.add(labs[a]);
          lastTxt[a] = key;
        }
        labs[a].visible = true;
        labs[a].position.copy(p).addScaledVector(dir, len + 0.05);
      }
    }
  }
  (window.__snapHooks = window.__snapHooks || []).push((s) => paint(s && s.force));
  btn.onclick = () => {
    on = !on;
    btn.classList.toggle("on", on);
    send({ type: "force", on });                    // server only computes the force when on
    if (!on) { hide(); for (const a in fsm) fsm[a] = null; }
  };
})();



// ── Motion tuning — jog/glide cruise speed + accel, contact sensitivity ─────
(function setupTuning() {
  const btn = $("btn-tune"), pop = $("tune-pop");
  if (!btn || !pop) return;
  const MAP = {                                   // slider id -> { bridge key, slider→value scale, decimals }
    "tn-jr": { key: "jog_rate",    scale: 0.01, dp: 2 },
    "tn-ja": { key: "jog_accel",   scale: 0.1,  dp: 1 },
    "tn-sr": { key: "seq_rate",    scale: 0.01, dp: 2 },
    "tn-sa": { key: "seq_accel",   scale: 0.1,  dp: 1 },
    "tn-ct": { key: "contact_tau", scale: 0.1,  dp: 1 },
  };
  const DEF = { jog_rate: 0.35, jog_accel: 3.0, seq_rate: 0.6, seq_accel: 2.0, contact_tau: 8.0 };
  function setRow(id, val) {
    const m = MAP[id], el = $(id), em = $(id + "-v");
    if (!el) return;
    el.value = Math.round(val / m.scale);
    if (em) em.textContent = val.toFixed(m.dp);
  }
  function sync(t) { if (t) for (const id in MAP) if (t[MAP[id].key] != null) setRow(id, t[MAP[id].key]); }
  for (const id in MAP) {
    const el = $(id), em = $(id + "-v"), m = MAP[id];
    if (!el) continue;
    el.oninput = () => {
      const v = +el.value * m.scale;
      if (em) em.textContent = v.toFixed(m.dp);
      send({ type: "tune", params: { [m.key]: v } });        // bridge clamps + applies live
    };
  }
  const rst = $("tn-reset");
  if (rst) rst.onclick = () => { for (const id in MAP) setRow(id, DEF[MAP[id].key]); send({ type: "tune", params: DEF }); };
  btn.onclick = (e) => {
    e.stopPropagation();
    const show = pop.style.display === "none" || !pop.style.display;
    pop.style.display = show ? "block" : "none";
    if (show && window.__lastTuning) sync(window.__lastTuning);   // reflect the live values on open
  };
  document.addEventListener("click", (e) => {
    if (!e.target.closest("#tune-pop") && !btn.contains(e.target)) pop.style.display = "none";
  });
  (window.__snapHooks = window.__snapHooks || []).push((s) => { if (s && s.tuning) window.__lastTuning = s.tuning; });
})();



// ── CSV download helper (shared by the trajectory + plot exports) ───────────
window.csvDownload = window.csvDownload || function (name, text) {
  const blob = new Blob([text], { type: "text/csv;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url; a.download = name; a.style.display = "none";
  document.body.appendChild(a); a.click(); a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1500);
};




// Virtual obstacles -- spawn a box, then drag its 3D gizmo to place it; the planner routes around them
(function setupObstacles() {
  const listEl = document.getElementById("obstacle-list");
  if (typeof renderer === "undefined" || !renderer || !renderer.domElement) return;
  const ray = new THREE.Raycaster();
  let on = false, lastObs = [], selId = null, lastMove = 0, selectNew = false, lastListSig = null;
  const HALF = 0.06, HALFH = 0.10;
  const grp = new THREE.Group(); grp.renderOrder = 5; scene.add(grp);
  const meshes = {};
  const mat     = new THREE.MeshBasicMaterial({ color: 0xE5484D, transparent: true, opacity: 0.22, depthWrite: false });
  const matSel  = new THREE.MeshBasicMaterial({ color: 0xF5A623, transparent: true, opacity: 0.30, depthWrite: false });
  const emat    = new THREE.LineBasicMaterial({ color: 0xE5484D, transparent: true, opacity: 0.8 });
  const ematSel = new THREE.LineBasicMaterial({ color: 0xF5A623, transparent: true, opacity: 1.0 });

  const otc = new TransformControls(camera, renderer.domElement);   // dedicated obstacle move-gizmo (separate from the IK one)
  otc.setMode("translate"); otc.setSize(0.6); otc.setSpace("world");
  scene.add(otc);
  function sendMove() { if (selId != null && otc.object) { const p = otc.object.position; send({ type: "obstacle_move", id: selId, p: [p.x, p.y, p.z] }); } }
  otc.addEventListener("dragging-changed", (e) => {
    controls.enabled = !e.value;
    const gh = document.getElementById("giz-hud"); if (gh) gh.style.display = e.value ? "block" : "none";
    if (!e.value) sendMove();                                       // commit final position on drop
  });
  otc.addEventListener("objectChange", () => {
    if (!otc.object) return;
    const p = otc.object.position, gh = document.getElementById("giz-hud");
    if (gh) { gh.style.display = "block"; gh.textContent = "OBSTACLE   " + Math.round(p.x * 1000) + " / " + Math.round(p.y * 1000) + " / " + Math.round(p.z * 1000) + " mm"; }
    const now = performance.now();
    if (now - lastMove > 60) { lastMove = now; sendMove(); }        // ~16 Hz while dragging
  });

  function selectBox(id) {
    selId = (id != null && meshes[id]) ? id : null;
    for (const k in meshes) { const s = (+k === selId); meshes[k].material = s ? matSel : mat; if (meshes[k].userData.edge) meshes[k].userData.edge.material = s ? ematSel : emat; }
    if (selId != null) otc.attach(meshes[selId]); else otc.detach();
    if (listEl && on) renderList(lastObs);
  }

  function render(list) {
    list = list || [];
    const ids = new Set(list.map((o) => o.id));
    for (const id in meshes) if (!ids.has(+id)) { if (+id === selId) { selId = null; otc.detach(); } grp.remove(meshes[id]); delete meshes[id]; }
    for (const o of list) {
      let m = meshes[o.id];
      const sizeKey = o.s.join(",");
      if (!m) {
        const geo = new THREE.BoxGeometry(2 * o.s[0], 2 * o.s[1], 2 * o.s[2]);
        m = new THREE.Mesh(geo, mat);
        const edge = new THREE.LineSegments(new THREE.EdgesGeometry(geo), emat);
        m.add(edge); m.userData.oid = o.id; m.userData.edge = edge; m.userData.sizeKey = sizeKey;
        grp.add(m); meshes[o.id] = m;
      } else if (m.userData.sizeKey !== sizeKey) {                       // live resize -> rebuild box geometry to the new dimensions
        m.geometry.dispose();
        m.geometry = new THREE.BoxGeometry(2 * o.s[0], 2 * o.s[1], 2 * o.s[2]);
        if (m.userData.edge) { m.remove(m.userData.edge); m.userData.edge.geometry.dispose(); }
        const edge = new THREE.LineSegments(new THREE.EdgesGeometry(m.geometry), (o.id === selId ? ematSel : emat));
        m.add(edge); m.userData.edge = edge; m.userData.sizeKey = sizeKey;
      }
      if (!(otc.object === m && otc.dragging)) m.position.set(o.p[0], o.p[1], o.p[2]);   // don't fight the gizmo mid-drag
    }
    if (selectNew && list.length) { selectNew = false; selectBox(list[list.length - 1].id); }
    lastObs = list;
    if (listEl) renderList(list);
  }

  function renderList(list) {
    if (!on) { listEl.style.display = "none"; listEl.innerHTML = ""; lastListSig = null; return; }
    listEl.style.display = "block";
    const _sig = list.map((o) => o.id + ":" + Math.round(o.p[0] * 1000) + "," + Math.round(o.p[1] * 1000) + "," + Math.round(o.p[2] * 1000)).join("|") + "#" + selId;
    if (_sig === lastListSig) return;     // skip the ~20 Hz innerHTML rebuild when unchanged so the + box button stays alive and real clicks land
    lastListSig = _sig;
    listEl.innerHTML = '<div class="ml-hd">OBSTACLES<span class="ml-acts"><button class="ml-add" title="Add a box in front of the robot">+ box</button><button class="ml-clr">clear</button></span></div>' +
      (list.length ? list.map((o) => {
        const w = Math.round(o.s[0] * 2000), d = Math.round(o.s[1] * 2000), h = Math.round(o.s[2] * 2000);
        const sel = (o.id === selId);
        const size = sel
          ? '<span class="ob-size"><input class="ob-dim" type="number" data-oid="' + o.id + '" value="' + w + '" min="20" max="1500" step="10" title="width (mm)"><i>x</i><input class="ob-dim" type="number" data-oid="' + o.id + '" value="' + d + '" min="20" max="1500" step="10" title="depth (mm)"><i>x</i><input class="ob-dim" type="number" data-oid="' + o.id + '" value="' + h + '" min="20" max="1500" step="10" title="height (mm)"><b>mm</b></span>'
          : '<em>' + w + ' x ' + d + ' x ' + h + ' mm</em>';
        return '<div class="ml-row' + (sel ? ' sel' : '') + '" data-sel="' + o.id + '"><span>BOX</span>' + size + '<button data-oid="' + o.id + '" title="Delete">&#10005;</button></div>';
      }).join("")
        : '<div class="ml-empty">click "+ box", then drag its handles to place it; select it to set W x D x H</div>');
    listEl.querySelector(".ml-add").onclick = addBox;
    listEl.querySelector(".ml-clr").onclick = () => { selId = null; otc.detach(); send({ type: "obstacle_clear" }); };
    listEl.querySelectorAll("button[data-oid]").forEach((b) => { b.onclick = (ev) => { ev.stopPropagation(); const id = +b.dataset.oid; if (id === selId) { selId = null; otc.detach(); } send({ type: "obstacle_del", id: id }); }; });
    listEl.querySelectorAll(".ob-dim").forEach((inp) => {
      inp.onclick = (ev) => ev.stopPropagation();                        // editing size -- don't toggle row selection
      inp.onchange = () => {
        const row = inp.closest(".ml-row"); if (!row) return;
        const id = +inp.dataset.oid;
        const dims = [...row.querySelectorAll(".ob-dim")].map((x) => Math.max(20, Math.min(1500, +x.value || 20)));
        send({ type: "obstacle_resize", id: id, s: [dims[0] / 2000, dims[1] / 2000, dims[2] / 2000] });
      };
    });
    listEl.querySelectorAll("[data-sel]").forEach((rw) => { rw.onclick = (ev) => { if (ev.target && ev.target.dataset && ev.target.dataset.oid != null) return; selectBox(+rw.dataset.sel); }; });
  }

  function addBox() { selectNew = true; send({ type: "obstacle_add", shape: "box", p: [0, 0.45, -0.15], s: [HALF, HALF, HALFH] }); }   // spawn in front of the robot at ~chest height

  renderer.domElement.addEventListener("pointerdown", (e) => {       // click a box to select it (gizmo); click empty to deselect. NO click-to-place.
    if (!on || e.button !== 0 || otc.dragging) return;
    const r = renderer.domElement.getBoundingClientRect();
    const ndc = new THREE.Vector2(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1);
    ray.setFromCamera(ndc, camera);
    const hit = ray.intersectObjects(Object.values(meshes), false)[0];
    if (hit) { let o = hit.object; while (o && o.userData.oid == null) o = o.parent; selectBox(o ? o.userData.oid : null); }
    else selectBox(null);                                            // empty space -> deselect (a drag still orbits the view)
  });
  document.addEventListener("keydown", (e) => { if (on && e.key === "Escape" && selId != null) selectBox(null); });

  (window.__snapHooks = window.__snapHooks || []).push((s) => render(s && s.obstacles));
  window.__obstacleActive = false;
  window.__obstacleMode = (v) => {
    on = v; window.__obstacleActive = v;
    if (typeof tc !== "undefined" && tc && tc.object) tc.detach();   // hand the gizmo over from the IK target
    if (!v) { selId = null; otc.detach(); if (listEl) listEl.style.display = "none"; }
    else if (listEl) renderList(lastObs);
  };
})();



// ── Scene save / load — markers (client) + obstacles (server) to a .json file ──
(function setupScene() {
  const inp = document.createElement("input");
  inp.type = "file"; inp.accept = ".json,application/json"; inp.style.display = "none";
  document.body.appendChild(inp);
  inp.onchange = () => {
    const f = inp.files && inp.files[0]; if (!f) return;
    const rd = new FileReader();
    rd.onload = () => {
      let d; try { d = JSON.parse(rd.result); } catch (e) { return; }
      send({ type: "obstacle_clear" });
      for (const o of (d.obstacles || [])) send({ type: "obstacle_add", shape: o.shape || "box", p: o.p, s: o.s });
      if (window.__loadMarkers) window.__loadMarkers(d.markers || []);
    };
    rd.readAsText(f); inp.value = "";
  };
  window.__sceneSave = () => {
    const markers = window.__getMarkers ? window.__getMarkers() : [];
    const obstacles = (typeof state !== "undefined" && state && state.obstacles)
      ? state.obstacles.map((o) => ({ shape: o.type || "box", p: o.p, s: o.s })) : [];
    const text = JSON.stringify({ version: 1, saved: new Date().toISOString(), markers, obstacles }, null, 2);
    if (window.csvDownload) window.csvDownload("skate_scene.json", text);
  };
  window.__sceneLoad = () => inp.click();
})();
