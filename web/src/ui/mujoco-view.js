// Replays episodes recorded from MuJoCo by scripts/train_mujoco_ppo.py.
//
// Nothing is simulated here: every frame is the recorded position and
// orientation of every MuJoCo geom. The browser only draws them.
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";

import { button, h, replace, s, segmented } from "./dom.js";
import { fmt, fmtSigned } from "./format.js";

const token = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim() || "#888888";

function webglAvailable() {
  try {
    const gl = document.createElement("canvas").getContext("webgl2");
    gl?.getExtension("WEBGL_lose_context")?.loseContext();
    return !!gl;
  } catch {
    return false;
  }
}

function gridTexture(line, bg) {
  const c = document.createElement("canvas");
  c.width = c.height = 128;
  const ctx = c.getContext("2d");
  ctx.fillStyle = bg;
  ctx.fillRect(0, 0, 128, 128);
  ctx.strokeStyle = line;
  ctx.lineWidth = 2;
  ctx.strokeRect(0, 0, 128, 128);
  const tex = new THREE.CanvasTexture(c);
  tex.colorSpace = THREE.SRGBColorSpace;
  tex.wrapS = tex.wrapT = THREE.RepeatWrapping;
  tex.anisotropy = 4;
  return tex;
}

// Three.js geometry for a MuJoCo geom, in the geom's own frame (MuJoCo capsules and cylinders run along local z).
function geometryFor(g) {
  const [a, b, c] = g.size;
  switch (g.type) {
    case "sphere":
      return new THREE.SphereGeometry(a, 24, 16);
    case "capsule":
      return new THREE.CapsuleGeometry(a, 2 * b, 8, 20).rotateX(Math.PI / 2);
    case "cylinder":
      return new THREE.CylinderGeometry(a, a, 2 * b, 24).rotateX(Math.PI / 2);
    case "box":
      return new THREE.BoxGeometry(2 * a, 2 * b, 2 * c);
    case "ellipsoid":
      return new THREE.SphereGeometry(1, 24, 16).scale(a, b, c);
    default:
      return null;
  }
}

const FOLLOW = /Hopper|Walker|HalfCheetah|Ant-|Humanoid|Swimmer/;
const SIDE_VIEW = /Hopper|Walker|HalfCheetah|InvertedPendulum|InvertedDoublePendulum|Swimmer/;

export class MujocoReplay {
  constructor(task) {
    this.task = task;
    this.checkpoint = task.checkpoints.length - 1;
    this.frame = 0;
    this.playing = false;
    this.speed = 1;
    this.el = h("div", { class: "mujoco" });
    this.ok = webglAvailable();

    this.canvasBox = h("div", { class: "scene3d mujoco-scene" });
    this.timeEl = h("span", { class: "note num" });
    this.scrub = h("input", { type: "range", min: 0, max: 0, step: 1, value: 0, id: `mj-scrub-${task.env_id}`, "aria-label": "Frame" });
    this.scrub.addEventListener("input", () => this.seek(Number(this.scrub.value)));
    this.playBtn = button("Play", { onClick: () => this.toggle() });
    const stepBack = button("‹ frame", { kind: "ghost", onClick: () => this.seek(this.frame - 1) });
    const stepFwd = button("frame ›", { kind: "ghost", onClick: () => this.seek(this.frame + 1) });
    const speed = segmented(
      [
        { value: 0.25, label: "¼×" },
        { value: 1, label: "1×" },
        { value: 3, label: "3×" },
      ],
      { value: 1, label: "Replay speed", onChange: (v) => (this.speed = v) },
    );
    this.picker = segmented(
      task.checkpoints.map((c, i) => ({ value: i, label: i === 0 ? "untrained" : i === task.checkpoints.length - 1 ? "final policy" : `after ${compact(c.env_steps)} steps` })),
      { value: this.checkpoint, label: "Policy", onChange: (i) => this.load(i) },
    );
    this.stats = h("div", { class: "counters" });
    this.actionBox = h("div", { class: "action-bars" });
    this.strip = s("svg", { class: "episode-strip", role: "img", "aria-label": "Reward and value over the episode" });

    this.el.append(
      h("div", { class: "toolbar" }, h("span", { class: "panel-title" }, "Policy"), this.picker),
      this.canvasBox,
      h("div", { class: "toolbar mujoco-controls" }, this.playBtn, stepBack, stepFwd, speed, h("div", { class: "scrub" }, this.scrub), this.timeEl),
      this.stats,
      h("div", { class: "two-col" }, h("div", { class: "figure" }, h("div", { class: "panel-title" }, "Action at this frame (one bar per actuator)"), this.actionBox), h("div", { class: "figure" }, h("div", { class: "panel-title" }, "Reward per step and the critic's V(s) over the episode"), this.strip)),
    );

    if (!this.ok) {
      this.canvasBox.append(h("p", { class: "scene3d-fallback" }, "The 3D replay needs WebGL, which this browser does not provide. The numbers below still follow the recorded episode."));
    } else {
      this.build();
    }
    this.load(this.checkpoint);
  }

  build() {
    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFShadowMap;
    this.canvasBox.append(this.renderer.domElement, h("p", { class: "scene3d-caption" }, "Recorded in MuJoCo · drag to look around"));
    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(35, 2, 0.05, 200);
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    Object.assign(this.controls, { enablePan: false, enableDamping: false, maxPolarAngle: 1.5 });
    this.controls.addEventListener("change", () => this.render());

    // MuJoCo is z-up; three.js is y-up.
    this.world = new THREE.Group();
    this.world.rotation.x = -Math.PI / 2;
    this.scene.add(this.world);
    const robot = new THREE.MeshStandardMaterial({ color: token("--series-2"), roughness: 0.45, metalness: 0.1 });
    const joint = new THREE.MeshStandardMaterial({ color: token("--series-1"), roughness: 0.5 });
    const wall = new THREE.MeshStandardMaterial({ color: new THREE.Color(token("--paper")).lerp(new THREE.Color(token("--ink")), 0.25), roughness: 0.85 });
    const floor = new THREE.MeshStandardMaterial({ map: gridTexture(token("--rule-strong"), token("--sheet")), roughness: 1 });
    this.meshes = new Map();
    for (const g of this.task.geoms) {
      let mesh;
      if (g.type === "plane") {
        const w = g.size[0] > 0 ? 2 * g.size[0] : 60;
        const d = g.size[1] > 0 ? 2 * g.size[1] : 60;
        const mat = floor.clone();
        mat.map = floor.map.clone();
        mat.map.repeat.set(w, d);
        mesh = new THREE.Mesh(new THREE.PlaneGeometry(w, d), mat);
        mesh.receiveShadow = true;
      } else {
        const geo = geometryFor(g);
        if (!geo) continue;
        const moving = g.body !== 0;
        mesh = new THREE.Mesh(geo, moving ? (g.body % 2 ? robot : joint) : wall);
        mesh.castShadow = true;
        mesh.receiveShadow = !moving;
      }
      this.meshes.set(g.id, mesh);
      this.world.add(mesh);
    }
    this.goal = new THREE.Mesh(new THREE.CylinderGeometry(0.18, 0.18, 0.01, 32).rotateX(Math.PI / 2), new THREE.MeshStandardMaterial({ color: token("--env"), emissive: token("--env"), emissiveIntensity: 0.3 }));
    this.goal.visible = false;
    this.world.add(this.goal);

    this.scene.add(new THREE.HemisphereLight(0xffffff, 0x5c6f8f, 1.6));
    this.sun = new THREE.DirectionalLight(0xffffff, 1.4);
    this.sun.castShadow = true;
    this.sun.shadow.mapSize.set(1024, 1024);
    Object.assign(this.sun.shadow.camera, { left: -6, right: 6, top: 6, bottom: -6, near: 0.5, far: 40 });
    this.scene.add(this.sun, this.sun.target);
    new ResizeObserver(() => requestAnimationFrame(() => this.resize())).observe(this.canvasBox);
    this.resize();
  }

  get episode() {
    return this.task.checkpoints[this.checkpoint];
  }

  load(i) {
    this.checkpoint = i;
    this.picker.select(i);
    const ep = this.episode;
    this.scrub.max = String(ep.pos.length - 1);
    if (this.ok) {
      ep.static.ids.forEach((id, k) => this.setPose(id, ep.static.pos, ep.static.quat, k));
      this.goal.visible = !!ep.goal;
      if (ep.goal) this.goal.position.set(ep.goal[0], ep.goal[1], 0.006);
      this.homeCamera();
    }
    this.drawStrip();
    this.seek(0);
  }

  setPose(id, pos, quat, k) {
    const mesh = this.meshes.get(id);
    if (!mesh) return;
    mesh.position.set(pos[3 * k] / 1000, pos[3 * k + 1] / 1000, pos[3 * k + 2] / 1000);
    mesh.quaternion.set(quat[4 * k + 1] / 1e4, quat[4 * k + 2] / 1e4, quat[4 * k + 3] / 1e4, quat[4 * k] / 1e4).normalize();
  }

  // The centre of the moving geoms, in three.js coordinates.
  robotCenter() {
    const c = new THREE.Vector3();
    let n = 0;
    for (const id of this.episode.dynamic) {
      const m = this.meshes.get(id);
      if (!m) continue;
      c.add(m.getWorldPosition(new THREE.Vector3()));
      n += 1;
    }
    return n ? c.divideScalar(n) : c;
  }

  homeCamera() {
    const id = this.task.env_id;
    this.setFrameGeometry(0);
    const center = this.robotCenter();
    const maze = /Maze/.test(id);
    const dist = maze ? 7.5 : /Ant|HalfCheetah/.test(id) ? 4.2 : /InvertedPendulum/.test(id) ? 3 : 3.6;
    // Mazes from almost straight above, so walls never hide the robot.
    const dir = maze ? new THREE.Vector3(0, 4, 1) : SIDE_VIEW.test(id) ? new THREE.Vector3(0, 0.25, 1) : new THREE.Vector3(0.7, 0.8, 1);
    this.controls.target.copy(maze ? new THREE.Vector3(0, 0, 0) : center);
    this.camera.position.copy(this.controls.target).addScaledVector(dir.normalize(), dist);
    this.controls.update();
    this.lastCenter = center;
  }

  setFrameGeometry(f) {
    const ep = this.episode;
    ep.dynamic.forEach((id, k) => this.setPose(id, ep.pos[f], ep.quat[f], k));
  }

  seek(f) {
    const ep = this.episode;
    this.frame = Math.max(0, Math.min(ep.pos.length - 1, f));
    this.scrub.value = String(this.frame);
    if (this.ok) {
      this.setFrameGeometry(this.frame);
      if (FOLLOW.test(this.task.env_id)) {
        const c = this.robotCenter();
        const delta = new THREE.Vector3(c.x - this.lastCenter.x, 0, c.z - this.lastCenter.z);
        this.controls.target.add(delta);
        this.camera.position.add(delta);
        this.lastCenter = c;
        this.controls.update();
      }
      const t = this.controls.target;
      this.sun.position.set(t.x - 3, t.y + 8, t.z + 4);
      this.sun.target.position.copy(t);
    }
    this.drawFrameInfo();
    this.render();
  }

  toggle() {
    this.playing = !this.playing;
    this.playBtn.textContent = this.playing ? "Pause" : "Play";
    if (this.playing) {
      if (this.frame >= this.episode.pos.length - 1) this.seek(0);
      let last = performance.now();
      let carry = 0;
      const tick = (now) => {
        if (!this.playing) return;
        carry += ((now - last) / 1000) * this.speed;
        last = now;
        const dt = this.episode.frame_dt;
        if (carry >= dt) {
          const steps = Math.floor(carry / dt);
          carry -= steps * dt;
          if (this.frame + steps >= this.episode.pos.length - 1) {
            this.seek(this.episode.pos.length - 1);
            this.toggle();
            return;
          }
          this.seek(this.frame + steps);
        }
        this.raf = requestAnimationFrame(tick);
      };
      this.raf = requestAnimationFrame(tick);
    } else {
      cancelAnimationFrame(this.raf);
    }
  }

  drawFrameInfo() {
    const ep = this.episode;
    const f = this.frame;
    const i = Math.min(f, ep.reward.length - 1);
    const seconds = f * ep.frame_dt;
    this.timeEl.textContent = `t = ${seconds.toFixed(2)} s · frame ${f + 1} of ${ep.pos.length}`;
    replace(
      this.stats,
      h("span", { class: "counter" }, "episode return", h("b", {}, fmt(ep.return, 1))),
      h("span", { class: "counter" }, "episode length", h("b", {}, `${ep.length} steps`)),
      h("span", { class: "counter" }, "recorded after", h("b", {}, `${ep.env_steps.toLocaleString("en-US")} training steps`)),
      h("span", { class: "counter" }, "reward at this frame", h("b", {}, i >= 0 ? fmtSigned(ep.reward[i], 2) : "—")),
      h("span", { class: "counter" }, "return so far", h("b", {}, i >= 0 ? fmt(ep.return_so_far[i], 1) : "—")),
    );
    const action = ep.action[Math.min(f, ep.action.length - 1)] ?? [];
    replace(
      this.actionBox,
      action.map((a, k) =>
        h(
          "div",
          { class: "action-bar" },
          h("span", { class: "note num" }, `a${k + 1}`),
          h("span", { class: "action-track" }, h("span", { class: "action-fill", style: { left: a < 0 ? `${50 + 50 * Math.max(-1, a)}%` : "50%", width: `${50 * Math.min(1, Math.abs(a))}%` } })),
          h("span", { class: "num" }, fmtSigned(a, 2)),
        ),
      ),
    );
    if (this.cursor) {
      const x = this.stripX(f);
      this.cursor.setAttribute("x1", x);
      this.cursor.setAttribute("x2", x);
    }
  }

  stripX(f) {
    return 40 + (f / Math.max(1, this.episode.pos.length - 1)) * (this.stripW - 50);
  }

  drawStrip() {
    const ep = this.episode;
    const W = (this.stripW = 520);
    const H = 168;
    const rows = [
      { key: "reward", label: "reward", color: "var(--series-2)", data: ep.reward },
      { key: "value", label: "V(s), normalized units", color: "var(--series-1)", data: ep.value },
    ];
    const parts = [];
    rows.forEach((row, r) => {
      const top = 20 + r * 78;
      const lo = Math.min(0, ...row.data);
      const hi = Math.max(0, ...row.data) || 1;
      const Y = (v) => top + 50 - ((v - lo) / (hi - lo || 1)) * 50;
      parts.push(s("line", { x1: 40, x2: W - 10, y1: Y(0), y2: Y(0), class: "chart-baseline" }));
      parts.push(s("text", { x: 36, y: Y(hi) + 4, "text-anchor": "end", class: "chart-label" }, fmt(hi, 1)));
      parts.push(s("text", { x: 36, y: Y(lo) + 4, "text-anchor": "end", class: "chart-label" }, fmt(lo, 1)));
      parts.push(s("text", { x: W - 10, y: top - 6, "text-anchor": "end", class: "chart-title-text" }, row.label));
      const d = row.data.map((v, i) => `${i ? "L" : "M"}${this.stripX(i).toFixed(1)},${Y(v).toFixed(1)}`).join("");
      parts.push(s("path", { d, fill: "none", stroke: row.color, "stroke-width": 1.8 }));
    });
    this.cursor = s("line", { x1: 40, x2: 40, y1: 6, y2: H - 6, stroke: "var(--ink)", "stroke-width": 1.2 });
    this.strip.setAttribute("viewBox", `0 0 ${W} ${H}`);
    replace(this.strip, parts, this.cursor);
    this.strip.onclick = (e) => {
      const rect = this.strip.getBoundingClientRect();
      const x = ((e.clientX - rect.left) / rect.width) * W;
      this.seek(Math.round(((x - 40) / (W - 50)) * (ep.pos.length - 1)));
    };
  }

  resize() {
    if (!this.ok) return;
    const w = this.canvasBox.clientWidth;
    if (!w) return;
    const hgt = Math.round(Math.min(420, Math.max(240, w * 0.45)));
    this.renderer.setSize(w, hgt, false);
    this.renderer.domElement.style.width = `${w}px`;
    this.renderer.domElement.style.height = `${hgt}px`;
    this.camera.aspect = w / hgt;
    this.camera.updateProjectionMatrix();
    this.render();
  }

  render() {
    if (this.ok && this.el.isConnected) this.renderer.render(this.scene, this.camera);
  }

  dispose() {
    this.playing = false;
    cancelAnimationFrame(this.raf);
    if (this.ok) {
      this.controls.dispose();
      this.renderer.dispose();
    }
  }
}

function compact(n) {
  return n >= 1e6 ? `${+(n / 1e6).toFixed(1)}M` : n >= 1e3 ? `${Math.round(n / 1e3)}k` : String(n);
}
