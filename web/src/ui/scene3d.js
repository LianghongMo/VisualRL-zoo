// The 3D view of a GridWorld: a warehouse floor, walls, a ledge down to the
// loading bay, a dock and a charger, and small delivery robots.
//
// This view only shows behavior. The dynamics are the discrete GridWorld: a
// robot gliding between two cells is an animation of one environment step, and
// every number stays in the 2D views next to it.
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";

import { h } from "./dom.js";

const DIRS = [
  [0, -1],
  [1, 0],
  [0, 1],
  [-1, 0],
];
const LEDGE_DEPTH = 0.75;
const WALL_HEIGHT = 0.22;
const ROBOT_SCALE = 1.25;

const token = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim() || "#888888";
const ease = (t) => (t < 0.5 ? 2 * t * t : 1 - (-2 * t + 2) ** 2 / 2);

function webglAvailable() {
  try {
    const gl = document.createElement("canvas").getContext("webgl2");
    gl?.getExtension("WEBGL_lose_context")?.loseContext(); // free the probe context right away
    return !!gl;
  } catch {
    return false;
  }
}

function labelTexture(text, { fg, bg }) {
  const canvas = document.createElement("canvas");
  canvas.width = 256;
  canvas.height = 128;
  const ctx = canvas.getContext("2d");
  ctx.fillStyle = bg;
  ctx.fillRect(0, 0, 256, 128);
  ctx.fillStyle = fg;
  ctx.font = '700 44px "Recursive", ui-sans-serif, system-ui, sans-serif';
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText(text, 128, 66);
  const tex = new THREE.CanvasTexture(canvas);
  tex.colorSpace = THREE.SRGBColorSpace;
  tex.anisotropy = 4;
  return tex;
}

function stripeTexture(a, b) {
  const canvas = document.createElement("canvas");
  canvas.width = 128;
  canvas.height = 16;
  const ctx = canvas.getContext("2d");
  ctx.fillStyle = b;
  ctx.fillRect(0, 0, 128, 16);
  ctx.fillStyle = a;
  for (let x = -16; x < 144; x += 16) {
    ctx.beginPath();
    ctx.moveTo(x, 16);
    ctx.lineTo(x + 8, 16);
    ctx.lineTo(x + 16, 0);
    ctx.lineTo(x + 8, 0);
    ctx.fill();
  }
  const tex = new THREE.CanvasTexture(canvas);
  tex.colorSpace = THREE.SRGBColorSpace;
  tex.wrapS = THREE.RepeatWrapping;
  return tex;
}

function arrowShape() {
  const s = new THREE.Shape();
  s.moveTo(-0.16, -0.06);
  s.lineTo(0.06, -0.06);
  s.lineTo(0.06, -0.14);
  s.lineTo(0.22, 0);
  s.lineTo(0.06, 0.14);
  s.lineTo(0.06, 0.06);
  s.lineTo(-0.16, 0.06);
  s.closePath();
  return new THREE.ShapeGeometry(s);
}

function frameShape(size = 0.9, width = 0.07) {
  const s = new THREE.Shape();
  const o = size / 2;
  s.moveTo(-o, -o);
  s.lineTo(o, -o);
  s.lineTo(o, o);
  s.lineTo(-o, o);
  s.closePath();
  const hole = new THREE.Path();
  const i = o - width;
  hole.moveTo(-i, -i);
  hole.lineTo(-i, i);
  hole.lineTo(i, i);
  hole.lineTo(i, -i);
  hole.closePath();
  s.holes.push(hole);
  return new THREE.ShapeGeometry(s);
}

class Robot {
  constructor(colorToken) {
    this.colorToken = colorToken;
    this.group = new THREE.Group();
    this.body = new THREE.Group();
    this.group.add(this.body);
    const dark = new THREE.MeshStandardMaterial({ roughness: 0.55, metalness: 0.15 });
    const accent = new THREE.MeshStandardMaterial({ roughness: 0.45 });
    const tire = new THREE.MeshStandardMaterial({ roughness: 0.9 });
    const eye = new THREE.MeshStandardMaterial({ roughness: 0.3, emissiveIntensity: 0.6 });
    this.materials = { dark, accent, tire, eye };

    const base = new THREE.Mesh(new THREE.CylinderGeometry(0.29, 0.31, 0.15, 36), dark);
    base.position.y = 0.12;
    const plate = new THREE.Mesh(new THREE.CylinderGeometry(0.25, 0.25, 0.035, 36), accent);
    plate.position.y = 0.21;
    const mast = new THREE.Mesh(new THREE.CylinderGeometry(0.03, 0.03, 0.2, 12), dark);
    mast.position.set(-0.07, 0.32, 0);
    const head = new THREE.Mesh(new THREE.BoxGeometry(0.12, 0.08, 0.2), dark);
    head.position.set(-0.04, 0.45, 0);
    const lens = new THREE.Mesh(new THREE.BoxGeometry(0.02, 0.04, 0.14), eye);
    lens.position.set(0.025, 0.45, 0);
    const bumper = new THREE.Mesh(new THREE.BoxGeometry(0.05, 0.06, 0.26), eye);
    bumper.position.set(0.29, 0.11, 0);
    this.wheels = [0.3, -0.3].map((z) => {
      const w = new THREE.Mesh(new THREE.CylinderGeometry(0.1, 0.1, 0.05, 24), tire);
      w.rotation.x = Math.PI / 2;
      w.position.set(0, 0.1, z);
      return w;
    });
    for (const m of [base, plate, mast, head, lens, bumper, ...this.wheels]) {
      m.castShadow = true;
      this.body.add(m);
    }
    this.heading = 0;
  }

  recolor() {
    this.materials.dark.color.set(token("--ink"));
    this.materials.accent.color.set(token(this.colorToken));
    this.materials.tire.color.set(token("--ink-2"));
    this.materials.eye.color.set(token("--env"));
    this.materials.eye.emissive.set(token("--env"));
  }

  face(dir) {
    const [dx, dz] = DIRS[dir];
    this.heading = Math.atan2(-dz, dx);
    this.group.rotation.y = this.heading;
  }

  spin(distance) {
    for (const w of this.wheels) w.rotation.y += distance / 0.1;
  }
}

export class WarehouseScene {
  // robots: [{ id, color: "--series-2", offset: [dx, dz] }]; offsets keep robots on the same cell apart
  constructor(env, { robots = [{ id: "main", color: "--series-2" }], caption, goalLabels = {} } = {}) {
    this.env = env;
    this.goalLabels = goalLabels;
    this.el = h("div", { class: "scene3d" });
    this.tweens = [];
    this.trails = new Map();
    this.ok = webglAvailable();
    if (!this.ok) {
      this.el.append(h("p", { class: "scene3d-fallback" }, "The 3D view needs WebGL, which this browser does not provide. The map below shows the same state."));
      return;
    }
    try {
      this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    } catch {
      this.ok = false;
      this.el.append(h("p", { class: "scene3d-fallback" }, "The 3D view could not start. The map below shows the same state."));
      return;
    }
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFShadowMap;
    this.el.append(this.renderer.domElement, caption ? h("p", { class: "scene3d-caption" }, caption) : null);

    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(30, 2, 0.1, 100);
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    Object.assign(this.controls, {
      enablePan: false,
      enableDamping: false,
      minPolarAngle: 0.25,
      maxPolarAngle: 1.2,
      minAzimuthAngle: -0.7,
      maxAzimuthAngle: 0.7,
    });
    this.controls.addEventListener("change", () => this.render());
    this.renderer.domElement.addEventListener("dblclick", () => this.fit());

    this.buildWorld();
    this.robots = new Map(robots.map((r) => [r.id, new Robot(r.color)]));
    this.offsets = new Map(robots.map((r) => [r.id, new THREE.Vector3(r.offset?.[0] ?? 0, 0, r.offset?.[1] ?? 0)]));
    for (const robot of this.robots.values()) {
      this.scene.add(robot.group);
      robot.group.visible = false;
    }
    this.arrow = new THREE.Mesh(arrowShape(), new THREE.MeshBasicMaterial({ transparent: true, opacity: 0.95 }));
    this.arrow.rotation.x = -Math.PI / 2;
    this.arrowHolder = new THREE.Group();
    this.arrowHolder.add(this.arrow);
    this.arrowHolder.visible = false;
    this.scene.add(this.arrowHolder);
    this.marker = new THREE.Mesh(frameShape(), new THREE.MeshBasicMaterial({ transparent: true, opacity: 0.95 }));
    this.marker.rotation.x = -Math.PI / 2;
    this.marker.visible = false;
    this.scene.add(this.marker);

    this.recolor();
    new ResizeObserver(() => this.resize()).observe(this.el);
  }

  pos(state, y = 0) {
    const [r, c] = this.env.toCell(state);
    return new THREE.Vector3(c - (this.env.width - 1) / 2, y, r - (this.env.height - 1) / 2);
  }

  posFor(id, state) {
    return this.pos(state).add(this.offsets.get(id));
  }

  setRobotColor(id, colorToken) {
    const robot = this.ok && this.robots.get(id);
    if (!robot) return;
    robot.colorToken = colorToken;
    robot.recolor();
    this.render();
  }

  buildWorld() {
    const env = this.env;
    const world = new THREE.Group();
    this.scene.add(world);
    this.mats = {
      floor: new THREE.MeshStandardMaterial({ roughness: 0.9 }),
      floorAlt: new THREE.MeshStandardMaterial({ roughness: 0.9 }),
      wall: new THREE.MeshStandardMaterial({ roughness: 0.8 }),
      bay: new THREE.MeshStandardMaterial({ roughness: 1 }),
      face: new THREE.MeshStandardMaterial({ roughness: 1 }),
      pad: new THREE.MeshStandardMaterial({ roughness: 0.5, emissiveIntensity: 0.25 }),
      trail: new THREE.MeshBasicMaterial({ transparent: true }),
    };
    const tile = new THREE.BoxGeometry(0.96, 0.08, 0.96);
    const wall = new THREE.BoxGeometry(1, WALL_HEIGHT, 1);
    for (let s = 0; s < env.nStates; s++) {
      const [r, c] = env.toCell(s);
      const p = this.pos(s);
      if (env.walls.includes(s)) {
        const m = new THREE.Mesh(wall, this.mats.wall);
        m.position.set(p.x, WALL_HEIGHT / 2 - 0.04, p.z);
        m.castShadow = m.receiveShadow = true;
        world.add(m);
      } else if (env.cliffs.includes(s)) {
        const m = new THREE.Mesh(tile, this.mats.bay);
        m.position.set(p.x, -LEDGE_DEPTH - 0.04, p.z);
        m.receiveShadow = true;
        world.add(m);
      } else {
        const m = new THREE.Mesh(tile, (r + c) % 2 ? this.mats.floorAlt : this.mats.floor);
        m.position.set(p.x, -0.04, p.z);
        m.receiveShadow = true;
        world.add(m);
      }
    }
    // The ledge: a vertical face under every floor edge that borders the bay, with a warning stripe on top.
    this.stripeMats = [];
    for (const s of env.cliffs) {
      const [r, c] = env.toCell(s);
      for (const [dr, dc] of [
        [-1, 0],
        [0, -1],
        [0, 1],
      ]) {
        const n = env.toState(r + dr, c + dc);
        if (env.walls.includes(n) || env.cliffs.includes(n)) continue;
        const p = this.pos(s);
        const face = new THREE.Mesh(new THREE.BoxGeometry(dr ? 1 : 0.02, LEDGE_DEPTH, dr ? 0.02 : 1), this.mats.face);
        face.position.set(p.x + dc * 0.5, -LEDGE_DEPTH / 2, p.z + dr * 0.5);
        world.add(face);
        const stripe = new THREE.Mesh(new THREE.PlaneGeometry(dr ? 0.96 : 0.12, dr ? 0.12 : 0.96), null);
        stripe.rotation.x = -Math.PI / 2;
        stripe.position.set(p.x + dc * 0.43, 0.002, p.z + dr * 0.43);
        this.stripeMats.push(stripe);
        world.add(stripe);
      }
    }
    // Dock and charger pads with floor labels.
    this.labels = [];
    const addPad = (s, text, round) => {
      const p = this.pos(s);
      const pad = round ? new THREE.Mesh(new THREE.CylinderGeometry(0.4, 0.42, 0.04, 40), this.mats.pad) : new THREE.Mesh(new THREE.BoxGeometry(0.8, 0.03, 0.8), this.mats.wall);
      pad.position.set(p.x, 0.02, p.z);
      pad.receiveShadow = true;
      world.add(pad);
      const label = new THREE.Mesh(new THREE.PlaneGeometry(0.8, 0.4), new THREE.MeshBasicMaterial({ transparent: true }));
      label.rotation.x = -Math.PI / 2;
      // A sign on top of a wall next to the pad (in front if possible), where no robot drives.
      const [r, c] = env.toCell(s);
      const wallAt = (dr) => r + dr >= 0 && r + dr < env.height && env.walls.includes(env.toState(r + dr, c));
      const side = wallAt(1) ? 1 : wallAt(-1) ? -1 : 0;
      label.position.set(p.x, side ? WALL_HEIGHT - 0.04 + 0.004 : 0.05, p.z + (side ? side : -0.62));
      label.userData.text = text;
      this.labels.push(label);
      world.add(label);
    };
    addPad(env.start, "DOCK", false);
    for (const g of env.goals) addPad(g, this.goalLabels[g] ?? "CHARGER", true);

    this.hemi = new THREE.HemisphereLight(0xffffff, 0x5c6f8f, 1.7);
    this.sun = new THREE.DirectionalLight(0xffffff, 1.3);
    this.sun.position.set(-3, 9, 5);
    this.sun.castShadow = true;
    this.sun.shadow.mapSize.set(1024, 1024);
    const half = Math.max(env.width, env.height) / 2 + 1;
    Object.assign(this.sun.shadow.camera, { left: -half, right: half, top: half, bottom: -half, near: 1, far: 30 });
    this.scene.add(this.hemi, this.sun);
  }

  recolor() {
    if (!this.ok) return;
    const paper = new THREE.Color(token("--paper"));
    const sheet = new THREE.Color(token("--sheet"));
    const ink = new THREE.Color(token("--ink"));
    this.mats.floor.color.copy(sheet);
    this.mats.floorAlt.color.copy(sheet).lerp(ink, 0.05);
    this.mats.wall.color.copy(paper).lerp(ink, 0.2);
    this.mats.bay.color.copy(paper).lerp(new THREE.Color(token("--neg")), 0.45);
    this.mats.face.color.copy(paper).lerp(ink, 0.55);
    this.mats.pad.color.set(token("--env"));
    this.mats.pad.emissive.set(token("--env"));
    this.mats.trail.color.set(token("--env"));
    const stripe = stripeTexture(token("--neg"), token("--sheet"));
    for (const m of this.stripeMats) m.material = new THREE.MeshBasicMaterial({ map: stripe });
    for (const l of this.labels) {
      l.material.map?.dispose();
      l.material.map = labelTexture(l.userData.text, { fg: token("--ink-2"), bg: token("--sheet") });
      l.material.needsUpdate = true;
    }
    this.arrow.material.color.set(token("--env"));
    this.marker.material.color.set(token("--learn"));
    this.hemi.color.set(0xffffff);
    this.hemi.groundColor.copy(paper).lerp(ink, 0.5);
    for (const robot of this.robots.values()) robot.recolor();
    this.render();
  }

  resize() {
    if (!this.ok) return;
    const width = this.el.clientWidth;
    if (!width) return;
    const height = Math.round(Math.min(420, Math.max(220, width * 0.4)));
    this.renderer.setSize(width, height, false);
    this.renderer.domElement.style.width = `${width}px`;
    this.renderer.domElement.style.height = `${height}px`;
    this.camera.aspect = width / height;
    this.fit();
  }

  // Frame the whole map from the near side: find the closest camera distance at which every
  // corner of the map projects inside the view, then center the map vertically.
  fit() {
    const W = this.env.width / 2;
    const H = this.env.height / 2;
    const corners = [];
    for (const x of [-W, W]) for (const z of [-H, H]) for (const y of [-LEDGE_DEPTH, WALL_HEIGHT]) corners.push(new THREE.Vector3(x, y, z));
    const dir = new THREE.Vector3(0, 1.35, 1).normalize();
    const target = new THREE.Vector3(0, 0, 0);
    this.camera.updateProjectionMatrix();
    const project = (d) => {
      this.camera.position.copy(target).addScaledVector(dir, d);
      this.camera.lookAt(target);
      this.camera.updateMatrixWorld();
      let ok = true;
      let lo = Infinity;
      let hi = -Infinity;
      for (const c of corners) {
        const p = c.clone().project(this.camera);
        if (Math.abs(p.x) > 0.95 || Math.abs(p.y) > 0.9) ok = false;
        lo = Math.min(lo, p.y);
        hi = Math.max(hi, p.y);
      }
      return { ok, center: (lo + hi) / 2 };
    };
    let d = 30;
    for (let pass = 0; pass < 2; pass++) {
      let near = 1;
      let far = 80;
      for (let i = 0; i < 40; i++) {
        const mid = (near + far) / 2;
        if (project(mid).ok) far = mid;
        else near = mid;
      }
      d = far;
      const { center } = project(d);
      const up = new THREE.Vector3(0, 1, 0).applyQuaternion(this.camera.quaternion);
      target.addScaledVector(up, center * d * Math.tan((this.camera.fov * Math.PI) / 360));
    }
    this.camera.position.copy(target).addScaledVector(dir, d);
    this.controls.target.copy(target);
    this.controls.minDistance = d * 0.5;
    this.controls.maxDistance = d * 1.5;
    this.controls.update();
    this.render();
  }

  render() {
    if (this.ok && this.el.isConnected) this.renderer.render(this.scene, this.camera);
  }

  animate() {
    if (this.frame) return;
    const tick = (now) => {
      this.tweens = this.tweens.filter((tw) => {
        const t = Math.min(1, (now - tw.start) / tw.duration);
        tw.update(t);
        return t < 1;
      });
      this.render();
      this.frame = this.tweens.length ? requestAnimationFrame(tick) : null;
    };
    this.frame = requestAnimationFrame(tick);
  }

  finishTweens(robotId) {
    this.tweens = this.tweens.filter((tw) => {
      if (tw.robot !== robotId) return true;
      tw.update(1);
      return false;
    });
  }

  place(id, state, dir) {
    if (!this.ok) return;
    const robot = this.robots.get(id);
    this.finishTweens(id);
    robot.group.visible = state !== null && state !== undefined;
    if (!robot.group.visible) return this.render();
    robot.group.position.copy(this.posFor(id, state));
    robot.body.position.set(0, 0, 0);
    robot.body.rotation.set(0, 0, 0);
    robot.group.scale.setScalar(ROBOT_SCALE);
    if (dir !== null && dir !== undefined) robot.face(dir);
    this.render();
  }

  // One environment step, animated: glide to the next cell, or roll off the ledge and reappear at the dock.
  move(id, { from, to, fellInto = null, action }, { duration = 240 } = {}) {
    if (!this.ok) return;
    const robot = this.robots.get(id);
    this.finishTweens(id);
    robot.group.visible = true;
    robot.face(action);
    const a = this.posFor(id, from);
    if (duration <= 0) return this.place(id, to, action);
    const edge = fellInto !== null ? this.posFor(id, fellInto) : null;
    const b = this.posFor(id, to);
    const bump = !edge && from === to;
    this.tweens.push({
      robot: id,
      start: performance.now(),
      duration: edge ? duration * 2.2 : duration,
      update: (t) => {
        robot.body.position.set(0, 0, 0);
        robot.body.rotation.set(0, 0, 0);
        robot.group.scale.setScalar(ROBOT_SCALE);
        if (edge) {
          if (t < 0.4) {
            const u = ease(t / 0.4);
            robot.group.position.lerpVectors(a, edge, u * 0.85);
          } else if (t < 0.75) {
            const u = (t - 0.4) / 0.35;
            robot.group.position.lerpVectors(a, edge, 0.85 + 0.15 * u);
            robot.group.position.y = -LEDGE_DEPTH * u * u;
            robot.body.rotation.z = -0.9 * u;
          } else {
            const u = ease((t - 0.75) / 0.25);
            robot.group.position.copy(b);
            robot.group.scale.setScalar(Math.max(0.01, u) * ROBOT_SCALE);
          }
        } else if (bump) {
          const [dx, dz] = DIRS[action];
          const k = Math.sin(Math.PI * t) * 0.18;
          robot.group.position.set(a.x + dx * k, 0, a.z + dz * k);
        } else {
          const u = ease(t);
          const prev = robot.group.position.clone();
          robot.group.position.lerpVectors(a, b, u);
          robot.spin(prev.distanceTo(robot.group.position));
        }
      },
    });
    this.animate();
  }

  // The flat arrow on the floor: the action the robot will take next.
  showNextAction(state, dir) {
    if (!this.ok) return;
    this.arrowHolder.visible = state !== null && state !== undefined && dir !== null && dir !== undefined;
    if (this.arrowHolder.visible) {
      const [dx, dz] = DIRS[dir];
      const p = this.pos(state, 0.012);
      this.arrowHolder.position.set(p.x + dx * 0.5, p.y, p.z + dz * 0.5);
      this.arrowHolder.rotation.y = Math.atan2(-dz, dx);
    }
    this.render();
  }

  // An amber frame on the floor: the state whose value the last learning step changed.
  markUpdated(state) {
    if (!this.ok) return;
    this.marker.visible = state !== null && state !== undefined;
    if (this.marker.visible) this.marker.position.copy(this.pos(state, 0.008));
    this.render();
  }

  // Small dots on the floor where a robot has been.
  trail(id, states) {
    if (!this.ok) return;
    let group = this.trails.get(id);
    if (!group) {
      group = new THREE.Group();
      this.trails.set(id, group);
      this.scene.add(group);
    }
    group.clear();
    const geo = new THREE.CircleGeometry(0.07, 16);
    states.forEach((s, i) => {
      if (this.env.cliffs.includes(s)) return;
      const m = new THREE.Mesh(geo, this.mats.trail.clone());
      m.material.opacity = 0.25 + (0.6 * (i + 1)) / states.length;
      m.material.color.set(token(this.robots.get(id)?.colorToken ?? "--env"));
      m.rotation.x = -Math.PI / 2;
      m.position.copy(this.posFor(id, s)).setY(0.006);
      group.add(m);
    });
    this.render();
  }

  // Drive every robot along its route, one cell per `stepMs`.
  playRoutes(routes, { stepMs = 260 } = {}) {
    if (!this.ok) return;
    this.stopRoutes();
    const ids = Object.keys(routes);
    const longest = Math.max(...ids.map((id) => routes[id].length));
    let i = 0;
    for (const id of ids) {
      this.place(id, routes[id][0], 1);
      this.trail(id, []);
    }
    const tick = () => {
      i += 1;
      if (i >= longest + 3) {
        i = 0;
        for (const id of ids) {
          this.place(id, routes[id][0], 1);
          this.trail(id, []);
        }
        return;
      }
      for (const id of ids) {
        const path = routes[id];
        if (i >= path.length) continue;
        const from = path[i - 1];
        const to = path[i];
        const [r0, c0] = this.env.toCell(from);
        const [r1, c1] = this.env.toCell(to);
        const action = DIRS.findIndex(([dx, dz]) => dx === c1 - c0 && dz === r1 - r0);
        this.move(id, { from, to, action: action < 0 ? 1 : action }, { duration: stepMs * 0.85 });
        this.trail(id, path.slice(0, i));
      }
    };
    this.routeTimer = setInterval(tick, stepMs);
  }

  // Replay recorded environment steps [{ from, to, fellInto, action }] one after another.
  playSteps(id, steps, { stepMs = 60, onDone } = {}) {
    if (!this.ok) return onDone?.();
    this.stopRoutes();
    let i = 0;
    let at = null;
    const tick = () => {
      if (i >= steps.length) {
        this.stopRoutes();
        onDone?.();
        return;
      }
      const st = steps[i++];
      if (at !== st.from) this.place(id, st.from, st.action); // an episode ended and the robot restarted at the dock
      this.move(id, st, { duration: stepMs * 0.9 });
      at = st.to;
    };
    tick();
    this.routeTimer = setInterval(tick, stepMs);
  }

  stopRoutes() {
    clearInterval(this.routeTimer);
    this.routeTimer = null;
  }

  dispose() {
    this.stopRoutes();
    if (this.frame) cancelAnimationFrame(this.frame);
    if (this.ok) {
      this.controls.dispose();
      this.renderer.dispose();
    }
  }
}
