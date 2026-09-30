// Mirrors visualrl/envs/gridworld.py.
//
//   S start   G goal (terminal)   . empty   # wall
//   C cliff: pays cliffReward and sends the agent back to the start
//
// Cell (r, c) is state r * width + c. Actions: 0 up, 1 right, 2 down, 3 left.
import MAZE_FILE from "../../../../visualrl/envs/maze_maps.json" with { type: "json" };
import { Rng } from "../rng.js";

// Maps shared with Python (visualrl/envs/maze_maps.json), in Gymnasium-Robotics PointMaze format.
export const MAZE_MAPS = Object.fromEntries(Object.entries(MAZE_FILE).filter(([k]) => !k.startsWith("_")));
const MAZE_CELLS = { 1: "#", 0: ".", r: "S", g: "G", h: "C", c: "." };

export function mazeLayout(mazeMap, { start = null, goal = null } = {}) {
  const rows = mazeMap.map((row) => row.map((cell) => MAZE_CELLS[cell]));
  if (start) rows[start[0]][start[1]] = "S";
  if (goal) rows[goal[0]][goal[1]] = "G";
  return rows.map((row) => row.join(""));
}

export class GridWorld {
  static ACTION_NAMES = ["up", "right", "down", "left"];
  static MOVES = [
    [-1, 0],
    [0, 1],
    [1, 0],
    [0, -1],
  ];

  constructor(layout, { stepReward = 0, goalReward = 1, cliffReward = -100, slip = 0, maxSteps = null, seed = 0, goalRewards = null } = {}) {
    this.layout = [...layout];
    this.height = layout.length;
    this.width = layout[0].length;
    if (layout.some((row) => row.length !== this.width)) throw new Error("every layout row must have the same length");
    const find = (ch) => {
      const out = [];
      layout.forEach((row, r) => [...row].forEach((c, col) => c === ch && out.push(r * this.width + col)));
      return out;
    };
    const starts = find("S");
    if (starts.length !== 1) throw new Error("the layout needs exactly one start cell 'S'");
    this.start = starts[0];
    this.goals = find("G");
    this.walls = find("#");
    this.cliffs = find("C");
    this.stepReward = stepReward;
    this.goalReward = goalReward;
    // goalRewards: [[row, col, reward], ...] for goals that pay differently from goalReward
    this.goalRewards = new Map((goalRewards ?? []).map(([r, c, v]) => [r * this.width + c, v]));
    this.cliffReward = cliffReward;
    this.slip = slip;
    this.maxSteps = maxSteps;
    this.nStates = this.height * this.width;
    this.nActions = 4;
    this.rng = new Rng(seed);
    this.state = null;
    this.steps = 0;
  }

  // Sutton & Barto, Example 6.6: -1 per step (the last one too), -100 for the cliff.
  static cliff(options = {}) {
    const layout = ["............", "............", "............", "SCCCCCCCCCCG"];
    return new GridWorld(layout, { stepReward: -1, goalReward: -1, cliffReward: -100, ...options });
  }

  static fromMazeMap(mazeMap, { start = null, goal = null, ...options } = {}) {
    return new GridWorld(mazeLayout(mazeMap, { start, goal }), options);
  }

  // The cliff as a warehouse: dock S, charger G, a loading ledge in between. Same dynamics as cliff().
  static warehouse(options = {}) {
    return GridWorld.fromMazeMap(MAZE_MAPS.warehouse_ledge, { stepReward: -1, goalReward: -1, cliffReward: -100, ...options });
  }

  // The world of Part I: a slow charger (+1) five moves from the dock, a fast one (+10) eight moves away,
  // a ledge (−10, back to the dock) in between, and no cost for moving.
  static chargingRoom(options = {}) {
    return GridWorld.fromMazeMap(MAZE_MAPS.charging_room, {
      stepReward: 0,
      cliffReward: -10,
      goalRewards: [
        [1, 6, 10],
        [4, 4, 1],
      ],
      ...options,
    });
  }

  toState(row, col) {
    return row * this.width + col;
  }

  toCell(state) {
    return [Math.floor(state / this.width), state % this.width];
  }

  move(state, action) {
    const [row, col] = this.toCell(state);
    const [dr, dc] = GridWorld.MOVES[action];
    let r = row + dr;
    let c = col + dc;
    if (r < 0 || r >= this.height || c < 0 || c >= this.width || this.layout[r][c] === "#") [r, c] = [row, col];
    const next = this.toState(r, c);
    if (this.goals.includes(next)) return [next, this.goalRewards.get(next) ?? this.goalReward, true];
    if (this.cliffs.includes(next)) return [this.start, this.cliffReward, false];
    return [next, this.stepReward, false];
  }

  reset() {
    this.state = this.start;
    this.steps = 0;
    return [this.state, {}];
  }

  step(action) {
    if (this.state === null) throw new Error("call reset() before step()");
    let executed = action;
    if (this.slip > 0 && this.rng.random() < this.slip) executed = this.rng.integers(4);
    const from = this.state;
    const [next, reward, terminated] = this.move(from, executed);
    this.state = next;
    this.steps += 1;
    const truncated = !terminated && this.maxSteps !== null && this.steps >= this.maxSteps;
    // Where the agent actually went before a cliff reset, for drawing the fall.
    const [dr, dc] = GridWorld.MOVES[executed];
    const [row, col] = this.toCell(from);
    const [r2, c2] = [row + dr, col + dc];
    const inside = r2 >= 0 && r2 < this.height && c2 >= 0 && c2 < this.width;
    const fellInto = inside && this.cliffs.includes(this.toState(r2, c2)) ? this.toState(r2, c2) : null;
    return [next, reward, terminated, truncated, { executed_action: executed, fell_into: fellInto }];
  }

  model() {
    const terminal = Array.from(
      { length: this.nStates },
      (_, s) => this.goals.includes(s) || this.walls.includes(s) || this.cliffs.includes(s),
    );
    const P = [];
    for (let s = 0; s < this.nStates; s++) {
      if (terminal[s]) {
        P.push(Array.from({ length: 4 }, () => [[1, s, 0, true]]));
        continue;
      }
      const row = [];
      for (let a = 0; a < 4; a++) {
        const merged = new Map();
        const add = (outcome, p) => {
          const key = outcome.join(",");
          merged.set(key, [(merged.get(key)?.[0] ?? 0) + p, ...outcome]);
        };
        add(this.move(s, a), 1 - this.slip);
        if (this.slip > 0) for (let e = 0; e < 4; e++) add(this.move(s, e), this.slip / 4);
        row.push([...merged.values()]);
      }
      P.push(row);
    }
    return { P, terminal, nStates: this.nStates, nActions: 4 };
  }
}
