"""Plain-text views for terminals and notebooks. The web app has the full visuals."""

from __future__ import annotations

import numpy as np

from visualrl.envs import Chain, GridWorld
from visualrl.policies import greedy_actions

ARROWS = ("↑", "→", "↓", "←")


def render_grid(env: GridWorld, values=None, Q=None) -> str:
    """One cell per state: walls '#', cliffs 'C', goals 'G', and either a value or greedy arrows."""
    rows = []
    for r in range(env.height):
        cells = []
        for c in range(env.width):
            s = env.to_state(r, c)
            if s in env.walls:
                cell = "#"
            elif s in env.cliffs:
                cell = "C"
            elif s in env.goals:
                cell = "G"
            elif Q is not None:
                cell = "".join(ARROWS[a] for a in greedy_actions(Q[s]))
            elif values is not None:
                cell = f"{values[s]:.2f}"
            else:
                cell = "S" if s == env.start else "."
            cells.append(cell)
        rows.append(cells)
    width = max(len(cell) for row in rows for cell in row)
    return "\n".join(" ".join(cell.center(width) for cell in row) for row in rows)


def render_path(env: GridWorld, Q, max_steps: int = 200) -> str:
    """The route the greedy policy takes from the start, drawn with '*'."""
    visited = []
    s = env.start
    for _ in range(max_steps):
        visited.append(s)
        s, _, done = env._move(s, int(np.argmax(Q[s])))
        if done:
            break
    lines = []
    for r in range(env.height):
        line = ""
        for c in range(env.width):
            s = env.to_state(r, c)
            if s in env.goals:
                line += "G"
            elif s in env.cliffs:
                line += "C"
            elif s in env.walls:
                line += "#"
            elif s in visited:
                line += "*"
            else:
                line += "."
        lines.append(line)
    return "\n".join(lines)


def render_chain(env: Chain, values) -> str:
    cells = []
    for s in range(env.n_interior + 2):
        cells.append("[T]" if s in env.terminal_states else f"{values[s]:.3f}")
    return "  ".join(cells)
