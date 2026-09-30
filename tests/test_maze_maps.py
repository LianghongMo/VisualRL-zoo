import importlib.util
import runpy
from pathlib import Path

import pytest

from visualrl.envs import MAZE_MAPS, GridWorld
from visualrl.reference import optimal_values


def test_warehouse_has_exactly_the_cliff_dynamics():
    """The warehouse is the cliff with walls around it: cell (r, c) of the cliff is (r+1, c+1)."""
    cliff, warehouse = GridWorld.cliff(), GridWorld.warehouse()
    to_w = lambda s: warehouse.to_state(*(x + 1 for x in cliff.to_cell(s)))
    P_c, P_w = cliff.model().P, warehouse.model().P
    for s in range(cliff.observation_space.n):
        for a in range(4):
            expected = [(p, to_w(s2), r, done) for p, s2, r, done in P_c[s][a]]
            assert P_w[to_w(s)][a] == expected
    V, _ = optimal_values(warehouse.model(), gamma=1.0)
    assert V[warehouse.start] == pytest.approx(-13.0)


def test_maps_without_marks_take_start_and_goal():
    env = GridWorld.from_maze_map(MAZE_MAPS["u_maze"], start=(3, 1), goal=(1, 1))
    assert env.to_cell(env.start) == (3, 1) and [env.to_cell(g) for g in env.goals] == [(1, 1)]
    V, _ = optimal_values(env.model(), gamma=0.9)
    assert V[env.start] == pytest.approx(0.9**5)  # six moves around the U, the last one pays +1


def test_maze_maps_match_gymnasium_robotics():
    spec = importlib.util.find_spec("gymnasium_robotics")
    if spec is None:
        pytest.skip("gymnasium-robotics is not installed")
    # maps.py has no imports; reading it directly avoids importing MuJoCo.
    maps = runpy.run_path(str(Path(spec.origin).parent / "envs" / "maze" / "maps.py"))
    assert MAZE_MAPS["u_maze"] == maps["U_MAZE"]
    assert MAZE_MAPS["medium_maze"] == maps["MEDIUM_MAZE"]
    assert MAZE_MAPS["large_maze"] == maps["LARGE_MAZE"]
