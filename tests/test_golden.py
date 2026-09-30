import json
import math
import runpy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def assert_close(saved, fresh, path="golden"):
    if isinstance(fresh, dict):
        assert isinstance(saved, dict) and saved.keys() == fresh.keys(), path
        for key in fresh:
            assert_close(saved[key], fresh[key], f"{path}.{key}")
    elif isinstance(fresh, list):
        assert isinstance(saved, list) and len(saved) == len(fresh), path
        for i, (a, b) in enumerate(zip(saved, fresh)):
            assert_close(a, b, f"{path}[{i}]")
    elif isinstance(fresh, float):
        assert math.isclose(saved, fresh, rel_tol=1e-12, abs_tol=1e-12), path
    else:
        assert saved == fresh, path


def test_golden_traces_are_up_to_date():
    """The JavaScript parity test reads this file. If this fails, an algorithm changed:
    run `python scripts/export_golden_traces.py` and check the diff."""
    build = runpy.run_path(str(ROOT / "scripts" / "export_golden_traces.py"))["build"]
    saved = json.loads((ROOT / "tests" / "golden" / "traces.json").read_text())
    assert_close(saved, json.loads(json.dumps(build())))
