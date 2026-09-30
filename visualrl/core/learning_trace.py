from __future__ import annotations

import dataclasses
import json
from collections.abc import Mapping
from typing import Any, Iterator

import numpy as np


class LearningTrace(Mapping):
    """A record of one real learning update.

    The algorithm fills it in while it performs the update; visualizations only
    read it. Fields keep the order in which the algorithm computed them, so a
    printed trace reads like the derivation of the update:

    >>> trace = LearningTrace("td0", prediction=0.3, target=0.792)
    >>> trace["target"]
    0.792
    """

    __slots__ = ("_fields",)

    def __init__(self, algorithm: str, **fields: Any) -> None:
        self._fields = {"algorithm": algorithm, **fields}

    @property
    def algorithm(self) -> str:
        return self._fields["algorithm"]

    def __getitem__(self, key: str) -> Any:
        return self._fields[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._fields)

    def __len__(self) -> int:
        return len(self._fields)

    def to_dict(self) -> dict:
        """Plain Python values only (no numpy), ready for JSON."""
        return {k: _plain(v) for k, v in self._fields.items()}

    def to_json(self, **kwargs: Any) -> str:
        return json.dumps(self.to_dict(), **kwargs)

    def explain(self) -> str:
        """The fields as an aligned two-column table."""
        width = max(len(k) for k in self._fields)
        return "\n".join(f"{k:<{width}}  {_format(v)}" for k, v in self._fields.items())

    def __repr__(self) -> str:
        return f"LearningTrace(\n{self.explain()}\n)"


def _plain(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {f.name: _plain(getattr(value, f.name)) for f in dataclasses.fields(value)}
    if isinstance(value, Mapping):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value


def _format(value: Any) -> str:
    value = _plain(value)
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, list) and len(value) > 8:
        return f"[{len(value)} items]"
    return str(value)
