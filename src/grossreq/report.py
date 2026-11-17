"""Deterministic writers for result files (LF line endings, fixed formats, sorted JSON keys)."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterable, Sequence


def vol(x: float) -> str:
    """Volume in m3."""
    return f'{x:.3f}'


def ratio(x: float) -> str:
    return f'{x:.6f}'


def pct(x: float) -> str:
    return f'{x:.4f}'


def sci(x: float) -> str:
    return f'{x:.3e}'


def write_csv(path: Path, header: Sequence[str], rows: Iterable[Sequence]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, 'w', encoding='utf-8', newline='\n') as f:
        f.write(','.join(header) + '\n')
        for r in rows:
            if len(r) != len(header):
                raise ValueError(f'{path.name}: row length {len(r)} != header length {len(header)}')
            f.write(','.join(_cell(c) for c in r) + '\n')
            n += 1
    return n


def _cell(c) -> str:
    s = str(c)
    if any(ch in s for ch in ',"\n'):
        s = '"' + s.replace('"', '""') + '"'
    return s


def _clean(obj):
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return str(obj)
        return obj
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if hasattr(obj, 'item') and not isinstance(obj, (str, bytes)):   # numpy scalar
        return _clean(obj.item())
    return obj


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8', newline='\n') as f:
        f.write(json.dumps(_clean(obj), indent=2, sort_keys=True, ensure_ascii=True) + '\n')
