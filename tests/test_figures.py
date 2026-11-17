"""Smoke test: the figure script runs on a quick pipeline run and writes every figure."""

import sys
from pathlib import Path

import pytest

from conftest import DATA, PARAMS, ROOT, SUMS
from grossreq.pipeline import run

pytest.importorskip('matplotlib')
sys.path.insert(0, str(ROOT / 'scripts'))
import make_figures  # noqa: E402

FIGURES = ('fig1_boundaries', 'fig2_benchmark', 'fig3_sobol', 'fig4_temporal')


def test_figures_are_written(tmp_path):
    run(DATA, PARAMS, tmp_path, checksums_path=SUMS, quick=True)
    assert make_figures.main(['--results', str(tmp_path)]) == 0
    for name in FIGURES:
        for ext in ('pdf', 'png'):
            f = Path(tmp_path) / 'figures' / f'{name}.{ext}'
            assert f.exists() and f.stat().st_size > 10000, f
