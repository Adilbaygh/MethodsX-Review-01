"""Checks on the Gone Abat Jap worked example and on the full pipeline."""

import csv
import filecmp
import os
from pathlib import Path

import numpy as np
import pytest

from conftest import DATA, PARAMS, SUMS
from grossreq.benchmarks import benchmark
from grossreq.dataset import sha256_file
from grossreq.edges import edge_table, efficiencies
from grossreq.pipeline import read_checksum, run
from grossreq.propagation import propagate
from grossreq.uncertainty import generators, sobol

# Optional cross-check against an independently produced edge table (same columns as
# results/tables/edge_parameters.csv, with 'efficiency_used' for the rule efficiency).
REFERENCE_EDGE_TABLE = os.environ.get('GROSSREQ_REFERENCE_EDGE_TABLE', '')


def test_dataset_checksum():
    assert read_checksum(SUMS, DATA.name) == sha256_file(DATA)


def test_network_structure(net):
    assert (len(net.nodes), len(net.edges), len(net.terminals), len(net.periods)) == (36, 35, 20, 16)
    assert len(net.edges_of_type('main')) == 15 and len(net.edges_of_type('terminal')) == 20
    assert net.leaves_without_demand == ('Tail1',)
    assert net.periods == tuple(str(k) for k in range(11, 27))
    assert [net.period_days[net.periods.index(p)] for p in ('15', '21', '24')] == [11, 11, 11]
    assert sum(net.edge_type[e] == 'terminal' for e in net.children['Of9']) == 6


def test_edge_rule(net, ps):
    rows, (lmin, lmax) = edge_table(net, ps.edge_rules)
    by = {r['edge']: r for r in rows}
    assert sum(r['adjusted'] for r in rows) == 18
    assert (lmin, round(lmax, 2)) == (0.0, 8.30)
    assert by['K33']['efficiency_rule'] == pytest.approx(0.99) and by['K33']['picket_adjusted']
    assert by['K5']['efficiency_rule'] == pytest.approx(0.95)
    assert by['K6']['efficiency_rule'] == by['K6']['efficiency_dataset'] == 0.85
    assert by['K25']['efficiency_rule'] == by['K25']['efficiency_dataset'] == 0.95
    assert sum(r['length_m'] for r in rows if r['type'] == 'main') == pytest.approx(1806.30, abs=0.01)
    assert all(r['length_m'] is None for r in rows if r['type'] == 'terminal')   # unknown length
    assert all(r['efficiency_rule'] == r['efficiency_dataset'] for r in rows if r['type'] == 'main')


@pytest.mark.skipif(not (REFERENCE_EDGE_TABLE and Path(REFERENCE_EDGE_TABLE).exists()),
                    reason='GROSSREQ_REFERENCE_EDGE_TABLE not set')
def test_edge_rule_matches_reference_table(net, ps):
    rows, _ = edge_table(net, ps.edge_rules)
    by = {r['edge']: r for r in rows}
    with open(REFERENCE_EDGE_TABLE, encoding='utf-8-sig') as f:
        ref = list(csv.DictReader(f))
    assert len(ref) == len(rows)
    for r in ref:
        mine = by[r['edge']]
        assert (mine['tail'], mine['head'], mine['type']) == (r['tail'], r['head'], r['type'])
        assert f"{mine['picket_length_m']:.2f}" == r['picket_length_m']
        if mine['type'] == 'main':
            assert f"{mine['length_m']:.2f}" == r['length_m']
        assert f"{mine['efficiency_rule']:.4f}" == r['efficiency_used']
        assert ('yes' if mine['adjusted'] else 'no') == r['adjusted']


def test_benchmark_identities(net, ps):
    rows, _ = edge_table(net, ps.edge_rules)
    f = propagate(net, efficiencies(net, rows, 'rule'), ps.nominal_scalars())
    bm = benchmark(net, f)
    assert np.max(np.abs(bm.G_M1 - bm.G_M0) / bm.G_M0) <= 1e-12
    assert abs(bm.dG_M0.sum()) / bm.G_P.sum() <= 1e-12
    assert bm.eta_a_bar * bm.eta_T_bar * bm.eta_M_bar == pytest.approx(bm.eta_p, rel=1e-12)
    assert np.allclose(bm.A_M0 / bm.A_P, bm.eta_path / bm.eta_p, rtol=1e-12, atol=0)


def test_sobol_groups_without_effect_are_exactly_zero(net, ps):
    rows, _ = edge_table(net, ps.edge_rules)
    eta = efficiencies(net, rows, 'rule')
    g = generators(1)
    res = sobol(net, ps, eta, 256, 10, g[2], g[3])
    for r in res:
        if r['output'] == 'G_h' and r['group'] in ('phi_E', 'phi_O', 'rho_S', 'rho_O', 'rho_F'):
            assert r['S1'] == 0.0 and r['ST'] == 0.0


@pytest.fixture(scope='module')
def quick_runs(tmp_path_factory):
    a = tmp_path_factory.mktemp('run_a')
    b = tmp_path_factory.mktemp('run_b')
    sa = run(DATA, PARAMS, a, checksums_path=SUMS, quick=True)
    sb = run(DATA, PARAMS, b, checksums_path=SUMS, quick=True)
    return a, b, sa, sb


def test_pipeline_all_checks_pass(quick_runs):
    _, _, sa, _ = quick_runs
    failed = [c['name'] for c in sa['checks'] if not c['passed']]
    assert not failed, failed
    assert sa['all_checks_passed']


def test_pipeline_is_deterministic(quick_runs):
    a, b, sa, sb = quick_runs
    assert sa['files'] == sb['files']
    for rel in sa['files']:
        assert filecmp.cmp(Path(a) / rel, Path(b) / rel, shallow=False), rel
