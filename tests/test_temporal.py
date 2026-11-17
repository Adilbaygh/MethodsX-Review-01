"""Checks on the temporal-scale experiment (discharge-dependent seepage on the main canal)."""

import math

import numpy as np
import pytest

from grossreq.balance import boundary_balances, edge_loss_identity, node_continuity
from grossreq.edges import edge_table, efficiencies
from grossreq.parameters import Scalars
from grossreq.propagation import propagate
from grossreq.temporal import (TemporalError, aggregate, month_of_dekad, period_groups,
                               propagate_flow_dependent, reference_discharge)

TOL, MAXIT = 1e-15, 200


def test_month_of_dekad():
    assert [month_of_dekad(d) for d in (1, 3, 4, 11, 12, 13, 15, 16, 24, 25, 26, 36)] == \
        [1, 1, 2, 4, 4, 5, 5, 6, 8, 9, 9, 12]
    for bad in (0, 37):
        with pytest.raises(TemporalError):
            month_of_dekad(bad)


def test_period_groups(net):
    assert period_groups(net, 'dekad') == [[i] for i in range(16)]
    assert period_groups(net, 'season') == [list(range(16))]
    assert period_groups(net, 'month') == [[0, 1], [2, 3, 4], [5, 6, 7], [8, 9, 10], [11, 12, 13], [14, 15]]
    N, days = aggregate(net, period_groups(net, 'month'))
    assert np.allclose(N.sum(axis=1), net.net_demand.sum(axis=1), rtol=1e-15, atol=0)
    assert list(days) == [20, 31, 30, 31, 31, 20]


def test_known_answer_single_main_edge(make_toy):
    # S -> J main edge with loss 0.1, all of it seepage; J -> T terminal edge without loss
    net = make_toy([('M', 'S', 'J'), ('K', 'J', 'T')], {'U': 'T'}, {'U': [1000.0, 4000.0]}, {'M': 0.9, 'K': 1.0})
    sc = Scalars(eta_a=1.0, phi_E=0.0, phi_O=0.0, rho_S=0.5, rho_O=0.5, rho_F=0.5)
    eta = np.array([0.9, 1.0])
    m = 0.5
    q_ref = reference_discharge(net, eta, sc)
    f, info = propagate_flow_dependent(net, eta, sc, net.net_demand, net.period_days, q_ref, m, 'main', TOL, MAXIT)
    s = 86400.0 * 10.0
    a = 0.1 * math.sqrt(s * q_ref[0])      # x - a sqrt(x) = Q_out with x = Q_in
    for k, q_out in enumerate((1000.0, 4000.0)):
        y = 0.5 * (a + math.sqrt(a * a + 4.0 * q_out))
        assert abs(f.Qin[0, k] - y * y) / (y * y) <= 1e-12
    assert info['fixed_point_residual'] <= 1e-12


@pytest.fixture(scope='module')
def primary_eta(net, ps):
    rows, _ = edge_table(net, ps.edge_rules)
    return efficiencies(net, rows, ps.primary_variant)


def test_m0_reproduces_linear(net, ps, primary_eta):
    sc = ps.nominal_scalars()
    lin = propagate(net, primary_eta, sc)
    q_ref = reference_discharge(net, primary_eta, sc)
    f, _ = propagate_flow_dependent(net, primary_eta, sc, net.net_demand, net.period_days, q_ref, 0.0, 'main', TOL, MAXIT)
    assert np.max(np.abs(f.Gh - lin.Gh) / lin.Gh) <= 1e-12


def test_season_scale_reproduces_linear(net, ps, primary_eta):
    sc = ps.nominal_scalars()
    lin = propagate(net, primary_eta, sc)
    q_ref = reference_discharge(net, primary_eta, sc)
    N, days = aggregate(net, period_groups(net, 'season'))
    for m in ps.temporal.m_values:
        f, _ = propagate_flow_dependent(net, primary_eta, sc, N, days, q_ref, m, 'main', TOL, MAXIT)
        assert abs(f.Gh[0] - lin.Gh.sum()) / lin.Gh.sum() <= 1e-12


def test_flow_dependent_chain_closes(net, ps, primary_eta):
    sc = ps.nominal_scalars()
    q_ref = reference_discharge(net, primary_eta, sc)
    f, info = propagate_flow_dependent(net, primary_eta, sc, net.net_demand, net.period_days, q_ref,
                                       max(ps.temporal.m_values), 'main', TOL, MAXIT)
    for b in boundary_balances(net, f):
        assert np.max(np.abs(b.residual)) <= 1e-12, b.code
    assert np.max(np.abs(node_continuity(net, f))) <= 1e-12
    assert np.max(np.abs(edge_loss_identity(f))) <= 1e-12
    assert info['fixed_point_residual'] <= 1e-12
    # terminal edges keep their constant loss fraction
    term = [net.edge_index[e] for e in net.edges_of_type('terminal')]
    assert np.allclose(f.lam[term], (1.0 - primary_eta[term])[:, None], rtol=0, atol=0)
