"""Checks on small networks whose answers are known in closed form (model, section 10)."""

import numpy as np
import pytest

from conftest import toy_dict
from grossreq.balance import boundary_balances, edge_loss_identity, node_continuity
from grossreq.dataset import DatasetError, network_from_dict
from grossreq.parameters import Scalars
from grossreq.propagation import closed_form_head_requirement, propagate

SC = Scalars(eta_a=0.6, phi_E=0.03, phi_O=0.25, rho_S=0.5, rho_O=0.5, rho_F=0.5)


def branched(make_toy):
    edges = [('A', 'S', 'J1'), ('B', 'J1', 'T1'), ('C', 'J1', 'J2'), ('D', 'J2', 'T2'), ('E', 'J2', 'T3'),
             ('F', 'J2', 'END')]
    users = {'U1': 'T1', 'U2': 'T2', 'U3': 'T3', 'U4': 'T3'}
    demand = {'U1': [100.0, 0.0], 'U2': [50.0, 80.0], 'U3': [10.0, 20.0], 'U4': [5.0, 0.0]}
    eff = {'A': 0.9, 'B': 0.8, 'C': 0.95, 'D': 0.85, 'E': 0.99, 'F': 0.9}
    return make_toy(edges, users, demand, eff)


def test_single_edge_single_terminal(make_toy):
    net = make_toy([('K', 'S', 'T')], {'U': 'T'}, {'U': [100.0, 30.0]}, {'K': 0.8})
    f = propagate(net, np.array([0.8]), SC)
    assert np.allclose(f.Gh, np.array([100.0, 30.0]) / (0.6 * 0.8), rtol=1e-15, atol=0)


def test_unit_efficiencies_give_net_demand(make_toy):
    net = branched(make_toy)
    f = propagate(net, np.ones(len(net.edges)), Scalars(1.0, 0.03, 0.25, 0.5, 0.5, 0.5))
    assert np.allclose(f.Gh, net.net_demand.sum(axis=0), rtol=1e-15, atol=0)
    assert np.allclose(f.Gbasin, net.net_demand.sum(axis=0), rtol=1e-15, atol=0)
    assert np.all(f.L == 0)


def test_users_sharing_a_terminal_are_summed(make_toy):
    net = branched(make_toy)
    assert net.terminals == ('T1', 'T2', 'T3')
    assert np.array_equal(net.net_demand[2], np.array([15.0, 20.0]))


def test_closed_form_equals_recursion(make_toy):
    net = branched(make_toy)
    eta = np.array([net.efficiency[e] for e in net.edges])
    f = propagate(net, eta, SC)
    g = closed_form_head_requirement(net, eta, SC.eta_a)
    assert np.max(np.abs(g - f.Gh) / f.Gh) <= 1e-12
    # hand calculation, period 1
    t1 = 100.0 / (0.6 * 0.9 * 0.8)
    t2 = 50.0 / (0.6 * 0.9 * 0.95 * 0.85)
    t3 = 15.0 / (0.6 * 0.9 * 0.95 * 0.99)
    assert abs(f.Gh[0] - (t1 + t2 + t3)) / f.Gh[0] <= 1e-12


def test_closure_and_continuity(make_toy):
    net = branched(make_toy)
    eta = np.array([net.efficiency[e] for e in net.edges])
    f = propagate(net, eta, SC)
    for b in boundary_balances(net, f):
        assert np.max(np.abs(b.residual)) <= 1e-12, b.code
    assert np.max(np.abs(node_continuity(net, f))) <= 1e-12
    assert np.max(np.abs(edge_loss_identity(f))) <= 1e-12
    # the dead-end edge F carries no requirement
    assert np.all(f.Qin[net.edge_index['F']] == 0)


def test_monotonic_in_every_edge_efficiency(net):
    eta = np.array([net.efficiency[e] for e in net.edges])
    base = propagate(net, eta, SC).Gh
    for i, e in enumerate(net.edges):
        lower = eta.copy()
        lower[i] -= 0.01
        g = propagate(net, lower, SC).Gh
        assert np.all(g >= base), e
        carries = propagate(net, eta, SC).Qin[i] > 0
        assert np.all(g[carries] > base[carries]), e


def test_non_radial_network_is_rejected():
    d = toy_dict([('A', 'S', 'J'), ('B', 'J', 'T'), ('C', 'S', 'T')], {'U': 'T'}, {'U': [1.0, 1.0]},
                 {'A': 0.9, 'B': 0.9, 'C': 0.9})
    with pytest.raises(DatasetError):
        network_from_dict(d)


def test_time_varying_efficiency_is_rejected():
    d = toy_dict([('A', 'S', 'T')], {'U': 'T'}, {'U': [1.0, 1.0]}, {'A': 0.9})
    d['efficiency']['2']['A'] = '0.8'
    with pytest.raises(DatasetError):
        network_from_dict(d)


def test_negative_demand_is_rejected():
    d = toy_dict([('A', 'S', 'T')], {'U': 'T'}, {'U': [1.0, -1.0]}, {'A': 0.9})
    with pytest.raises(DatasetError):
        network_from_dict(d)


def test_reassigned_volume_two_terminals(make_toy):
    from grossreq.benchmarks import benchmark
    edges = [('K1', 'S', 'J1'), ('K2', 'J1', 'T1'), ('K3', 'J1', 'J2'), ('K4', 'J2', 'T2')]
    eff = {'K1': 0.9, 'K2': 0.8, 'K3': 0.95, 'K4': 0.85}
    net = make_toy(edges, {'U1': 'T1', 'U2': 'T2'}, {'U1': [100.0, 0.0], 'U2': [50.0, 80.0]}, eff)
    bm = benchmark(net, propagate(net, np.array([eff[e] for e in net.edges]), SC))
    p1, p2 = 0.6 * 0.9 * 0.8, 0.6 * 0.9 * 0.95 * 0.85
    eta_p = 230.0 / (100.0 / p1 + 130.0 / p2)
    expected = abs(100.0 / eta_p - 100.0 / p1)
    assert abs(bm.reassigned - expected) / expected <= 1e-12
