"""Water-balance closure at each system boundary (model, sections 4-5).

For a boundary B and period k the relative residual is

    eps_Bk = (I - O - dS - C - NC) / I

with I the inflow, O the outflow to the next boundary (or the return to the basin for B5),
dS the storage change, C the consumed water and NC the non-consumed water that leaves the
boundary. Every term is summed from the stored flows, independently of how the flows were
computed, so a non-zero residual exposes a bookkeeping error.

Canal storage change is neglected at the dekad scale (dS = 0, model section 5). In a
requirement propagation, leaves without demand (here the main-canal tail) carry no flow;
this is asserted rather than assumed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from .dataset import Network
from .propagation import Flows

BOUNDARIES = (
    ('B1', 'fields'),
    ('B2', 'farm gates'),
    ('B3a', 'terminal canals'),
    ('B3b', 'main canal'),
    ('B4', 'canal network (head to farm gates)'),
    ('B4+', 'irrigation system including fields'),
    ('B5', 'basin'),
)


@dataclass
class BoundaryBalance:
    code: str
    name: str
    I: np.ndarray
    O: np.ndarray
    dS: np.ndarray
    C: np.ndarray
    NC: np.ndarray

    @property
    def residual(self) -> np.ndarray:
        return relative(self.I - self.O - self.dS - self.C - self.NC, self.I)


def relative(num, den) -> np.ndarray:
    """num / den where den > 0; 0 where both are 0; inf where only den is 0."""
    num = np.asarray(num, dtype=float)
    den = np.asarray(den, dtype=float)
    num, den = np.broadcast_arrays(num, den)
    out = np.zeros(num.shape)
    pos = den > 0
    np.divide(num, den, out=out, where=pos)
    out[~pos & (num != 0)] = np.inf
    return out


def _rows(net: Network, edges) -> List[int]:
    return [net.edge_index[e] for e in edges]


def _assert_no_tail_flow(net: Network, f: Flows) -> None:
    for v in net.leaves_without_demand:
        q = f.Qout[net.edge_index[net.parent_edge[v]]]
        if np.any(q != 0):
            raise AssertionError(f'leaf {v} has no demand but carries flow')


def boundary_balances(net: Network, f: Flows) -> List[BoundaryBalance]:
    _assert_no_tail_flow(net, f)
    zero = np.zeros(f.Gh.shape[0])
    main = _rows(net, net.edges_of_type('main'))
    term = _rows(net, net.edges_of_type('terminal'))
    gate = _rows(net, [net.terminal_edge[t] for t in net.terminals])
    main_in = _rows(net, [e for e in net.children[net.source] if net.edge_type[e] == 'main'])
    main_nodes = {net.head[e] for e in net.edges_of_type('main')}
    offtakes = _rows(net, [e for e in net.edges_of_type('terminal') if net.tail[e] in main_nodes])
    if not main_in or len(offtakes) != len(term):
        raise AssertionError('every terminal edge must branch off a main-canal node')

    def s(a, rows):
        return a[rows].sum(axis=0) if rows else zero.copy()

    sumN, sumD, sumF, sumRF = f.N.sum(axis=0), f.D.sum(axis=0), f.F.sum(axis=0), f.RF.sum(axis=0)
    sumEv, sumSO = f.Ev.sum(axis=0), (f.S + f.O).sum(axis=0)
    b = [
        BoundaryBalance('B1', 'fields', sumD, zero, zero, sumN, sumF),
        BoundaryBalance('B2', 'farm gates', s(f.Qout, gate), sumD, zero, zero, zero),
        BoundaryBalance('B3a', 'terminal canals', s(f.Qin, term), s(f.Qout, term), zero,
                        s(f.Ev, term), s(f.S + f.O, term)),
        BoundaryBalance('B3b', 'main canal', s(f.Qin, main_in), s(f.Qin, offtakes), zero,
                        s(f.Ev, main), s(f.S + f.O, main)),
        BoundaryBalance('B4', 'canal network (head to farm gates)', f.Gh, sumD, zero, sumEv, sumSO),
        BoundaryBalance('B4+', 'irrigation system including fields', f.Gh, zero, zero,
                        sumN + sumEv, sumSO + sumF),
        BoundaryBalance('B5', 'basin', f.Gh, f.R.sum(axis=0) + sumRF, zero,
                        sumN + sumEv, f.U.sum(axis=0) + (sumF - sumRF)),
    ]
    assert tuple((x.code, x.name) for x in b) == BOUNDARIES
    return b


def node_continuity(net: Network, f: Flows) -> np.ndarray:
    """Relative continuity residual at every node, shape (V, K); the source row checks G^h."""
    res = []
    for v in net.nodes:
        if v == net.source:
            head_out = np.zeros_like(f.Gh)
            for c in net.children[v]:
                head_out = head_out + f.Qin[net.edge_index[c]]
            res.append(relative(f.Gh - head_out, f.Gh))
            continue
        inflow = f.Qout[net.edge_index[net.parent_edge[v]]]
        outflow = np.zeros_like(inflow)
        for c in net.children[v]:
            outflow = outflow + f.Qin[net.edge_index[c]]
        if v in net.terminal_index:
            outflow = outflow + f.D[net.terminal_index[v]]
        res.append(relative(inflow - outflow, inflow))
    return np.array(res)


def edge_loss_identity(f: Flows) -> np.ndarray:
    """Relative difference between L_e and lambda_e Qin_e, shape (E, K).

    lambda_e = 1 - eta_e, or the period-specific loss fraction when the flows carry one.
    """
    lam = f.lam if f.lam is not None else (1.0 - f.eta_e)[:, None]
    return relative(f.L - lam * f.Qin, f.Qin)


def basin_depletion_identity(f: Flows) -> np.ndarray:
    """Relative difference between G^basin and consumed plus non-recoverable water, shape (K,)."""
    consumed = f.N.sum(axis=0) + f.Ev.sum(axis=0)
    nonrec = f.U.sum(axis=0) + (f.F - f.RF).sum(axis=0)
    return relative(f.Gbasin - consumed - nonrec, f.Gbasin)
