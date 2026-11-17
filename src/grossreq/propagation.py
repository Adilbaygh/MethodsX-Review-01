"""Deterministic propagation of net field demand up a radial canal network.

Equations (model, section 3), for every period k:

* farm-gate demand          D_t = N_t / eta_a
* edge outflow requirement  Qout_e = sum of Qin over edges leaving head(e) + D_t if head(e) is terminal t
* edge inflow requirement   Qin_e = Qout_e / eta_e
* head requirement          G^h = sum of Qin over edges leaving the source
* edge loss and its parts   L_e = Qin_e - Qout_e;  E = phi_E L, S = phi_S L, O = phi_O L
* recoverable / not         R_e = rho_S S + rho_O O;  U_e = (1 - rho_S) S + (1 - rho_O) O
* field loss                F_t = D_t - N_t;  recoverable part RF_t = rho_F F_t
* basin depletion           G^basin = G^h - sum R - sum RF

All quantities are volumes per period (m3).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .dataset import Network
from .parameters import Scalars


@dataclass
class Flows:
    N: np.ndarray       # (T, K) net field demand
    D: np.ndarray       # (T, K) farm-gate demand
    F: np.ndarray       # (T, K) field losses
    RF: np.ndarray      # (T, K) recoverable field losses
    Qin: np.ndarray     # (E, K)
    Qout: np.ndarray    # (E, K)
    L: np.ndarray       # (E, K) edge losses
    Ev: np.ndarray      # (E, K) evaporation from canal surfaces
    S: np.ndarray       # (E, K) seepage
    O: np.ndarray       # (E, K) operational spills
    R: np.ndarray       # (E, K) recoverable part of edge losses
    U: np.ndarray       # (E, K) non-recoverable, non-consumed part of edge losses
    Gh: np.ndarray      # (K,) head requirement
    Gbasin: np.ndarray  # (K,) basin depletion
    eta_e: np.ndarray   # (E,) nominal edge efficiencies
    scalars: Scalars
    lam: Optional[np.ndarray] = None   # (E, K) loss fractions when they vary over periods


def propagate(net: Network, eta_e: np.ndarray, sc: Scalars, N: Optional[np.ndarray] = None) -> Flows:
    eta_e = np.asarray(eta_e, dtype=float)
    if eta_e.shape != (len(net.edges),) or np.any(eta_e <= 0) or np.any(eta_e > 1):
        raise ValueError('edge efficiencies must be a vector in (0, 1] with one value per edge')
    if not 0.0 < sc.eta_a <= 1.0:
        raise ValueError('eta_a must lie in (0, 1]')
    N = net.net_demand if N is None else np.asarray(N, dtype=float)
    n_e, n_k = len(net.edges), N.shape[1]

    D = N / sc.eta_a
    F = D - N
    RF = sc.rho_F * F

    Qin = np.zeros((n_e, n_k))
    Qout = np.zeros((n_e, n_k))
    for e in net.post_order:
        i = net.edge_index[e]
        v = net.head[e]
        q = np.zeros(n_k)
        if v in net.terminal_index:
            q = q + D[net.terminal_index[v]]
        for c in net.children[v]:
            q = q + Qin[net.edge_index[c]]
        Qout[i] = q
        Qin[i] = q / eta_e[i]
    Gh = np.zeros(n_k)
    for c in net.children[net.source]:
        Gh = Gh + Qin[net.edge_index[c]]

    L = Qin - Qout
    Ev = sc.phi_E * L
    S = sc.phi_S * L
    O = sc.phi_O * L
    R = sc.rho_S * S + sc.rho_O * O
    U = (1.0 - sc.rho_S) * S + (1.0 - sc.rho_O) * O
    Gbasin = Gh - R.sum(axis=0) - RF.sum(axis=0)
    return Flows(N=N, D=D, F=F, RF=RF, Qin=Qin, Qout=Qout, L=L, Ev=Ev, S=S, O=O, R=R, U=U,
                 Gh=Gh, Gbasin=Gbasin, eta_e=eta_e, scalars=sc)


def path_efficiencies(net: Network, eta_e: np.ndarray, eta_a: float):
    """Per-terminal efficiency of the main-canal part, the terminal part and the whole path.

    Returns arrays (T,) eta_main, eta_terminal and eta_path = eta_a * eta_main * eta_terminal.
    """
    eta_main = np.ones(len(net.terminals))
    eta_term = np.ones(len(net.terminals))
    for j, t in enumerate(net.terminals):
        for e in net.path_edges[t]:
            if net.edge_type[e] == 'main':
                eta_main[j] *= eta_e[net.edge_index[e]]
            else:
                eta_term[j] *= eta_e[net.edge_index[e]]
    return eta_main, eta_term, eta_a * eta_main * eta_term


def closed_form_head_requirement(net: Network, eta_e: np.ndarray, eta_a: float,
                                 N: Optional[np.ndarray] = None) -> np.ndarray:
    """G^h_k = sum_t N_tk / (eta_a * prod_{e in P(t)} eta_e), summed terminal by terminal."""
    N = net.net_demand if N is None else np.asarray(N, dtype=float)
    _, _, eta_path = path_efficiencies(net, eta_e, eta_a)
    G = np.zeros(N.shape[1])
    for j in range(len(net.terminals)):
        G = G + N[j] / eta_path[j]
    return G


def head_attributed(net: Network, eta_e: np.ndarray, eta_a: float, N: Optional[np.ndarray] = None) -> np.ndarray:
    """H_ek: head requirement attributable to the terminals served through edge e (E, K)."""
    N = net.net_demand if N is None else np.asarray(N, dtype=float)
    _, _, eta_path = path_efficiencies(net, eta_e, eta_a)
    G_t = N / eta_path[:, None]
    H = np.zeros((len(net.edges), N.shape[1]))
    for j, t in enumerate(net.terminals):
        for e in net.path_edges[t]:
            H[net.edge_index[e]] += G_t[j]
    return H
