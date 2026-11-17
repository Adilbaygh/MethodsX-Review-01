"""Efficiency-coefficient benchmarks on identical inputs (model, section 8).

M0  single overall efficiency:       G^M0_k = sum_t N_tk / eta_p
M1  level-wise efficiency chain:     G^M1_k = sum_t N_tk / (eta_a_bar * eta_T_bar * eta_M_bar)
P   proposed workflow:               per-terminal paths on the network (propagation.py)

The coefficients of M0 and M1 are calibrated on the seasonal volumes of P for the same
dataset (Bos-Nugteren style volume ratios). This is the most favourable setting for the
benchmarks: both reproduce the seasonal head total exactly, so every remaining difference
comes from how the requirement is distributed over periods and over users.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .dataset import Network
from .propagation import Flows, path_efficiencies


@dataclass
class Benchmark:
    eta_p: float
    eta_a_bar: float
    eta_T_bar: float
    eta_M_bar: float
    G_P: np.ndarray          # (K,)
    G_M0: np.ndarray         # (K,)
    G_M1: np.ndarray         # (K,)
    eta_main: np.ndarray     # (T,)
    eta_term: np.ndarray     # (T,)
    eta_path: np.ndarray     # (T,)
    A_P: np.ndarray          # (T,) seasonal head requirement attributed to each terminal by P
    A_M0: np.ndarray         # (T,)
    A_M1: np.ndarray         # (T,)

    @property
    def dG_M0(self) -> np.ndarray:
        return self.G_M0 - self.G_P

    @property
    def dG_M0_pct(self) -> np.ndarray:
        return 100.0 * self.dG_M0 / self.G_P

    @property
    def attribution_pct(self) -> np.ndarray:
        return 100.0 * (self.A_M0 / self.A_P - 1.0)

    @property
    def reassigned(self) -> float:
        """Seasonal head requirement that M0 attributes to other terminals than P does, m3.

        Both attributions add up to the seasonal head requirement, so the volume M0 adds to
        some terminals equals the volume it removes from the others.
        """
        return 0.5 * float(np.abs(self.A_M0 - self.A_P).sum())


def benchmark(net: Network, f: Flows) -> Benchmark:
    term = [net.edge_index[e] for e in net.edges_of_type('terminal')]
    main_in = [net.edge_index[e] for e in net.children[net.source] if net.edge_type[e] == 'main']
    main_nodes = {net.head[e] for e in net.edges_of_type('main')}
    offtakes = [net.edge_index[e] for e in net.edges_of_type('terminal') if net.tail[e] in main_nodes]

    vol_N = f.N.sum()
    vol_D = f.D.sum()
    vol_T_in = f.Qin[term].sum()
    vol_T_out = f.Qout[term].sum()
    vol_M_in = f.Qin[main_in].sum()
    vol_M_out = f.Qin[offtakes].sum()
    vol_G = f.Gh.sum()

    eta_p = vol_N / vol_G
    eta_a_bar = vol_N / vol_D
    eta_T_bar = vol_T_out / vol_T_in
    eta_M_bar = vol_M_out / vol_M_in

    sumN_k = f.N.sum(axis=0)
    G_M0 = sumN_k / eta_p
    G_M1 = sumN_k / (eta_a_bar * eta_T_bar * eta_M_bar)

    eta_main, eta_term, eta_path = path_efficiencies(net, f.eta_e, f.scalars.eta_a)
    N_t = f.N.sum(axis=1)
    A_P = N_t / eta_path
    A_M0 = N_t / eta_p
    A_M1 = N_t / (eta_a_bar * eta_T_bar * eta_M_bar)
    return Benchmark(eta_p=eta_p, eta_a_bar=eta_a_bar, eta_T_bar=eta_T_bar, eta_M_bar=eta_M_bar,
                     G_P=f.Gh.copy(), G_M0=G_M0, G_M1=G_M1, eta_main=eta_main, eta_term=eta_term,
                     eta_path=eta_path, A_P=A_P, A_M0=A_M0, A_M1=A_M1)
