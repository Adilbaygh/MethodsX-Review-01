"""Uncertainty propagation and grouped Sobol sensitivity indices (model, section 7).

Uncertain inputs (DATA/parameters.json):

* eta_a                       triangular
* edge loss fractions         lambda_e = 1 - eta_e, uniform on lambda_e_nom * (1 - r, 1 + r);
                              independent across edges ("independent") or driven by one common
                              factor ("common", a bounding case)
* phi_E, phi_O, rho_S, rho_O, rho_F   uniform; phi_S = 1 - phi_E - phi_O

Fast evaluator (closed forms, valid for network-wide phi and rho; checked against the
edge-by-edge chain in the pipeline):

    G^h_k     = sum_t N_tk / (eta_a * prod_{e in P(t)} eta_e)
    G^basin_k = G^h_k - (rho_S phi_S + rho_O phi_O) (G^h_k - sum_t D_tk) - rho_F (sum_t D_tk - sum_t N_tk)

Random numbers: PCG64 streams spawned from the master seed. Only uniform doubles are drawn
(Generator.random) and they are transformed by explicit inverse CDFs, so the samples do not
depend on numpy's distribution algorithms.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from .dataset import Network
from .parameters import Distribution, ParameterSet
from .propagation import closed_form_head_requirement, head_attributed

SCALARS = ('phi_E', 'phi_O', 'rho_S', 'rho_O', 'rho_F')
OUTPUTS = ('G_h', 'G_basin')

# fixed assignment of spawned random streams
STREAM_MC = {'independent': 0, 'common': 1}
STREAM_SOBOL_AB = 2
STREAM_SOBOL_BOOT = 3
N_STREAMS = 4


def generators(master_seed: int) -> List[np.random.Generator]:
    children = np.random.SeedSequence(master_seed).spawn(N_STREAMS)
    return [np.random.Generator(np.random.PCG64(c)) for c in children]


@dataclass(frozen=True)
class Layout:
    n_edges: int
    correlation: str

    def __post_init__(self):
        if self.correlation not in ('independent', 'common'):
            raise ValueError(f'unknown correlation case {self.correlation!r}')

    @property
    def n_edge_cols(self) -> int:
        return self.n_edges if self.correlation == 'independent' else 1

    @property
    def dim(self) -> int:
        return 1 + self.n_edge_cols + len(SCALARS)

    @property
    def edge_cols(self) -> List[int]:
        return list(range(1, 1 + self.n_edge_cols))

    def scalar_col(self, name: str) -> int:
        return 1 + self.n_edge_cols + SCALARS.index(name)


@dataclass
class Samples:
    eta_a: np.ndarray    # (n,)
    eta_e: np.ndarray    # (n, E)
    phi_E: np.ndarray
    phi_O: np.ndarray
    rho_S: np.ndarray
    rho_O: np.ndarray
    rho_F: np.ndarray


def ppf(dist: Distribution, u: np.ndarray) -> np.ndarray:
    """Inverse CDF of a uniform or triangular distribution."""
    a, b = dist.low, dist.high
    if dist.type == 'uniform':
        return a + u * (b - a)
    c = dist.mode
    fc = (c - a) / (b - a)
    left = a + np.sqrt(np.clip(u, 0.0, 1.0) * (b - a) * (c - a))
    right = b - np.sqrt(np.clip(1.0 - u, 0.0, 1.0) * (b - a) * (b - c))
    return np.where(u < fc, left, right)


def transform(U: np.ndarray, layout: Layout, ps: ParameterSet, eta_nom: np.ndarray) -> Samples:
    p = ps.params
    r = p['edge_loss_rel_halfwidth'].nominal
    lam_nom = 1.0 - np.asarray(eta_nom, dtype=float)
    u_e = 2.0 * U[:, layout.edge_cols] - 1.0          # (n, E) or (n, 1)
    eta_e = 1.0 - lam_nom[None, :] * (1.0 + r * u_e)
    vals = {n: ppf(p[n].distribution, U[:, layout.scalar_col(n)]) for n in SCALARS}
    return Samples(eta_a=ppf(p['eta_a'].distribution, U[:, 0]), eta_e=eta_e, **vals)


def nominal_samples(ps: ParameterSet, eta_nom: np.ndarray) -> Samples:
    sc = ps.nominal_scalars()
    one = np.ones(1)
    return Samples(eta_a=sc.eta_a * one, eta_e=np.asarray(eta_nom, dtype=float)[None, :],
                   phi_E=sc.phi_E * one, phi_O=sc.phi_O * one, rho_S=sc.rho_S * one,
                   rho_O=sc.rho_O * one, rho_F=sc.rho_F * one)


def evaluate(net: Network, s: Samples) -> Tuple[np.ndarray, np.ndarray]:
    """Head requirement and basin depletion for every sample, each (n, K).

    Products and sums run in a fixed order without BLAS calls, so results do not depend on
    the linear-algebra library or its thread count.
    """
    N = net.net_demand
    Gh = np.zeros((s.eta_a.shape[0], N.shape[1]))
    for j, t in enumerate(net.terminals):
        path_eff = s.eta_a.copy()
        for e in net.path_edges[t]:
            path_eff = path_eff * s.eta_e[:, net.edge_index[e]]
        Gh += (1.0 / path_eff)[:, None] * N[j][None, :]
    N_k = N.sum(axis=0)
    D_k = N_k[None, :] / s.eta_a[:, None]
    phi_S = 1.0 - s.phi_E - s.phi_O
    r_L = s.rho_S * phi_S + s.rho_O * s.phi_O
    Gb = Gh - r_L[:, None] * (Gh - D_k) - s.rho_F[:, None] * (D_k - N_k[None, :])
    return Gh, Gb


def summarize(Y: np.ndarray) -> Dict[str, np.ndarray]:
    return {'mean': Y.mean(axis=0), 'sd': Y.std(axis=0, ddof=1),
            'p05': np.percentile(Y, 5, axis=0), 'p50': np.percentile(Y, 50, axis=0),
            'p95': np.percentile(Y, 95, axis=0)}


def monte_carlo(net: Network, ps: ParameterSet, eta_nom: np.ndarray, correlation: str,
                n: int, rng: np.random.Generator) -> Dict[str, Dict[str, Dict[str, np.ndarray]]]:
    """Statistics of G_h and G_basin per period and for the season."""
    layout = Layout(len(net.edges), correlation)
    U = rng.random((n, layout.dim))
    Gh, Gb = evaluate(net, transform(U, layout, ps, eta_nom))
    out = {}
    for name, Y in (('G_h', Gh), ('G_basin', Gb)):
        out[name] = {'period': summarize(Y), 'season': summarize(Y.sum(axis=1))}
    return out


def first_order_sd(net: Network, ps: ParameterSet, eta_nom: np.ndarray, correlation: str):
    """First-order (linearised) standard deviation of G_h per period and for the season."""
    sc = ps.nominal_scalars()
    eta_nom = np.asarray(eta_nom, dtype=float)
    Gh = closed_form_head_requirement(net, eta_nom, sc.eta_a)
    H = head_attributed(net, eta_nom, sc.eta_a)
    sd_a = ps.params['eta_a'].distribution.sd
    r = ps.params['edge_loss_rel_halfwidth'].nominal
    sd_e = r * (1.0 - eta_nom) / np.sqrt(3.0)
    ga = Gh * sd_a / sc.eta_a
    ge = H * (sd_e / eta_nom)[:, None]
    ga_s = Gh.sum() * sd_a / sc.eta_a
    ge_s = H.sum(axis=1) * sd_e / eta_nom
    if correlation == 'independent':
        var, var_s = ga ** 2 + (ge ** 2).sum(axis=0), ga_s ** 2 + (ge_s ** 2).sum()
    elif correlation == 'common':
        var, var_s = ga ** 2 + ge.sum(axis=0) ** 2, ga_s ** 2 + ge_s.sum() ** 2
    else:
        raise ValueError(f'unknown correlation case {correlation!r}')
    return np.sqrt(var), float(np.sqrt(var_s))


def sobol_groups(net: Network, layout: Layout) -> Dict[str, List[int]]:
    if layout.correlation != 'independent':
        raise ValueError('grouped Sobol indices are computed for independent edge inputs')
    cols = layout.edge_cols
    groups = {'eta_a': [0],
              'eta_e_main': [cols[i] for i, e in enumerate(net.edges) if net.edge_type[e] == 'main'],
              'eta_e_terminal': [cols[i] for i, e in enumerate(net.edges) if net.edge_type[e] == 'terminal']}
    for n in SCALARS:
        groups[n] = [layout.scalar_col(n)]
    used = sorted(c for g in groups.values() for c in g)
    assert used == list(range(layout.dim)), 'groups must partition the inputs'
    return groups


def _indices(ya: np.ndarray, yb: np.ndarray, yab: np.ndarray) -> Tuple[float, float]:
    """Saltelli (2010) first-order and Jansen (1999) total-effect estimators.

    The outputs are centred on their sample mean first. This leaves the expectation of both
    estimators unchanged and removes the large variance the first-order estimator has when
    the output mean is large compared with its spread.
    """
    c = 0.5 * (ya.mean() + yb.mean())
    ya, yb, yab = ya - c, yb - c, yab - c
    v = np.var(np.concatenate([ya, yb]), ddof=1)
    if v == 0.0:
        return 0.0, 0.0
    s1 = float(np.mean(yb * (yab - ya)) / v)
    st = float(0.5 * np.mean((ya - yab) ** 2) / v)
    return s1, st


def sobol(net: Network, ps: ParameterSet, eta_nom: np.ndarray, base_n: int, n_boot: int,
          rng_ab: np.random.Generator, rng_boot: np.random.Generator) -> List[Dict]:
    """Grouped first-order and total Sobol indices of the seasonal G_h and G_basin."""
    layout = Layout(len(net.edges), 'independent')
    groups = sobol_groups(net, layout)
    A = rng_ab.random((base_n, layout.dim))
    B = rng_ab.random((base_n, layout.dim))

    def f(U):
        Gh, Gb = evaluate(net, transform(U, layout, ps, eta_nom))
        return {'G_h': Gh.sum(axis=1), 'G_basin': Gb.sum(axis=1)}

    YA, YB = f(A), f(B)
    YAB = {}
    for g, cols in groups.items():
        AB = A.copy()
        AB[:, cols] = B[:, cols]
        YAB[g] = f(AB)
    pairs = [(out, g) for out in OUTPUTS for g in groups]
    boot = {pg: (np.empty(n_boot), np.empty(n_boot)) for pg in pairs}
    for b in range(n_boot):
        i = np.floor(rng_boot.random(base_n) * base_n).astype(np.int64)
        for out, g in pairs:
            boot[(out, g)][0][b], boot[(out, g)][1][b] = _indices(YA[out][i], YB[out][i], YAB[g][out][i])
    rows = []
    for out, g in pairs:
        s1, st = _indices(YA[out], YB[out], YAB[g][out])
        b1, bt = boot[(out, g)]
        rows.append({'output': out, 'group': g, 'n_inputs': len(groups[g]), 'S1': s1, 'ST': st,
                     'S1_ci_low': float(np.percentile(b1, 2.5)), 'S1_ci_high': float(np.percentile(b1, 97.5)),
                     'ST_ci_low': float(np.percentile(bt, 2.5)), 'ST_ci_high': float(np.percentile(bt, 97.5))})
    return rows
