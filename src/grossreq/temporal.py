"""Temporal-scale experiment: discharge-dependent seepage on the main canal (model, section 6).

For edges of the configured type (the main canal) the loss fraction depends on the mean
inflow discharge q of the period:

    lambda_e(q) = lambda_e_nom * [(1 - phi_S) + phi_S * (q / q_ref_e) ** (-m)]

lambda_e_nom = 1 - eta_e is the dataset loss fraction and q_ref_e is the seasonal mean inflow
discharge of edge e in the linear model. The seepage part follows Kostiakov's dependence on
discharge (seepage fraction proportional to Q**-m); evaporation and operational spills stay
proportional to flow. At q = q_ref_e the loss fraction equals the dataset value, so a single
seasonal period reproduces the linear model. All other edges keep constant efficiency.

For every edge and period the implicit equation Q_in = Q_out / (1 - lambda(Q_in)) is solved by
fixed-point iteration. The calculation is repeated with dekads, calendar months and the whole
season as periods, so that the seasonal totals of the three temporal scales can be compared.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np

from .dataset import Network
from .parameters import Scalars
from .propagation import Flows, propagate

SCALES = ('dekad', 'month', 'season')
SECONDS_PER_DAY = 86400.0


class TemporalError(ValueError):
    """Raised when the flow-dependent calculation leaves its valid range."""


def month_of_dekad(dekad: int) -> int:
    """Calendar month of a dekad numbered 1-36 within the year."""
    if not 1 <= dekad <= 36:
        raise TemporalError(f'dekad number {dekad} outside 1-36')
    return (dekad - 1) // 3 + 1


def period_groups(net: Network, scale: str) -> List[List[int]]:
    """Indices of the dataset periods that form each period of the requested scale."""
    k = len(net.periods)
    if scale == 'dekad':
        return [[i] for i in range(k)]
    if scale == 'season':
        return [list(range(k))]
    if scale == 'month':
        groups: Dict[int, List[int]] = {}
        for i, p in enumerate(net.periods):
            groups.setdefault(month_of_dekad(int(p)), []).append(i)
        return [groups[mo] for mo in sorted(groups)]
    raise TemporalError(f'unknown scale {scale!r}')


def aggregate(net: Network, groups: Sequence[Sequence[int]]) -> Tuple[np.ndarray, np.ndarray]:
    """Net demand (T, G) and duration in days (G,) of the grouped periods."""
    N = np.stack([net.net_demand[:, list(g)].sum(axis=1) for g in groups], axis=1)
    days = np.array([net.period_days[list(g)].sum() for g in groups])
    return N, days


def reference_discharge(net: Network, eta_nom: np.ndarray, sc: Scalars) -> np.ndarray:
    """Seasonal mean inflow discharge of every edge in the linear model, m3/s (E,)."""
    f = propagate(net, eta_nom, sc)
    return f.Qin.sum(axis=1) / (SECONDS_PER_DAY * net.period_days.sum())


def _loss_fraction(lam_nom: float, phi_S: float, ratio: np.ndarray, m: float) -> np.ndarray:
    return lam_nom * ((1.0 - phi_S) + phi_S * ratio ** (-m))


def propagate_flow_dependent(net: Network, eta_nom: np.ndarray, sc: Scalars, N: np.ndarray,
                             days: np.ndarray, q_ref: np.ndarray, m: float, edge_type: str,
                             tol: float, max_iter: int) -> Tuple[Flows, Dict[str, float]]:
    eta_nom = np.asarray(eta_nom, dtype=float)
    N = np.asarray(N, dtype=float)
    seconds = SECONDS_PER_DAY * np.asarray(days, dtype=float)
    n_e, n_k = len(net.edges), N.shape[1]
    lam_nom_all = 1.0 - eta_nom

    D = N / sc.eta_a
    F = D - N
    RF = sc.rho_F * F
    Qin = np.zeros((n_e, n_k))
    Qout = np.zeros((n_e, n_k))
    lam = np.zeros((n_e, n_k))
    max_iter_used, max_residual = 0, 0.0

    for e in net.post_order:
        i = net.edge_index[e]
        v = net.head[e]
        q = np.zeros(n_k)
        if v in net.terminal_index:
            q = q + D[net.terminal_index[v]]
        for c in net.children[v]:
            q = q + Qin[net.edge_index[c]]
        Qout[i] = q
        lam_nom = lam_nom_all[i]
        if net.edge_type[e] == edge_type and m != 0.0 and lam_nom > 0.0 and np.any(q > 0.0):
            if q_ref[i] <= 0.0:
                raise TemporalError(f'edge {e} carries flow here but none in the linear model')
            flowing = q > 0.0
            qin = q / eta_nom[i]
            for it in range(1, max_iter + 1):
                ratio = np.ones(n_k)
                ratio[flowing] = qin[flowing] / seconds[flowing] / q_ref[i]
                lam_i = _loss_fraction(lam_nom, sc.phi_S, ratio, m)
                if np.any(lam_i[flowing] >= 1.0):
                    raise TemporalError(f'loss fraction of edge {e} reaches 1 at low discharge')
                new = np.where(flowing, q / (1.0 - lam_i), 0.0)
                change = float(np.max(np.abs(new[flowing] - qin[flowing]) / new[flowing])) if flowing.any() else 0.0
                qin = new
                if change <= tol:
                    break
            else:
                raise TemporalError(f'fixed-point iteration for edge {e} did not converge')
            ratio = np.ones(n_k)
            ratio[flowing] = qin[flowing] / seconds[flowing] / q_ref[i]
            lam_i = _loss_fraction(lam_nom, sc.phi_S, ratio, m)
            if flowing.any():
                res = np.abs(qin[flowing] * (1.0 - lam_i[flowing]) - q[flowing]) / q[flowing]
                max_residual = max(max_residual, float(np.max(res)))
            max_iter_used = max(max_iter_used, it)
            Qin[i] = qin
            lam[i] = np.where(flowing, lam_i, lam_nom)
        else:
            Qin[i] = q / eta_nom[i]
            lam[i] = lam_nom

    Gh = np.zeros(n_k)
    for c in net.children[net.source]:
        Gh = Gh + Qin[net.edge_index[c]]
    L = Qin - Qout
    Ev = sc.phi_E * lam_nom_all[:, None] * Qin      # proportional to flow
    O = sc.phi_O * lam_nom_all[:, None] * Qin       # proportional to flow
    S = L - Ev - O                                    # seepage: the discharge-dependent part
    R = sc.rho_S * S + sc.rho_O * O
    U = (1.0 - sc.rho_S) * S + (1.0 - sc.rho_O) * O
    Gbasin = Gh - R.sum(axis=0) - RF.sum(axis=0)
    flows = Flows(N=N, D=D, F=F, RF=RF, Qin=Qin, Qout=Qout, L=L, Ev=Ev, S=S, O=O, R=R, U=U,
                  Gh=Gh, Gbasin=Gbasin, eta_e=eta_nom, scalars=sc, lam=lam)
    return flows, {'iterations': max_iter_used, 'fixed_point_residual': max_residual}


def experiment(net: Network, eta_nom: np.ndarray, sc: Scalars, m_values: Sequence[float],
               edge_type: str, tol: float, max_iter: int):
    """Linear reference and flow-dependent runs for every exponent and temporal scale.

    m = 0 is always run as well; it must reproduce the linear model and serves as a check.
    Returns (linear flows, q_ref, {(m, scale): (flows, info, groups)}).
    """
    q_ref = reference_discharge(net, eta_nom, sc)
    lin = propagate(net, eta_nom, sc)
    runs = {}
    for m in sorted({0.0} | {float(x) for x in m_values}):
        for scale in SCALES:
            groups = period_groups(net, scale)
            N, days = aggregate(net, groups)
            f, info = propagate_flow_dependent(net, eta_nom, sc, N, days, q_ref, m, edge_type, tol, max_iter)
            runs[(m, scale)] = (f, info, groups)
    return lin, q_ref, runs
