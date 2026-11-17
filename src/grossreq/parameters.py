"""Parameter file: values, uncertainty ranges and provenance.

Every parameter carries a provenance label (dataset, literature, assumed, author decision),
its source or the reasoning behind the assumption, and a status. Parameters whose status is
not "confirmed" are reported as provisional on every run.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REQUIRED = ('eta_a', 'edge_loss_rel_halfwidth', 'phi_E', 'phi_O', 'rho_S', 'rho_O', 'rho_F', 'delta_S')
SOBOL_GROUPS = ('eta_a', 'eta_e_main', 'eta_e_terminal', 'phi_E', 'phi_O', 'rho_S', 'rho_O', 'rho_F')


class ParameterError(ValueError):
    """Raised when the parameter file is inconsistent."""


@dataclass(frozen=True)
class Distribution:
    type: str                 # 'uniform' or 'triangular'
    low: float
    high: float
    mode: Optional[float] = None

    @property
    def sd(self) -> float:
        a, b = self.low, self.high
        if self.type == 'uniform':
            return (b - a) / math.sqrt(12.0)
        c = self.mode
        return math.sqrt((a * a + b * b + c * c - a * b - a * c - b * c) / 18.0)

    @property
    def mean(self) -> float:
        if self.type == 'uniform':
            return 0.5 * (self.low + self.high)
        return (self.low + self.high + self.mode) / 3.0


@dataclass(frozen=True)
class Parameter:
    name: str
    symbol: str
    description: str
    unit: str
    nominal: float
    distribution: Optional[Distribution]
    provenance: str
    source: str
    status: str


@dataclass(frozen=True)
class EdgeRules:
    short_picket_threshold_m: float
    eta_short_max: float
    eta_short_min: float
    outlets_at_junction: Tuple[str, ...]


@dataclass(frozen=True)
class TemporalSettings:
    """Temporal-scale experiment: Kostiakov-type discharge dependence of seepage."""
    flow_dependent_edge_type: str
    m_nominal: float
    m_values: Tuple[float, ...]          # nominal and sensitivity values, sorted
    tolerance: float
    max_iterations: int
    m_status: str


@dataclass(frozen=True)
class UncertaintySettings:
    master_seed: int
    n_monte_carlo: int
    sobol_base_n: int
    sobol_bootstrap: int
    edge_correlation_cases: Tuple[str, ...]
    sobol_groups: Tuple[str, ...]


@dataclass(frozen=True)
class Scalars:
    """Network-wide scalar parameters used by the deterministic chain."""
    eta_a: float
    phi_E: float
    phi_O: float
    rho_S: float
    rho_O: float
    rho_F: float

    @property
    def phi_S(self) -> float:
        return 1.0 - self.phi_E - self.phi_O


@dataclass(frozen=True)
class ParameterSet:
    dataset_file: str
    primary_variant: str
    sensitivity_variant: str
    edge_rules: EdgeRules
    params: Dict[str, Parameter]
    uncertainty: UncertaintySettings
    temporal: TemporalSettings
    edge_rule_status: Dict[str, str]

    @property
    def variants(self) -> Tuple[str, str]:
        return (self.primary_variant, self.sensitivity_variant)

    def nominal_scalars(self) -> Scalars:
        p = self.params
        return Scalars(eta_a=p['eta_a'].nominal, phi_E=p['phi_E'].nominal, phi_O=p['phi_O'].nominal,
                       rho_S=p['rho_S'].nominal, rho_O=p['rho_O'].nominal, rho_F=p['rho_F'].nominal)

    def provisional(self) -> List[str]:
        names = [n for n, p in self.params.items() if p.status != 'confirmed']
        names += [f'edge_rules.{n}' for n, s in self.edge_rule_status.items() if s != 'confirmed']
        if self.temporal.m_status != 'confirmed':
            names.append('temporal.kostiakov_m')
        return names


def _dist(raw: Optional[dict], name: str) -> Optional[Distribution]:
    if raw is None:
        return None
    kind = raw['type']
    if kind == 'uniform':
        dist = Distribution('uniform', float(raw['low']), float(raw['high']))
    elif kind == 'triangular':
        dist = Distribution('triangular', float(raw['low']), float(raw['high']), float(raw['mode']))
        if not dist.low <= dist.mode <= dist.high:
            raise ParameterError(f'{name}: triangular mode outside [low, high]')
    else:
        raise ParameterError(f'{name}: unknown distribution type {kind!r}')
    if not dist.low < dist.high:
        raise ParameterError(f'{name}: distribution needs low < high')
    return dist


def _check_range(p: Parameter, lo: float, hi: float, lo_open: bool = False) -> None:
    values = [p.nominal]
    if p.distribution is not None:
        values += [p.distribution.low, p.distribution.high]
        if not p.distribution.low <= p.nominal <= p.distribution.high:
            raise ParameterError(f'{p.name}: nominal value outside its distribution range')
    for v in values:
        if v > hi or v < lo or (lo_open and v == lo):
            raise ParameterError(f'{p.name}: value {v} outside the admissible range')


def parameters_from_dict(raw: dict) -> ParameterSet:
    if raw.get('schema') != 'grossreq-parameters/1':
        raise ParameterError('unknown parameter schema')
    er = raw['edge_rules']
    rules = EdgeRules(
        short_picket_threshold_m=float(er['short_picket_threshold_m']['value']),
        eta_short_max=float(er['eta_short_max']['value']),
        eta_short_min=float(er['eta_short_min']['value']),
        outlets_at_junction=tuple(str(t) for t in er['outlets_at_junction']['value']))
    if not 0.0 < rules.eta_short_min <= rules.eta_short_max <= 1.0:
        raise ParameterError('edge rules: need 0 < eta_short_min <= eta_short_max <= 1')
    if rules.short_picket_threshold_m <= 0:
        raise ParameterError('edge rules: threshold must be positive')
    var = raw['variants']
    primary, sensitivity = str(var['primary']), str(var['sensitivity'])
    if {primary, sensitivity} != {'dataset', 'rule'}:
        raise ParameterError('variants: primary and sensitivity must be "dataset" and "rule"')
    rule_status = {k: str(v['status']) for k, v in er.items()}

    params: Dict[str, Parameter] = {}
    for name, rec in raw['parameters'].items():
        params[name] = Parameter(name=name, symbol=str(rec['symbol']), description=str(rec['description']),
                                 unit=str(rec['unit']), nominal=float(rec['nominal']),
                                 distribution=_dist(rec.get('distribution'), name),
                                 provenance=str(rec['provenance']), source=str(rec['source']),
                                 status=str(rec['status']))
    missing = [n for n in REQUIRED if n not in params]
    if missing:
        raise ParameterError(f'missing parameters: {missing}')

    _check_range(params['eta_a'], 0.0, 1.0, lo_open=True)
    for n in ('phi_E', 'phi_O', 'rho_S', 'rho_O', 'rho_F'):
        _check_range(params[n], 0.0, 1.0)
    pe, po = params['phi_E'], params['phi_O']
    hi_e = pe.distribution.high if pe.distribution else pe.nominal
    hi_o = po.distribution.high if po.distribution else po.nominal
    if hi_e + hi_o > 1.0:
        raise ParameterError('phi_E + phi_O can exceed 1 within their ranges')
    r = params['edge_loss_rel_halfwidth'].nominal
    if not 0.0 <= r < 1.0:
        raise ParameterError('edge_loss_rel_halfwidth must lie in [0, 1)')
    if params['delta_S'].nominal != 0.0:
        raise ParameterError('the worked example models no canal storage; delta_S must be 0')
    for n in ('eta_a', 'phi_E', 'phi_O', 'rho_S', 'rho_O', 'rho_F'):
        if params[n].distribution is None:
            raise ParameterError(f'{n}: an uncertainty distribution is required')

    u = raw['uncertainty']
    unc = UncertaintySettings(master_seed=int(u['master_seed']), n_monte_carlo=int(u['n_monte_carlo']),
                              sobol_base_n=int(u['sobol_base_n']), sobol_bootstrap=int(u['sobol_bootstrap']),
                              edge_correlation_cases=tuple(u['edge_correlation_cases']),
                              sobol_groups=tuple(u['sobol_groups']))
    if set(unc.edge_correlation_cases) - {'independent', 'common'}:
        raise ParameterError('edge_correlation_cases must be "independent" and/or "common"')
    if tuple(unc.sobol_groups) != SOBOL_GROUPS:
        raise ParameterError(f'sobol_groups must be {list(SOBOL_GROUPS)}')
    if min(unc.n_monte_carlo, unc.sobol_base_n) < 2 or unc.sobol_bootstrap < 1:
        raise ParameterError('sample sizes too small')

    t = raw['temporal']
    km = t['kostiakov_m']
    m_values = tuple(sorted({float(km['nominal'])} | {float(x) for x in km['sensitivity']}))
    temporal = TemporalSettings(flow_dependent_edge_type=str(t['flow_dependent_edge_type']),
                                m_nominal=float(km['nominal']), m_values=m_values,
                                tolerance=float(t['fixed_point_tolerance']),
                                max_iterations=int(t['max_iterations']), m_status=str(km['status']))
    if temporal.flow_dependent_edge_type not in ('main', 'terminal'):
        raise ParameterError('temporal: flow_dependent_edge_type must be "main" or "terminal"')
    if any(not 0.0 <= m < 1.0 for m in m_values):
        raise ParameterError('temporal: Kostiakov exponents must lie in [0, 1)')
    if not 0.0 < temporal.tolerance < 1e-6 or temporal.max_iterations < 1:
        raise ParameterError('temporal: invalid fixed-point settings')

    return ParameterSet(dataset_file=str(raw['dataset']['file']), primary_variant=primary,
                        sensitivity_variant=sensitivity, edge_rules=rules, params=params,
                        uncertainty=unc, temporal=temporal, edge_rule_status=rule_status)


def load_parameters(path: Path) -> ParameterSet:
    with open(path, encoding='utf-8') as f:
        return parameters_from_dict(json.load(f))
