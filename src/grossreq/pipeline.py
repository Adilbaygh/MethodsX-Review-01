"""End-to-end run of the worked example: compute, check and write results/.

Checks are identities that must hold to round-off (tolerance 1e-12) or exact properties of
the computation; a failed check means a programming error and the run exits with status 1.
Indicators are the numbers the article reports or discusses; they are written, not judged.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from . import __version__
from .balance import (basin_depletion_identity, boundary_balances, edge_loss_identity,
                      node_continuity)
from .benchmarks import benchmark
from .dataset import Network, load_network, sha256_file
from .edges import edge_table, efficiencies
from .parameters import ParameterSet, load_parameters
from .propagation import Flows, closed_form_head_requirement, propagate
from .report import pct, ratio, sci, vol, write_csv, write_json
from .temporal import SCALES, experiment
from .uncertainty import (OUTPUTS, STREAM_MC, STREAM_SOBOL_AB, STREAM_SOBOL_BOOT, evaluate,
                          first_order_sd, generators, monte_carlo, nominal_samples, sobol)

TOL = 1e-12


def read_checksum(path: Path, filename: str) -> str:
    with open(path, encoding='utf-8') as f:
        for line in f:
            parts = line.split()
            if len(parts) == 2 and parts[1].lstrip('*').split('/')[-1] == filename:
                return parts[0].lower()
    raise KeyError(f'{filename} not listed in {path}')


class Recorder:
    def __init__(self, out_dir: Path):
        self.out_dir = out_dir
        self.checks: List[Dict] = []
        self.files: List[str] = []

    def check(self, name: str, value: float, tol: float = TOL) -> None:
        value = float(value)
        self.checks.append({'name': name, 'value': value, 'tolerance': tol,
                            'passed': bool(np.isfinite(value) and abs(value) <= tol)})

    def check_true(self, name: str, ok: bool) -> None:
        self.checks.append({'name': name, 'value': bool(ok), 'tolerance': None, 'passed': bool(ok)})

    def csv(self, rel: str, header, rows) -> None:
        write_csv(self.out_dir / rel, header, rows)
        self.files.append(rel)


def _edge_order(rows: List[Dict]) -> List[str]:
    return [r['edge'] for r in sorted(rows, key=lambda r: (r['type'] != 'main', r['picket_tail_m'], r['edge']))]


def _rel_max(a, b) -> float:
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return float(np.max(np.abs(a - b) / np.abs(b)))


def _variant_tables(rec: Recorder, net: Network, f: Flows, variant: str, order: List[str]) -> Dict:
    K = net.periods
    sumN, sumD, sumF, sumRF = f.N.sum(0), f.D.sum(0), f.F.sum(0), f.RF.sum(0)
    sumL, sumEv, sumS, sumO = f.L.sum(0), f.Ev.sum(0), f.S.sum(0), f.O.sum(0)
    sumR, sumU = f.R.sum(0), f.U.sum(0)
    seconds = 86400.0 * net.period_days
    cap_src = net.source_capacity
    rows = []
    for k, p in enumerate(K):
        rows.append([p, int(net.period_days[k]), vol(sumN[k]), vol(sumD[k]), vol(sumF[k]), vol(sumRF[k]),
                     vol(sumL[k]), vol(sumEv[k]), vol(sumS[k]), vol(sumO[k]), vol(sumR[k]), vol(sumU[k]),
                     vol(f.Gh[k]), ratio(f.Gh[k] / seconds[k]), vol(f.Gbasin[k]),
                     ratio(sumN[k] / f.Gh[k]),
                     vol(cap_src[k]) if cap_src is not None else '',
                     ratio(f.Gh[k] / cap_src[k]) if cap_src is not None else ''])
    rows.append(['season', int(net.period_days.sum()), vol(sumN.sum()), vol(sumD.sum()), vol(sumF.sum()),
                 vol(sumRF.sum()), vol(sumL.sum()), vol(sumEv.sum()), vol(sumS.sum()), vol(sumO.sum()),
                 vol(sumR.sum()), vol(sumU.sum()), vol(f.Gh.sum()), ratio(f.Gh.sum() / seconds.sum()),
                 vol(f.Gbasin.sum()), ratio(sumN.sum() / f.Gh.sum()),
                 vol(cap_src.sum()) if cap_src is not None else '',
                 ratio(f.Gh.sum() / cap_src.sum()) if cap_src is not None else ''])
    rec.csv(f'tables/{variant}/dekad_totals.csv',
            ['dekad', 'days', 'N', 'D', 'F', 'RF', 'L', 'E', 'S', 'O', 'R', 'U', 'G_h', 'G_h_m3s',
             'G_basin', 'N_over_G_h', 'source_capacity', 'G_h_over_source_capacity'], rows)

    rows = []
    for k, p in enumerate(K):
        for e in order:
            i = net.edge_index[e]
            cap = net.edge_capacity[i, k] if net.edge_capacity is not None else None
            rows.append([p, e, net.edge_type[e], net.tail[e], net.head[e], ratio(f.eta_e[i]),
                         vol(f.Qin[i, k]), vol(f.Qout[i, k]), vol(f.L[i, k]), vol(f.Ev[i, k]),
                         vol(f.S[i, k]), vol(f.O[i, k]), vol(f.R[i, k]), vol(f.U[i, k]),
                         vol(cap) if cap is not None else '',
                         ratio(f.Qin[i, k] / cap) if cap else ''])
    rec.csv(f'tables/{variant}/edge_flows.csv',
            ['dekad', 'edge', 'type', 'tail', 'head', 'eta_e', 'Q_in', 'Q_out', 'L', 'E', 'S', 'O', 'R', 'U',
             'capacity', 'Q_in_over_capacity'], rows)

    rows = []
    for k, p in enumerate(K):
        for j, t in enumerate(net.terminals):
            rows.append([p, t, net.terminal_edge[t], vol(f.N[j, k]), vol(f.D[j, k]), vol(f.F[j, k]), vol(f.RF[j, k])])
    rec.csv(f'tables/{variant}/terminal_flows.csv', ['dekad', 'terminal', 'terminal_edge', 'N', 'D', 'F', 'RF'], rows)

    bal = boundary_balances(net, f)
    rows = []
    for k, p in enumerate(K):
        for b in bal:
            rows.append([p, b.code, b.name, vol(b.I[k]), vol(b.O[k]), vol(b.dS[k]), vol(b.C[k]), vol(b.NC[k]),
                         sci(b.residual[k])])
    rec.csv(f'tables/{variant}/closure.csv', ['dekad', 'boundary', 'name', 'I', 'O', 'dS', 'C', 'NC', 'residual'], rows)

    rows, n_exceed = [], 0
    if net.edge_capacity is not None:
        for k, p in enumerate(K):
            for e in order:
                i = net.edge_index[e]
                if f.Qin[i, k] > net.edge_capacity[i, k]:
                    n_exceed += 1
                    rows.append([p, e, vol(f.Qin[i, k]), vol(net.edge_capacity[i, k]),
                                 ratio(f.Qin[i, k] / net.edge_capacity[i, k])])
    n_src = 0
    if cap_src is not None:
        for k, p in enumerate(K):
            if f.Gh[k] > cap_src[k]:
                n_src += 1
                rows.append([p, net.source, vol(f.Gh[k]), vol(cap_src[k]), ratio(f.Gh[k] / cap_src[k])])
    rec.csv(f'tables/{variant}/capacity_exceedances.csv', ['dekad', 'element', 'required', 'capacity', 'ratio'], rows)
    return {'edge_capacity_exceedances': n_exceed, 'source_capacity_exceedances': n_src,
            'closure_max_abs_residual_by_boundary': {b.code: float(np.max(np.abs(b.residual))) for b in bal}}


def run(data_path: Path, params_path: Path, out_dir: Path, *, checksums_path: Optional[Path] = None,
        quick: bool = False) -> Dict:
    ps: ParameterSet = load_parameters(params_path)
    digest = sha256_file(data_path)
    if checksums_path is not None:
        expected = read_checksum(checksums_path, data_path.name)
        if expected != digest:
            raise RuntimeError(f'checksum mismatch for {data_path.name}: expected {expected}, found {digest}')
    net = load_network(data_path)
    unc = ps.uncertainty
    n_mc = 2000 if quick else unc.n_monte_carlo
    n_sobol = 1024 if quick else unc.sobol_base_n
    n_boot = 50 if quick else unc.sobol_bootstrap

    rec = Recorder(out_dir)
    ind: Dict = {}
    primary, variants = ps.primary_variant, ps.variants

    # ---- edges -------------------------------------------------------------------------
    rows_e, (lmin, lmax) = edge_table(net, ps.edge_rules)
    order = _edge_order(rows_e)
    by_edge = {r['edge']: r for r in rows_e}
    rec.csv('tables/edge_parameters.csv',
            ['edge', 'tail', 'head', 'type', 'picket_tail_m', 'picket_head_m', 'picket_length_m', 'length_m',
             'picket_adjusted', 'efficiency_dataset', 'efficiency_rule', 'adjusted'],
            [[r['edge'], r['tail'], r['head'], r['type'], f"{r['picket_tail_m']:.2f}", f"{r['picket_head_m']:.2f}",
              f"{r['picket_length_m']:.2f}", '' if r['length_m'] is None else f"{r['length_m']:.2f}",
              'yes' if r['picket_adjusted'] else 'no',
              f"{r['efficiency_dataset']:.4f}", f"{r['efficiency_rule']:.4f}", 'yes' if r['adjusted'] else 'no']
             for r in (by_edge[e] for e in order)])
    ind['network'] = {'nodes': len(net.nodes), 'edges': len(net.edges),
                      'main_edges': len(net.edges_of_type('main')),
                      'terminal_edges': len(net.edges_of_type('terminal')),
                      'terminals': len(net.terminals), 'periods': len(net.periods),
                      'main_canal_length_m': sum(r['length_m'] for r in rows_e if r['type'] == 'main'),
                      'edges_adjusted_by_rule': sum(r['adjusted'] for r in rows_e),
                      'short_group_picket_range_m': [lmin, lmax]}

    # ---- parameters table -------------------------------------------------------------
    prow = []
    for n, p in ps.params.items():
        d = p.distribution
        dist = '' if d is None else (f'uniform({d.low:g}, {d.high:g})' if d.type == 'uniform'
                                     else f'triangular({d.low:g}, {d.mode:g}, {d.high:g})')
        prow.append([n, p.symbol, p.description, p.unit, f'{p.nominal:g}', dist, p.provenance, p.source, p.status])
    rec.csv('tables/parameters.csv',
            ['name', 'symbol', 'description', 'unit', 'nominal', 'distribution', 'provenance', 'source', 'status'], prow)

    # ---- deterministic chain, closure and identities per variant ------------------------
    sc = ps.nominal_scalars()
    flows: Dict[str, Flows] = {}
    for variant in variants:
        eta = efficiencies(net, rows_e, variant)
        f = propagate(net, eta, sc)
        flows[variant] = f
        info = _variant_tables(rec, net, f, variant, order)
        res = np.array([b.residual for b in boundary_balances(net, f)])
        rec.check(f'closure_max_abs_residual[{variant}]', np.max(np.abs(res)))
        rec.check(f'node_continuity_max_abs[{variant}]', np.max(np.abs(node_continuity(net, f))))
        rec.check(f'edge_loss_identity_max_abs[{variant}]', np.max(np.abs(edge_loss_identity(f))))
        rec.check(f'basin_depletion_identity_max_abs[{variant}]', np.max(np.abs(basin_depletion_identity(f))))
        rec.check(f'closed_form_max_rel_diff[{variant}]', _rel_max(closed_form_head_requirement(net, eta, sc.eta_a), f.Gh))
        fs = propagate(net, eta, sc, N=net.net_demand.sum(axis=1, keepdims=True))
        rec.check(f'linearity_rel_diff[{variant}]', abs(f.Gh.sum() - fs.Gh[0]) / f.Gh.sum())
        gh_fast, gb_fast = evaluate(net, nominal_samples(ps, eta))
        rec.check(f'fast_evaluator_max_rel_diff[{variant}]', max(_rel_max(gh_fast[0], f.Gh), _rel_max(gb_fast[0], f.Gbasin)))
        consumed = f.N.sum(0) + f.Ev.sum(0)
        viol = max(0.0, float(np.max((consumed - f.Gbasin) / f.Gh)), float(np.max((f.Gbasin - f.Gh) / f.Gh)))
        rec.check(f'basin_bounds_violation[{variant}]', viol)
        f0 = propagate(net, eta, replace(sc, rho_S=0.0, rho_O=0.0, rho_F=0.0))
        rec.check(f'basin_equals_head_at_rho0[{variant}]', _rel_max(f0.Gbasin, f0.Gh))
        f1 = propagate(net, eta, replace(sc, rho_S=1.0, rho_O=1.0, rho_F=1.0))
        rec.check(f'basin_equals_consumed_at_rho1[{variant}]', _rel_max(f1.Gbasin, f1.N.sum(0) + f1.Ev.sum(0)))
        info.update({'G_h_season': float(f.Gh.sum()), 'G_basin_season': float(f.Gbasin.sum()),
                     'N_season': float(f.N.sum()), 'D_season': float(f.D.sum()),
                     'consumed_season': float(consumed.sum()),
                     'G_basin_season_rho0': float(f0.Gbasin.sum()), 'G_basin_season_rho1': float(f1.Gbasin.sum()),
                     'overall_efficiency_season': float(f.N.sum() / f.Gh.sum())})
        ind[variant] = info
    rec.check_true('dataset_variant_requires_more_water_every_dekad',
                   bool(np.all(flows['dataset'].Gh > flows['rule'].Gh)))
    ind['variant_difference_pct'] = {
        'season': float(100.0 * (flows['dataset'].Gh.sum() / flows['rule'].Gh.sum() - 1.0)),
        'max_dekad': float(np.max(100.0 * (flows['dataset'].Gh / flows['rule'].Gh - 1.0)))}

    # ---- benchmarks ---------------------------------------------------------------------
    rows_bd, rows_bt, ind['benchmark'] = [], [], {}
    for variant in variants:
        f = flows[variant]
        bm = benchmark(net, f)
        rec.check(f'M1_equals_M0_max_rel_diff[{variant}]', _rel_max(bm.G_M1, bm.G_M0))
        rec.check(f'benchmark_season_sum_rel[{variant}]', abs(bm.dG_M0.sum()) / bm.G_P.sum())
        rec.check(f'attribution_P_sums_to_head_rel[{variant}]', abs(bm.A_P.sum() - f.Gh.sum()) / f.Gh.sum())
        rec.check(f'attribution_M0_sums_to_head_rel[{variant}]', abs(bm.A_M0.sum() - f.Gh.sum()) / f.Gh.sum())
        sumN = f.N.sum(0)
        for k, p in enumerate(net.periods):
            rows_bd.append([variant, p, vol(sumN[k]), vol(bm.G_P[k]), vol(bm.G_M0[k]), vol(bm.G_M1[k]),
                            vol(bm.dG_M0[k]), pct(bm.dG_M0_pct[k])])
        N_t = f.N.sum(1)
        for j, t in enumerate(net.terminals):
            rows_bt.append([variant, t, ratio(bm.eta_main[j]), ratio(bm.eta_term[j]), ratio(bm.eta_path[j]),
                            vol(N_t[j]), vol(bm.A_P[j]), vol(bm.A_M0[j]), vol(bm.A_M1[j]), pct(bm.attribution_pct[j])])
        kmax = int(np.argmax(np.abs(bm.dG_M0_pct)))
        jmin, jmax = int(np.argmin(bm.attribution_pct)), int(np.argmax(bm.attribution_pct))
        ind['benchmark'][variant] = {
            'eta_p': bm.eta_p, 'eta_a_bar': bm.eta_a_bar, 'eta_T_bar': bm.eta_T_bar, 'eta_M_bar': bm.eta_M_bar,
            'max_abs_dG_pct': float(np.abs(bm.dG_M0_pct[kmax])), 'max_abs_dG_pct_signed': float(bm.dG_M0_pct[kmax]),
            'max_abs_dG_dekad': net.periods[kmax],
            'dG_pct_range': [float(bm.dG_M0_pct.min()), float(bm.dG_M0_pct.max())],
            'attribution_pct_min': float(bm.attribution_pct[jmin]), 'attribution_terminal_min': net.terminals[jmin],
            'attribution_pct_max': float(bm.attribution_pct[jmax]), 'attribution_terminal_max': net.terminals[jmax],
            'attribution_terminals_over': [t for j, t in enumerate(net.terminals) if bm.attribution_pct[j] > 0.0],
            'reassigned_m3': bm.reassigned,
            'reassigned_pct_of_head': 100.0 * bm.reassigned / float(f.Gh.sum()),
            'path_efficiency_range': [float(bm.eta_path.min()), float(bm.eta_path.max())]}
    rec.csv('tables/benchmark_dekad.csv', ['variant', 'dekad', 'N', 'G_P', 'G_M0', 'G_M1', 'dG_M0', 'dG_M0_pct'], rows_bd)
    rec.csv('tables/benchmark_terminals.csv',
            ['variant', 'terminal', 'eta_main', 'eta_terminal', 'eta_path', 'N_season', 'A_P', 'A_M0', 'A_M1',
             'A_M0_vs_A_P_pct'], rows_bt)

    # ---- step-by-step table for the peak dekad (primary variant) -------------------------
    f = flows[primary]
    kp = int(np.argmax(f.N.sum(0)))
    ind['peak_dekad'] = net.periods[kp]
    rec.csv('tables/stepwise_peak_terminals.csv',
            ['dekad', 'terminal', 'N', 'eta_a', 'D', 'F', 'RF', 'terminal_edge', 'eta_terminal_edge',
             'Q_in_terminal_edge'],
            [[net.periods[kp], t, vol(f.N[j, kp]), ratio(sc.eta_a), vol(f.D[j, kp]), vol(f.F[j, kp]), vol(f.RF[j, kp]),
              net.terminal_edge[t], ratio(f.eta_e[net.edge_index[net.terminal_edge[t]]]),
              vol(f.Qin[net.edge_index[net.terminal_edge[t]], kp])] for j, t in enumerate(net.terminals)])
    rec.csv('tables/stepwise_peak_edges.csv',
            ['dekad', 'edge', 'type', 'tail', 'head', 'Q_out', 'eta_e', 'Q_in', 'L', 'E', 'S', 'O', 'R', 'U'],
            [[net.periods[kp], e, net.edge_type[e], net.tail[e], net.head[e], vol(f.Qout[i, kp]), ratio(f.eta_e[i]),
              vol(f.Qin[i, kp]), vol(f.L[i, kp]), vol(f.Ev[i, kp]), vol(f.S[i, kp]), vol(f.O[i, kp]),
              vol(f.R[i, kp]), vol(f.U[i, kp])] for e in order for i in [net.edge_index[e]]])

    # ---- uncertainty ----------------------------------------------------------------------
    rows_u, ind['uncertainty'] = [], {}
    for corr in unc.edge_correlation_cases:
        for variant in variants:
            f = flows[variant]
            rng = generators(unc.master_seed)[STREAM_MC[corr]]      # common random numbers across variants
            mc = monte_carlo(net, ps, f.eta_e, corr, n_mc, rng)
            fo_k, fo_s = first_order_sd(net, ps, f.eta_e, corr)
            nominal = {'G_h': f.Gh, 'G_basin': f.Gbasin}
            for out in OUTPUTS:
                st_p, st_s = mc[out]['period'], mc[out]['season']
                for k, p in enumerate(net.periods):
                    fo = fo_k[k] if out == 'G_h' else None
                    rows_u.append([variant, corr, out, p, vol(nominal[out][k]), vol(st_p['mean'][k]), vol(st_p['sd'][k]),
                                   vol(st_p['p05'][k]), vol(st_p['p50'][k]), vol(st_p['p95'][k]),
                                   vol(fo) if fo is not None else '', ratio(st_p['sd'][k] / fo) if fo is not None else ''])
                fo = fo_s if out == 'G_h' else None
                rows_u.append([variant, corr, out, 'season', vol(nominal[out].sum()), vol(st_s['mean']), vol(st_s['sd']),
                               vol(st_s['p05']), vol(st_s['p50']), vol(st_s['p95']),
                               vol(fo) if fo is not None else '', ratio(st_s['sd'] / fo) if fo is not None else ''])
            gh_p, gh_s = mc['G_h']['period'], mc['G_h']['season']
            gb_s = mc['G_basin']['season']
            ind['uncertainty'][f'{variant}|{corr}'] = {
                'G_h_season_p05_p50_p95': [float(gh_s['p05']), float(gh_s['p50']), float(gh_s['p95'])],
                'G_basin_season_p05_p50_p95': [float(gb_s['p05']), float(gb_s['p50']), float(gb_s['p95'])],
                'G_h_season_cv': float(gh_s['sd'] / gh_s['mean']),
                'G_h_season_mean_over_nominal_minus_1': float(gh_s['mean'] / f.Gh.sum() - 1.0),
                'sd_ratio_mc_over_first_order_season': float(gh_s['sd'] / fo_s),
                'sd_ratio_mc_over_first_order_max_abs_dev_dekads': float(np.max(np.abs(gh_p['sd'] / fo_k - 1.0)))}
    rec.csv('tables/uncertainty.csv',
            ['variant', 'edge_correlation', 'output', 'dekad', 'nominal', 'mc_mean', 'mc_sd', 'mc_p05', 'mc_p50',
             'mc_p95', 'first_order_sd', 'mc_sd_over_first_order_sd'], rows_u)

    # ---- global sensitivity (primary variant, independent edges) ----------------------------
    gens = generators(unc.master_seed)
    srows = sobol(net, ps, flows[primary].eta_e, n_sobol, n_boot, gens[STREAM_SOBOL_AB], gens[STREAM_SOBOL_BOOT])
    rec.csv('tables/sobol_indices.csv',
            ['output', 'group', 'n_inputs', 'S1', 'S1_ci_low', 'S1_ci_high', 'ST', 'ST_ci_low', 'ST_ci_high'],
            [[r['output'], r['group'], r['n_inputs'], ratio(r['S1']), ratio(r['S1_ci_low']), ratio(r['S1_ci_high']),
              ratio(r['ST']), ratio(r['ST_ci_low']), ratio(r['ST_ci_high'])] for r in srows])
    no_effect = [r for r in srows if r['output'] == 'G_h' and r['group'] in ('phi_E', 'phi_O', 'rho_S', 'rho_O', 'rho_F')]
    rec.check_true('sobol_phi_rho_have_no_effect_on_G_h', all(r['S1'] == 0.0 and r['ST'] == 0.0 for r in no_effect))
    ind['sobol_ranking_by_ST'] = {out: [r['group'] for r in sorted((r for r in srows if r['output'] == out),
                                                                   key=lambda r: (-r['ST'], r['group']))]
                                  for out in OUTPUTS}

    # ---- temporal scale: discharge-dependent seepage on the main canal ----------------------
    tset = ps.temporal
    rows_ts, rows_td, ind['temporal'] = [], [], {}
    for variant in variants:
        f = flows[variant]
        lin, _, runs = experiment(net, f.eta_e, sc, tset.m_values, tset.flow_dependent_edge_type,
                                  tset.tolerance, tset.max_iterations)
        lin_total = lin.Gh.sum()
        rec.check(f'temporal_m0_reproduces_linear[{variant}]',
                  max([_rel_max(runs[(0.0, 'dekad')][0].Gh, lin.Gh)]
                      + [abs(runs[(0.0, sc_)][0].Gh.sum() - lin_total) / lin_total for sc_ in SCALES]))
        rec.check(f'temporal_season_scale_equals_linear[{variant}]',
                  max(abs(runs[(m, 'season')][0].Gh.sum() - lin_total) / lin_total for m in tset.m_values))
        worst_closure = worst_identity = worst_fp = 0.0
        for key in sorted(runs):
            fr, info, _ = runs[key]
            res = np.array([b.residual for b in boundary_balances(net, fr)])
            worst_closure = max(worst_closure, float(np.max(np.abs(res))),
                                float(np.max(np.abs(node_continuity(net, fr)))))
            worst_identity = max(worst_identity, float(np.max(np.abs(edge_loss_identity(fr)))),
                                 float(np.max(np.abs(basin_depletion_identity(fr)))))
            worst_fp = max(worst_fp, info['fixed_point_residual'])
        rec.check(f'temporal_closure_max_abs[{variant}]', worst_closure)
        rec.check(f'temporal_edge_and_basin_identity_max_abs[{variant}]', worst_identity)
        rec.check(f'temporal_fixed_point_residual_max[{variant}]', worst_fp)
        ind['temporal'][variant] = {}
        for m in tset.m_values:
            totals = {sc_: float(runs[(m, sc_)][0].Gh.sum()) for sc_ in SCALES}
            for sc_ in SCALES:
                rows_ts.append([variant, f'{m:g}', sc_, len(runs[(m, sc_)][2]), vol(totals[sc_]),
                                pct(100.0 * (totals[sc_] / totals['season'] - 1.0))])
            fd = runs[(m, 'dekad')][0]
            dpct = 100.0 * (fd.Gh / lin.Gh - 1.0)
            for k, p in enumerate(net.periods):
                rows_td.append([variant, f'{m:g}', p, vol(lin.Gh[k]), vol(fd.Gh[k]), pct(dpct[k])])
            kmax, kmin = int(np.argmax(dpct)), int(np.argmin(dpct))
            ind['temporal'][variant][f'{m:g}'] = {
                'G_h_total_by_scale': totals,
                'dekad_vs_season_pct': 100.0 * (totals['dekad'] / totals['season'] - 1.0),
                'month_vs_season_pct': 100.0 * (totals['month'] / totals['season'] - 1.0),
                'order_dekad_le_month_le_season': bool(totals['dekad'] <= totals['month'] <= totals['season']),
                'dekad_diff_pct_max': float(dpct[kmax]), 'dekad_diff_pct_max_dekad': net.periods[kmax],
                'dekad_diff_pct_min': float(dpct[kmin]), 'dekad_diff_pct_min_dekad': net.periods[kmin],
                'max_fixed_point_iterations': max(runs[(m, sc_)][1]['iterations'] for sc_ in SCALES)}
    rec.csv('tables/temporal_scale.csv',
            ['variant', 'kostiakov_m', 'scale', 'n_periods', 'G_h_total', 'diff_vs_season_pct'], rows_ts)
    rec.csv('tables/temporal_dekads.csv',
            ['variant', 'kostiakov_m', 'dekad', 'G_h_linear', 'G_h_flow_dependent', 'diff_pct'], rows_td)

    summary = {
        'package_version': __version__,
        'dataset': {'file': ps.dataset_file, 'sha256': digest},
        'parameters_file': {'file': params_path.name, 'sha256': sha256_file(params_path)},
        'quick': quick,
        'primary_variant': primary,
        'variants': list(variants),
        'temporal_settings': {'flow_dependent_edge_type': tset.flow_dependent_edge_type,
                              'kostiakov_m_nominal': tset.m_nominal, 'kostiakov_m_values': list(tset.m_values),
                              'fixed_point_tolerance': tset.tolerance},
        'sample_sizes': {'monte_carlo': n_mc, 'sobol_base_n': n_sobol, 'sobol_bootstrap': n_boot,
                         'master_seed': unc.master_seed},
        'nominal_scalars': {'eta_a': sc.eta_a, 'phi_E': sc.phi_E, 'phi_O': sc.phi_O, 'phi_S': sc.phi_S,
                            'rho_S': sc.rho_S, 'rho_O': sc.rho_O, 'rho_F': sc.rho_F},
        'checks': rec.checks,
        'all_checks_passed': all(c['passed'] for c in rec.checks),
        'indicators': ind,
        'provisional_parameters': ps.provisional(),
        'files': sorted(rec.files + ['run_summary.json']),
    }
    write_json(out_dir / 'run_summary.json', summary)
    return summary
