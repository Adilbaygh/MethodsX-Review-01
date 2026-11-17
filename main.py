"""Run the worked example end to end and write results/.

Usage (from the repository root):
    python main.py                 # full run, sample sizes from DATA/parameters.json
    python main.py --quick         # small samples, for a fast smoke test

Exit status 0 when every check passes, 1 otherwise.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'src'))

from grossreq import __version__  # noqa: E402
from grossreq.pipeline import run  # noqa: E402


def _fmt(x) -> str:
    return f'{x:,.0f}'.replace(',', ' ')


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data', default=str(ROOT / 'DATA' / 'gone_abat_jap_am2a.json'))
    ap.add_argument('--params', default=str(ROOT / 'DATA' / 'parameters.json'))
    ap.add_argument('--checksums', default=str(ROOT / 'DATA' / 'SHA256SUMS'))
    ap.add_argument('--out', default=str(ROOT / 'results'))
    ap.add_argument('--quick', action='store_true', help='small samples for a fast smoke test')
    a = ap.parse_args(argv)

    s = run(Path(a.data), Path(a.params), Path(a.out), checksums_path=Path(a.checksums), quick=a.quick)
    ind = s['indicators']
    net = ind['network']
    print(f'grossreq {__version__}  worked example')
    print(f"dataset   {s['dataset']['file']}  sha256 {s['dataset']['sha256'][:16]}... verified")
    print(f"network   {net['nodes']} nodes, {net['edges']} edges ({net['main_edges']} main, "
          f"{net['terminal_edges']} terminal), {net['terminals']} terminals, {net['periods']} dekads")
    print(f"variants  primary '{s['primary_variant']}'; sensitivity '{s['variants'][1]}' "
          f"(rule adjusts {net['edges_adjusted_by_rule']} terminal edges, "
          f"picket {net['short_group_picket_range_m'][0]:.2f}-{net['short_group_picket_range_m'][1]:.2f} m)")
    print(f"samples   Monte Carlo {s['sample_sizes']['monte_carlo']}, Sobol base {s['sample_sizes']['sobol_base_n']}, "
          f"bootstrap {s['sample_sizes']['sobol_bootstrap']}, seed {s['sample_sizes']['master_seed']}"
          + ('  [QUICK]' if s['quick'] else ''))
    print('checks')
    for c in s['checks']:
        val = c['value'] if isinstance(c['value'], bool) else f"{c['value']:.2e}"
        print(f"  [{'PASS' if c['passed'] else 'FAIL'}] {c['name']}: {val}")
    print('indicators')
    for v in s['variants']:
        r = ind[v]
        print(f"  {v:8s} season: N {_fmt(r['N_season'])} m3, G_h {_fmt(r['G_h_season'])} m3, "
              f"G_basin {_fmt(r['G_basin_season'])} m3 (rho=0: {_fmt(r['G_basin_season_rho0'])}, "
              f"rho=1: {_fmt(r['G_basin_season_rho1'])}), N/G_h {r['overall_efficiency_season']:.4f}")
        print(f"           capacity exceedances: edges {r['edge_capacity_exceedances']}, "
              f"source {r['source_capacity_exceedances']}")
    print(f"  dataset vs rule G_h: season {ind['variant_difference_pct']['season']:+.2f}%, "
          f"max dekad {ind['variant_difference_pct']['max_dekad']:+.2f}%")
    pv = s['primary_variant']
    b = ind['benchmark'][pv]
    print(f"  benchmark ({pv}): eta_p {b['eta_p']:.4f} = {b['eta_a_bar']:.4f} x {b['eta_T_bar']:.4f} x {b['eta_M_bar']:.4f}; "
          f"dG_M0 range {b['dG_pct_range'][0]:+.2f}% .. {b['dG_pct_range'][1]:+.2f}% "
          f"(max |dG| in dekad {b['max_abs_dG_dekad']})")
    print(f"  attribution M0 vs P ({pv}): {b['attribution_pct_min']:+.1f}% ({b['attribution_terminal_min']}) .. "
          f"{b['attribution_pct_max']:+.1f}% ({b['attribution_terminal_max']}); "
          f"reassigned {_fmt(b['reassigned_m3'])} m3 ({b['reassigned_pct_of_head']:.2f}% of G_h)")
    for key, u in ind['uncertainty'].items():
        lo, mid, hi = u['G_h_season_p05_p50_p95']
        print(f"  MC {key:22s} G_h season p05/p50/p95 {_fmt(lo)} / {_fmt(mid)} / {_fmt(hi)}; "
              f"sd MC/first-order {u['sd_ratio_mc_over_first_order_season']:.3f}")
    for out, ranking in ind['sobol_ranking_by_ST'].items():
        print(f"  Sobol ST ranking {out}: {', '.join(ranking[:4])}, ...")
    t = s['temporal_settings']
    for m in t['kostiakov_m_values']:
        tm = ind['temporal'][pv][f'{m:g}']
        tot = tm['G_h_total_by_scale']
        print(f"  temporal ({pv}, m={m:g}): G_h season {_fmt(tot['season'])}, month {_fmt(tot['month'])} "
              f"({tm['month_vs_season_pct']:+.2f}%), dekad {_fmt(tot['dekad'])} ({tm['dekad_vs_season_pct']:+.2f}%); "
              f"dekad vs linear {tm['dekad_diff_pct_min']:+.1f}% ({tm['dekad_diff_pct_min_dekad']}) .. "
              f"{tm['dekad_diff_pct_max']:+.1f}% ({tm['dekad_diff_pct_max_dekad']})")
    print(f"  peak dekad for the step-by-step table: {ind['peak_dekad']}")
    print(f"results   {len(s['files'])} files in {a.out}")
    if s['provisional_parameters']:
        print('provisional parameters (author to confirm): ' + ', '.join(s['provisional_parameters']))
    print('ALL CHECKS PASSED' if s['all_checks_passed'] else 'SOME CHECKS FAILED')
    return 0 if s['all_checks_passed'] else 1


if __name__ == '__main__':
    sys.exit(main())
