"""Draw the article figures from the result tables written by main.py.

Usage (from the repository root, after main.py):
    python scripts/make_figures.py                       # reads results/, writes results/figures/
    python scripts/make_figures.py --results DIR         # reads DIR, writes DIR/figures/

Writes fig1_boundaries.{pdf,png} ... fig4_temporal.{pdf,png}. Every value is read from the result
files; nothing is computed here except unit conversions. The primary variant is taken from
run_summary.json.

Colours follow a categorical palette validated for colour-vision deficiency; every series also
carries its own marker or line style, so the figures remain readable in greyscale.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / 'results'
TAB = RES / 'tables'
OUT = RES / 'figures'
MM = 1.0 / 25.4
MINUS = '\N{MINUS SIGN}'


def signed(v: float, digits: int, plus: bool = True) -> str:
    text = f'{v:+.{digits}f}%' if plus else f'{v:.{digits}f}%'
    return text.replace('-', MINUS)

# validated categorical slots (light mode): blue, orange, aqua, yellow
SLOT = ('#2a78d6', '#eb6834', '#1baf7a', '#eda100')
INK, INK2, MUTED = '#0b0b0b', '#52514e', '#898781'
GRID, AXIS = '#e1e0d9', '#c3c2b7'
MONTHS = ((11, 12, 'Apr'), (13, 15, 'May'), (16, 18, 'Jun'), (19, 21, 'Jul'), (22, 24, 'Aug'), (25, 26, 'Sep'))

plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 7.5, 'axes.titlesize': 8, 'axes.labelsize': 7.5,
    'axes.edgecolor': AXIS, 'axes.linewidth': 0.8, 'axes.labelcolor': INK, 'text.color': INK,
    'xtick.color': INK2, 'ytick.color': INK2, 'xtick.major.size': 0, 'ytick.major.size': 0,
    'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6, 'grid.linestyle': '-',
    'axes.spines.top': False, 'axes.spines.right': False, 'legend.frameon': False,
    'legend.fontsize': 7, 'lines.linewidth': 1.4, 'lines.solid_capstyle': 'round',
    'lines.solid_joinstyle': 'round', 'savefig.facecolor': 'white', 'figure.facecolor': 'white',
    'pdf.fonttype': 42, 'svg.fonttype': 'none',
})


def rows(path: Path):
    with open(path, encoding='utf-8') as f:
        return list(csv.DictReader(f))


def save(fig, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f'{name}.pdf', metadata={'CreationDate': None, 'ModDate': None})
    fig.savefig(OUT / f'{name}.png', dpi=300, metadata={'Software': None})
    plt.close(fig)
    print(f'wrote {OUT / name}.pdf and .png')


def month_axis(ax, dekads) -> None:
    ax.set_xticks(dekads)
    ax.set_xlim(dekads[0] - 0.6, dekads[-1] + 0.6)
    for a, b, name in MONTHS:
        if a >= dekads[0] and b <= dekads[-1]:
            ax.annotate(name, xy=((a + b) / 2, 0), xycoords=('data', 'axes fraction'),
                        xytext=(0, -17), textcoords='offset points', ha='center', va='top',
                        color=MUTED, fontsize=6.5)
    ax.set_xlabel('Dekad (ten-day period) of the year', labelpad=14)


def fig1_boundaries(primary: str) -> None:
    tot = [r for r in rows(TAB / primary / 'dekad_totals.csv') if r['dekad'] != 'season']
    unc = {(r['output'], r['dekad']): r for r in rows(TAB / 'uncertainty.csv')
           if r['variant'] == primary and r['edge_correlation'] == 'independent'}
    k = np.array([int(r['dekad']) for r in tot])
    m = 1e-6
    series = [  # slot order follows the importance of the boundary in the article
        ('Head works, $G^{h}$', np.array([float(r['G_h']) for r in tot]) * m, 'G_h', SLOT[0], 'D', '-'),
        ('Basin depletion, $G^{\\mathrm{basin}}$', np.array([float(r['G_basin']) for r in tot]) * m, 'G_basin', SLOT[1], '^', '-'),
        ('Farm gates, $\\sum D$', np.array([float(r['D']) for r in tot]) * m, None, SLOT[2], 's', '--'),
        ('Fields (net), $\\sum N$', np.array([float(r['N']) for r in tot]) * m, None, SLOT[3], 'o', ':'),
    ]
    fig, ax = plt.subplots(figsize=(140 * MM, 78 * MM))
    for label, y, key, col, mk, ls in series:
        if key:
            lo = np.array([float(unc[(key, str(d))]['mc_p05']) for d in k]) * m
            hi = np.array([float(unc[(key, str(d))]['mc_p95']) for d in k]) * m
            ax.fill_between(k, lo, hi, color=col, alpha=0.12, linewidth=0)
        ax.plot(k, y, color=col, linestyle=ls, marker=mk, markersize=4, markeredgecolor='white',
                markeredgewidth=0.8, label=label)
    month_axis(ax, k)
    ax.set_ylabel('Water requirement (million m$^3$ per dekad)')
    ax.set_ylim(0, ax.get_ylim()[1] * 1.12)
    ax.fill_between([], [], [], color=MUTED, alpha=0.25, linewidth=0, label='Shading: Monte Carlo 5-95% range')
    ax.legend(loc='upper left', ncol=2, handlelength=2.6, columnspacing=1.2)
    fig.tight_layout()
    save(fig, 'fig1_boundaries')


def fig2_benchmark(primary: str) -> None:
    bd = [r for r in rows(TAB / 'benchmark_dekad.csv') if r['variant'] == primary]
    bt = [r for r in rows(TAB / 'benchmark_terminals.csv') if r['variant'] == primary]
    fig, (a, b) = plt.subplots(1, 2, figsize=(190 * MM, 72 * MM), gridspec_kw={'width_ratios': [16, 20]})
    k = np.array([int(r['dekad']) for r in bd])
    dg = np.array([float(r['dG_M0_pct']) for r in bd])
    a.bar(k, dg, width=0.62, color=SLOT[0], linewidth=0, zorder=2)
    a.axhline(0, color=AXIS, linewidth=0.8, zorder=1)
    month_axis(a, k)
    a.set_ylabel('$(G^{M0}_k - G^{P}_k)/G^{P}_k$ (%)')
    a.set_title('(a) Head requirement per dekad', loc='left')
    span = dg.max() - dg.min()
    a.set_ylim(dg.min() - 0.12 * span, dg.max() + 0.12 * span)
    i = int(np.argmin(dg))
    a.annotate(signed(dg[i], 1, plus=False), (k[i], dg[i]), xytext=(0, -3), textcoords='offset points',
               ha='center', va='top', color=INK2, fontsize=6.5)
    names = [r['terminal'] for r in bt]
    att = np.array([float(r['A_M0_vs_A_P_pct']) for r in bt])
    x = np.arange(len(names))
    b.bar(x, att, width=0.62, color=SLOT[0], linewidth=0, zorder=2)
    b.axhline(0, color=AXIS, linewidth=0.8, zorder=1)
    b.set_xticks(x)
    b.set_xticklabels(names, rotation=90)
    b.set_xlim(-0.6, len(names) - 0.4)
    b.set_xlabel('Terminal, ordered from the head works to the tail')
    b.set_ylabel('$(A^{M0}_t - A^{P}_t)/A^{P}_t$ (%)')
    b.set_title('(b) Seasonal head requirement attributed to each terminal', loc='left')
    span = att.max() - att.min()
    b.set_ylim(att.min() - 0.12 * span, att.max() + 0.12 * span)
    for j in (int(np.argmax(att)), int(np.argmin(att))):
        b.annotate(signed(att[j], 0), (x[j], att[j]), xytext=(0, 3 if att[j] > 0 else -3),
                   textcoords='offset points', ha='center', va='bottom' if att[j] > 0 else 'top',
                   color=INK2, fontsize=6.5)
    for ax in (a, b):
        ax.grid(axis='x', visible=False)
    fig.tight_layout()
    save(fig, 'fig2_benchmark')


def fig3_sobol() -> None:
    sb = rows(TAB / 'sobol_indices.csv')
    order = ['eta_a', 'eta_e_main', 'eta_e_terminal', 'phi_E', 'phi_O', 'rho_S', 'rho_O', 'rho_F']
    label = {'eta_a': '$\\eta^{a}$ field application', 'eta_e_main': '$\\eta_e$ main canal (15 edges)',
             'eta_e_terminal': '$\\eta_e$ terminal canals (20 edges)', 'phi_E': '$\\phi^{E}$ evaporation share',
             'phi_O': '$\\phi^{O}$ spill share', 'rho_S': '$\\rho^{S}$ seepage recoverable',
             'rho_O': '$\\rho^{O}$ spills recoverable', 'rho_F': '$\\rho^{F}$ field losses recoverable'}
    fig, axes = plt.subplots(1, 2, figsize=(190 * MM, 70 * MM), sharey=True)
    for ax, out, title in zip(axes, ('G_h', 'G_basin'),
                              ('(a) Head requirement $G^{h}$, season', '(b) Basin depletion $G^{\\mathrm{basin}}$, season')):
        d = {r['group']: r for r in sb if r['output'] == out}
        y = np.arange(len(order))[::-1]
        st = np.array([float(d[g]['ST']) for g in order])
        lo = st - np.array([float(d[g]['ST_ci_low']) for g in order])
        hi = np.array([float(d[g]['ST_ci_high']) for g in order]) - st
        s1 = np.array([float(d[g]['S1']) for g in order])
        ax.barh(y, st, height=0.55, color=SLOT[0], linewidth=0, zorder=2, label='Total effect $S_T$')
        ax.errorbar(st, y, xerr=[lo, hi], fmt='none', ecolor=INK2, elinewidth=0.8, capsize=1.5, zorder=3)
        ax.plot(s1, y, linestyle='none', marker='o', markersize=4, color=SLOT[1], markeredgecolor='white',
                markeredgewidth=0.8, zorder=4, label='First order $S_1$')
        ax.set_xlim(0, 1)
        ax.set_xlabel('Sobol index (whiskers: 95% bootstrap interval of $S_T$)')
        ax.set_title(title, loc='left')
        ax.grid(axis='y', visible=False)
        for yy, v in zip(y, st):
            if v >= 0.03:
                ax.annotate(f'{v:.2f}', (v, yy), xytext=(4, 0), textcoords='offset points', va='center',
                            color=INK2, fontsize=6.5)
    axes[0].set_yticks(np.arange(len(order))[::-1])
    axes[0].set_yticklabels([label[g] for g in order])
    axes[1].legend(loc='lower right')
    fig.tight_layout()
    save(fig, 'fig3_sobol')


def fig4_temporal(primary: str, m_nominal: float) -> None:
    td = [r for r in rows(TAB / 'temporal_dekads.csv') if r['variant'] == primary]
    ms = sorted({float(r['kostiakov_m']) for r in td}, key=lambda v: (v != m_nominal, v))
    fig, ax = plt.subplots(figsize=(140 * MM, 75 * MM))
    styles = [('-', 'D', SLOT[0]), ('--', 'o', SLOT[1])]
    for (ls, mk, col), m in zip(styles, ms):
        sel = [r for r in td if float(r['kostiakov_m']) == m]
        k = np.array([int(r['dekad']) for r in sel])
        y = np.array([float(r['diff_pct']) for r in sel])
        tag = 'nominal' if m == m_nominal else 'sensitivity'
        ax.plot(k, y, linestyle=ls, marker=mk, markersize=4, color=col, markeredgecolor='white',
                markeredgewidth=0.8, label=f'$m = {m:g}$ ({tag})')
        j = int(np.argmax(y))
        ax.annotate(signed(y[j], 0), (k[j], y[j]), xytext=(6, 0), textcoords='offset points',
                    ha='left', va='center', color=INK2, fontsize=6.5)
    ax.axhline(0, color=AXIS, linewidth=0.8)
    month_axis(ax, k)
    ax.set_xlim(k[0] - 0.6, k[-1] + 1.4)
    ax.set_ylabel('Head requirement, discharge-dependent\nversus constant losses (%)')
    ax.legend(loc='upper center')
    fig.tight_layout()
    save(fig, 'fig4_temporal')


def main(argv=None) -> int:
    global RES, TAB, OUT
    ap = argparse.ArgumentParser(description='Draw the article figures from the result tables.')
    ap.add_argument('--results', default=str(RES), help='directory written by main.py')
    a = ap.parse_args(argv)
    RES = Path(a.results)
    TAB, OUT = RES / 'tables', RES / 'figures'
    s = json.loads((RES / 'run_summary.json').read_text(encoding='utf-8'))
    primary = s['primary_variant']
    fig1_boundaries(primary)
    fig2_benchmark(primary)
    fig3_sobol()
    fig4_temporal(primary, float(s['temporal_settings']['kostiakov_m_nominal']))
    return 0


if __name__ == '__main__':
    sys.exit(main())
