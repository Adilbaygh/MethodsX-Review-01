"""Edge lengths from pickets and the efficiency rule for short terminal edges.

Authors' decisions and assumptions (2026-11-10):

* The x coordinate of a node in node_positions is its picket along the main canal; y is a
  drawing offset. Pickets are counted from the source.
* Outlets listed in ``outlets_at_junction`` are placed at the picket of their junction node,
  because their drawn x value is a layout position.
* Main-canal edge lengths are picket differences. Terminal edges are earthen canals of
  unknown length; their picket difference only locates the outlet along the main canal.
* Sensitivity variant "rule": terminal edges whose picket difference is below
  ``short_picket_threshold_m`` receive an efficiency that decreases linearly with the picket
  difference, from ``eta_short_max`` for the shortest edge of the group to ``eta_short_min``
  for the longest. In the primary variant "dataset" every edge keeps the dataset efficiency.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from .dataset import Network
from .parameters import EdgeRules

VARIANTS = ('dataset', 'rule')


def edge_table(net: Network, rules: EdgeRules) -> Tuple[List[Dict], Tuple[float, float]]:
    """Per-edge pickets, lengths and both efficiency variants, in dataset edge order."""
    unknown = sorted(set(rules.outlets_at_junction) - set(net.terminals))
    if unknown:
        raise ValueError(f'outlets_at_junction lists nodes that are not terminals: {unknown}')
    x0 = net.x_position[net.source]
    rows: List[Dict] = []
    for e in net.edges:
        kind = net.edge_type[e]
        pk_t = net.x_position[net.tail[e]] - x0
        pk_h = net.x_position[net.head[e]] - x0
        moved = net.head[e] in rules.outlets_at_junction
        if moved:
            pk_h = pk_t
        picket_len = abs(pk_h - pk_t)
        rows.append({'edge': e, 'tail': net.tail[e], 'head': net.head[e], 'type': kind,
                     'picket_tail_m': pk_t, 'picket_head_m': pk_h, 'picket_length_m': picket_len,
                     'length_m': picket_len if kind == 'main' else None,
                     'picket_adjusted': moved, 'efficiency_dataset': net.efficiency[e]})
    short = [r for r in rows if r['type'] == 'terminal' and r['picket_length_m'] < rules.short_picket_threshold_m]
    if short:
        lmin = min(r['picket_length_m'] for r in short)
        lmax = max(r['picket_length_m'] for r in short)
    else:
        lmin = lmax = float('nan')
    short_ids = {r['edge'] for r in short}
    for r in rows:
        if r['edge'] in short_ids:
            frac = 0.0 if lmax == lmin else (r['picket_length_m'] - lmin) / (lmax - lmin)
            r['efficiency_rule'] = rules.eta_short_max - (rules.eta_short_max - rules.eta_short_min) * frac
            r['adjusted'] = True
        else:
            r['efficiency_rule'] = r['efficiency_dataset']
            r['adjusted'] = False
    return rows, (lmin, lmax)


def efficiencies(net: Network, rows: List[Dict], variant: str) -> np.ndarray:
    """Edge efficiencies in dataset edge order for the requested variant."""
    if variant not in VARIANTS:
        raise ValueError(f'unknown variant {variant!r}')
    key = 'efficiency_rule' if variant == 'rule' else 'efficiency_dataset'
    by_edge = {r['edge']: r[key] for r in rows}
    return np.array([by_edge[e] for e in net.edges], dtype=float)
