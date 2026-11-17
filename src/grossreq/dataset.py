"""Loading and validating the canal-network dataset.

The network must be a radial tree rooted at the source: every other node has exactly one
incoming edge and every node is reachable from the source. Values in the dataset file are
stored as strings and are converted here. Nothing in the dataset file is modified.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


class DatasetError(ValueError):
    """Raised when the dataset violates an assumption of the model."""


class Network:
    """Radial canal network with net demand per terminal and period."""

    def __init__(self, *, name: str, source: str, nodes: Sequence[str], edges: Sequence[str],
                 tail: Dict[str, str], head: Dict[str, str], terminals: Sequence[str],
                 periods: Sequence[str], period_days: np.ndarray, net_demand: np.ndarray,
                 efficiency: Dict[str, float], x_position: Dict[str, float],
                 edge_capacity: Optional[np.ndarray], source_capacity: Optional[np.ndarray]):
        self.name = name
        self.source = source
        self.nodes: Tuple[str, ...] = tuple(nodes)
        self.edges: Tuple[str, ...] = tuple(edges)
        self.tail = dict(tail)
        self.head = dict(head)
        self.terminals: Tuple[str, ...] = tuple(terminals)
        self.periods: Tuple[str, ...] = tuple(periods)
        self.period_days = np.asarray(period_days, dtype=float)
        self.net_demand = np.asarray(net_demand, dtype=float)
        self.efficiency = dict(efficiency)
        self.x_position = dict(x_position)
        self.edge_capacity = edge_capacity
        self.source_capacity = source_capacity

        self.edge_index = {e: i for i, e in enumerate(self.edges)}
        self.terminal_index = {t: j for j, t in enumerate(self.terminals)}
        children: Dict[str, List[str]] = {v: [] for v in self.nodes}
        parent: Dict[str, Optional[str]] = {v: None for v in self.nodes}
        for e in self.edges:
            children[self.tail[e]].append(e)
            parent[self.head[e]] = e
        self.children: Dict[str, Tuple[str, ...]] = {v: tuple(c) for v, c in children.items()}
        self.parent_edge = parent
        self.edge_type = {e: ('terminal' if self.head[e] in self.terminal_index else 'main')
                          for e in self.edges}
        self.terminal_edge = {t: parent[t] for t in self.terminals}
        self.leaves_without_demand = tuple(v for v in self.nodes
                                           if not self.children[v] and v not in self.terminal_index
                                           and v != self.source)
        self.post_order = self._post_order()
        self.path_edges = {t: self._path_to(t) for t in self.terminals}
        inc = np.zeros((len(self.edges), len(self.terminals)))
        for j, t in enumerate(self.terminals):
            for e in self.path_edges[t]:
                inc[self.edge_index[e], j] = 1.0
        self.path_incidence = inc

    @property
    def n_periods(self) -> int:
        return len(self.periods)

    def edges_of_type(self, kind: str) -> Tuple[str, ...]:
        return tuple(e for e in self.edges if self.edge_type[e] == kind)

    def _post_order(self) -> Tuple[str, ...]:
        """Edges ordered so that every edge comes after all edges below it."""
        pre: List[str] = []
        stack = list(reversed(self.children[self.source]))
        while stack:
            e = stack.pop()
            pre.append(e)
            stack.extend(reversed(self.children[self.head[e]]))
        return tuple(reversed(pre))

    def _path_to(self, node: str) -> Tuple[str, ...]:
        path: List[str] = []
        v = node
        while v != self.source:
            e = self.parent_edge[v]
            path.append(e)
            v = self.tail[e]
        return tuple(reversed(path))


def _num(value, what: str) -> float:
    try:
        x = float(value)
    except (TypeError, ValueError) as exc:
        raise DatasetError(f'{what}: not a number ({value!r})') from exc
    if not math.isfinite(x):
        raise DatasetError(f'{what}: not finite ({value!r})')
    return x


def network_from_dict(d: dict) -> Network:
    """Build and validate a Network from the dataset dictionary."""
    nodes = [str(v) for v in d['nodes']]
    if len(set(nodes)) != len(nodes):
        raise DatasetError('duplicate node ids')
    node_set = set(nodes)
    source = str(d['source'])
    if source not in node_set:
        raise DatasetError(f'source {source} is not a node')

    edges, tail, head = [], {}, {}
    for rec in d['edges']:
        e, u, v = str(rec['id']), str(rec['tail']), str(rec['head'])
        if e in tail:
            raise DatasetError(f'duplicate edge id {e}')
        if u not in node_set or v not in node_set:
            raise DatasetError(f'edge {e} refers to an unknown node')
        if u == v:
            raise DatasetError(f'edge {e} is a self-loop')
        edges.append(e)
        tail[e], head[e] = u, v

    indegree = {v: 0 for v in nodes}
    for e in edges:
        indegree[head[e]] += 1
    if indegree[source] != 0:
        raise DatasetError('the source has an incoming edge')
    bad = sorted(v for v in nodes if v != source and indegree[v] != 1)
    if bad:
        raise DatasetError(f'nodes without exactly one incoming edge (network is not radial): {bad}')
    if len(edges) != len(nodes) - 1:
        raise DatasetError('number of edges is not number of nodes minus one')
    reach, frontier = {source}, [source]
    out = {v: [] for v in nodes}
    for e in edges:
        out[tail[e]].append(head[e])
    while frontier:
        u = frontier.pop()
        for v in out[u]:
            if v not in reach:
                reach.add(v)
                frontier.append(v)
    if reach != node_set:
        raise DatasetError(f'nodes not reachable from the source: {sorted(node_set - reach)}')

    # users -> terminals (several users may share a terminal; demand is summed)
    user_terminal: Dict[str, str] = {}
    terminals: List[str] = []
    for rec in d['users']:
        uid, t = str(rec['id']), str(rec['terminal'])
        if uid in user_terminal:
            raise DatasetError(f'duplicate user id {uid}')
        if t not in node_set:
            raise DatasetError(f'user {uid} is attached to an unknown node {t}')
        if out[t]:
            raise DatasetError(f'user {uid} is attached to {t}, which is not a leaf')
        user_terminal[uid] = t
        if t not in terminals:
            terminals.append(t)
    t_index = {t: j for j, t in enumerate(terminals)}

    periods = [str(p) for p in d['periods']]
    if len(set(periods)) != len(periods):
        raise DatasetError('duplicate period ids')
    days = np.array([_num(d['period_days'][p], f'period_days[{p}]') for p in periods])
    if np.any(days <= 0):
        raise DatasetError('period_days must be positive')

    demand = np.zeros((len(terminals), len(periods)))
    for k, p in enumerate(periods):
        dp = d['demand'][p]
        missing = sorted(set(user_terminal) - set(map(str, dp)))
        if missing:
            raise DatasetError(f'demand[{p}] has no value for users {missing}')
        for uid, val in dp.items():
            uid = str(uid)
            if uid not in user_terminal:
                raise DatasetError(f'demand[{p}] refers to unknown user {uid}')
            x = _num(val, f'demand[{p}][{uid}]')
            if x < 0:
                raise DatasetError(f'demand[{p}][{uid}] is negative')
            demand[t_index[user_terminal[uid]], k] += x

    eff_by_period = d['efficiency']
    first = periods[0]
    efficiency = {}
    for e in edges:
        x = _num(eff_by_period[first][e], f'efficiency[{first}][{e}]')
        if not 0.0 < x <= 1.0:
            raise DatasetError(f'efficiency of {e} outside (0, 1]')
        efficiency[e] = x
    for p in periods[1:]:
        for e in edges:
            if _num(eff_by_period[p][e], f'efficiency[{p}][{e}]') != efficiency[e]:
                raise DatasetError(f'efficiency of {e} varies over periods; the worked example assumes constant efficiency')

    x_position = {str(v): _num(pos[0], f'node_positions[{v}]') for v, pos in d['node_positions'].items()}
    missing_pos = sorted(node_set - set(x_position))
    if missing_pos:
        raise DatasetError(f'node_positions missing for {missing_pos}')

    edge_capacity = None
    if 'edge_capacity' in d:
        edge_capacity = np.array([[_num(d['edge_capacity'][p][e], f'edge_capacity[{p}][{e}]')
                                   for p in periods] for e in edges])
    source_capacity = None
    if 'source_capacity' in d:
        source_capacity = np.array([_num(d['source_capacity'][p], f'source_capacity[{p}]') for p in periods])

    return Network(name=str(d.get('name', '')), source=source, nodes=nodes, edges=edges,
                   tail=tail, head=head, terminals=terminals, periods=periods, period_days=days,
                   net_demand=demand, efficiency=efficiency, x_position=x_position,
                   edge_capacity=edge_capacity, source_capacity=source_capacity)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def load_network(path: Path) -> Network:
    with open(path, encoding='utf-8') as f:
        return network_from_dict(json.load(f))
