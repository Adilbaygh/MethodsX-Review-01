import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from grossreq.dataset import load_network, network_from_dict  # noqa: E402
from grossreq.parameters import load_parameters  # noqa: E402

DATA = ROOT / 'DATA' / 'gone_abat_jap_am2a.json'
PARAMS = ROOT / 'DATA' / 'parameters.json'
SUMS = ROOT / 'DATA' / 'SHA256SUMS'


@pytest.fixture(scope='session')
def net():
    return load_network(DATA)


@pytest.fixture(scope='session')
def ps():
    return load_parameters(PARAMS)


def toy_dict(edges, users, demand, eff, periods=('1', '2'), positions=None):
    """Small dataset dictionary in the format of the real file.

    edges:  list of (id, tail, head); the source is 'S'
    users:  dict user -> terminal
    demand: dict user -> list of values per period
    eff:    dict edge -> efficiency (constant over periods)
    """
    nodes = ['S']
    for _, u, v in edges:
        for n in (u, v):
            if n not in nodes:
                nodes.append(n)
    positions = positions or {n: [float(i), 0.0] for i, n in enumerate(nodes)}
    return {
        'name': 'toy', 'nodes': nodes, 'source': 'S',
        'edges': [{'id': e, 'tail': u, 'head': v} for e, u, v in edges],
        'users': [{'id': uid, 'terminal': t} for uid, t in users.items()],
        'periods': list(periods),
        'period_days': {p: '10' for p in periods},
        'demand': {p: {uid: str(vals[k]) for uid, vals in demand.items()} for k, p in enumerate(periods)},
        'efficiency': {p: {e: str(x) for e, x in eff.items()} for p in periods},
        'node_positions': positions,
    }


@pytest.fixture
def make_toy():
    def _make(*args, **kwargs):
        return network_from_dict(toy_dict(*args, **kwargs))
    return _make
