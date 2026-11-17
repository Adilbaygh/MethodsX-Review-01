"""Record the computing environment of a run in results/environment.json.

Usage (from the repository root):
    python scripts/record_environment.py
"""

from __future__ import annotations

import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _version(module: str) -> str:
    try:
        mod = __import__(module)
        return str(getattr(mod, '__version__', 'unknown'))
    except ImportError:
        return 'not installed'


def main() -> int:
    sys.path.insert(0, str(ROOT / 'src'))
    env = {
        'recorded_utc': datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        'os': platform.platform(),
        'machine': platform.machine(),
        'processor': platform.processor(),
        'cpu_count': os.cpu_count(),
        'python': sys.version.split()[0],
        'python_implementation': platform.python_implementation(),
        'packages': {name: _version(name) for name in ('grossreq', 'numpy', 'matplotlib', 'pytest')},
    }
    out = ROOT / 'results' / 'environment.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'w', encoding='utf-8', newline='\n') as f:
        f.write(json.dumps(env, indent=2, sort_keys=True) + '\n')
    print(json.dumps(env, indent=2, sort_keys=True))
    return 0


if __name__ == '__main__':
    sys.exit(main())
