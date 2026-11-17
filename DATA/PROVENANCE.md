# Data provenance

## gone_abat_jap_am2a.json

| Item | Value |
|---|---|
| Content | Gone Abat Jap open-canal irrigation network: 36 nodes, 35 edges, 20 terminals, 16 dekads (11-26) |
| Record | derived from the file `gone_abat_jap.json` of the Mendeley Data record doi:10.17632/xt3gsf89n9.1 (V1) |
| File version | `am2a`: the V1 file with `period_days` added and `edge_capacity` changed (see the differences below) |
| Licence | CC BY 4.0, as the data record |
| SHA-256 | listed in `SHA256SUMS`; verified by `main.py` on every run |
| Description field of the file | "Transparent operational assessment adaptation of the Gone Abat Jap real-case dataset; not an observed historical scarcity record." |

### Differences from the V1 file

- `period_days` is added: dekads 15, 21 and 24 (21-31 May, July and August) have 11 days, the
  others 10.
- `edge_capacity` follows the calendar for the 11-day dekads. In V1 the capacities of dekads 15,
  20 and 25 were the 11-day values; in `am2a` those of dekads 15, 21 and 24 are. The capacities of
  dekads 20 and 25 are therefore 10/11 and those of dekads 21 and 24 are 11/10 of the V1 values
  (K6 in dekad 25 is unchanged).
- The capacity of edge K11 is 0.9654 times the V1 value in every dekad.
- `nodes`, `source`, `edges`, `users`, `periods`, `demand`, `efficiency`, `node_positions` and
  `source_capacity` are identical.

The capacities do not enter the computed requirements; they are only compared with them
(`capacity_exceedances.csv`).

### Fields used

| Field | Meaning in the worked example |
|---|---|
| `nodes`, `edges`, `source` | radial network rooted at the head works `Of1` |
| `users` | one user per terminal `T1`-`T20` |
| `periods`, `period_days` | dekads 11-26; dekads 15, 21 and 24 have 11 days |
| `demand` | net field irrigation requirement per user, m3 per dekad (authors' confirmation) |
| `efficiency` | edge conveyance efficiency, constant over the dekads |
| `node_positions` | x is the picket along the main canal; y is a drawing offset (authors' confirmation) |
| `edge_capacity`, `source_capacity` | scenario values of the file, m3 per dekad; compared with the computed requirement for information only |

### Not in the file

No measured head-works discharge is included, so the worked example verifies the closure of the
calculation chain and makes no accuracy claim. The field application efficiency, the loss
partition and the recoverable fractions are parameters of `parameters.json`, each with its
provenance. Edge lengths and the efficiency rule for short terminal edges are derived by
`src/grossreq/edges.py` from the fields above and the rules in `parameters.json`; the dataset
file itself is not modified.
