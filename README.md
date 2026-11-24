# Worked example: from field-level net irrigation demand to boundary-specific gross requirements

Reproducibility package of the worked example. The code takes the net field irrigation demand of
the 20 terminals of the Gone Abat Jap open-canal network for 16 dekads (ten-day periods, dekads
11-26). It computes the farm-gate, network, head-works and basin gross requirements, checks the
water balance at every system boundary, compares the head requirement with two
efficiency-coefficient methods on identical inputs, and propagates parameter uncertainty.

The edge efficiencies are taken from the dataset; the package does not estimate them. A
temporal-scale experiment repeats the calculation with discharge-dependent seepage on the main
canal at three temporal scales (dekad, month, season).

## Requirements

Python 3.9 or later, numpy, matplotlib for the figures, and pytest for the tests (`requirements.txt`).
`requirements-lock.txt` lists the exact package versions of the reported run.

## How to run

Windows PowerShell, from the repository root:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe main.py
.\.venv\Scripts\python.exe scripts\record_environment.py
.\.venv\Scripts\python.exe scripts\make_figures.py
```

On Linux or macOS use `python3 -m venv .venv` and `.venv/bin/python` instead.

`main.py` writes `results/` and exits with status 0 only if every check passes.
`main.py --quick` uses small samples for a fast smoke test; its uncertainty numbers are not
the reported ones.

## Repository layout

| Path | Content |
|---|---|
| `main.py` | entry point: runs the whole pipeline and prints checks and headline indicators |
| `pytest.ini` | test configuration (tests are collected from `tests/` only) |
| `src/grossreq/dataset.py` | loads the dataset and verifies that the network is a radial tree |
| `src/grossreq/edges.py` | edge pickets and lengths; efficiency rule for short terminal edges |
| `src/grossreq/parameters.py` | reads and validates `DATA/parameters.json` |
| `src/grossreq/propagation.py` | deterministic chain from net demand to head and basin requirement |
| `src/grossreq/balance.py` | balance terms and residuals at boundaries B1-B5, node continuity |
| `src/grossreq/benchmarks.py` | efficiency-coefficient benchmarks M0 and M1 |
| `src/grossreq/uncertainty.py` | Monte Carlo, first-order propagation, grouped Sobol indices |
| `src/grossreq/temporal.py` | temporal-scale experiment with discharge-dependent seepage |
| `src/grossreq/pipeline.py` | runs everything, records checks and writes the result files |
| `src/grossreq/report.py` | deterministic CSV and JSON writers |
| `scripts/record_environment.py` | records machine, OS, Python and package versions |
| `scripts/make_figures.py` | draws the article figures from the result tables |
| `tests/` | known-answer tests on small networks and checks on the worked example |
| `DATA/` | dataset, parameter file, checksum, provenance |
| `results/` | result tables (written by `main.py`) and figures (written by `scripts/make_figures.py`) |
| `LICENSE` | MIT License of the code |
| `CITATION.cff` | how to cite this package |
| `.gitattributes` | keeps `DATA/` and `results/` byte-identical on every platform |

## What is computed

For every terminal $t$, edge $e=(u,v)$ and dekad $k$, all quantities in m3 per dekad:

- farm-gate demand $D_{t,k}=N_{t,k}/\eta^{a}$;
- edge requirements $Q^{\mathrm{out}}_{e,k}=\sum_{e'\text{ leaving }v}Q^{\mathrm{in}}_{e',k}+D_{v,k}$ and $Q^{\mathrm{in}}_{e,k}=Q^{\mathrm{out}}_{e,k}/\eta_e$, from the leaves to the source;
- head requirement $G^{h}_{k}$, the sum of $Q^{\mathrm{in}}$ over the edges leaving the source;
- edge losses $L_{e,k}=Q^{\mathrm{in}}_{e,k}-Q^{\mathrm{out}}_{e,k}$, split into evaporation, seepage and operational spills ($\phi^{E}$, $\phi^{S}=1-\phi^{E}-\phi^{O}$, $\phi^{O}$); the recoverable part $R=\rho^{S}S+\rho^{O}O$;
- field losses $F_{t,k}=D_{t,k}-N_{t,k}$; the recoverable part $R^{F}=\rho^{F}F$;
- basin depletion $G^{\mathrm{basin}}_{k}=G^{h}_{k}-\sum_e R_{e,k}-\sum_t R^{F}_{t,k}$.

Boundaries: B1 fields, B2 farm gates, B3a terminal canals, B3b main canal, B4 canal network
(head to farm gates), B4+ irrigation system including fields, B5 basin. Canal storage change
is neglected at the dekad scale.

Benchmarks on identical inputs: M0, one overall efficiency; M1, a chain of level
efficiencies (field, terminal canals, main canal). Both are calibrated on the seasonal
volumes of the proposed workflow, which is the most favourable setting for them.

## Variants

- `dataset` (primary): all edge efficiencies as in the dataset file. Terminal edges are
  earthen canals of unknown length.
- `rule` (sensitivity): terminal edges whose picket difference along the main canal is below
  10 m receive an efficiency decreasing linearly with that difference from 0.99 to 0.95;
  outlets of T9-T14 and T18 are placed at the picket of their junction.

Both are authors' decisions recorded in `DATA/parameters.json` (sections `variants` and
`edge_rules`).

## Temporal-scale experiment

On main-canal edges the loss fraction is made to depend on the mean inflow discharge $q$ of
the period:
$\lambda_e(q)=\lambda^{\mathrm{nom}}_e\left[(1-\phi^{S})+\phi^{S}(q/\bar q_e)^{-m}\right]$,
where $\lambda^{\mathrm{nom}}_e=1-\eta_e$ and $\bar q_e$ is the seasonal mean inflow discharge of
the edge in the linear model. The seepage part thus follows Kostiakov's discharge dependence;
evaporation and operational spills stay proportional to flow. At $q=\bar q_e$ the loss fraction
equals the dataset value. The calculation is run with dekads, calendar months and the whole
season as periods, for the exponents listed in `DATA/parameters.json` (section `temporal`).
The implicit equation $Q^{\mathrm{in}}=Q^{\mathrm{out}}/(1-\lambda(Q^{\mathrm{in}}))$ is solved by
fixed-point iteration.

## Outputs

| File | Content |
|---|---|
| `results/run_summary.json` | checks, indicators, sample sizes, seed, provisional parameters, file list |
| `results/tables/parameters.csv` | parameter values, uncertainty ranges, provenance and status |
| `results/tables/edge_parameters.csv` | pickets, main-canal lengths, dataset and rule efficiencies per edge |
| `results/tables/<variant>/dekad_totals.csv` | per-dekad totals: $N$, $D$, $F$, $R^{F}$, $L$, $E$, $S$, $O$, $R$, $U$, $G^{h}$ (also in m3/s), $G^{\mathrm{basin}}$ |
| `results/tables/<variant>/edge_flows.csv` | per edge and dekad: $\eta_e$, $Q^{\mathrm{in}}$, $Q^{\mathrm{out}}$, loss components, dataset capacity |
| `results/tables/<variant>/terminal_flows.csv` | per terminal and dekad: $N$, $D$, $F$, $R^{F}$ |
| `results/tables/<variant>/closure.csv` | balance terms $I$, $O$, $\Delta S$, $C$, $NC$ and residual for B1-B5 per dekad |
| `results/tables/<variant>/capacity_exceedances.csv` | dekads where a requirement exceeds the dataset's capacity value |
| `results/tables/benchmark_dekad.csv` | $G^{P}$, $G^{M0}$, $G^{M1}$ and their difference per dekad |
| `results/tables/benchmark_terminals.csv` | path efficiencies and the head requirement attributed to each terminal by P, M0 and M1 |
| `results/tables/stepwise_peak_terminals.csv` | step-by-step calculation for the dekad with the largest net demand: terminals |
| `results/tables/stepwise_peak_edges.csv` | the same dekad: edges |
| `results/tables/uncertainty.csv` | Monte Carlo mean, sd, 5/50/95 percentiles; first-order sd of $G^{h}$ |
| `results/tables/sobol_indices.csv` | grouped first-order and total Sobol indices with bootstrap 95% intervals |
| `results/tables/temporal_scale.csv` | seasonal head requirement at the dekad, month and season scale per exponent $m$ |
| `results/tables/temporal_dekads.csv` | dekad head requirement with constant and with discharge-dependent losses |
| `results/environment.json` | machine, OS, Python and package versions |
| `results/figures/fig1_boundaries.{pdf,png}` | dekad requirement at the field, farm-gate, head-works and basin boundaries, with Monte Carlo 5-95% ranges |
| `results/figures/fig2_benchmark.{pdf,png}` | efficiency-coefficient benchmark: dekad head requirement and attribution to terminals |
| `results/figures/fig3_sobol.{pdf,png}` | grouped Sobol indices of the seasonal head requirement and basin depletion |
| `results/figures/fig4_temporal.{pdf,png}` | dekad head requirement with discharge-dependent versus constant losses |

## Figures and tables of Section 9

Every figure and table of the worked example (Section 9 of the article) comes from the files
below. Tables and numbers are written by `main.py`; the figures by `scripts/make_figures.py`,
which reads those tables and computes nothing except unit conversions.

| Section 9 | Object | File |
|---|---|---|
| 9.3 Parameters | parameter table | `results/tables/parameters.csv` |
| 9.4 Verification of the water balance | the 40 checks | `results/run_summary.json` |
| 9.4 Verification of the water balance | balance table (dekad 22 and the largest residual over the dekads) | `results/tables/dataset/closure.csv` |
| 9.4 Verification of the water balance | the chain traced for dekad 22 | `results/tables/stepwise_peak_terminals.csv`, `results/tables/stepwise_peak_edges.csv` |
| 9.5 Gross requirements at the boundaries | requirements at four boundaries; seasonal totals | `results/figures/fig1_boundaries.{pdf,png}`; `results/tables/dataset/dekad_totals.csv` |
| 9.6 Uncertainty and sensitivity | grouped Sobol indices; uncertainty table | `results/figures/fig3_sobol.{pdf,png}`; `results/tables/sobol_indices.csv`, `results/tables/uncertainty.csv` |
| 9.7 Benchmark against efficiency-coefficient methods | benchmark figure and attribution to terminals | `results/figures/fig2_benchmark.{pdf,png}`; `results/tables/benchmark_dekad.csv`, `results/tables/benchmark_terminals.csv` |
| 9.8 Temporal scale | temporal-scale figure | `results/figures/fig4_temporal.{pdf,png}`; `results/tables/temporal_scale.csv`, `results/tables/temporal_dekads.csv` |
| 9.9 Reproducibility and limitations | sensitivity to the efficiencies of short terminal edges | `results/tables/rule/` compared with `results/tables/dataset/` |

## Checks

Every run evaluates the following and stops with status 1 if any fails (tolerance $10^{-12}$
on relative values):

- balance residual at every boundary and dekad; continuity at every node; $L_e=(1-\eta_e)Q^{\mathrm{in}}_e$;
- basin depletion equals consumed plus non-recoverable water;
- recursive head requirement equals the closed form $\sum_t N_{t,k}/(\eta^{a}\prod_{e\in P(t)}\eta_e)$;
- the sum of dekad results equals the result for the seasonal demand (linearity);
- the fast evaluator used for uncertainty equals the edge-by-edge chain at nominal values;
- $\sum_t N+\sum_e E\le G^{\mathrm{basin}}\le G^{h}$, with equality at $\rho=1$ and $\rho=0$ respectively;
- M1 equals M0 under seasonal calibration, and their seasonal sum equals that of P;
- the head requirements attributed to terminals add up to $G^{h}$;
- parameters that do not enter $G^{h}$ have Sobol indices of exactly zero for $G^{h}$;
- temporal experiment: with $m=0$ it reproduces the linear model; one seasonal period
  reproduces the linear seasonal total for every $m$; balance, node continuity, edge-loss and
  basin identities hold; the fixed-point residual is below the tolerance.

## Parameters and provenance

`DATA/parameters.json` holds every parameter with its value, uncertainty distribution,
provenance (dataset, literature, assumed, authors' decision), source and status.
Parameters whose status is not `confirmed` are listed as provisional on every run.

## Reproducibility

- Random numbers come from PCG64 streams spawned from one master seed
  (`DATA/parameters.json`). Only uniform doubles are drawn and they are transformed by
  explicit inverse distribution functions.
- The uncertainty evaluator uses fixed-order products and sums without BLAS calls.
- Result files are written with fixed number formats and LF line endings; two runs on the
  same machine give byte-identical files (checked in quick mode by
  `tests/test_worked_example.py`).
- `main.py` verifies the dataset against `DATA/SHA256SUMS` before computing.
- `.gitattributes` stores `DATA/` and `results/` without line-ending conversion, so the
  checksum and the expected outputs match byte for byte after a clone on any platform.
- An optional cross-check against an independently produced edge table runs when the
  environment variable `GROSSREQ_REFERENCE_EDGE_TABLE` points to such a table.

## Data

See `DATA/PROVENANCE.md`.

## Licence

The code (`main.py`, `src/`, `scripts/`, `tests/`) is released under the MIT License
(`LICENSE`). The dataset file, the parameter file and the result files (`DATA/`, `results/`)
are released under the Creative Commons Attribution 4.0 International licence (CC BY 4.0,
https://creativecommons.org/licenses/by/4.0/), the licence of the dataset record on
Mendeley Data.

## Citation

This package, version 0.1.0, is archived in Zenodo: https://doi.org/10.5281/zenodo.24496308.
Please cite it together with the dataset: A. Kudaybergenov, A. Kabulov, M. Kazimbetova,
G. Ametova, J. Ispanova, Input Data and Controlled-Scenario Outputs for the Gone Abat Jap
Irrigation Allocation Model, Mendeley Data, V1, 2026, https://doi.org/10.17632/xt3gsf89n9.1.
`CITATION.cff` gives the citation of the package; the article will be added when it is
published.
