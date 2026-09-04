# Task prompt — generate the cross-geometry event set

_Hand this file to an agent with shell access on a machine with internet and
≥8 CPU cores. It is self-contained. Do not paraphrase the acceptance criteria._

---

## What you are producing

**One** event set: **600 muon-neutrino interactions at exactly 10 TeV, placed at
three fixed points in the detector, replayed through all six geometries** plus a
medium control and a photon-noise null.

Not one simulation per geometry — **one injection, replayed**. The entire value
is in the pairing: if arm A's event 37 is not arm B's event 37, the output is
worthless no matter how good the plots look.

The design deliberately fixes everything except the geometry:

| held fixed | varied |
|---|---|
| energy (10 TeV, exactly) | detector geometry (6 arms) |
| vertex (3 named points) | direction (isotropic, but the same per event across arms) |
| the injected events themselves | medium (1 control arm) |
| | photon seed (1 null arm) |

## Get the code

```bash
git clone https://github.com/seanxia8/ORACLE.git
cd ORACLE
git checkout dev
```

Everything you need is under `src/prometheus_simulation/`. Read its `README.md`
before starting. If that directory is missing, stop — you are on the wrong
branch and nothing below will work.

Run every command below from the **repository root**, with `PYTHONPATH=src`.

## Why three fixed points rather than a sampled volume

The six geometries span 34× in horizontal radius (57.7 m to 1949.6 m). There is
**no balanced injection cylinder** — run
`PYTHONPATH=src python -c "from prometheus_simulation.geometry import *;G=load_geometries(default_geodir());[print(r) for r in sweep_cylinders(G)]"`
and read the two worst-case columns: at every radius, either the small
geometries see almost none of the injected events, or the large ones are probed
in only their innermost fraction of a percent. Fixed points remove the
acceptance confound instead of trading it off.

The points are **detector-relative**, applied after recentring, so each means
the same thing in every detector. They must lie in the common region —
radius ≤ 57.7 m (set by `triangle`), |z| ≤ 95.4 m (set by `flower_s`).
`simulate.plan()` refuses to run if any point falls outside even one geometry.

| point | (x, y, z) m | what it probes |
|---|---|---|
| `centre` | (0, 0, 0) | on axis, mid-depth — best case everywhere |
| `radial` | (40, 0, 0) | 69% of the common radius |
| `vertical` | (0, 0, 70) | 73% of the common half-height |

## What is already tested, and what is not

Run this first — it needs nothing but numpy/pandas/h5py/awkward/pyarrow/pytest:

```bash
PYTHONPATH=src python -m pytest src/prometheus_simulation/tests/ -q   # 11 tests
```

They need no Prometheus clone: the geofiles are read from
`src/prometheus_simulation/data/geofiles/` when `external/` is absent. They cover the
geometry parsing, the containment gate, the h5 vertex surgery,
the readout, the pairing check (including that it FAILS on a deliberately
broken pairing), and the reconstruction. **What they do not cover is
`Prometheus().sim()` itself** — the two calls that need PROPOSAL,
LeptonInjector and PPC have never been executed. Treat step 1 below as
debugging, not as production.

## Setup — use `install.sh`, NOT `pip install -r requirements.txt`

`requirements.txt` covers the Python deps only. PROPOSAL and **LeptonInjector**
are C++ and must be built; PPC (needed for the ice arm) is a separate step.
Prometheus ships an installer that does all of it via micromamba.

```bash
cd <repo>
bash src/prometheus_simulation/fetch_prometheus.sh          # clone + pin
cd src/prometheus_simulation/external/prometheus
bash install.sh --with-ppc                                  # --with-ppc is REQUIRED
source scripts/activate.sh .prometheus_env
bash scripts/check_install.sh
pip install awkward pyarrow pandas                          # readout only
```

A container is also shipped and is the more reliable route on a cluster:
`docker build -f container/Dockerfile -t prometheus:cpu .`
(needs `build-essential cmake libhdf5-serial-dev libboost-all-dev libgsl-dev
libsuitesparse-dev`).

**If the build is blocked** — `install.sh` needs `micro.mamba.pm` and
`conda.anaconda.org`, and PROPOSAL's pip path needs `conan`; a restricted
network will refuse all three — use this: **`import LeptonInjector` is lazy.**
It happens inside `make_new_LI_injection`, i.e. only on the `inject=True` path
(`prometheus/injection/lepton_injector_utils.py:48`).
`injection_from_LI_output` is pure h5py. So **only the one-off injection step
needs LeptonInjector**; the eight replay arms do not. If the full build only
works in one place, generate the injection there once and run the arms
anywhere PROPOSAL is available. Report if you have to split it this way.

Verify before anything else:

```bash
python examples/02_basic_ice.py     # 3 events, demo ice geometry, PPC, CPU
```

If that fails, **stop and report the error**. Do not work around it, do not
substitute another photon propagator, do not drop the ice arm.

## Step 0 — dry run, no simulation

```bash
cd <repo>
PYTHONPATH=src python -m prometheus_simulation.simulate --out runs/pilot
```

Writes `plan.json` and `physics_record.json` without touching Prometheus.
**Read `plan.json` and check all of this before spending CPU:**

- `design` = "fixed injection points, single energy, isotropic direction"
- `reference_geometry` = `flower_xl`
- `energy_gev` = `10000.0` — a **float, not a string**. If it is a string the
  YAML was edited to `1.0e4`, which PyYAML parses as text; write `10000.0`.
- `point_check[*].inside_all_geometries` — all `true`
- `event_point_assignment` — 200 each of `centre`, `radial`, `vertical`
- `arms` — 8 entries: six geometries, `hexagon_ice_le` (`medium_control`),
  `flower_xl__seed2` (`photon_null`)

## Step 1 — the null first, at reduced N

**Run this before the full set. Do not skip it.** Photon propagation reruns per
arm and is seedable only to Poisson level, so the same event in the *same*
detector already gives different light. That spread is the floor every
cross-geometry number is measured against. If it is as large as the geometry
effect, the design fails — and you learn that for ~10% of the cost.

```bash
PYTHONPATH=src python -m prometheus_simulation.simulate --out runs/null --n-events 60 --execute
```

Then:

```python
from prometheus_simulation.readout import load_event_set, check_pairing
truth, hits, plan = load_event_set("runs/null")
print(check_pairing(truth, plan))
```

**Acceptance:** every row `paired = True`, and `plan.json:vertex_residual_max_m`
< 1e-6 (the vertices really are on the three points). If either fails, STOP and
report which arm and what the residual is. Do not loosen `atol`.

## Step 2 — the set

```bash
PYTHONPATH=src python -m prometheus_simulation.simulate --out runs/pilot --execute
```

600 events × 8 arms = 4,800 event-simulations. At 1–10 CPU-s each that is
roughly 1.5–13 CPU-hours, embarrassingly parallel over arms. The ice arm
(`hexagon_ice_le`, PPC) is the slow one; `flower_xl` is the memory-heavy one
(24,220 modules).

To scale later, raise `events_per_point` in
`src/prometheus_simulation/config/physics_default.yaml`. **If you shard, shard
the arms, never the injection** — every arm must replay the same file.

## Step 3 — run the analysis

```bash
PYTHONPATH=src python -m prometheus_simulation.analyze --run runs/pilot
```

Writes `runs/pilot/analysis/` with `REPORT.md`, five CSVs and three figures,
and **exits non-zero if a gate fails**. It needs `matplotlib` and `tabulate`
in addition to the readout deps.

Read `REPORT.md` before reporting back. Three things in it decide whether the
run is usable:

- **Gates** — both must be `True`.
- **`null_separation`** — `auc_vs_null` per arm. Above 0.75 the geometry change
  is separated from photon noise at this statistics; near 0.5 it is not, and
  nothing may be claimed from that arm without more events. Report the numbers
  either way; a null result here is a real result, not a failure.
- **`light_yield.point_spread_ratio`** — how much each geometry's yield changes
  between the three vertices. This is the position-uniformity of that geometry
  over the common region and is the cleanest single quantity the set measures.

## What to report back

1. `runs/pilot/analysis/` in full — `REPORT.md`, the CSVs, the figures.
2. `plan.json` and `physics_record.json`.
3. Wall-clock time per arm, and peak memory for `flower_xl` (24,220 modules is
   the memory-heavy arm).
4. The `DECLARED DEVIATIONS` list the run prints at the end. Those are the
   parameters NuBench never published; they are our choices and belong in the
   paper's deviations section.
5. Anything that surprised you, and anything you had to change to make it run.
   If you changed code, say exactly what and why — a silent fix to the
   injection or recentring path invalidates the whole set.

## Rules

- **Never regenerate the injection file.** It is written once, its vertices are
  overwritten onto the three points, and it is checksummed into `plan.json`.
  Every arm reads that same file translated by its recentring delta. An arm
  that regenerates rather than copies breaks the pairing silently, and
  `check_pairing` is the only thing that will catch it.
- **Never patch `external/prometheus/`.** LGPL-2.1 upstream, pinned by commit.
  Changes go in `src/prometheus_simulation/` and pass through config.
- **Never commit `external/` or `runs/`.** Both are gitignored.
- **Do not move an injection point outside the common region** to make a
  geometry produce more light. A point outside a detector produces nothing
  there, which looks like a geometry effect and is not.
- If an arm produces zero hits for every event, suspect the recentring: check
  `plan.json:arms[*].recentre_delta_m` against the offsets printed by
  `PYTHONPATH=src python -m prometheus_simulation.geometry`. It is **not** a reason to
  enlarge anything.
- Report failures. Do not silently reduce scope, swap a geometry, or fall back
  to independent per-geometry injections — independent injections are exactly
  what this experiment exists to avoid.
