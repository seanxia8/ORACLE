# `prometheus_simulation` — one event set, every geometry

Generates the cross-geometry event set that NuBench's released data cannot
provide: **600 events at exactly 10 TeV, placed at three fixed vertices, and
reconstructed by six different detector geometries.**

| held fixed | varied |
|---|---|
| energy — 10 TeV exactly | detector geometry (6 arms) |
| vertex — 3 named points | direction (isotropic; same per event across arms) |
| the injected events themselves | medium (1 control arm) |
| | photon seed (1 null arm) |

NuBench's seven datasets are independently injected, so comparing geometry A to
geometry B compares two *populations* and any difference in reconstruction
error mixes the geometry effect with the sample difference. Prometheus supports
replaying a stored injection (`injection.lepton_injector.inject = False` plus
`paths.injection_file`), which makes the comparison within-event.

## What is in here

Two layers, one package. `oracle_paired` was merged in on 2026-09-04; keeping
them apart split one experiment across two packages.

| layer | modules |
|---|---|
| **geometry & injection** | `geometry.py` (.geo parsing, offsets, common region, containment), `simulate.py` (inject once, recentre, replay every arm), `physics.py` (the provenance-tagged parameter record), `plans.py` (Prometheus run-plan JSON for cluster dispatch, no Prometheus import) |
| **detector response & cells** | `detector.py` (`DetectorGeometry`, the OM table), `events.py`, `response.py` (NuBench §3.2 emulation), `interventions.py` (N1–N5), `strata.py` (S and U families), `matching.py` (content-matched clean twins), `export.py` (Parquet with `injection_id`), `toy.py` |
| **readout & analysis** | `readout.py`, `recon.py` (geometry-free baselines), `analyze.py` (headless report) |

`geometry.Geometry` parses the `.geo` files and owns offsets and containment;
`detector.DetectorGeometry` is the OM-position table the response stage works
on. Different jobs, deliberately different types.

## The defect the merge fixed

`plans.prometheus_config_pairs` (formerly `oracle_paired/config.py`) used to
say *"Only `detector.geo_file` differs otherwise — that is the entire
experimental manipulation."* It is not, and nothing in that package handled a
detector offset.

Prometheus applies the offset to the injection file **in place** and **only**
on the `inject=True` path (`lepton_injector_utils.apply_detector_offset`),
while `injection_from_LI_output` takes `**_` and **ignores** `detector_offset`
on load. A stored injection therefore lives in the first detector's absolute
frame. ORCA is centred at z = +95 m and ARCA at z = −3194 m, so replaying the
ORCA injection into ARCA with only `geo_file` changed places every event about
3.3 km away and **every replayed arm yields zero hits**.

`prometheus_config_pairs` now emits a per-arm injection path plus the
`recentre_delta_m` that produces it, and `simulate.build_event_set` applies it.
`tests/test_configs.py::test_replay_recentre_delta_is_non_zero_for_offset_geometries`
is the regression guard: it asserts the ORCA→ARCA translation really is a
kilometre-scale shift.

## Quick start

```bash
# Geometry work needs no clone: the .geo files fall back to
# data/geofiles/. Only running Prometheus needs this.
bash src/prometheus_simulation/fetch_prometheus.sh
cd src/prometheus_simulation/external/prometheus
bash install.sh --with-ppc          # NOT pip install -r requirements.txt:
                                    # PROPOSAL and LeptonInjector are C++ builds,
                                    # and the ice arm needs PPC
source scripts/activate.sh .prometheus_env && cd -

PYTHONPATH=src python -m prometheus_simulation.geometry                      # geometry survey
PYTHONPATH=src python -m prometheus_simulation.simulate --out runs/pilot     # plan only, no CPU
PYTHONPATH=src python -m prometheus_simulation.simulate --out runs/pilot --execute
```

Then open `notebooks/geometry_survey.ipynb`.

To hand the production run to an agent on a cluster, give it `AGENT_PROMPT.md`
verbatim.

## The geometries

Parsed from the geofiles Prometheus ships. String counts match NuBench Table 1
exactly, which is the check that these are the right files.

| dataset | geofile | modules | strings | offset z (m) | r_horiz (m) | half-height (m) | header medium |
|---|---|---:|---:|---:|---:|---:|---|
| flower_s (ORCA) | `orca.geo` | 3300 | 150 | +95.4 | 100.2 | 95.4 | mediterranean |
| flower_l (ARCA) | `arca.geo` | 2070 | 115 | −3194.0 | 500.9 | 306.0 | mediterranean |
| flower_xl (TRIDENT) | `trident.geo` | 24220 | 1211 | −3090.0 | 1949.6 | 285.0 | water |
| triangle (P-ONE) | `pone_triangle.geo` | 60 | 3 | −0.0 | 57.7 | 500.0 | water |
| cluster (Baikal-GVD) | `gvd.geo` | 288 | 8 | −970.0 | 60.0 | 270.0 | water |
| hexagon (IceCube) | `icecube.geo` | 5160 | 86 | −1972.0 | 596.3 | 518.7 | ice |

NuBench simulated **all six in water** and added **one ice dataset** on the
Hexagon geometry, so the shipped headers are overridden per arm — the header is
what selects PPC (ice) vs olympus (water).

## The four things that make this non-trivial

1. **The geometries do not share a coordinate frame.** Detector centres run
   from z = +95 m to z = −3194 m. Prometheus applies the detector offset to the
   injection file *in place*, and only when generating it
   (`apply_detector_offset`); `injection_from_LI_output` takes `**_` and
   **ignores** `detector_offset` on load. Replaying verbatim puts every event
   kilometres away. → `geometry.recentre_delta`.

2. **No injection cylinder can serve all six.** They span 34× in radius, so at
   every radius either the small geometries see almost none of the injected
   events or the large ones are probed in a fraction of a percent of their
   volume — the two never rise together, and the best crossing point is ~2.5%
   on both (`geometry.sweep_cylinders`). The design therefore uses **three
   fixed injection points** instead of a sampled volume, which turns a
   trade-off into a containment check. The points are detector-relative and
   must lie in the **common region** — radius ≤ 57.7 m (set by `triangle`),
   |z| ≤ 95.4 m (set by `flower_s`); `simulate.plan()` refuses to run
   otherwise.

   | point | (x, y, z) m | probes |
   |---|---|---|
   | `centre` | (0, 0, 0) | on axis, mid-depth |
   | `radial` | (40, 0, 0) | 69% of the common radius |
   | `vertical` | (0, 0, 70) | 73% of the common half-height |

3. **There is no zero-point.** Photon propagation is re-run per arm and is
   seedable only to Poisson level. The `photon_null` arm — reference geometry,
   same events, different photon seed — measures that floor. Every
   cross-geometry number is reported as a multiple of its p95, never raw.

4. **Depth changes with the geometry.** The Earth model is built at
   `-detector_offset[2]`, so recentring changes the overburden and
   LeptonInjector weights do not transfer across arms. Report **unweighted,
   per-event, truth-referenced** metrics only.

## Upstream quirks we work around

`external/prometheus` is LGPL-2.1 and pinned, so nothing there is patched.
Two behaviours are compensated for in `simulate.run_arm` instead:

- **`ppc.paths.force` does not make a re-run safe.** `prometheus.py` guards the
  PPC tmpdir with `if tmpdir.exists() and not force: raise
  PpcTmpdirExistsError`, then calls `mkdir(parents=True, exist_ok=False)`
  unconditionally — so `force` only converts a typed error into a raw
  `FileExistsError`. A `--arm <name>` retry after a crash or OOM would fail on
  the leftover directory instead of resuming. We remove it first.
- **The default tmpdir is `./.ppc_tmp`, relative to the working directory,** so
  concurrent ice arms would share one scratch directory. Each arm now gets its
  own under its output directory.

`tests/test_ppc_tmpdir.py` pins both, including a retry over a leftover
directory as the regression guard.

## Hardware

Two independent GPU paths, neither on by default. Seven of the eight arms are
**water** → olympus (JAX), which uses the GPU only if a CUDA `jaxlib` is
installed — `requirements.txt` pins the CPU build, so a GPU box runs on the CPU
silently. One arm (`hexagon_ice_le`) is **ice** → PPC, whose CUDA binary
`install.sh --with-ppc` does *not* build (`make cpu` only); use
`container/Dockerfile.gpu` or `make gpu arch=<SM>`, and match `SM_ARCH` to the
card. `use_gpu: true` switches only the ice arm.

CPU is a legitimate way to run this: ~1.5–13 CPU-hours for the full set,
parallel over arms.

`olympus_max_distance_m` (300 m) is **physics, not memory** — it drops
source-module pairs before propagation, so a central event in `flower_xl`
(r = 1950 m) illuminates only the inner ~300 m. Lower `olympus_photon_chunk`
for memory instead.

## Recorded physics

Every parameter carries a provenance tag — `nubench` (stated in
arXiv:2511.13111), `ours`, or `ask` (never published). `physics_record.json` is
written beside every run with the full set, the tags, a config fingerprint, and
the repo and Prometheus commits. Each run prints its `ask` list at the end;
that list is the paper's declared-deviations section.

Currently `ask`: spectral index (inert at fixed energy), zenith/azimuth range,
endcap length, Earth model. Also unpublished and not modelled here at all: the NuBench detector
response of their §3.2 (per-OM noise rate, per-dataset TTS, ice angular
acceptance) — this package produces *photon hits*, not NuBench-format pulses.

## Layout

```
geometry.py         geofile parsing, offsets, recentring, cylinder choice
physics.py          PhysicsParameters + provenance + the run record
simulate.py         plan() then build_event_set(): inject once, replay all arms
readout.py          parquet -> tidy truth/hits frames; check_pairing
recon.py            geometry-free baselines: vertex, direction, light yield
config/             physics_default.yaml
tests/              11 tests: geometry, h5 vertex surgery, readout, recon
notebooks/          geometry_survey.ipynb
fetch_prometheus.sh clone + pin upstream (LGPL-2.1) into external/ [gitignored]
AGENT_PROMPT.md     hand this to a remote agent to run production
external/, runs/    gitignored
```

## Licence position

Prometheus is LGPL-2.1: used as a library, never patched, pinned by commit.
The geometries ship with Prometheus, not with NuBench. GraphNeT (DynEdge) is
Apache-2.0. The NuBench repository states **no licence**, so its datasets,
predictions and checkpoints are not used by this package at all — which is
what makes this path free of any permission question.
