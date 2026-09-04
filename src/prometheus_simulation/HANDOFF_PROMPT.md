# Handoff — apply two fixes, build the environment, generate the event set

_Continuation of `AGENT_PROMPT.md`. Read that first for the experimental design;
this file records two code defects that block the run, and everything learned
from a full build and first-contact debugging elsewhere._

Written 2026-09-04. You are on a **different Linux server** from where this was
debugged, so **none of the built environment transfers** — you are building from
scratch. What does transfer is knowledge: two code bugs that will block you
regardless of host, and a set of environment failures that cost hours to
diagnose and are cheap to avoid if you know the signatures.

**Read §1 and §2 before running anything.** §1 is host-independent. §2 is
"here is what to check on your host, and what to do if it bites."

---

## 0. The situation

The design and the plan are sound and fully tested. Two defects block execution,
both found only by actually calling `Prometheus().sim()` — which no test does.
Neither is subtle once seen, and the fix for each is decided and specified.

Once they are in: build the environment, run a PPC CPU-vs-CUDA equivalence
check, the 60-event null, then the 600-event set and the analysis.

---

## 1. Fix these two things first — host-independent

These are code bugs. They will block you on any machine.

### 1.1 The injection cannot express a fixed energy (BLOCKING)

Symptom:

```
ValueError: injection minimal energy (10000.0) must be < maximal energy (10000.0)
```

**Do not "fix" this by widening the energy into a band.** The design is not the
problem. LeptonInjector supports monoenergetic injection *natively* — it is the
first branch of the sampler
(`resources/LeptonInjector/private/LeptonInjector/LeptonInjector.cxx:121`):

```cpp
double LeptonInjectorBase::SampleEnergy(){
    if(config.energyMinimum==config.energyMaximum)
        return(config.energyMinimum); //return the only allowed energy
```

The bug is that Prometheus' validator is **stricter than the library it wraps**:
`prometheus/utils/config_mims.py:117-123` rejects `min_e >= max_e`. It should be
`min_e > max_e`. `external/` is LGPL and pinned, so we do not patch it — we work
around it in our code.

Why a narrow band is actively wrong: the next branch of `SampleEnergy` reads

```cpp
if(config.powerlawIndex==1.0) //sample uniformly in log space
    return(pow(10.0,this->random->Uniform(log10(min),log10(max))));
```

and our `power_law` **is** 1.0. A band would bypass the exact monoenergetic
branch and sample log-uniformly, converting an exactly fixed energy into a
spread — the opposite of the design.

**The fix — verified working, bit-exact.** `check_consistency` runs inside
`Prometheus.__init__` (`prometheus.py:233`), while `inject()` reads
`config.injection[...].simulation` **live** afterwards (`prometheus.py:307-328`
→ `lepton_injector_utils.make_new_LI_injection`). `config_mims` only *reads* the
energies to validate; it derives and caches nothing from them. So construct with
a placeholder band, then set the exact single energy before `sim()`.

In `simulate.inject_once`, replace the single `Prometheus().sim()` call with:

```python
    # LeptonInjector supports a single fixed energy natively -- SampleEnergy()
    # opens with `if(energyMinimum==energyMaximum) return energyMinimum`. But
    # Prometheus' check_consistency (utils/config_mims.py) rejects min >= max,
    # i.e. it is stricter than the library it wraps. That check runs in
    # Prometheus.__init__; inject() re-reads this config afterwards. So pass a
    # placeholder band through the constructor, then set the real single
    # energy before sim().
    #
    # DO NOT collapse this into a narrow band: with power_law == 1.0 LI samples
    # LOG-UNIFORMLY between min and max, so a band silently turns the fixed
    # energy into a distribution. The two-step is what keeps it exact.
    sim.minimal_energy = params.energy_gev
    sim.maximal_energy = params.energy_gev * 2.0     # placeholder, never sampled
    prom = Prometheus()                              # check_consistency runs here
    sim.minimal_energy = sim.maximal_energy = params.energy_gev
    prom.sim()
```

`* 2.0` is deliberate: an obvious placeholder rather than an epsilon someone
later mistakes for intended physics.

This exact two-step was run here on a 3-event injection and produced:

```
unique energies : [10000.]
all == 10000.0  : True
max |E - 10000.0| : 0.000e+00
```

**Regression test.** Assert both fields equal `energy_gev` after the two-step,
and that injected `properties["totalEnergy"]` is exactly `10000.0` for every
event. Also file an upstream issue against Prometheus (`>=` → `>`).

### 1.2 `--n-events` raises AttributeError (BLOCKING the null)

```
simulate.py:424  params.n_events = a.n_events
AttributeError: property 'n_events' of 'PhysicsParameters' object has no setter
```

`n_events` became a derived read-only property
(`events_per_point * len(injection_points)`, `physics.py:130-132`) but the CLI
still assigns to it. The prompt's own Step 1 command cannot run.

**The fix (decided):** set the underlying field, and **reject** a count not
divisible by the number of injection points. The assignment is contiguous blocks
of `events_per_point`, so a non-divisible value silently produces an unbalanced
set and breaks the design's equal-per-point invariant. Fail loudly.

```python
    if a.n_events:
        npts = len(params.injection_points)
        if a.n_events % npts:
            raise SystemExit(
                f"--n-events {a.n_events} is not divisible by the {npts} "
                f"injection points. The design needs an equal number of events "
                f"at each point; use a multiple of {npts}.")
        params.events_per_point = a.n_events // npts
```

Regression test: `--n-events 60` → `events_per_point == 20`; `--n-events 61`
exits non-zero.

### 1.3 Small cleanup — ALREADY APPLIED, just don't undo it

`.gitignore:73` used to read `src/prometheus_simulation/external/` **with a
trailing slash**, which matches a directory but **not a symlink**. Relocating
`external/` (see §2.3) then made it untracked, and `git add -A` would have
committed an absolute scratch path into the repo — violating "never commit
`external/`". The trailing slash has been removed; nothing to do.

---

## 2. Environment — survey your host first, then build

### 2.0 Run this survey before building. It decides three things.

```bash
ldd --version | head -1                 # glibc  -> §2.2
nproc
nvidia-smi --query-gpu=index,name,compute_cap,driver_version,memory.total --format=csv
nvcc --version | tail -2
df -h . && df -i .
getfattr -n ceph.quota.max_files -n ceph.dir.rfiles . 2>/dev/null   # if on CephFS -> §2.3
```

Three decisions come out of it:

1. **glibc < 2.34?** → you will hit the Conan `b2` failure. Pre-empt it with §2.2.
2. **GPU `compute_cap`?** → that is your PPC `arch=` value (8.6 → `arch=86`,
   7.5 → `arch=75`). And it decides whether JAX can use your GPUs at all (§2.4).
3. **Inode-quota filesystem?** → relocate `external/` per §2.3.

### 2.1 The build

```bash
git clone https://github.com/seanxia8/ORACLE.git && cd ORACLE && git checkout dev
bash src/prometheus_simulation/fetch_prometheus.sh     # pins 8c199384062012009094862bc244fa55f7694ee0
cd src/prometheus_simulation/external/prometheus
bash install.sh --with-ppc                              # --with-ppc REQUIRED
source scripts/activate.sh .prometheus_env
bash scripts/check_install.sh                           # expect "All imports successful"
```

If `install.sh` completes, skip to §2.4. If it dies, §2.2 is almost certainly why.

### 2.2 If glibc < 2.34: the Conan `b2` failure

Signature — `install.sh` dies building PROPOSAL:

```
b2: /lib64/libc.so.6: version `GLIBC_2.34' not found (required by b2)
ERROR: Failed building wheel for proposal
```

Conan ships **`b2` (Boost.Build) as a prebuilt binary needing glibc >= 2.34**.
Boost 1.85.0 requires exactly `b2/5.5.3`. Build it from source into the Conan
cache instead:

```bash
pip install conan && conan profile detect --force
conan install --tool-requires="b2/[*]" --build="b2/*"
```

Then run `install.sh`'s Linux steps 2–5 **individually** rather than re-running
`install.sh`, which would rebuild the 189-package micromamba env for nothing:

```bash
bash scripts/install_proposal.sh
bash scripts/install_leptoninjector_legacy.sh
bash scripts/install_ppc.sh
bash scripts/check_install.sh
bash scripts/fixes.sh "$PWD/.prometheus_env"
```

This worked on RHEL 8.10 / glibc 2.28 with **system gcc 8.5** — Boost and
PROPOSAL both compiled, no C++17 failure, no need for a newer toolset. **If you
do need a newer gcc, rebuild Boost AND PROPOSAL with it, never PROPOSAL alone**
— an ABI mix fails at runtime in ways far harder to diagnose than a compile
error.

**Escape hatch.** If the native build fights you at more than one step, stop and
switch to the shipped container rather than fixing moles one at a time:
`container/Dockerfile` (`ubuntu:22.04`) and `container/Dockerfile.gpu`
(`nvidia/cuda:12.3.2-devel-ubuntu22.04`, `--build-arg SM_ARCH=<your cap>`) are
glibc 2.35, where none of this arises. Apptainer works rootless if you lack
Docker daemon access. The container is also better provenance — an image hash
beats "I rebuilt b2 locally".

### 2.3 If your filesystem has an inode quota

Signature: a confusing `Disk quota exceeded` while linking conda packages, with
terabytes of space free. Check `ceph.quota.max_files` vs `ceph.dir.rfiles`.
A conda env plus the C++ builds is easily 100k+ inodes.

Layout that works:

- `external/` → **symlink to a filesystem with inodes to spare** (local scratch
  is fine: it is reproducible from `fetch_prometheus.sh` at a pinned commit, so
  losing it costs nothing, and it is what actually eats inodes). Point
  `CONDA_PKGS_DIRS` there too.
- `runs/` → **somewhere durable, never scratch.** It is the expensive,
  irreproducible artifact and it is few large files, so it does not stress an
  inode quota.

If you relocate `external/`, apply §1.3 or the symlink will show as untracked.

### 2.4 GPUs — check both paths separately, they are independent

There are two unrelated GPU paths and neither is on by default:

**Water (7 of 8 arms) → olympus → JAX.** `requirements.txt` pins the **CPU**
jaxlib, so a GPU box runs on CPU silently. Test on your host:

```bash
pip install "jax[cuda12]"       # cuda12 ONLY, never cuda13
python -c "import jax; print(jax.devices(), jax.default_backend())"
```

- `[CudaDevice(id=0), ...]` → water arms get the GPU. Good; the §4 timings do
  not apply to you and you should re-measure.
- `[CpuDevice(id=0)]` or a `no supported devices found for platform CUDA`
  error → your cards are below jaxlib's floor.

**On the debug host this failed**, and the failure mode matters: with 4× GTX
TITAN X (compute_cap **5.2**, Maxwell, driver 525.147.05),
`jax[cuda12]==0.11.1` installed cleanly then refused every device
(`Failed to create stream executor for device CUDA:0..3: CUDA_ERROR_INVALID_VALUE`,
`no supported devices found for platform CUDA`) — while a raw CUDA binary drove
the same cards fine. jaxlib has moved its floor above Maxwell independent of
CUDA version.

**Critical:** with the CUDA plugin installed but unusable, **JAX hard-fails
instead of falling back to CPU**, so every water arm dies. If that is your
situation, `pip uninstall jax-cuda12-pjrt jax-cuda12-plugin` and confirm
`jax.devices()` reads `[CpuDevice(id=0)]`. Do not leave it installed.

**Ice (1 arm) → PPC.** `install.sh --with-ppc` runs `make cpu` only. Build the
CUDA binary with **your** compute capability:

```bash
cd resources/PPC_executables/PPC_CUDA && make gpu arch=<your compute_cap without the dot>
./ppc            # should print "Found N devices" and list them
```

The Makefile defaults to `arch = 50`, so pass it explicitly. A mismatch fails at
run time, not build time. Then set `use_gpu: true` in
`config/physics_default.yaml` — it routes the ice arm to `ppc_cuda` and affects
**nothing else** (no config controls the water path).

### 2.5 Upstream quirk already compensated for

Already fixed on `dev` in `768c1dd`, but know it exists: `ppc.paths.force` does
**not** make a re-run safe. `prometheus.py` guards the PPC tmpdir then calls
`mkdir(exist_ok=False)` unconditionally, so `force` only converts a typed error
into a raw `FileExistsError`. `run_arm` removes the directory first and gives
each arm its own tmpdir.

---

## 3. Verify before spending CPU

```bash
export PYTHONPATH=src
python -m pytest src/prometheus_simulation/tests -q        # 53 + your new tests
python -m prometheus_simulation.simulate --out runs/pilot  # plan only
python examples/02_basic_ice.py                            # THE install gate
```

`02_basic_ice.py` is the gate. **If it fails, stop and report the error.** Do
not work around it, substitute another photon propagator, or drop the ice arm.
It passed on the debug host (3 events, exit 0).

Then read `plan.json` and check all of this:

- `design` = "fixed injection points, single energy, isotropic direction"
- `reference_geometry` = `flower_xl`
- `energy_gev` = `10000.0` — a **float, not a string**
- `point_check[*].inside_all_geometries` — all `true`
- `event_point_assignment` — 200 each of `centre`, `radial`, `vertical`
- `arms` — 8 entries, `hexagon_ice_le` = `medium_control`,
  `flower_xl__seed2` = `photon_null`
- recentring deltas kilometre-scale for the offset geometries

---

## 4. Timing — the original estimate is wrong by 1–2 orders of magnitude

`AGENT_PROMPT.md` budgets **1–10 CPU-s per event-simulation**. Measured on the
debug host (**CPU only** — that host had no usable JAX GPU):

| arm | propagator | measured | basis |
|---|---|---|---|
| demo ice (4860 modules) | PPC, CPU | **~280 s/event** | 839.58 s prop / 3 events |
| `flower_xl` (24,220 modules) | olympus, CPU | **~95 s/event** | 286.30 s prop / 3 events |

Both are averages over 3 events and **include first-call compilation**, so
steady-state is lower — how much lower is exactly what the null must separate.
Treat them as upper bounds, but do not assume they collapse.

Naive extrapolation at those rates, 600 events: `flower_xl` and
`flower_xl__seed2` ~16 h each; `hexagon_ice_le` on CPU PPC ~47 h — which is why
the CUDA PPC build matters.

**These numbers are host-specific.** If JAX uses your GPUs, the water arms will
be far faster and you should re-measure rather than plan against the table.

**Report per-event cost for every arm, first-event and steady-state separated.**
If the water arms really are 10–95× the budget on your host too, the whole
schedule needs rewriting, not just the ice arm. `olympus` has a `warm_up` flag
(documented "Does not change results") that compiles up front and makes
per-event time flat from the first event — useful for a clean steady-state
number.

**If the ice arm still projects to tens of hours after CUDA, stop and ask**
before running it. Reducing the control arm to a 200-of-600 subset is on the
table but is the owner's scope call, not yours.

---

## 5. What to run, in order

### Step A — the fixes
Apply §1.1, §1.2, §1.3 with regression tests. `pytest` stays green and gains the
new cases. Re-run the dry run and re-check §3.

### Step B — PPC CPU vs CUDA equivalence (paper-grade)

The ice arm is about to switch propagator implementation, so verify rather than
assume. Run ~21 events (7 per point) of `hexagon_ice_le` through **CPU PPC** and
**`ppc_cuda`** from the **same injection**, and compare hit-count and
time-residual distributions.

They will not be bit-identical — different RNG streams, different float
ordering. Agreement within Poisson is the pass condition. **If they diverge
beyond Poisson, stop and report: that is a PPC build problem, not a physics
result.** Report the comparison either way.

Each arm writes `arm_record.json` recording the propagator it *actually* used,
and `plan.json` carries `propagator_family` per arm (as of `db18d33`). Cite
those in the comparison rather than asserting which binary ran — that is the
whole point of the record.

Framing worth keeping straight: the ice arm is *already* a different
implementation from the water arms, because Prometheus picks the propagator from
the medium (water → olympus, ice → PPC). So `hexagon` vs `hexagon_ice_le`
changes propagation code as well as medium — the "medium control" is not
medium-only. That is structural, predates the GPU question, and is a **declared
limitation**. CPU-PPC vs CUDA-PPC is two builds of one algorithm on the same
side of a difference that already exists, so it adds no new confound.

### Step C — the null, at reduced N

```bash
python -m prometheus_simulation.simulate --out runs/null --n-events 60 --execute
```

Then:

```python
from prometheus_simulation.readout import load_event_set, check_pairing
truth, hits, plan = load_event_set("runs/null")
print(check_pairing(truth, plan))
```

**Acceptance: every row `paired = True`, and `plan.json:vertex_residual_max_m`
< 1e-6.** Do not loosen `atol`. If either fails, stop and report which arm and
what the residual is.

Both gates are real on both code paths as of `91b0122` — a plan.json missing
`vertex_residual_max_m` now **fails** rather than passing vacuously. If you see
"NOT RECORDED — gate cannot be satisfied", that is a stale plan.json:
regenerate, do not override.

This null is not a formality. Photon propagation reruns per arm and is seedable
only to Poisson level, so the same event in the *same* detector already gives
different light. That spread is the floor every cross-geometry number is
measured against. If it is as large as the geometry effect, the design fails —
and you learn that for ~10% of the cost.

### Step D — the 600-event set

```bash
python -m prometheus_simulation.simulate --out runs/pilot --execute --jobs 8 --gpus <idle ids>
```

`--jobs 8` because there are 8 independent arms. Pick `--gpus` from idle cards
only; leave headroom if the box is shared, because with
`XLA_PYTHON_CLIENT_PREALLOCATE=false` a returning job causes a mid-run OOM
rather than a clean refusal. If an arm dies, re-run just it with `--arm <name>`
— that replays the pinned injection and does **not** regenerate it.

If water runs on CPU, watch for core contention: JAX on CPU is multi-threaded,
so 8 concurrent arms may oversubscribe. Cap per-arm threads if throughput
disappoints.

If you shard, **shard the arms, never the injection** — every arm must replay
the same file.

### Step E — analysis

```bash
python -m prometheus_simulation.analyze --run runs/pilot
```

Read `REPORT.md`. Both gates must be `True`. Report `null_separation`
(`auc_vs_null` per arm — above 0.75 is separated from photon noise; near 0.5 is
not, and a null result there is a real result, not a failure) and
`light_yield.point_spread_ratio`.

---

## 6. Rules that still bind

- **Never regenerate the injection.** Written once, checksummed into
  `plan.json`, read-only thereafter. Every arm reads that file translated by its
  recentring delta. `check_pairing` is the only thing that catches a violation.
- **Never patch `external/prometheus/`.** LGPL-2.1, pinned by commit. Fixes go
  in `src/prometheus_simulation/` and pass through config.
- **Never commit `external/` or `runs/`.**
- **Do not touch `olympus_max_distance_m`** (300 m). It is physics, not memory —
  it drops source-module pairs before propagation, so a central `flower_xl`
  event only illuminates the inner ~300 m. That is a real property of the
  comparison and must be stated, not tuned away. For memory, lower
  `olympus_photon_chunk`.
- **Do not move an injection point outside the common region.** A point outside
  a detector produces nothing there, which looks like a geometry effect and is
  not.
- If an arm produces zero hits for every event, suspect the recentring — check
  `plan.json:arms[*].recentre_delta_m` against
  `python -m prometheus_simulation.geometry`. It is **not** a reason to enlarge
  anything.
- **Escalate rather than decide:** anything touching the injection, the gates,
  or `olympus_max_distance_m`. Storage layout and GPU selection are yours.
- **Report failures.** Do not silently reduce scope, swap a geometry, or fall
  back to independent per-geometry injections — independent injections are
  exactly what this experiment exists to avoid.

## 7. What to report back

1. `runs/pilot/analysis/` in full — `REPORT.md`, the CSVs, the figures.
2. `plan.json` and `physics_record.json`.
3. The PPC CPU-vs-CUDA comparison (Step B), pass or fail.
4. **Per-arm wall clock and per-event cost, first-event and steady-state
   separated**, plus peak memory for `flower_xl` (24,220 modules).
5. Your host's survey results (§2.0) and which route produced the data —
   native build or container, and whether JAX used a GPU. `physics_record.json`
   captures the toolchain and JAX backend automatically as of `ced5f40`, and
   each arm's `arm_record.json` captures the propagator it actually used as of
   `db18d33`; say it explicitly anyway.
6. The `DECLARED DEVIATIONS` list the run prints at the end.
7. Anything you changed and why. A silent fix to the injection or recentring
   path invalidates the whole set.
