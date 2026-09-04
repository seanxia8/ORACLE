"""Generate ONE event set and replay it through every geometry.

    step 1  inject once, in the reference geometry's frame, into the shared
            cylinder chosen by `geometry.injection_cylinder`
    step 2  for each geometry: recentre the injection, override the medium,
            run photon propagation with `inject=False`
    step 3  re-run ONE geometry with a different photon seed -- the pairing
            null, which is the scale every cross-geometry number is measured in

Nothing here regenerates the injection. The injection file is the pairing key:
it is written once, checksummed, and thereafter read-only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np

from .geometry import (NUBENCH_GEOFILES, NUBENCH_MEDIA, check_points,
                       common_region, default_geodir, load_geometries,
                       recentre_delta)
from .physics import PhysicsParameters

HERE = Path(__file__).resolve().parent


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def plan(params: PhysicsParameters, geodir: Path, out: Path) -> dict:
    """Everything decided before a single event is simulated.

    Runs without Prometheus installed, which is what makes it testable and
    what the remote agent must run and read before spending any CPU.
    """
    geoms = load_geometries(geodir, list(params.datasets))
    region = common_region(geoms)
    point_check = check_points(params.injection_points, geoms)
    bad = [r["point"] for r in point_check if not r["inside_all_geometries"]]
    if bad:
        raise ValueError(
            f"injection point(s) {bad} lie outside at least one geometry. "
            f"Common region: radius <= {region['max_radius_m']:.1f} m "
            f"(set by {region['binding_radius']}), |z| <= "
            f"{region['max_abs_z_m']:.1f} m (set by {region['binding_height']}). "
            "A point outside a detector produces no light there, which looks "
            "like a geometry effect and is not.")

    # Reference frame: the geometry the injection file is written in. The
    # largest, so the Earth model is built at a realistic deep-detector depth.
    ref = max(geoms, key=lambda k: geoms[k].r_horizontal)

    # Event -> injection point assignment. Contiguous blocks, so event_id
    # ordering is stable and every arm inherits the same assignment.
    names = list(params.injection_points)
    assignment = [n for n in names for _ in range(params.events_per_point)]

    arms = []
    for key, g in geoms.items():
        arms.append({
            "arm": key,
            "geofile": NUBENCH_GEOFILES[key],
            "medium": NUBENCH_MEDIA[key],
            "recentre_delta_m": recentre_delta(geoms[ref], g).tolist(),
            "depth_m": g.depth,
            "n_modules": g.n_modules,
            "n_strings": g.n_strings,
            "r_horizontal_m": g.r_horizontal,
            "photon_seed": params.photon_seed,
            "role": "geometry",
        })
    if params.include_ice_control and "hexagon" in geoms:
        arms.append({**arms[[a["arm"] for a in arms].index("hexagon")],
                     "arm": "hexagon_ice_le", "medium": "ice",
                     "role": "medium_control"})
    # The pairing null: the REFERENCE geometry re-run with the same events and
    # a different photon seed. This is the scale in which every cross-geometry
    # displacement is reported -- run it first, at reduced N.
    ref_arm = arms[[a["arm"] for a in arms].index(ref)]
    arms.append({**ref_arm, "arm": f"{ref}__seed{params.photon_seed + 1}",
                 "photon_seed": params.photon_seed + 1, "role": "photon_null"})

    return {
        "design": "fixed injection points, single energy, isotropic direction",
        "reference_geometry": ref,
        "reference_offset_m": geoms[ref].offset.tolist(),
        "common_region": region,
        "injection_points": params.injection_points,
        "point_check": point_check,
        "events_per_point": params.events_per_point,
        "n_events": params.n_events,
        "energy_gev": params.energy_gev,
        "event_point_assignment": assignment,
        "arms": arms,
        "fingerprint": params.fingerprint(),
        "out": str(out),
    }


# --------------------------------------------------------------------- run --
def inject_once(params: PhysicsParameters, plan_d: dict, geodir: Path,
                out: Path) -> Path:
    from prometheus import Prometheus, config   # noqa: PLC0415

    out.mkdir(parents=True, exist_ok=True)
    config.run.nevents = params.n_events
    config.run.random_state_seed = params.injection_seed
    config.run.storage_prefix = str(out) + "/"
    config.detector.geo_file = str(
        geodir / NUBENCH_GEOFILES[plan_d["reference_geometry"]])

    config.injection.name = "LeptonInjector"
    config.injection.lepton_injector.inject = True
    sim = config.injection.lepton_injector.simulation
    sim.is_ranged = params.is_ranged
    sim.final_state_1 = params.final_state_1
    sim.final_state_2 = params.final_state_2
    # Single energy: min == max. The sampled vertices are then OVERWRITTEN by
    # set_vertices(), so the cylinder only has to be a legal volume for LI --
    # it is sized to the common region so the pre-overwrite vertices are
    # already in the right neighbourhood.
    sim.minimal_energy = params.energy_gev
    sim.maximal_energy = params.energy_gev
    sim.power_law = params.power_law
    sim.min_zenith, sim.max_zenith = params.min_zenith_deg, params.max_zenith_deg
    sim.min_azimuth, sim.max_azimuth = params.min_azimuth_deg, params.max_azimuth_deg
    sim.cylinder_radius = plan_d["common_region"]["max_radius_m"]
    sim.cylinder_height = 2 * plan_d["common_region"]["max_abs_z_m"]
    sim.endcap_length = params.endcap_length_m
    Prometheus().sim()

    li = sorted(out.glob("*.h5")) or sorted(out.glob("*.hdf5"))
    if not li:
        raise FileNotFoundError(f"no injection file produced in {out}")
    return li[0]


_LI_KEYS = ("final_1", "final_2", "initial")


def _shift_h5(path: Path, delta: np.ndarray) -> None:
    """Translate every stored position by ``delta``.

    Mirrors ``lepton_injector_utils.apply_detector_offset`` exactly -- the three
    particle groups plus the scalar x/y/z in ``properties``. ``delta`` is either
    a single (3,) vector or a per-event (n_events, 3) array.
    """
    import h5py   # noqa: PLC0415

    delta = np.asarray(delta, dtype=float)
    with h5py.File(path, "r+") as h5f:
        inj = h5f[list(h5f.keys())[0]]
        d = delta if delta.ndim == 1 else delta.reshape(-1, 3)
        for key in _LI_KEYS:
            if key in inj:
                inj[key]["Position"] = inj[key]["Position"] + d
        if "properties" in inj:
            for i, ax in enumerate("xyz"):
                col = d[i] if delta.ndim == 1 else d[:, i]
                inj["properties"][ax] = inj["properties"][ax] + col


def read_vertices(path: Path) -> np.ndarray:
    import h5py   # noqa: PLC0415

    with h5py.File(path, "r") as h5f:
        inj = h5f[list(h5f.keys())[0]]
        return np.asarray(inj["initial"]["Position"], dtype=float)


def set_vertices(path: Path, targets: np.ndarray) -> np.ndarray:
    """Move each event's vertex to its assigned injection point.

    LeptonInjector samples vertices in a volume; we overwrite them so every
    event sits at one of three known points. The per-event shift is applied to
    the initial state AND both final states, so the interaction stays
    internally consistent -- PROPOSAL then propagates the secondaries from the
    new vertex when the arm runs, which is why the physics remains valid.

    Returns the applied per-event delta, for the record.
    """
    current = read_vertices(path)
    targets = np.asarray(targets, dtype=float)
    if targets.shape != current.shape:
        raise ValueError(f"targets {targets.shape} != vertices {current.shape}")
    delta = targets - current
    _shift_h5(path, delta)
    return delta


def recentre_injection(src: Path, dst: Path, delta: np.ndarray) -> None:
    """Copy the reference injection and translate it into another detector's
    frame. Operates on a COPY: the reference injection is the pairing key and
    is never modified."""
    shutil.copy(src, dst)
    _shift_h5(dst, np.asarray(delta, dtype=float))


def override_medium(geofile: Path, medium: str, dst: Path) -> Path:
    """Rewrite the `Medium:` header. Prometheus picks PPC (ice) vs olympus
    (water) from it, so this selects the whole photon-propagation branch."""
    lines = geofile.read_text().splitlines()
    dst.write_text("\n".join(
        f"Medium:\t{medium}" if ln.strip().lower().startswith("medium") else ln
        for ln in lines) + "\n")
    return dst


def run_arm(arm: dict, params: PhysicsParameters, injection: Path, lic: Path,
            geodir: Path, out: Path) -> None:
    from prometheus import Prometheus, config   # noqa: PLC0415

    arm_dir = out / arm["arm"]
    arm_dir.mkdir(parents=True, exist_ok=True)
    local_inj = arm_dir / "injection.h5"
    recentre_injection(injection, local_inj, np.asarray(arm["recentre_delta_m"]))
    geo = override_medium(geodir / arm["geofile"], arm["medium"],
                          arm_dir / f"{arm['arm']}.geo")

    config.detector.geo_file = str(geo)
    config.run.storage_prefix = str(arm_dir) + "/"
    config.run.random_state_seed = arm["photon_seed"]
    config.injection.name = "LeptonInjector"
    config.injection.lepton_injector.inject = False       # <- the pairing switch
    config.injection.lepton_injector.paths.injection_file = str(local_inj)
    config.injection.lepton_injector.paths.lic_file = str(lic)
    if arm["medium"] == "ice":
        # PPC (g++, CPU) vs ppc_cuda (nvcc). install.sh --with-ppc builds only
        # the CPU one; the CUDA binary comes from container/Dockerfile.gpu or
        # `make gpu arch=<SM>` in resources/PPC_executables/PPC_CUDA.
        #
        # Two problems with the shipped tmpdir handling, both fixed here rather
        # than by patching external/prometheus (LGPL, pinned by commit):
        #
        # 1. `force` does NOT make a re-run safe. prometheus.py guards with
        #    `if tmpdir.exists() and not force: raise PpcTmpdirExistsError`,
        #    then calls `mkdir(exist_ok=False)` unconditionally -- so force
        #    only converts a typed error into a raw FileExistsError. A
        #    `--arm hexagon_ice_le` retry after a crash or an OOM would fail on
        #    the leftover directory instead of resuming. We remove it first.
        # 2. The default tmpdir is `./.ppc_tmp`, relative to the working
        #    directory, so two ice arms running concurrently would share one
        #    scratch directory and corrupt each other. Only hexagon_ice_le is
        #    ice today, which is the only reason this has not bitten; giving
        #    each arm its own tmpdir removes the latent trap.
        ppc_tmp = arm_dir / ".ppc_tmp"
        if ppc_tmp.exists():
            shutil.rmtree(ppc_tmp)          # Prometheus mkdirs it with exist_ok=False
        if params.use_gpu:
            config.photon_propagator.name = "ppc_cuda"
            config.photon_propagator.ppc_cuda.paths.ppc_tmpdir = str(ppc_tmp)
            config.photon_propagator.ppc_cuda.paths.force = True
        else:
            config.photon_propagator.name = "PPC"
            config.photon_propagator.ppc.paths.ppc_tmpdir = str(ppc_tmp)
            config.photon_propagator.ppc.paths.force = True
    else:
        # Water -> olympus (JAX). It runs on the GPU only if a CUDA jaxlib is
        # installed; there is no config switch for that.
        oly = config.photon_propagator.olympus.simulation
        oly.max_distance = params.olympus_max_distance_m
        oly.photon_chunk = params.olympus_photon_chunk
    Prometheus().sim()


def prepare_injection(params: PhysicsParameters, plan_d: dict, geodir: Path,
                      out: Path) -> "tuple[Path, Path]":
    """Inject once, place the vertices, and record the provenance.

    ONE function for both the serial and the parallel path. They used to do
    this separately and the parallel copy silently omitted
    ``vertex_residual_max_m`` -- which analyze.py then treated as a pass, so
    running in parallel made the `vertices_on_points` gate vacuously true. A
    gate that can only be satisfied by one of two code paths is not a gate, so
    the duplication is removed rather than the second copy patched.

    Returns (injection_file, lic_file) and updates ``plan_d`` in place.
    """
    inj_dir = out / "_injection"
    injection = inject_once(params, plan_d, geodir, inj_dir)
    lics = sorted(inj_dir.glob("*.lic"))
    if not lics:
        raise FileNotFoundError(f"no .lic file produced in {inj_dir}")

    # Place every vertex on its assigned point, in the REFERENCE frame.
    ref_offset = np.asarray(plan_d["reference_offset_m"])
    wanted = np.array([np.asarray(params.injection_points[n])
                       for n in plan_d["event_point_assignment"]])
    applied = set_vertices(injection, ref_offset + wanted)
    achieved = read_vertices(injection) - ref_offset

    plan_d["vertex_shift_max_m"] = float(np.abs(applied).max())
    plan_d["vertex_residual_max_m"] = float(np.abs(achieved - wanted).max())
    plan_d["injection_file"] = str(injection)
    plan_d["injection_sha256"] = _sha256(injection)
    (out / "plan.json").write_text(json.dumps(plan_d, indent=2))
    return injection, lics[0]


GPU_ENV_NOTE = """\
JAX preallocates ~75% of a GPU by default, which fails immediately when another
process already holds part of the card (and on this box GPU 0 usually does).
XLA_PYTHON_CLIENT_PREALLOCATE=false makes it allocate on demand instead."""


def _arm_env(gpu: "int | None") -> dict:
    env = dict(os.environ)
    env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    env.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    return env


def dispatch_arms(out: Path, arms: "list[str]", gpus: "list[int] | None",
                  jobs: int) -> dict:
    """Run arms as separate processes, round-robin over the given GPUs.

    Separate processes rather than threads or a ProcessPool: each arm needs its
    own JAX runtime pinned to one device via CUDA_VISIBLE_DEVICES, and JAX does
    not survive being forked after initialisation. The arms are independent by
    construction -- they all read the same immutable injection -- so this is
    embarrassingly parallel with no coordination.
    """
    import subprocess   # noqa: PLC0415

    pending = list(arms)
    running: list[tuple] = []
    results: dict[str, int] = {}
    while pending or running:
        while pending and len(running) < jobs:
            arm = pending.pop(0)
            gpu = gpus[len(running) % len(gpus)] if gpus else None
            log = out / arm / "run.log"
            log.parent.mkdir(parents=True, exist_ok=True)
            fh = open(log, "w")
            cmd = [sys.executable, "-m", "prometheus_simulation.simulate",
                   "--out", str(out), "--arm", arm, "--execute"]
            proc = subprocess.Popen(cmd, env=_arm_env(gpu), stdout=fh,
                                    stderr=subprocess.STDOUT)
            print(f"  -> {arm}  (gpu={gpu if gpu is not None else 'cpu'})  log: {log}")
            running.append((arm, proc, fh))
        time.sleep(1.0)
        for entry in list(running):
            arm, proc, fh = entry
            if proc.poll() is not None:
                fh.close()
                running.remove(entry)
                results[arm] = proc.returncode
                status = "ok" if proc.returncode == 0 else f"FAILED rc={proc.returncode}"
                print(f"  <- {arm}: {status}")
    return results


def build_event_set(params: PhysicsParameters, geodir: Path, out: Path,
                    execute: bool = False) -> dict:
    plan_d = plan(params, geodir, out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "plan.json").write_text(json.dumps(plan_d, indent=2))
    params.record(out, extra={"plan": plan_d})
    if not execute:
        return plan_d

    injection, lic = prepare_injection(params, plan_d, geodir, out)

    for arm in plan_d["arms"]:
        print(f"--- arm {arm['arm']} ({arm['role']}, medium={arm['medium']}) ---")
        run_arm(arm, params, injection, lic, geodir, out)
    return plan_d


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--params", type=Path,
                    default=HERE / "config" / "physics_default.yaml")
    ap.add_argument("--geodir", type=Path, default=None,
                    help="defaults to the Prometheus clone, else the geofiles "
                         "shipped in data/geofiles/")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-events", type=int, default=None,
                    help="override n_events (use a small value for a smoke test)")
    ap.add_argument("--execute", action="store_true",
                    help="actually run Prometheus; without it only the plan is written")
    ap.add_argument("--arm", default=None,
                    help="run ONE arm from an existing plan.json (used by --jobs)")
    ap.add_argument("--jobs", type=int, default=1,
                    help="run this many arms concurrently, as separate processes")
    ap.add_argument("--gpus", default=None,
                    help="comma-separated GPU ids to spread arms over, e.g. 1,2,3. "
                         "Omit to run on the CPU. Skip any GPU already in use.")
    a = ap.parse_args()

    geodir = a.geodir or default_geodir()
    params = PhysicsParameters.from_yaml(a.params) if a.params.exists() \
        else PhysicsParameters()

    # single-arm mode: replay one arm of an existing set, in this process
    if a.arm:
        plan_d = json.loads((a.out / "plan.json").read_text())
        arm = next((x for x in plan_d["arms"] if x["arm"] == a.arm), None)
        if arm is None:
            raise SystemExit(f"no arm {a.arm!r} in {a.out/'plan.json'}")
        injection = Path(plan_d["injection_file"])
        lics = sorted((a.out / "_injection").glob("*.lic"))
        if not lics:
            raise SystemExit(f"no .lic in {a.out/'_injection'}")
        run_arm(arm, params, injection, lics[0], geodir, a.out)
        return

    if a.n_events:
        params.n_events = a.n_events
    gpus = [int(g) for g in a.gpus.split(",")] if a.gpus else None
    parallel = a.execute and (a.jobs > 1 or gpus)

    # In parallel mode, inject once here and let the dispatcher run the arms.
    p = build_event_set(params, geodir, a.out, a.execute and not parallel)
    if parallel:
        p = build_event_set(params, geodir, a.out, execute=False)
        prepare_injection(params, p, geodir, a.out)
        print(f"injection written and pinned: {p['injection_sha256'][:16]}  "
              f"vertex residual {p['vertex_residual_max_m']:.3e} m")
        print(GPU_ENV_NOTE)
        rc = dispatch_arms(a.out, [x["arm"] for x in p["arms"]], gpus,
                           max(a.jobs, len(gpus) if gpus else 1))
        bad = {k: v for k, v in rc.items() if v != 0}
        if bad:
            print(f"\nARMS FAILED: {bad} -- see each arm's run.log", file=sys.stderr)
            raise SystemExit(1)
    print(json.dumps({k: v for k, v in p.items() if k != "arms"}, indent=2))
    print(f"arms: {[x['arm'] for x in p['arms']]}")
    if params.unresolved():
        print("\nDECLARED DEVIATIONS (parameters NuBench never published):")
        for k in params.unresolved():
            print(f"  {k} = {getattr(params, k)}")


if __name__ == "__main__":
    main()
