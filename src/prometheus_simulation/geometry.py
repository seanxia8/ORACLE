"""Detector geometries from the Prometheus geofiles.

Three facts, read from the Prometheus source, drive everything here:

1. ``detector.py`` defines the detector *offset* as the mean module position.
2. ``lepton_injector_utils.apply_detector_offset`` adds that offset to every
   vertex **in place in the injection file**, and only on the ``inject=True``
   path.
3. ``LI_injection.injection_from_LI_output`` signature is ``(LI_file, **_)`` --
   it **ignores** ``detector_offset`` when loading an existing injection.

Therefore a stored injection lives in the *reference* detector's absolute
frame. The shipped geofiles are centred anywhere from z = +95 m (orca) to
z = -3194 m (arca), so replaying an injection verbatim in another geometry
puts every event kilometres away and nothing triggers. `recentre_delta` is
the correction, and it is what makes the event set paired rather than merely
similar.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

#: The six NuBench geometries, mapped to the geofiles Prometheus ships.
NUBENCH_GEOFILES = {
    "flower_s": "orca.geo",           # ORCA-inspired
    "flower_l": "arca.geo",           # ARCA-inspired
    "flower_xl": "trident.geo",       # TRIDENT-inspired
    "triangle": "pone_triangle.geo",  # P-ONE-inspired
    "cluster": "gvd.geo",             # Baikal-GVD-inspired
    "hexagon": "icecube.geo",         # IceCube-inspired
}

#: NuBench simulated all six geometries in WATER, then added one ICE dataset
#: on the Hexagon geometry. The shipped .geo headers disagree (orca/arca say
#: "mediterranean", icecube says "ice") and the header is what selects the
#: photon propagator, so it must be rewritten per run.
NUBENCH_MEDIA = {k: "water" for k in NUBENCH_GEOFILES}
NUBENCH_MEDIA["hexagon_ice_le"] = "ice"


@dataclass(frozen=True)
class Geometry:
    name: str
    medium_header: str
    coords: np.ndarray            # (n_modules, 3), metres
    string_ids: np.ndarray        # (n_modules,), from column 4 of the geofile

    @property
    def offset(self) -> np.ndarray:
        """Exactly Prometheus' ``detector.offset``: the mean module position."""
        return self.coords.mean(axis=0)

    @property
    def r_horizontal(self) -> float:
        return float(np.linalg.norm(self.coords[:, :2] - self.offset[:2], axis=1).max())

    @property
    def half_height(self) -> float:
        return float((self.coords[:, 2].max() - self.coords[:, 2].min()) / 2.0)

    @property
    def depth(self) -> float:
        """Depth the Earth model is built at (= -offset_z)."""
        return float(-self.offset[2])

    @property
    def n_modules(self) -> int:
        return int(len(self.coords))

    @property
    def n_strings(self) -> int:
        # Column 4 of the geofile, not unique (x, y): IceCube strings are
        # tilted, so distinct modules on one string differ in x and y.
        return int(len(np.unique(self.string_ids)))

    def bounding_volume(self) -> float:
        return float(np.pi * self.r_horizontal ** 2 * 2 * self.half_height)

    def as_row(self) -> dict:
        o = self.offset
        return {
            "name": self.name,
            "n_modules": self.n_modules,
            "n_strings": self.n_strings,
            "offset_x": float(o[0]), "offset_y": float(o[1]), "offset_z": float(o[2]),
            "r_horizontal_m": self.r_horizontal,
            "half_height_m": self.half_height,
            "depth_m": self.depth,
            "bounding_volume_m3": self.bounding_volume(),
            "medium_header": self.medium_header,
        }


def load_geo(path: str | Path) -> Geometry:
    path = Path(path)
    medium, rows, sids = "unknown", [], []
    for line in path.read_text().splitlines():
        s = line.strip()
        if not s:
            continue
        low = s.lower()
        if low.startswith("medium"):
            medium = s.split(":", 1)[1].strip() if ":" in s else "unknown"
            continue
        if s.startswith("#") or low.startswith("dom"):
            continue
        parts = s.split()
        try:
            rows.append([float(parts[0]), float(parts[1]), float(parts[2])])
        except (ValueError, IndexError):
            continue
        sids.append(parts[3] if len(parts) > 3 else str(len(rows)))
    if not rows:
        raise ValueError(f"no module coordinates parsed from {path}")
    return Geometry(path.stem, medium, np.asarray(rows, dtype=float),
                    np.asarray(sids))


def load_geometries(geodir: str | Path,
                    names: list[str] | None = None) -> dict[str, Geometry]:
    """Load the NuBench geometries by dataset key (``flower_s``, ...)."""
    geodir = Path(geodir)
    names = names or list(NUBENCH_GEOFILES)
    return {n: load_geo(geodir / NUBENCH_GEOFILES[n]) for n in names}


def recentre_delta(src: Geometry, dst: Geometry) -> np.ndarray:
    """Translation carrying an injection from ``src``'s frame into ``dst``'s.

    Positions move; energies and directions do not. That is the whole point:
    the same physical event, seen by a different detector.
    """
    return dst.offset - src.offset


def _coaxial_overlap(r_a: float, h_a: float, r_b: float, h_b: float) -> float:
    """Volume shared by two coaxial, co-centred cylinders. Exact.

    Both the injection cylinder and each detector's bounding cylinder are
    centred on the detector after recentring, so the overlap is simply the
    cylinder of the smaller radius and the smaller half-height. A plain volume
    ratio is NOT equivalent: `triangle` is tall and thin (r 58 m, half-height
    500 m) while `hexagon` is wide and shallow, so comparing volumes alone
    silently overstates their overlap.
    """
    return float(np.pi * min(r_a, r_b) ** 2 * 2 * min(h_a, h_b))


def common_region(geoms: dict[str, Geometry]) -> dict:
    """The detector-relative envelope that lies inside EVERY geometry.

    This is the region a fixed injection point must sit in for the point to
    mean the same thing in every detector. It is set by the smallest detector
    on each axis independently -- radius by `triangle` (57.7 m), half-height by
    `flower_s` (95.4 m) -- so it is smaller than any single detector.
    """
    r = min(g.r_horizontal for g in geoms.values())
    h = min(g.half_height for g in geoms.values())
    return {
        "max_radius_m": float(r),
        "max_abs_z_m": float(h),
        "binding_radius": min(geoms, key=lambda k: geoms[k].r_horizontal),
        "binding_height": min(geoms, key=lambda k: geoms[k].half_height),
        "volume_m3": float(np.pi * r ** 2 * 2 * h),
    }


def check_points(points: dict[str, list[float]], geoms: dict[str, Geometry]) -> "list[dict]":
    """Verify each fixed injection point lies inside every geometry.

    A point outside even one detector breaks the comparison for that arm --
    the event is still simulated, it just lands outside the instrumented
    volume and produces nothing, which looks like a geometry effect and is not.
    """
    reg = common_region(geoms)
    rows = []
    for name, xyz in points.items():
        x, y, z = xyz
        rho = float(np.hypot(x, y))
        rows.append({
            "point": name, "x": x, "y": y, "z": z,
            "radius_m": rho, "abs_z_m": abs(float(z)),
            "radius_headroom": float(rho / reg["max_radius_m"]),
            "z_headroom": float(abs(z) / reg["max_abs_z_m"]),
            "inside_all_geometries": bool(rho <= reg["max_radius_m"]
                                          and abs(z) <= reg["max_abs_z_m"]),
        })
    return rows


def injection_cylinder(geoms: dict[str, Geometry], mode: str = "intersection",
                       margin_r: float = 0.0, margin_h: float = 0.0) -> dict:
    """Shared injection cylinder for a RANDOM-VOLUME event set.

    Kept for the volume-sampled design. The fixed-point design in
    `physics.PhysicsParameters.injection_points` does not use it, and for a
    six-geometry set that is the better choice: there is no balanced radius.
    `sweep_cylinders` shows why -- the worst-case `probes_detector` never rises
    above ~0.07 at any radius, because flower_xl is 34x triangle.

    ``detector_sees``   share of injected events this detector can contain.
    ``probes_detector`` share of this detector's own volume the set explores.
    Both use the exact coaxial overlap, not a volume ratio.
    """
    if mode not in {"intersection", "union"}:
        raise ValueError(f"mode must be 'intersection' or 'union', got {mode!r}")
    agg = min if mode == "intersection" else max
    radius = agg(g.r_horizontal for g in geoms.values()) + margin_r
    half_h = agg(g.half_height for g in geoms.values()) + margin_h
    v_cyl = float(np.pi * radius ** 2 * 2 * half_h)
    sees, probes = {}, {}
    for k, g in geoms.items():
        v_o = _coaxial_overlap(radius, half_h, g.r_horizontal, g.half_height)
        sees[k] = v_o / v_cyl
        probes[k] = v_o / g.bounding_volume()
    return {
        "mode": mode,
        "cylinder_radius_m": float(radius),
        "cylinder_half_height_m": float(half_h),
        "cylinder_height_m": float(2 * half_h),
        "cylinder_volume_m3": v_cyl,
        "detector_sees": sees,
        "probes_detector": probes,
        "worst_detector_sees": float(min(sees.values())),
        "worst_probes_detector": float(min(probes.values())),
    }


def sweep_cylinders(geoms: dict[str, Geometry],
                    radii: "list[float] | None" = None) -> "list[dict]":
    """Evidence that no balanced cylinder exists for a six-geometry set."""
    radii = radii or [57.7, 100, 150, 200, 250, 300, 500, 1000, 1949.6]
    out = []
    for r in radii:
        h = min(max(r, 95.4), max(g.half_height for g in geoms.values()))
        v_cyl = float(np.pi * r ** 2 * 2 * h)
        sees, probes = [], []
        for g in geoms.values():
            v_o = _coaxial_overlap(r, h, g.r_horizontal, g.half_height)
            sees.append(v_o / v_cyl)
            probes.append(v_o / g.bounding_volume())
        out.append({"radius_m": r, "half_height_m": h,
                    "worst_detector_sees": min(sees),
                    "worst_probes_detector": min(probes)})
    return out


def survey(geodir: str | Path, names: list[str] | None = None) -> None:
    """Print the geometry table and both cylinder choices. Reproduces the
    numbers quoted in AGENT_PROMPT.md and the Tier-2 runbook."""
    import json
    geoms = load_geometries(geodir, names)
    print(f"{'dataset':12s} {'geofile':16s} {'mods':>6s} {'str':>5s} "
          f"{'off_z':>9s} {'r_h':>8s} {'half_h':>8s} {'header':>14s}")
    for k, g in geoms.items():
        r = g.as_row()
        print(f"{k:12s} {NUBENCH_GEOFILES[k]:16s} {r['n_modules']:6d} "
              f"{r['n_strings']:5d} {r['offset_z']:9.1f} {r['r_horizontal_m']:8.1f} "
              f"{r['half_height_m']:8.1f} {r['medium_header']:>14s}")
    for mode in ("intersection", "union"):
        print(f"\n{mode}:")
        print(json.dumps(injection_cylinder(geoms, mode), indent=2))


def default_geodir() -> Path:
    """Where to read the .geo files from.

    Prefers the Prometheus clone when present, and falls back to the copies
    shipped as package data in `data/geofiles/` — so the geometry survey, the
    plan and the whole test suite work with no clone at all. Only actually
    running Prometheus needs `fetch_prometheus.sh`.
    """
    here = Path(__file__).resolve().parent
    clone = here / "external" / "prometheus" / "resources" / "geofiles"
    if clone.is_dir():
        return clone
    vendored = here / "data" / "geofiles"
    if vendored.is_dir():
        return vendored
    raise FileNotFoundError(
        "No geofiles found. Run src/prometheus_simulation/fetch_prometheus.sh, "
        "or place .geo files in src/prometheus_simulation/data/geofiles/.")


if __name__ == "__main__":
    import sys
    survey(sys.argv[1] if len(sys.argv) > 1 else default_geodir())
