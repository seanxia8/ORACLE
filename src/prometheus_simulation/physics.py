"""The recorded physics configuration.

Every value carries a provenance tag, because the difference between "NuBench
said so" and "we chose it" is the difference between a reproduction and an
inspired-by. The tags are:

    nubench   stated in arXiv:2511.13111
    ours      our choice; NuBench does not constrain it
    ask       NOT published -- we chose a value, and the choice is a deviation
              that must be declared (and is what to ask the authors for)

`PhysicsParameters.record()` writes the full set, tags included, next to the
simulation output. Nothing about a produced event set should ever have to be
reconstructed from memory.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import yaml

PROVENANCE = {"nubench", "ours", "ask"}


@dataclass
class PhysicsParameters:
    # --- interaction -------------------------------------------------------
    final_state_1: str = "MuMinus"          # nubench: nu_mu CC
    final_state_2: str = "Hadrons"          # nubench
    interactions: tuple[str, ...] = ("CC", "NC")   # nubench

    # --- spectrum: ONE energy, not a spectrum ------------------------------
    # Fixed at 10 TeV. Bright enough that the sparse geometries trigger
    # reliably (flower_xl, ~90 m string spacing), not so bright that the small
    # dense ones saturate. Sampling a spectrum instead would mean every arm's
    # reconstruction error is a mixture over energy, and the geometry effect
    # would have to be disentangled from it.
    energy_gev: float = 1.0e4               # ours
    power_law: float = 1.0                  # ask -- inert while min == max

    # --- direction ---------------------------------------------------------
    # Isotropic. Directional acceptance is one of the largest geometry effects
    # -- a 3-string detector is far more direction-sensitive than a hexagonal
    # array -- so fixing the direction would hide the thing we are measuring.
    min_zenith_deg: float = 0.0             # ask
    max_zenith_deg: float = 180.0           # ask
    min_azimuth_deg: float = 0.0            # ask
    max_azimuth_deg: float = 360.0          # ask

    # --- injection points --------------------------------------------------
    # DETECTOR-RELATIVE coordinates, applied after recentring, so a point means
    # the same thing in every geometry. They must lie inside the common region
    # (geometry.common_region): radius <= 57.7 m, |z| <= 95.4 m, set by
    # `triangle` and `flower_s` respectively.
    #
    #   centre    on-axis, mid-depth        -- the best case for every detector
    #   radial    40 m out (69% of common radius)
    #   vertical  70 m up  (73% of common |z|)
    #
    # Three points, not a sampled volume: for six geometries spanning 34x in
    # radius there is NO balanced injection cylinder -- see
    # geometry.sweep_cylinders. Fixed points remove the acceptance confound
    # instead of trading it off.
    injection_points: dict = field(default_factory=lambda: {
        "centre":   [0.0, 0.0, 0.0],
        "radial":   [40.0, 0.0, 0.0],
        "vertical": [0.0, 0.0, 70.0],
    })                                       # ours
    events_per_point: int = 200             # ours
    is_ranged: bool = False                 # ours: volume mode
    endcap_length_m: float = 500.0          # ask (inert in volume mode)
    earth_model: str | None = None          # ask -- not published

    # --- photon propagation backend ---------------------------------------
    # Two INDEPENDENT GPU paths, and neither is on by default:
    #   water (olympus) is JAX -- needs `pip install "jax[cuda12]"`; the jax
    #     pinned in requirements.txt is the CPU build, so a GPU box silently
    #     runs on CPU until that is installed.
    #   ice (PPC) needs the separately compiled PPC_CUDA binary. install.sh
    #     --with-ppc builds `make cpu` ONLY; the CUDA binary comes from
    #     container/Dockerfile.gpu or a manual `make gpu arch=<SM>`.
    # Seven of our eight arms are water, so the JAX path is the one that
    # matters most; only hexagon_ice_le uses PPC.
    use_gpu: bool = False                   # ours

    # olympus `max_distance`: source-module pairs beyond this are dropped
    # BEFORE propagation. Its own docstring says this "changes physics, not
    # just memory" -- it is not a tuning knob. At 300 m in water it is many
    # absorption lengths and defensible, but it means a central event in
    # flower_xl (r = 1950 m) only ever illuminates the inner ~300 m, so the
    # cross-geometry light-yield comparison is really comparing local string
    # density near the vertex. State that rather than raising it.
    olympus_max_distance_m: float = 300.0   # ours
    olympus_photon_chunk: int = 2 ** 18     # ours -- pure memory; lower on a small GPU

    # --- run ---------------------------------------------------------------
    injection_seed: int = 20260904          # ours -- ONE seed for the shared set
    photon_seed: int = 1                    # ours -- vary for the pairing null

    # --- geometries --------------------------------------------------------
    datasets: tuple[str, ...] = (
        "flower_s", "flower_l", "flower_xl", "triangle", "cluster", "hexagon",
    )                                        # ours: all six
    include_ice_control: bool = True        # ours: hexagon re-run in ice, the
                                            # medium-only negative control

    provenance: dict = field(default_factory=lambda: {
        "final_state_1": "nubench", "final_state_2": "nubench",
        "interactions": "nubench",
        "energy_gev": "ours", "power_law": "ask",
        "min_zenith_deg": "ask", "max_zenith_deg": "ask",
        "min_azimuth_deg": "ask", "max_azimuth_deg": "ask",
        "injection_points": "ours", "events_per_point": "ours",
        "is_ranged": "ours",
        "endcap_length_m": "ask", "earth_model": "ask",
        "use_gpu": "ours", "olympus_max_distance_m": "ours",
        "olympus_photon_chunk": "ours",
        "injection_seed": "ours", "photon_seed": "ours",
        "datasets": "ours", "include_ice_control": "ours",
    })

    @property
    def n_events(self) -> int:
        return int(self.events_per_point * len(self.injection_points))

    # ---------------------------------------------------------------- io ---
    @classmethod
    def from_yaml(cls, path: str | Path) -> "PhysicsParameters":
        d = yaml.safe_load(Path(path).read_text()) or {}
        prov = d.pop("provenance", None)
        obj = cls(**{k: (tuple(v) if isinstance(v, list) and k != "injection_points" else v)
                     for k, v in d.items()})
        obj._coerce()
        if prov:
            obj.provenance.update(prov)
        return obj

    def _coerce(self) -> None:
        """PyYAML does not parse `1.0e4` as a float (it wants `1.0e+4`), so a
        plausible-looking config can silently deliver a string into the
        simulation. Coerce the numeric fields rather than trusting the file."""
        for name in ("energy_gev", "power_law", "min_zenith_deg", "max_zenith_deg",
                     "min_azimuth_deg", "max_azimuth_deg", "endcap_length_m",
                     "olympus_max_distance_m"):
            v = getattr(self, name)
            if v is not None:
                setattr(self, name, float(v))
        self.events_per_point = int(self.events_per_point)
        self.injection_seed = int(self.injection_seed)
        self.photon_seed = int(self.photon_seed)
        self.injection_points = {
            k: [float(c) for c in v] for k, v in self.injection_points.items()}

    def to_dict(self) -> dict:
        d = asdict(self)
        d["n_events"] = self.n_events
        for k, v in d.items():
            if isinstance(v, tuple):
                d[k] = list(v)
        return d

    def unresolved(self) -> list[str]:
        """Parameters tagged `ask`: our values are guesses, and every one is a
        declared deviation from NuBench until the authors supply the real one."""
        return sorted(k for k, v in self.provenance.items() if v == "ask")

    def fingerprint(self) -> str:
        """Stable hash of the physics configuration. Two event sets with the
        same fingerprint were generated under the same physics."""
        payload = {k: v for k, v in self.to_dict().items() if k != "provenance"}
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode()
        ).hexdigest()[:16]

    def record(self, out_dir: str | Path, extra: dict | None = None) -> Path:
        """Write the full parameter record beside the simulation output."""
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        rec = {
            "physics": self.to_dict(),
            "provenance": self.provenance,
            "unresolved_parameters": self.unresolved(),
            "fingerprint": self.fingerprint(),
            "environment": _environment(),
            "written_utc": datetime.now(timezone.utc).isoformat(),
        }
        if extra:
            rec.update(extra)
        path = out_dir / "physics_record.json"
        path.write_text(json.dumps(rec, indent=2, default=str))
        return path


def _environment() -> dict:
    def _git(cwd: Path) -> str | None:
        try:
            return subprocess.run(
                ["git", "rev-parse", "HEAD"], cwd=cwd, capture_output=True,
                text=True, timeout=10, check=True).stdout.strip()
        except Exception:
            return None

    here = Path(__file__).resolve().parent
    prom = here / "external" / "prometheus"
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "repo_commit": _git(here),
        "prometheus_commit": _git(prom) if prom.exists() else None,
        "prometheus_upstream": "https://github.com/Harvard-Neutrino/prometheus",
        "prometheus_licence": "LGPL-2.1",
    }


if __name__ == "__main__":
    p = PhysicsParameters()
    print(json.dumps(p.to_dict(), indent=2))
    print("total events:", p.n_events,
          f"({p.events_per_point} x {len(p.injection_points)} points"
          f" at {p.energy_gev:.0f} GeV)")
    print("fingerprint:", p.fingerprint())
    print("unresolved (ask the NuBench authors):", p.unresolved())
