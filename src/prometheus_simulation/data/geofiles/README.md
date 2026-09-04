# Detector geometry files — provenance

These six `.geo` files are copied **verbatim** from the Prometheus neutrino
telescope simulation, `resources/geofiles/`, at commit `8c199384062012009094862bc244fa55f7694ee0`
(https://github.com/Harvard-Neutrino/prometheus). Prometheus is released under
the **GNU LGPL v2.1**; these files are redistributed here under that licence,
unmodified, with this notice. They are NOT covered by this package's MIT
licence.

Format: a `### Metadata ###` header (medium, DOM radius) followed by
`### Modules ###` and one row per optical module:
`x_m  y_m  z_m  string_id  om_id` (tab-separated).

| File | NuBench name | Inspired by | Modules |
|---|---|---|---|
| `orca.geo` | Flower S | KM3NeT/ORCA | 3300 |
| `arca.geo` | Flower L | KM3NeT/ARCA | 2070 |
| `trident.geo` | Flower XL | TRIDENT | 24220 |
| `pone_triangle.geo` | Triangle | P-ONE | 60 |
| `gvd.geo` | Cluster | Baikal-GVD | 288 |
| `icecube.geo` | Hexagon | IceCube (+DeepCore) | 5160 |

Module counts match NuBench Table 1 (arXiv:2511.13111) exactly, which is how
the Flower/Triangle/Cluster/Hexagon names were mapped to files (see
`docs/archive/OPEN_DECISIONS.md` D1). The depth (z) conventions differ between
files; `DetectorGeometry.centered()` places a detector at its own centroid,
which is the "common site" convention of D1 caveat (b).
