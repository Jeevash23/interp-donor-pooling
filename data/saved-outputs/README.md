# Saved outputs

The studies' results, packed as deterministic tar archives compressed with xz. Member paths are relative to the release root, so `make unpack` (or `python tools/unpack.py`) puts every file where the study code reads it and checks each SHA-256 against `data/PACKED-FILES.json`. Text records inside the archives had personal strings replaced, as listed in `RELEASE-CHANGES.md`.

| Archive | Files | Packed (MB) | Unpacked (MB) | Contents |
|---|---|---|---|---|
| `p1-pilot.tar.xz` | 8 | 0.6 | 0.6 | P1: per-model analyses and audits (raw per-group arrays not included) |
| `p2-six-model-audit.tar.xz` | 27 | 0.2 | 2.2 | P2: per-model analyses, head, BLiMP, precision and scale summaries (component arrays and head-pair intervals not included) |
| `study1-grouping-and-circuit.tar.xz` | 647 | 3.7 | 5.9 | Study 1: analyses, design and audit records, circuit results, environment and checkpoint digests (raw token arrays not included) |
| `study2-reference-and-sweeps.tar.xz` | 2900 | 24.9 | 53.2 | Study 2: reference experiment, sweeps and post hoc results with every raw array |
| `study3-head-confirmation.tar.xz` | 260 | 9.0 | 10.0 | Study 3: calibration, development, test and replay arrays, analyses, gate |
| `study4-component-confirmation.tar.xz` | 84 | 8.7 | 9.2 | Study 4: calibration, development, test and replay arrays, analyses |
| `study5-larger-model.tar.xz` | 394 | 0.4 | 2.7 | Study 5: every per-row record with its array digest, analyses, gates, calibration, cost decision, verification (raw arrays stayed on the cloud volume) |

Total packed size: 47.4 MB. Rendered figures are not included; `make figures` draws them from these outputs.
