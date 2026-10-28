# TrackingQuick

A CPU pipeline for the SPAICE single-drone tracking challenge, combining
**DINOv2 appearance verification, Lucas-Kanade point tracking and Kalman filtering**.
It produces image-space trajectories, track status and timing measurements, with
optional depth estimates based on a physical-size prior.

Version 0.2 supports fixed-camera video with a user-specified initial target box
and controlled synthetic evaluation. The [system outline](OUTLINE.md) describes
the proposed extension to calibrated 3D tracking.

## Setup

Python 3.12+ and [uv](https://docs.astral.sh/uv/):

```powershell
uv sync --locked --extra research
uv run python scripts/fetch_dino.py
uv run pytest -q
```

DINOv2-small uses a 24.4 MB INT8 ONNX export with CPU inference. The downloader
verifies the pinned SHA-256 and reuses existing verified weights. Dependency
versions are recorded in `uv.lock`.

## Processing videos

The first command exports a frame for target selection. The second initializes
tracking from a bounding box specified as `X Y WIDTH HEIGHT` in original pixels:

```powershell
uv run trackingquick-video data/drone.mp4 --output runs/drone
# Replace the example coordinates with the target's first-frame bounding box.
uv run trackingquick-video data/drone.mp4 --output runs/drone --roi 84 125 32 20 --max-frames 300
```

Foreground proposals provide candidate boxes. Forward/backward point flow
tracks local motion between detections, while a constant-velocity Kalman filter
uses source timestamps and spatial gating to estimate the target center.
DINO verifies candidate crops asynchronously against target and background
references. Scores measure each crop's relative cosine similarity to the two
references.

| Output | Contents |
| --- | --- |
| `first_frame.jpg`, `inspection.json` | Target-selection frame and video dimensions/fps |
| `track.csv` | Source timestamps, track status, center, optional depth, verification age and timing |
| `manifest.json` | Configuration, initial box, setup time, timestamp fallback and verifier statistics |
| `annotated.mp4` | Tracking overlay, enabled with `--write-video` |

Each source frame produces one CSV row. Confirmed tracks emit coordinates;
other rows retain blank position fields. Status distinguishes detections, flow
updates, predictions, tentative tracks and loss. `--mode flow` selects the
point-flow/Kalman baseline. The default `--negative-limit 1` retires a track after
one negative DINO verdict; `--negative-limit 2` retains it through one negative
verdict with `suspect` status. Positive verification has a finite lifetime.

Optional depth requires both `--focal-px` and `--width-m`. The range filter uses
`Z = focal_px * assumed_width_m / bounding_box_width_px` and preserves uncertainty
in the physical-size prior. Camera-relative X/Y use an image-center principal
point. Depth estimates are conditional on these camera and size assumptions.

## Evaluating tracking

```powershell
uv run trackingquick-evaluate
```

The benchmark compares three policies on four 180-frame synthetic clips at a
paced 30 Hz: two lighting/blur/occlusion sequences, a bird-only sequence, and a
drone with a larger bird distractor. DINO reference fitting uses 16 training and
12 validation crops with separate seeds from the same renderer. The results
characterize performance within this synthetic setting.

| Policy | Mixed 42 success | Mixed 73 success | Distractor success | Bird-only false alarms |
| --- | ---: | ---: | ---: | ---: |
| Flow + Kalman | 98.8% | 100.0% | 0.0% | 100.0% |
| + DINO, retire after one negative | 64.8% | 55.8% | 75.6% | 8.3% |
| + DINO, retire after two negatives | 83.6% | 84.8% | 95.0% | 28.3% |

Success is the fraction of all visible-target frames with an output within
8 pixels of the true center; missing outputs count as failures. False alarms
are the fraction of target-absent frames emitting a position. DINO improves
distractor rejection, while the retirement policy controls the observed
tradeoff between continuity and false alarms.

On the Windows CPU used for evaluation, per-run p95 DINO-policy processing
was 13.3–23.8 ms, p95 output age relative to the replay schedule was
13.6–247.8 ms, and p95 background DINO crop inference was 98.5–126.6 ms.
Output age includes replay backlog. Measurements exclude setup and physical
capture/display; asynchronous scheduling and system load affect both timing
and track decisions. A paced 30 Hz input therefore remains distinct from a
guaranteed 30 Hz output deadline.

[Metrics](reports/v0_2/metrics.json), [per-frame CSVs](reports/v0_2/),
[comparison figure](reports/v0_2/success.png) and
[evaluation methodology](docs/REVIEW_V0_2.md) accompany the benchmark.

## Possible avenues to build on this

| Area | Extension |
| --- | --- |
| Detection and validation | Evaluate YOLO and YOLO + DINO on drone, bird and empty clips, with training, validation and test splits separated by flight. |
| Appearance verification | Calibrate similarity thresholds and identity lifetime on held-out footage; assess continuity, wrong tracks and false alarms jointly. |
| Latency | Introduce latest-frame capture and bounded queues; measure capture-to-output age on the intended hardware. |
| Point tracking | Compare TAPNext with Lucas-Kanade flow using matched initialization, visibility handling, drift and latency measurements. |
| 3D estimation | Extend the geometry routines to synchronized stereo, propagate correspondence uncertainty and use bearing-only updates at weak disparity. |
| Motion modelling | Develop the conditional flow-matching experiment into a calibrated motion prior for occlusion handling. |

## Layout and large files

| Path | Purpose |
| --- | --- |
| `src/trackingquick/` | Filters, vision, reusable pipeline, video runner and experiments |
| `scripts/`, `tests/` | Model download, TAPNext trial preparation, report tools and regression tests |
| `reports/v0_2/` | Current benchmark results; [v0.1 PDF](reports/TrackingQuick_research.pdf) and [Markdown report](reports/RESEARCH.md) document the initial experiments |
| `docs/`, [OUTLINE.md](OUTLINE.md) | Evaluation methodology, research sources and system design |
| `data/`, `weights/`, `runs/` | Local footage, weights and generated outputs; [data provenance](data/README.md) |

The scale, triangulation and conditional flow-matching experiments are available
with `uv run trackingquick`. Large videos, weights and local run outputs are
excluded from version control.
