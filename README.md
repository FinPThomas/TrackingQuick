# TrackingQuick

A small CPU research project for the SPAICE single-drone tracking challenge.
It combines **DINOv2-small appearance verification, point flow and a Kalman filter**,
with an optional distance estimate conditional on an assumed drone size.

**v0.2 is a synthetic research prototype with a user-seeded video runner.**
Real-drone accuracy, YOLO detection, TAPNext inference and full stereo tracking
remain unvalidated. The [review](docs/REVIEW_V0_2.md) separates changes from open issues.

## Setup

Python 3.12+ and [uv](https://docs.astral.sh/uv/):

```powershell
uv sync --locked --extra research
uv run python scripts/fetch_dino.py
uv run pytest -q
```

The downloader checks a pinned SHA-256 for the 24.4 MB INT8 DINOv2-small ONNX
export. Inference uses CPU; dependency versions are in `uv.lock`. A fresh model
download requires 700 MB free. Existing verified weights are reused.

## Processing videos

Start with a short **fixed-camera** video. The first command saves a frame for
inspection; rerun with the target box in original-image pixels:

```powershell
uv run trackingquick-video data/drone.mp4 --output runs/drone
# Replace the example X Y WIDTH HEIGHT below with the drone's first-frame box.
uv run trackingquick-video data/drone.mp4 --output runs/drone --roi 84 125 32 20 --max-frames 300
```

The runner uses foreground proposals, forward/backward Lucas-Kanade point flow,
and a constant-velocity Kalman filter with source timestamps and spatial gating.
DINO checks candidate crops asynchronously, using the selected target and
background crops as appearance references. These scores are **not calibrated
drone probabilities**, and this is not an automatic YOLO detector.

| Output | Contents |
| --- | --- |
| `first_frame.jpg`, `inspection.json` | First-run ROI inspection and video dimensions/fps |
| `track.csv` | One row per frame: source time, status, center, optional depth, verification age and timing |
| `manifest.json` | Configuration, ROI, setup time, timestamp fallback and verifier statistics |
| `annotated.mp4` | Optional overlay, enabled with `--write-video` |

Only confirmed tracks emit coordinates; missing outputs stay blank. Status
distinguishes detections, flow updates, predictions, tentative tracks and loss.
`--mode flow` disables DINO. The default `--negative-limit 1` retires a track on
one failed check; `--negative-limit 2` tolerates one failure and labels it
`suspect`, but increases false-alarm risk. Negative checks never renew identity.

Distance stays blank unless **both** `--focal-px` and `--width-m` are supplied.
It uses `Z = focal_px * assumed_width_m / bounding_box_width_px`, with persistent size
uncertainty. X/Y assume the principal point is the image center. This requires
an appropriate camera model and size prior; DINO does not recover metric scale.

## Evaluating

```powershell
uv run trackingquick-evaluate
```

Runs three policies on four 180-frame synthetic clips at a paced 30 Hz: two
mixed lighting/blur/occlusion cases, a bird-only negative case, and a drone with
a larger bird distractor. Reference fitting uses 16 training and 12 validation
crops with separate seeds, but the same renderer. No real footage was evaluated.

| Policy | Mixed 42 success | Mixed 73 success | Distractor success | Bird-only false alarms |
| --- | ---: | ---: | ---: | ---: |
| Flow + Kalman | 98.8% | 100.0% | 0.0% | 100.0% |
| + DINO, reject after one negative | 64.8% | 55.8% | 75.6% | 8.3% |
| + DINO, reject after two negatives | 83.6% | 84.8% | 95.0% | 28.3% |

Success means an output within 8 pixels on **all visible-target frames**;
missing outputs count as failures. False alarms mean the fraction of
target-absent frames emitting a position. DINO helps reject the larger bird,
but continuity and false alarms trade off. The tolerant policy is optional.

On this Windows CPU run, DINO policies had per-run p95 processing times of
13.3–23.8 ms, but p95 output age relative to the replay schedule was
13.6–247.8 ms. DINO crop inference itself took 98.5–126.6 ms at p95. These
are separate measurements: asynchronous inference does not guarantee a 30 Hz
deadline. Timing varies with load and excludes initial setup and physical
capture/display. Inputs are repeatable; asynchronous decisions can vary.

[Raw metrics](reports/v0_2/metrics.json), [per-frame CSVs](reports/v0_2/),
[comparison figure](reports/v0_2/success.png) and
[measurement details](docs/REVIEW_V0_2.md) are checked in.

## Current issues and next work

| Issue | Next useful experiment |
| --- | --- |
| Synthetic evidence; no trained YOLO | Label a small fixed-camera drone/bird/empty set, split by flight, then compare YOLO alone with YOLO + DINO. |
| Fragile appearance agreement | Calibrate scores and identity lifetime on held-out clips; report missed frames, wrong tracks and false alarms together. |
| Replay can fall behind | Add latest-frame capture with bounded queues, then measure capture-to-output age on the target hardware. |
| Size and orientation bias depth | Test calibrated stereo correspondences and covariance; retain bearing-only output when disparity is weak. |
| TAPNext has not run | Provide storage, then execute the bounded trial and compare visibility, drift and latency against point flow. |
| Flow matching remains a toy | Calibrate its uncertainty before using it as a motion prior; do not fuse sampled futures as camera measurements. |

TAPNext's [published ONNX artifact](https://github.com/healthonrails/annolid/releases/tag/v1.6.6)
is 984,253,311 bytes. The [recorded preflight](reports/v0_2/tapnext_preflight.json)
found only 114,409,472 bytes free, so no download or inference was attempted.
CPU/RAM requirements and latency remain unmeasured. Run
`uv run python scripts/tapnext_trial.py` to recheck storage; its inference branch
is prepared but unvalidated. Another drive or roughly 2 GB free is needed first.

## Layout and large files

| Path | Purpose |
| --- | --- |
| `src/trackingquick/` | Filters, vision, reusable pipeline, video runner and experiments |
| `scripts/`, `tests/` | Model downloads, TAPNext preflight, report tools and regression tests |
| `reports/v0_2/` | Current measurements; [v0.1 PDF](reports/TrackingQuick_research.pdf) and [Markdown](reports/RESEARCH.md) are historical snapshots |
| `docs/`, [OUTLINE.md](OUTLINE.md) | Review, research sources and original challenge/design |
| `data/`, `weights/`, `runs/` | Ignored footage, model weights and local outputs; [data provenance](data/README.md) |

The original scale/triangulation/flow-matching ablations remain available with
`uv run trackingquick`. Large videos and weights are not committed.
