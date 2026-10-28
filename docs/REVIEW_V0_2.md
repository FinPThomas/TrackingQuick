# v0.2 review — 28 October 2026

The project now has a reusable, timestamped tracking pipeline and a video-file
entry point. The strongest result is a clearer account of when it fails:
the old low processing-time and conditional-error figures did not establish
reliable, low-latency drone tracking.

## What changed

- Separated frame ingestion and tracking from ground-truth evaluation. The new
  pipeline receives images, source timestamps and frame indices, never labels.
  Kalman prediction uses elapsed source time rather than an assumed frame step.
- Added first-frame inspection and explicit user ROI initialization for video
  files. Input dimensions/fps are respected; missing or repeated decoder
  timestamps fall back to fps and are counted in the manifest. This path uses
  MOG2 foreground proposals and requires a fixed camera.
- Removed a duplicate verifier implementation. The asynchronous queue holds
  at most one job; stale results and results for a previous track generation
  are discarded. Positive evidence expires; negative evidence cannot renew it.
- Made one- versus two-negative retirement selectable. Rejected locations are
  briefly excluded during reacquisition so a large distractor cannot continually
  monopolize proposals. The video CLI defaults to strict rejection after one
  negative because the tolerant policy produced more false alarms.
- Disabled distance output by default for supplied videos. Enabling it needs
  both focal length and physical width; uncertainty in assumed size persists
  across repeated observations. The image-center principal-point assumption
  remains a limitation, alongside unmodelled lens distortion and orientation.
- Added target-absent and large-distractor cases, all-visible-frame success,
  wrong-track fractions, common-frame error comparisons, processing stage times,
  replay lateness, output age and deadline misses. Kept v0.1 reports separately.
- Added a size/checksum-pinned TAPNext preflight and bounded optional clip trial.
  Preflight was run; model inference was not possible with the available disk.

## Evaluation protocol

`trackingquick-evaluate` uses the same synthetic proposals, frame sequences,
point flow, Kalman settings and DINO references for all three policies. The
only differences are whether DINO is used and the negative-verdict limit.
It is a new comparison, not a numerically interchangeable rerun of v0.1.

Each clip contains 180 frames at 640×360 and 30 Hz. Mixed clips use seeds 42
and 73 and contain lighting change, directional blur and 15 frames of
occlusion. The bird-only clip has no drone at any point. The distractor clip
contains both a drone and a larger bird. Negative-case birds are large enough
to pass the proposal generator; the regression suite checks this explicitly.

DINOv2-small is frozen, INT8 ONNX, CPU, with 112×112 crops. Prototypes use
16 reference crops and a threshold chosen on 12 separate validation crops.
All these images share the synthetic renderer. No YOLO was trained and no
real-image augmentation or real-drone evaluation has been completed. Existing
augmentation stays small: brightness, orientation/size and directional blur.

Metrics in [metrics.json](../reports/v0_2/metrics.json):

- `success_8px_all_visible`: correct center within 8 px divided by every visible
  target frame. Missing outputs fail. `coverage` counts any reported center,
  including an incorrect one, so it is not accuracy.
- `wrong_track_fraction_15px`: output more than 15 px from truth divided by
  every visible target frame. This is a displacement proxy, not identity labels.
- `empty_false_alarm_fraction`: reported centers divided by target-absent
  frames. It is a frame fraction, not false-alarm events per hour.
- `rmse_reported_*`: error only where an output exists. The `paired` entries
  compare the two DINO policies on their common visible outputs and state the
  number of frames. Neither replaces the all-frame success measure.
- `processing_ms`: synchronous tracking work after a frame is available.
  Stage means separate proposals, point flow/reseeding, filters and other work.
  `dino_crop_ms` measures background crop inference separately.
- `scheduled_output_age_ms`: delay from the frame's intended replay time to
  its output, including replay backlog. `deadline_miss_fraction` counts age
  above 33.33 ms. File replay preserves frames; it does not implement a live
  camera queue or frame dropping. Setup/model loading and physical capture,
  exposure, transport, current-frame encoding and display are outside this age.
- `first_confirmation_wall_s`: time from replay start to the first emitted
  center. On an empty clip, this is the first false confirmation.

There are only four short clips and one recorded run per policy. Asynchronous
completion, OS scheduling and the shared CPU change which frame consumes a
verdict, so even accuracy can vary between runs. No confidence intervals,
deployment accuracy, end-to-end camera latency or hard real-time claim follows.

## Findings and limits

The flow/Kalman baseline follows the easy mixed sequences well but consistently
selects the larger bird in the distractor sequence. DINO improves that case.
Strict rejection also loses true tracks during appearance changes. Tolerating
one negative raised all-frame success from 64.8% to 83.6% and 55.8% to 84.8%
on the two mixed clips, but bird-only false outputs rose from 8.3% to 28.3%.
That tradeoff needs real validation, not a universal agreement percentage.

Per-run p95 DINO-policy processing was 13.3–23.8 ms, while scheduled output age
was 13.6–247.8 ms. The worst DINO run missed 35.6% of 30 Hz replay deadlines.
The background verifier is bounded, but CPU contention and frame production
can still delay the whole loop. Next steps are latest-frame capture, controlled
thread budgets and a capture-to-output measurement on the intended device.

Depth remains conditional on a size prior; bounding-box width changes with
orientation, occlusion and segmentation, and these biases are not fully
represented by the filter. There is an analytic triangulation/covariance
routine, but no stereo image association, synchronized rig or gimbal motion
compensation. DINO features do not make monocular scale observable.

The v0.1 flow-matching toy's nominal 95% marginal intervals covered only 88.3%
of held-out synthetic residual components. It remains outside the measurement
update; treating its sampled futures as independent observations would
double-count assumptions and overstate confidence.

TAPNext's [Annolid ONNX export](https://annolid.com/portal/tapnext/) was checked
against its [release metadata](https://github.com/healthonrails/annolid/releases/tag/v1.6.6).
The artifact is 984,253,311 bytes; the measured free space was 114,409,472 bytes.
[Preflight](../reports/v0_2/tapnext_preflight.json) therefore stopped before any
download. This establishes a storage blocker, not a GPU requirement. The
optional trial supports that export's short-clip interface, not the official
streaming Torch state interface; its unexecuted inference branch is not a
working TAPNext integration claim. More disk space is the next requirement.

## Validation and next order of work

The regression suite has 12 passing tests covering filter/geometry behavior,
uncertainty, track identity and stale results, missing-output metrics, source
timing, negative-case proposals and a decoded-video integration path. The DINO
video path also processed 90 frames of the existing synthetic encoded demo,
with 84 reported centers and every distance field blank. That smoke test used
the two-negative policy; it is integration evidence, not real-drone accuracy.

1. Add a small licensed, fixed-camera real dataset with drone, bird and empty
   clips; record source/calibration and split by flight before tuning.
2. Compare a drone-trained YOLO baseline with DINO verification. Calibrate on
   validation flights and retain held-out test flights for the final comparison.
3. Measure a latest-frame capture pipeline; compare TAPNext after storage is
   available, with identical points, visibility handling and time accounting.
4. Test calibrated stereo correspondence and depth uncertainty. Use bearing-only
   updates when depth is weak; revisit flow-matching priors only after calibration.
