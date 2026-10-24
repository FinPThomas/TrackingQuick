# SPAICE Technical Challenge: 3D Tracking of a Single Drone

**Project:** TrackingQuick
**Stage:** outline only; architecture and thresholds are proposals, not measured results.

## Mission and deliverables

Design a real-time, visual-only perception system for a fixed or gimbaled ground camera. It must detect one flying drone and estimate its camera-relative 3D position `(x, y, z)` as the drone changes altitude, range, orientation, background, and lighting.

The requested submission is a short **PDF report** plus a **Python proof of concept v0.1** delivered through GitHub or PDF plus code ZIP. The report has no page limit. The proof of concept only needs **2D detection and tracking** on a public dataset or synthetic data. The goal is clear engineering reasoning about sensing, autonomy, latency, simplicity, and failure modes; it is not a request to train a perfect model.

## 1. System design overview — proposed report section

- **Primary visual setup:** synchronized, calibrated stereo RGB cameras with global shutters, a rigid known baseline, and lenses chosen for the required field of view and target pixel size. Mount both cameras on the same fixed head or gimbal. Record accurate frame timestamps and, for a gimbal, orientation/encoder telemetry. The 3D output remains in a defined camera frame; gimbal motion must be compensated before temporal filtering or world-frame reporting.
- **Why stereo:** two simultaneous views provide metric depth by triangulation. A single moving drone viewed by a fixed monocular camera has scale/depth ambiguity without a known target size or other prior. An optional wider view can help reacquire a fast target outside the narrow tracking field of view, at extra hardware and compute cost.
- **Range/accuracy tradeoff:** focal length and resolution determine the drone's pixel footprint; stereo baseline and pixel localization accuracy determine disparity and depth uncertainty, which increases sharply at long range. Specify an operating range only after testing actual optics, baseline, and target sizes. At poor disparity, report high uncertainty or bearing-only tracking rather than a confident metric range.
- **Calibration:** estimate each camera's intrinsics and distortion, stereo rotation/translation and baseline, rectification maps, and timestamp offset. Check calibration again after mounting or gimbal changes. Transform triangulated points to the stated camera coordinate convention; keep calibration version and uncertainty with results.
- **Operational constraints:** measure capture-to-output latency, detection recall at small angular sizes, tracking continuity, and compute/power cost on the target device. Prefer lightweight inference and deadline-aware scheduling. The PDF should show a latency budget and measured percentiles once hardware is chosen.

## 2. Algorithmic approach — proposed report section

### 2D detection and model agreement

1. Run a lightweight YOLO detector on full frames or a predicted region of interest (ROI). Periodically scan the full frame so a filter prediction cannot permanently hide a newly visible or displaced drone.
2. Use DINOv2 features on selected candidate crops with a small drone/background classifier or reference-embedding score. DINOv2 provides crop features; it does not itself produce a drone bounding box. Calibrate scores on held-out sequences.
3. At track initiation, recovery, and ambiguous detections, require both models to support the drone hypothesis. Define agreement as a calibrated YOLO confidence, a DINOv2 drone score, and a spatial consistency test between the candidate box and the filter's predicted ROI (for example, normalized center error or IoU). The allowed percentage/threshold is **to be selected on validation data**, not guessed now.
4. During a stable, high-confidence track, skip or reduce DINOv2 calls to protect latency; the Kalman prediction and periodic verification provide continuity. Compare this schedule with always-on two-model agreement. Record cases where the verifier rejects true tiny/blurred drones.

### 3D position from stereo

- Associate left/right detections using rectified epipolar geometry, appearance, size, and the predicted 3D track. Estimate disparity from matched centers or local features, then triangulate with calibrated projection matrices.
- Reject invalid/low disparity and inconsistent pairs. Carry measurement covariance from pixel and calibration uncertainty into the tracker; range uncertainty is not constant.
- If one view is lost, retain the image bearing and predict the 3D state with growing covariance. A 2D bearing alone does not restore metric depth. Make a full 3D correction when reliable stereo returns.

### Kalman filtering: main technical emphasis

- **3D state:** `s = [X, Y, Z, Vx, Vy, Vz]^T` in one explicit camera or stabilized reference frame. Start with a constant-velocity transition `F(dt)` using actual frame timestamps, with process noise `Q(dt)` for drone manoeuvres. Consider acceleration or interacting motion models only if the baseline misses turns.
- **Measurement:** stereo yields `m = [X, Y, Z]^T` with range-dependent covariance `R`. A projection-based extended Kalman filter (EKF) could instead update directly with paired image coordinates and the calibrated camera model. Compare numerical stability and error before choosing.
- **Update discipline:** predict every frame; associate observations with an uncertainty-aware Mahalanobis gate; correct only from valid measurements. Increase covariance through missed detections and occlusion. Use innovation residuals to detect misassociation or abrupt manoeuvres; tune `Q`, `R`, and gates on validation sequences.
- **Track lifecycle:** tentative initiation needs repeated agreement; one confirmed track is maintained, predicted through short gaps, and retired/reacquired after a documented timeout. Report position, velocity, covariance, and a status flag (`observed`, `predicted`, or `lost`) so predictions are never presented as measured points.
- **2D v0.1 analogue:** a constant-velocity Kalman filter over image center `(u, v)` and velocity `(du/dt, dv/dt)`, with box size or scale tracked separately if useful. Use timestamp-aware prediction, detection-to-track gating, missed-frame prediction, reacquisition, and explicit confidence/status. This gives a runnable bridge to the 3D design.

### Version 0.1 proof of concept

- Python/OpenCV pipeline for one video or image sequence: load frames and optional 2D labels, detect the drone, track its 2D center/box over time, and export an annotated video plus CSV/JSON track with timestamps and `observed/predicted/lost` state.
- Start with a compact YOLO baseline and a Kalman filter. Add DINOv2 crop verification only if a measured comparison shows a useful precision/latency tradeoff. No metric 3D claim is required for v0.1.
- Keep dataset conversion, inference, tracking, and evaluation separate so the detector and verifier can be swapped without changing track logic.

## 3. Failure modes and robustness — proposed report section

| Challenge | Planned response and remaining limit |
| --- | --- |
| Partial/full occlusion | Predict through short gaps with growing covariance; use reacquisition gates and a timeout. Long occlusion can break identity. |
| Small/far drone | Choose optics for target pixels; use a high-resolution ROI around prediction and scheduled full-frame scans. Depth may become unobservable. |
| Low light / glare / changing illumination | Exposure control where available, balanced lighting augmentations, confidence calibration, and optional thermal camera in a later multisensor design. |
| Motion blur / fast manoeuvres | Prefer short exposure and sufficient frame rate at capture; train with realistic directional blur and varying shutter lengths; increase process noise on large innovations. Evaluate any deblurring or frame integration against its latency and artifacts before adding it. |
| Clouds, birds, clutter, moving background | YOLO proposals plus selective DINOv2 verification, temporal and epipolar consistency, negative examples, and false-alarm measurement. |
| Stereo mismatch / calibration drift | Epipolar and reprojection checks, calibration checks, uncertainty-aware rejection, and bearing-only fallback. |
| Gimbal motion / asynchronous views | Timestamp and pose compensation; reject stereo pairs with uncorrected motion or large time offset. |

## 4. Further improvements with more sensors — proposed report section

- **Radar** for coarse range/range-rate and all-weather cueing, **thermal** for visibility in darkness or against difficult RGB backgrounds, and optionally **acoustic** cues for sector-level detection. Each adds cost, calibration, synchronization, and false-alarm modes.
- Transform all observations into a common frame and time base. Use a centralized EKF/UKF with sensor-specific observation functions and covariances, or track-to-track fusion when raw measurements cannot be shared. Gate associations and avoid counting correlated camera observations twice.
- Measure whether added sensors improve 3D error, detection range, continuity, and false alarms enough to justify extra latency and hardware.

## 5. Data, augmentation, and evaluation plan

- Candidate source: [Cenek Albl et al., multiview drone tracking datasets](https://github.com/CenekAlbl/drone-tracking-datasets). Its README lists 2D labels for datasets 1–4, synchronization and camera locations for dataset 3, and a harder dataset 4 with moving clouds. Dataset 3 is a candidate for later 3D checks; datasets 1–4 can support 2D v0.1 after checking actual files and annotation format. Dataset 5 has multiple drones and no listed 2D labels, so it is outside the initial single-drone proof of concept.
- The repository's [CC BY-NC-SA 4.0 license](https://github.com/CenekAlbl/drone-tracking-datasets/blob/master/LICENSE) includes noncommercial and attribution/share-alike conditions. Confirm that the intended challenge submission and any derived images/weights comply before downloading, augmenting, or redistributing data. Keep large source videos and derived frames out of Git.
- Split by flight/sequence and preferably camera/session before augmentation to avoid adjacent-frame leakage. Document provenance and conditions for every split.
- Train-only augmentation candidates: synthetic partial occluders, varied exposure/contrast/gamma, backlight and haze, sensor noise, realistic directional motion blur at several lengths, and changes in apparent drone size. Transform labels consistently; keep unaugmented real holdouts for evaluation. Include difficult negatives such as birds/cloud fragments if permitted by source licenses.
- Compare detector-only, detector plus selective verifier, and detector plus always-on verifier. Report 2D center error, detection precision/recall and false alarms, tracking continuity through occlusion, 3D position error when later implemented, and end-to-end latency (median, p95, p99) on a named device. Break down capture, preprocessing, detector, verifier, triangulation, and filter time.

## Build sequence and decisions to make later

1. Confirm available hardware, approximate range/field of view, target latency/FPS, and whether the proof of concept should run on CPU or GPU.
2. Check candidate dataset files, license fit, and train/validation/test split; choose v0.1 sequence(s).
3. Implement a basic 2D detector + Kalman filter and measure it on held-out sequences.
4. Test selective DINOv2 verification, tune agreement thresholds, and keep it only if the measured operating point improves the result.
5. Add calibrated stereo, triangulation, and 3D filtering when cameras/calibration or suitable multiview data are available.
6. Write the final PDF with diagrams, measured tradeoffs, failure cases, limitations, and reproducible run instructions. Package code and artifacts for GitHub or ZIP delivery.

**Current status:** these steps are planned only. No implementation, dataset download, model training, PDF, or performance result exists yet.
