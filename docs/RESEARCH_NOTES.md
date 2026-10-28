# Research decisions and limits

These are the original v0.1 design notes. See the [v0.2 review](REVIEW_V0_2.md)
for the current pipeline, measurements and verified TAPNext storage blocker.

This quick project prioritizes a measured CPU prototype and correct handling of uncertainty. The proposed complete system remains a calibrated stereo rig, a drone-specific detector, point tracking between detections, and uncertainty-aware filtering. The runnable experiment implements the permitted 2D proof of concept plus a conditional scale experiment.

## DINOv2 and detection

We use DINOv2-small as a frozen crop feature encoder, through a checksum-pinned community INT8 ONNX export of the Meta backbone. Positive and negative reference embeddings are averaged and normalized. The score is the difference between cosine similarity to the two prototypes. A threshold is selected on independent synthetic validation crops. It is a similarity margin, not a calibrated probability or a percentage agreement between independent detectors. [DINOv2 source](https://github.com/facebookresearch/dinov2); [ONNX export source](https://huggingface.co/onnx-community/dinov2-small).

Cheap proposals are evaluated every five source frames and DINO verification is requested approximately every fifteen. The asynchronous variant permits one pending crop; requests are skipped while busy. Each result carries its source frame and track generation. Results older than 23 frames or belonging to a replaced track are discarded. Tracks remain tentative until verified; confirmation expires after 45 frames without a sufficiently recent verdict. Background verification lowers the time spent updating a track, but does not shorten the model's inference time or remove delayed identity decisions.

A suitable next detector is a small YOLO trained for the drone class and difficult negatives. The current heuristic proposal generator makes no claim to work on arbitrary skies, night scenes, or moving cameras. A generic COCO checkpoint is not evidence of drone detection capability. Test YOLO-only, YOLO plus DINO, and YOLO plus point tracking using the same flight-level split and original-image coordinates. [Official detection documentation](https://docs.ultralytics.com/tasks/detect/).

## Size, depth and Kalman filtering

For an object with projected physical width W, calibrated focal length f in pixels, and observed width p in pixels, the pinhole approximation is Z = fW/p. Orientation, propeller motion, truncation, blur and unknown drone dimensions all break an exact width-to-range interpretation. We use it only as an uncertain prior. DINO appearance similarity does not establish physical size or camera intrinsics. A future reference bank could associate known drone models with dimension distributions, but confident recognition and trustworthy dimensions would both be required.

The range filter uses state [q, q_rate, L], where q = log(Z/W) and L = log(W). Image width observes q = log(f/p); it never observes L. The reported median depth is exp(q + L), with a lognormal conditional interval. Keeping the size prior as a persistent state prevents repeated images from erasing the global scale ambiguity. Pixel-width noise and a viewpoint term enter measurement noise. Velocity in log depth predicts short gaps. This remains an approximation for a changing orientation and is not a full 3D EKF.

The image filter uses [u, v, du/dt, dv/dt], source timestamps, piecewise constant acceleration noise, a Mahalanobis gate and the Joseph covariance update. Point flow passes forward/backward consistency and photometric checks; its uncertainty grows with time since detection. Correlation between successive flow observations is only approximated by this inflation. Only fresh detected widths update depth; predicted positions and flow guesses are not new range observations. Full 3D deployment should use [X,Y,Z,Vx,Vy,Vz], consistent camera/rig frames and either calibrated 3D observations or an EKF with image projection measurements.

The exact name "ScaleGPT" did not resolve to a reliable primary source in this research. [ScaleDepth](https://github.com/RuijieZhu94/ScaleDepth) is a related, identifiable work that separates scale prediction from relative depth estimation. It is a research comparison, not an implemented dependency here. A learned monocular metric model would require an out-of-domain test on small drones against sky, with no scale fitting on held-out labels.

## Stereo design

Prefer synchronized global-shutter RGB cameras with intrinsics, distortion, stereo extrinsics and time offset calibrated together. The camera convention is x-right, y-down, z-forward; camera-relative depth Z differs from Euclidean range. Choose focal length and baseline from the required drone pixel footprint and range uncertainty. A narrow field of view improves target pixels but makes reacquisition harder; an optional wide view can cue the rig.

For rectified stereo, Z = fB/d. With independent per-image localization noise sigma_px, sigma_Z is approximately sqrt(2) * sigma_px * Z^2 / (fB). The implemented geometry routine rejects tiny or negative disparity and epipolar inconsistency and returns a first-order XYZ covariance. Its synthetic calculations assume perfect calibration, simultaneous images and correct correspondence. A real pipeline must also reject poor reprojection, calibration drift and mismatched appearance; bbox centers need not identify the same physical 3D point when the drone rotates. There is no full stereo-video experiment in this version.

For a gimbal, compensate rotation and any moving optical-center translation before applying a constant-velocity model. Record pose and time uncertainty. Loss of one view leaves a bearing update and growing depth uncertainty; it does not create a new metric range observation.

## Point tracking and flow matching

"NextTap" is interpreted as [TAPNext](https://tap-next.github.io/), an online causal point tracker. The [official repository](https://github.com/google-deepmind/tapnet) provides PyTorch and JAX demos and newer checkpoints. Those implementations have not been installed or benchmarked on this laptop. Published accelerator timing must not be transferred to this CPU. The next controlled test would replace the `PointFlow` backend with TAPNext, seed several stable points inside each accepted box, apply visibility checks, preserve original-image coordinates, and compare occlusion recovery, drift and complete per-frame latency. Points on fast-spinning propellers are poor stable landmarks.

The implemented point backend is OpenCV pyramidal Lucas-Kanade flow, with forward/backward rejection and a robust displacement median. It is separate from generative flow matching. [OpenCV primary documentation](https://docs.opencv.org/4.x/d4/dee/tutorial_optical_flow.html).

For the flow-matching request, a small original NumPy network learns v(x_t,t,c) from x_t=(1-t)x_0+t*x_1 with squared loss against x_1-x_0. Conditions describe motion and gap duration; the endpoints are residual displacement after constant-velocity prediction. Sixteen Euler steps produce 64 possible residuals. Training draws fresh synthetic samples; evaluation uses a separate random seed and 150 cases. This demonstrates a distribution of possible futures, not an image tracker. The implementation is independent of any downloaded flow-matching code. [Flow matching guide](https://arxiv.org/abs/2412.06264).

[Generative Point Tracking with Flow Matching (GenPT)](https://arxiv.org/abs/2510.20951) is the relevant full research system for multimodal occluded point trajectories. Our toy does not reproduce it. A sampled future is not a sensor reading. Future integration could compare a motion prior or a mixture of hypotheses against constant-velocity filtering; undercovered uncertainty is a reason to keep this out of the main filter until calibrated.

## Data, augmentation and limits

No third-party footage is downloaded or redistributed. Original synthetic drone/bird drawings and seeded sky backgrounds provide deterministic, small test clips. Reference crops vary brightness, width and small image-plane orientation; every third reference seed uses a three-pixel directional blur. Test clips have a modest lighting change, five-pixel blur, and one fifteen-frame full occlusion. This is deliberately limited augmentation. There is no proof of robustness to real-world nuisance factors, and only two trajectories are tested.

The original [multiview drone dataset](https://github.com/CenekAlbl/drone-tracking-datasets) remains the real-data candidate. Its [dataset-1 annotation description](https://github.com/CenekAlbl/drone-tracking-datasets/blob/master/dataset1/README.md) specifies center labels, not bounding boxes. Box supervision would require additional annotation or a separately justified method; test labels must not be used to create training targets. The repository [license](https://github.com/CenekAlbl/drone-tracking-datasets/blob/master/LICENSE) is CC BY-NC-SA 4.0. Confirm terms for intended submission and derived artifacts. Split related views of the same flight together.

Current failure modes include missed proposals against dark clutter, DINO domain shift, drift onto an occluder, delayed verification, and depth bias from the size/orientation assumption. Short exposure mitigates blur at capture but increases low-light noise. Thermal can improve night visibility; radar can provide range/range-rate. Fuse timestamped measurements in a common frame with sensor-specific observation functions and covariance, including cross-correlation where needed. An IMU/encoder helps gimbal motion compensation; acoustics may provide coarse cues rather than precise depth.

## Next smallest useful experiment

Acquire one licensed fixed-camera drone clip and box/center annotations, build a flight-level held-out split, and test a drone-trained detector plus the existing tracker. Calibrate DINO verification on real positives and birds before accepting its score. Compare TAPNext only after this baseline exposes a concrete point-tracking failure. For metric depth, prefer a calibrated stereo pair or a separately measured drone size; do not claim accuracy from the synthetic size prior.
