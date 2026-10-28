"""Reusable timestamped tracker. Image source metadata contains no annotations."""
from dataclasses import dataclass
import time
import cv2
import numpy as np
from .filters import PixelKalman, ScaleRangeKalman
from .vision import AsyncVerifier, PointFlow, crop_box, proposals


@dataclass
class Frame:
    index: int
    time_s: float
    image: np.ndarray


@dataclass
class Config:
    fps: float = 30.
    detector_period_s: float = 1/6
    verification_period_s: float = .5
    verification_ttl_s: float = 1.5
    result_max_age_s: float = .75
    retire_after_s: float = 1.
    negative_limit: int = 2
    focal_px: float | None = None
    width_m: float | None = None
    size_log_sigma: float = .30

    def __post_init__(self):
        if not np.isfinite(self.fps) or self.fps <= 0:
            raise ValueError("fps must be positive and finite")
        if (self.focal_px is None) != (self.width_m is None):
            raise ValueError("Distance needs both calibrated focal_px and a physical width prior")
        if self.focal_px is not None and (not np.isfinite(self.focal_px) or self.focal_px <= 0 or not np.isfinite(self.width_m) or self.width_m <= 0):
            raise ValueError("Focal length and width must be positive and finite")
        if self.negative_limit < 1:
            raise ValueError("negative_limit must be at least one")


class IdentityState:
    """Expire positive evidence and retire after the configured negative count.

    Negative verdicts never extend the lifetime of the last positive evidence.
    """
    def __init__(self, threshold, ttl_s=1.5, negative_limit=2):
        self.threshold, self.ttl_s, self.negative_limit = threshold, ttl_s, negative_limit
        self.last_positive = None
        self.failures = 0
        self.margin = None

    def apply(self, score, source_time):
        self.margin = score
        if score >= self.threshold:
            self.last_positive, self.failures = source_time, 0
            return False
        self.failures += 1
        return self.last_positive is None or self.failures >= self.negative_limit

    def valid(self, now):
        return self.last_positive is not None and self.failures < self.negative_limit and now-self.last_positive <= self.ttl_s


def replay(frames, config, verifier=None, proposal_fn=proposals, initial_box=None,
           use_flow=True, paced=True, video_path=None):
    """Process Frame objects, with no access to evaluation labels.

    Off-line replay preserves all frames. It does not simulate a camera's capture
    queue. Age/deadline telemetry makes delays visible rather than hiding them.
    """
    worker = AsyncVerifier(verifier, max_age=config.result_max_age_s) if verifier else None
    identity = IdentityState(verifier.threshold if verifier else 0., config.verification_ttl_s, config.negative_limit)
    flow, pixel, depth = PointFlow(), None, None
    generation = 0
    last_direct = last_detect = last_submit = -np.inf
    previous_time = first_source_time = None
    wall_start = time.perf_counter()
    banned = []
    rows = []
    writer = None
    stage_keys = ("proposal_ms", "flow_ms", "filter_ms")
    try:
        for packet in frames:
            t, image = float(packet.time_s), packet.image
            if not np.isfinite(t) or (previous_time is not None and t <= previous_time):
                raise ValueError("Source timestamps must be finite and strictly increasing")
            if first_source_time is None:
                first_source_time, wall_start = t, time.perf_counter()
            relative_t = t-first_source_time
            if paced:
                time.sleep(max(0., wall_start+relative_t-time.perf_counter()))
            start = time.perf_counter()
            lateness = max(0., (start-wall_start-relative_t)*1000) if paced else None
            dt = t-previous_time if previous_time is not None else 1/config.fps
            previous_time = t
            stage = dict.fromkeys(stage_keys, 0.)
            status = "lost"
            if pixel is not None and t-last_direct > config.retire_after_s:
                pixel = depth = None
                flow = PointFlow()
                identity = IdentityState(identity.threshold, config.verification_ttl_s, config.negative_limit)
                generation += 1
            if worker:
                verdict = worker.poll(t, generation)
                if verdict and identity.apply(*verdict):
                    if pixel is not None:
                        banned.append((pixel.x[:2].copy(), t+.65))
                    pixel = depth = None
                    flow = PointFlow()
                    generation += 1
                    identity = IdentityState(identity.threshold, config.verification_ttl_s, config.negative_limit)
            if pixel is not None:
                stamp = time.perf_counter()
                pixel.predict(dt)
                if depth:
                    depth.predict(dt)
                stage["filter_ms"] += (time.perf_counter()-stamp)*1000
                status = "predicted"
            stamp = time.perf_counter()
            flow_measurement = flow.step(image) if use_flow and pixel is not None else None
            stage["flow_ms"] = (time.perf_counter()-stamp)*1000
            selected = None
            if pixel is None or t-last_detect >= config.detector_period_s-1e-9:
                stamp = time.perf_counter()
                candidates = list(proposal_fn(image))
                last_detect = t
                if packet.index == 0 and initial_box is not None:
                    candidates.insert(0, tuple(initial_box))
                else:
                    candidates.sort(key=lambda b: pixel.distance([b[0]+b[2]/2, b[1]+b[3]/2]) if pixel else -b[2])
                banned = [(center, expiry) for center, expiry in banned if expiry > t]
                for box in candidates:
                    x, y, w, h = box
                    center = np.array([x+w/2, y+h/2])
                    if any(np.linalg.norm(center-old) < 45 for old, _ in banned):
                        continue
                    if pixel is not None and pixel.distance(center) > 13.82:
                        continue
                    selected = box
                    break
                stage["proposal_ms"] = (time.perf_counter()-stamp)*1000
                if selected is not None:
                    x, y, w, h = selected
                    center = [x+w/2, y+h/2]
                    stamp = time.perf_counter()
                    if pixel is None:
                        generation += 1
                        pixel = PixelKalman(center)
                        if config.focal_px is not None:
                            depth = ScaleRangeKalman(w, config.focal_px, config.width_m, config.size_log_sigma)
                        accepted = True
                    else:
                        accepted = pixel.update(center)
                        if accepted and depth:
                            depth.update(w)
                    stage["filter_ms"] += (time.perf_counter()-stamp)*1000
                    if accepted:
                        status, last_direct = "detected", t
                        if use_flow:
                            stamp = time.perf_counter()
                            flow.seed(image, selected)
                            stage["flow_ms"] += (time.perf_counter()-stamp)*1000
                        if worker and (not identity.valid(t) or identity.failures or t-last_submit >= config.verification_period_s-1e-9):
                            if worker.submit(crop_box(image, selected), t, generation):
                                last_submit = t
            if pixel is not None and status != "detected" and flow_measurement is not None:
                stamp = time.perf_counter()
                if pixel.update(*flow_measurement):
                    status = "flow"
                stage["filter_ms"] += (time.perf_counter()-stamp)*1000
            confirmed = pixel is not None and (verifier is None or identity.valid(t))
            u = v = z = low = high = x_m = y_m = None
            if confirmed:
                u, v = map(float, pixel.x[:2])
                if identity.failures:
                    status = "suspect"
                if depth:
                    z, (low, high) = depth.estimate()
                    x_m = (u-image.shape[1]/2)*z/config.focal_px
                    y_m = (v-image.shape[0]/2)*z/config.focal_px
            elif pixel is not None:
                status = "tentative"
            end = time.perf_counter()
            elapsed_ms = (end-start)*1000
            stage["other_ms"] = max(0.,elapsed_ms-sum(stage.values()))
            row = {"frame": packet.index, "time_s": t, "dt_s": dt, "status": status,
                "u_px": u, "v_px": v, "depth_m": z, "depth_low95_m": low, "depth_high95_m": high,
                "x_m": x_m, "y_m": y_m, "distance_source": "assumed_size_prior" if depth else "unavailable",
                "dino_margin": identity.margin, "negative_streak": identity.failures,
                "verification_age_s": t-identity.last_positive if identity.last_positive is not None else None,
                "processing_ms": elapsed_ms, "output_wall_s": end-wall_start,
                "scheduler_lateness_ms": lateness,
                "scheduled_output_age_ms": lateness+elapsed_ms if paced else None,
                "deadline_missed": int(paced and lateness+elapsed_ms > 1000/config.fps), **stage}
            rows.append(row)
            if video_path:
                if writer is None:
                    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), config.fps,
                                             (image.shape[1], image.shape[0]))
                    if not writer.isOpened():
                        raise RuntimeError("Cannot open output video")
                annotated = image.copy()
                if u is not None:
                    cv2.circle(annotated, (round(u), round(v)), 10, (40, 220, 50), 1)
                cv2.putText(annotated, status, (12, 24), 0, .5, (10, 10, 10), 1)
                writer.write(annotated)
    finally:
        if writer:
            writer.release()
        if worker:
            worker.close()
    return rows, {"verification": worker.stats if worker else {}, "paced": paced,
                  "source_frames": len(rows), "wall_s": time.perf_counter()-wall_start}
