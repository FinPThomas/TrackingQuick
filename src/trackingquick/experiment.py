"""Reproduce the bounded DINOv2/Kalman/scale experiment on original synthetic scenes."""
import argparse
import csv
import json
from pathlib import Path
import platform
import time
import cv2
import numpy as np
from .filters import PixelKalman, ScaleRangeKalman, triangulate_rectified
from .vision import DinoVerifier, PointFlow, AsyncVerifier, proposals, crop_box
from .synthetic import sequence, reference_crop, WIDTH, HEIGHT, FPS, FOCAL
from .flowmatch import experiment as flow_experiment


def percentiles(values):
    a = np.asarray(values, float)
    return {f"p{p}_ms": float(np.percentile(a, p)) for p in (50, 95, 99)} if len(a) else {}


def reference_data(seeds):
    return [(reference_crop(kind, seed), kind == "drone") for seed in seeds for kind in ("drone", "bird")]


def track(frames, mode, verifier=None, save_video=None):
    pixel = depth = None
    flow = PointFlow()
    last_direct = -1000
    rows = []
    writer = None
    if save_video:
        writer = cv2.VideoWriter(str(save_video), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (WIDTH, HEIGHT))
        if not writer.isOpened():
            raise RuntimeError("Could not open annotated video output")
    asynchronous = mode == "dino_async_flow_kf"
    use_dino, use_flow = mode != "proposals_kf", mode in ("dino_flow_kf", "dino_async_flow_kf")
    worker = AsyncVerifier(verifier) if asynchronous else None
    confirmed, generation, verified_frame = False, 0, -1000
    clock_start = time.perf_counter()
    try:
        for i, (frame, _) in enumerate(frames):
            # Truth is intentionally never read in this inference function.
            if asynchronous:
                time.sleep(max(0., clock_start+i/FPS-time.perf_counter()))
            start = time.perf_counter()
            lateness = max(0., (start-clock_start-i/FPS)*1000) if asynchronous else None
            submitted = 0
            status, box, margin = "lost", None, None
            if worker:
                result = worker.poll(i, generation)
                if result is not None:
                    score, verified_frame = result
                    confirmed = score >= verifier.threshold
                    if not confirmed:
                        pixel = depth = None
                        flow = PointFlow()
            if pixel is not None and i-last_direct > FPS:
                pixel = depth = None
                flow = PointFlow()
                confirmed = False
            if asynchronous and i-verified_frame > 45:
                confirmed = False
            if pixel is not None:
                pixel.predict(1/FPS)
                depth.predict(1/FPS)
                status = "predicted"
            before_calls = len(verifier.times) if verifier else 0
            flow_measurement = flow.step(frame) if use_flow and pixel is not None else None
            if i % 5 == 0 or pixel is None:
                candidates = proposals(frame)
                candidates.sort(key=lambda b: pixel.distance([b[0]+b[2]/2, b[1]+b[3]/2]) if pixel else -b[2])
                accepted = []
                for candidate in candidates:
                    x, y, w, h = candidate
                    center = np.array([x+w/2, y+h/2])
                    if pixel is not None and pixel.distance(center) > 13.82:
                        continue
                    verify = use_dino and (pixel is None or i % 15 == 0 or i-last_direct > 10)
                    score = verifier.score(crop_box(frame, candidate)) if verify and not asynchronous else None
                    if verify and not asynchronous and score < verifier.threshold:
                        continue
                    accepted.append((candidate, score))
                    if pixel is not None:
                        break
                if accepted:
                    if pixel is None and use_dino and not asynchronous:
                        accepted.sort(key=lambda a: a[1], reverse=True)
                    box, margin = accepted[0]
                    x, y, w, h = box
                    center = np.array([x+w/2, y+h/2])
                    if pixel is None:
                        generation += 1
                        pixel = PixelKalman(center)
                        depth = ScaleRangeKalman(w, FOCAL)
                        status = "detected"
                    elif pixel.update(center):
                        depth.update(w)
                        status = "detected"
                    if status == "detected":
                        last_direct = i
                        if use_flow:
                            flow.seed(frame, box)
                        if worker and (not confirmed or i % 15 == 0):
                            submitted = int(worker.submit(crop_box(frame, box), i, generation))
            if status != "detected" and flow_measurement is not None:
                center, sigma = flow_measurement
                if pixel.update(center, sigma):
                    status = "flow"
            u = v = z = low = high = x_m = y_m = None
            if pixel is not None and asynchronous and not confirmed:
                status = "tentative"
            if pixel is not None and (not asynchronous or confirmed):
                u, v = map(float, pixel.x[:2])
                z, (low, high) = depth.estimate()
                x_m, y_m = (u-WIDTH/2)*z/FOCAL, (v-HEIGHT/2)*z/FOCAL
            elapsed = (time.perf_counter()-start)*1000
            rows.append({"frame": i, "time_s": i/FPS, "status": status, "u_px": u, "v_px": v,
                         "depth_m": z, "depth_low95_m": low, "depth_high95_m": high,
                         "x_m": x_m, "y_m": y_m, "distance_source": "assumed_size_prior" if z else "unavailable",
                         "dino_margin": margin, "dino_calls": submitted if asynchronous else len(verifier.times)-before_calls if verifier else 0,
                         "verification_age_frames": i-verified_frame if asynchronous and confirmed else None,
                         "scheduler_lateness_ms": lateness, "processing_ms": elapsed})
            if writer:
                overlay = frame.copy()
                if u is not None:
                    cv2.circle(overlay, (round(u), round(v)), 12, (40, 220, 50), 1)
                    cv2.putText(overlay, f"{status}  Z~{z:.1f}m [{low:.1f},{high:.1f}]", (12, 24), 0, .5, (25, 25, 25), 1)
                cv2.putText(overlay, "SYNTHETIC | distance assumes W=0.6m +/-30% log-scale", (12, HEIGHT-14), 0, .43, (25, 25, 25), 1)
                writer.write(overlay)
    finally:
        if writer:
            writer.release()
        if worker:
            worker.close()
    return rows


def evaluate(rows, truth):
    # Scoring happens only after tracking; annotation data never enters inference.
    visible_errors, hidden_errors, depth_errors, coverage = [], [], [], []
    false_associations = detections = correct_detections = correct_scheduled = eligible = visible_count = tracked_visible = 0
    for row, gt in zip(rows, truth, strict=True):
        visible_count += int(gt["visible"])
        eligible += int(gt["visible"] and gt["frame"] % 5 == 0)
        if row["u_px"] is None:
            continue
        error = float(np.hypot(row["u_px"]-gt["u"], row["v_px"]-gt["v"]))
        if gt["visible"]:
            visible_errors.append(error)
            tracked_visible += 1
            depth_errors.append(abs(row["depth_m"]-gt["z"])/gt["z"])
            coverage.append(row["depth_low95_m"] <= gt["z"] <= row["depth_high95_m"])
            false_associations += int(error > 15)
        else:
            hidden_errors.append(error)
        if row["status"] == "detected":
            detections += 1
            correct_detections += int(gt["visible"] and error < 8)
            correct_scheduled += int(gt["visible"] and error < 8 and gt["frame"] % 5 == 0)
    return {"frames": len(rows), "visible_frames": visible_count,
            "visible_center_rmse_px": float(np.sqrt(np.mean(np.square(visible_errors)))) if visible_errors else None,
            "occluded_center_rmse_px": float(np.sqrt(np.mean(np.square(hidden_errors)))) if hidden_errors else None,
            "visible_track_coverage": tracked_visible/max(1, visible_count),
            "false_association_frames": false_associations,
            "detection_precision_8px": correct_detections/max(1, detections),
            "scheduled_detection_recall_8px": correct_scheduled/max(1, eligible),
            "depth_absrel": float(np.mean(depth_errors)) if depth_errors else None,
            "conditional_depth_interval_coverage": float(np.mean(coverage)) if coverage else None,
            "dino_calls": sum(r["dino_calls"] for r in rows),
            "first_confirmed_time_s": next((r["time_s"] for r in rows if r["u_px"] is not None), None),
            "max_scheduler_lateness_ms": max((r["scheduler_lateness_ms"] for r in rows if r["scheduler_lateness_ms"] is not None), default=None),
            **percentiles([r["processing_ms"] for r in rows]),
            "timing_scope": "main-thread per-frame processing; excludes acquisition, encoding and paced input waiting; background-verification delay reported separately"}


def plot_results(output, runs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    palette = ["#777777", "#e59435", "#177e89", "#9b4d96"]
    for (mode, rows, truth), color in zip(runs, palette, strict=True):
        t = [r["time_s"] for r in rows]
        error = [np.hypot(r["u_px"]-g["u"], r["v_px"]-g["v"]) if r["u_px"] is not None else np.nan for r, g in zip(rows, truth)]
        axes[0, 0].plot(t, error, label=mode, color=color)
        axes[1, 0].plot(t, [r["processing_ms"] for r in rows], label=mode, color=color, alpha=.85)
    rows, truth = runs[-1][1:]
    t = [r["time_s"] for r in rows]
    axes[0, 1].plot(t, [g["z"] for g in truth], color="black", label="True depth")
    axes[0, 1].plot(t, [r["depth_m"] for r in rows], color=palette[-1], label="Size-prior estimate")
    axes[0, 1].fill_between(t, [r["depth_low95_m"] or np.nan for r in rows], [r["depth_high95_m"] or np.nan for r in rows], color=palette[-1], alpha=.2, label="Conditional 95% interval")
    axes[1, 1].plot([g["u"] for g in truth], [g["v"] for g in truth], color="black", label="True center")
    axes[1, 1].plot([r["u_px"] for r in rows], [r["v_px"] for r in rows], color=palette[-1], label="Tracked center")
    axes[1, 1].invert_yaxis()
    titles = ["2D center error", "Depth retains physical-size uncertainty", "CPU processing latency (verification causes spikes)", "Image trajectory"]
    labels = [("Time (s)", "Error (px)"), ("Time (s)", "Optical-axis depth (m)"), ("Time (s)", "Processing (ms)"), ("u (px)", "v (px)")]
    for ax, title, (xlab, ylab) in zip(axes.flat, titles, labels):
        ax.set(title=title, xlabel=xlab, ylabel=ylab)
        ax.grid(alpha=.2)
        ax.legend(fontsize=7)
    fig.suptitle("TrackingQuick: synthetic held-out sequence, seed 42", fontsize=14)
    fig.savefig(output / "tracking_results.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="weights/dinov2-small-int8.onnx")
    parser.add_argument("--output", default="reports/results")
    parser.add_argument("--frames", type=int, default=180)
    parser.add_argument("--crop-size", type=int, default=112, help="DINO input side; use 224 for the reference-resolution ablation")
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    cv2.setNumThreads(2)
    verifier = DinoVerifier(args.model, crop_size=args.crop_size)
    fit = verifier.fit(reference_data(range(100, 108)), reference_data(range(300, 306)))
    print("DINO reference calibration:", json.dumps(fit), flush=True)
    np.savez(output / "dino_prototypes.npz", positive=verifier.positive, negative=verifier.negative, threshold=verifier.threshold)
    metrics = {"reference_calibration": fit, "dino_input_size": args.crop_size,
               "hardware": {"processor": platform.processor(), "platform": platform.platform(), "onnx_threads": 2}, "runs": []}
    figures = []
    for seed in (42, 73):
        # Data are deterministic and materialized equally for each ablation.
        frames = list(sequence(seed, args.frames))
        truth = [gt for _, gt in frames]
        for mode in ("proposals_kf", "dino_kf", "dino_flow_kf", "dino_async_flow_kf"):
            video = output / "demo.mp4" if seed == 42 and mode == "dino_async_flow_kf" else None
            rows = track(frames, mode, verifier, video)
            measured = {"seed": seed, "mode": mode, **evaluate(rows, truth)}
            metrics["runs"].append(measured)
            with (output / f"{mode}_seed{seed}.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
            print(json.dumps(measured), flush=True)
            if seed == 42:
                figures.append((mode, rows, truth))
    metrics["dino_crop_latency"] = percentiles(verifier.times[1:])
    metrics["dino_crop_latency"]["warmup_excluded"] = 1
    depth_rows, depth_truth = figures[-1][1:]
    metrics["size_prior_sensitivity"] = []
    for size in (.4, .6, .9):
        errors = [abs(r["depth_m"]*(size/.6)-g["z"])/g["z"] for r, g in zip(depth_rows, depth_truth)
                  if r["depth_m"] is not None and g["visible"]]
        metrics["size_prior_sensitivity"].append({"assumed_width_m": size, "true_width_m": .6,
            "depth_absrel": float(np.mean(errors)), "method": "exact multiplicative sensitivity with observations held fixed"})
    metrics["flow_matching"], hypotheses, actual = flow_experiment()
    metrics["stereo_geometry"] = []
    for z in (20, 50, 100, 200):
        try:
            xyz, cov = triangulate_rectified([330, 185], [330-FOCAL/z, 185], FOCAL, 1., [320, 180])
            metrics["stereo_geometry"].append({"depth_m": z, "sigma_z_m": float(np.sqrt(cov[2, 2]))})
        except ValueError as exc:
            metrics["stereo_geometry"].append({"depth_m": z, "rejected": str(exc)})
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=False), encoding="utf-8")
    plot_results(output, figures)
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    ax.scatter(hypotheses[:, 0], hypotheses[:, 1], alpha=.4, label="Flow samples")
    ax.scatter([0], [0], marker="+", s=100, label="Constant-velocity residual")
    ax.scatter(*actual, marker="*", s=120, label="One held-out future")
    ax.set(title="Toy flow-matching residual hypotheses", xlabel="Residual u (px)", ylabel="Residual v (px)")
    ax.legend()
    fig.savefig(output / "flow_matching.png", dpi=140)
    plt.close(fig)
    print("Finished:", output.resolve(), flush=True)


if __name__ == "__main__":
    main()
