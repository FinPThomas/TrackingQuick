"""Controlled v0.2 comparisons; missing outputs count against all-frame success."""
import argparse
from dataclasses import asdict
import csv
import json
from pathlib import Path
import platform
import cv2
import numpy as np
from .pipeline import Config, Frame, replay
from .vision import DinoVerifier
from .synthetic import sequence, reference_crop, FPS, FOCAL


def save_csv(path, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows, truth):
    if len(rows) != len(truth):
        raise ValueError("Every source frame must have an output row, including lost frames")
    visible = np.array([g["visible"] for g in truth], bool)
    present = np.array([g.get("present", True) for g in truth], bool)
    error = np.array([np.hypot(r["u_px"]-g["u"], r["v_px"]-g["v"]) if r["u_px"] is not None else np.nan
                      for r, g in zip(rows, truth, strict=True)])
    reported = np.isfinite(error)
    def fraction(mask, denominator):
        return float(np.sum(mask)/np.sum(denominator)) if np.any(denominator) else None
    def rmse(mask):
        return float(np.sqrt(np.mean(error[mask]**2))) if np.any(mask) else None
    processing = [r["processing_ms"] for r in rows]
    ages = [r["scheduled_output_age_ms"] for r in rows if r.get("scheduled_output_age_ms") is not None]
    metric = {"frames": len(rows), "visible_frames": int(visible.sum()),
        "coverage": fraction(reported & visible, visible),
        "success_8px_all_visible": fraction(visible & reported & (error <= 8), visible),
        "wrong_track_fraction_15px": fraction(visible & reported & (error > 15), visible),
        "rmse_reported_visible_px": rmse(visible & reported),
        "rmse_reported_occluded_px": rmse(present & ~visible & reported),
        "empty_false_alarm_fraction": fraction(~present & reported, ~present),
        "first_confirmation_wall_s": next((r.get("output_wall_s") for r in rows if r["u_px"] is not None), None),
        "deadline_miss_fraction": float(np.mean([r.get("deadline_missed", 0) for r in rows])) if ages else None,
        "suspect_frames": sum(r.get("status") == "suspect" for r in rows),
        "processing_ms": {f"p{q}": float(np.percentile(processing, q)) for q in (50,95,99)},
        "scheduled_output_age_ms": {f"p{q}": float(np.percentile(ages, q)) for q in (50,95,99)} if ages else {}}
    metric["stage_mean_ms"] = {key:float(np.mean([r[key] for r in rows]))
        for key in ("proposal_ms","flow_ms","filter_ms","other_ms") if key in rows[0]}
    metric["by_condition"] = {}
    for condition in sorted({g["condition"] for g in truth}):
        mask = np.array([g["condition"] == condition for g in truth]) & visible
        metric["by_condition"][condition] = {"visible_frames": int(mask.sum()),
            "coverage": fraction(mask & reported, mask),
            "success_8px_all_visible": fraction(mask & reported & (error <= 8), mask)}
    return metric


def common_frame_comparison(a, b, truth):
    common = [(x,y,g) for x,y,g in zip(a,b,truth,strict=True) if g["visible"] and x["u_px"] is not None and y["u_px"] is not None]
    def error(which):
        return float(np.sqrt(np.mean([(p[which]["u_px"]-p[2]["u"])**2+(p[which]["v_px"]-p[2]["v"])**2 for p in common]))) if common else None
    return {"common_visible_frames": len(common), "one_negative_rmse_px": error(0), "two_negative_rmse_px": error(1)}


def source(seed, count, condition):
    for i, (image, _) in enumerate(sequence(seed, count, condition)):
        yield Frame(i, i/FPS, image)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="reports/v0_2")
    parser.add_argument("--model", default="weights/dinov2-small-int8.onnx")
    parser.add_argument("--frames", type=int, default=180)
    args = parser.parse_args(argv)
    if args.frames < 1:
        parser.error("--frames must be positive")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    cv2.setNumThreads(2)
    verifier = DinoVerifier(args.model)
    references = lambda seeds: [(reference_crop(kind, seed), kind == "drone") for seed in seeds for kind in ("drone","bird")]
    calibration = verifier.fit(references(range(100,108)), references(range(300,306)))
    config = Config(focal_px=FOCAL, width_m=.6)
    result = {"version": "0.2", "calibration": calibration, "config": asdict(config),
        "platform": platform.platform(), "runs": [], "paired": [],
        "scope": "synthetic replay; paced 30 Hz; source-aligned output age includes replay lateness, not physical camera/display latency"}
    for seed, condition in [(42,"mixed"),(73,"mixed"),(42,"empty"),(42,"distractor")]:
        truth = [g for _,g in sequence(seed,args.frames,condition)]
        case_rows = {}
        for name, model, limit in [("flow_kf",None,2),("dino_one_negative",verifier,1),("dino_two_negative",verifier,2)]:
            config.negative_limit = limit
            first_crop = len(verifier.times)
            rows, diagnostics = replay(source(seed,args.frames,condition), config, verifier=model)
            measured = {"seed":seed,"condition":condition,"mode":name,**summarize(rows,truth), **diagnostics}
            crop_times = verifier.times[first_crop:]
            measured["dino_crop_ms"] = {f"p{q}":float(np.percentile(crop_times,q)) for q in (50,95,99)} if crop_times else {}
            case_rows[name] = rows
            result["runs"].append(measured)
            save_csv(output/f"{condition}_{seed}_{name}.csv",rows)
            print(json.dumps({k:measured[k] for k in ("seed","condition","mode","coverage","success_8px_all_visible","empty_false_alarm_fraction","processing_ms")}), flush=True)
        result["paired"].append({"seed":seed,"condition":condition,
            **common_frame_comparison(case_rows["dino_one_negative"],case_rows["dino_two_negative"],truth)})
    (output/"metrics.json").write_text(json.dumps(result,indent=2,allow_nan=False),encoding="utf-8")
    plot_results(result, output)
    print(output.resolve(),flush=True)


def plot_results(result, output):
    """Render the recorded comparison without rerunning inference."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cases = [(42,"mixed"),(73,"mixed"),(42,"distractor")]
    modes = ["flow_kf","dino_one_negative","dino_two_negative"]
    fig, ax = plt.subplots(figsize=(9,4),constrained_layout=True)
    for j, mode in enumerate(modes):
        values = [next(r["success_8px_all_visible"] for r in result["runs"] if r["seed"]==s and r["condition"]==c and r["mode"]==mode) for s,c in cases]
        ax.bar(np.arange(3)+(j-1)*.25,np.array(values)*100,width=.24,label=mode)
    ax.set_xticks(np.arange(3),[f"{c}, seed {s}" for s,c in cases])
    ax.set(ylabel="Visible frames with correct output within 8 px (%)",ylim=(0,110),title="v0.2: missing outputs count as failures")
    ax.legend(fontsize=8,loc="upper center",bbox_to_anchor=(.5,-.13),ncol=3)
    fig.savefig(output/"success.png",dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
