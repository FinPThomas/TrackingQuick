"""Train an appearance verifier on real labelled clips and evaluate the tracker on held-out ones."""
import argparse
import json
import math
from pathlib import Path
import time
import cv2
import numpy as np
from .pipeline import Config, Frame, replay
from .video import ForegroundProposals, TargetProposals, first_frame_references
from .vision import DinoVerifier, crop_box

MIN_CROP_PX = 40
SAMPLE_EVERY = 20


def load_index(root):
    return json.loads((Path(root)/"index.json").read_text(encoding="utf-8"))


def clip_frames(path):
    capture = cv2.VideoCapture(str(path))
    try:
        while True:
            ok, image = capture.read()
            if not ok:
                return
            yield image
    finally:
        capture.release()


def jittered(rng, box, shape):
    """Approximate a tight proposal box: labels are looser than foreground blobs."""
    x, y, w, h = box
    scale = rng.uniform(.35, 1.)
    new_w, new_h = max(3., w*scale), max(3., h*scale)
    new_x = x+rng.uniform(0, max(0., w-new_w))
    new_y = y+rng.uniform(0, max(0., h-new_h))
    return clamp((new_x, new_y, new_w, new_h), shape)


def clamp(box, shape):
    x, y, w, h = box
    x, y = min(max(0., x), shape[1]-2.), min(max(0., y), shape[0]-2.)
    return int(x), int(y), max(2, int(round(min(w, shape[1]-x)))), max(2, int(round(min(h, shape[0]-y))))


def contains(outer, point):
    x, y, w, h = outer
    return x <= point[0] <= x+w and y <= point[1] <= y+h


def clip_crops(rng, root, name, clip):
    """Yield (crop, label). Negatives include other classes, background and real false proposals."""
    proposer = ForegroundProposals()
    for index, image in enumerate(clip_frames(root/f"{name}.mp4")):
        candidates = proposer(image)
        box = clip["boxes"][index] if index < len(clip["boxes"]) else None
        if index % SAMPLE_EVERY or box is None or min(box[2:]) < 2:
            continue
        yield crop_box(image, jittered(rng, box, image.shape), MIN_CROP_PX), clip["class"] == "DRONE"
        center = (box[0]+box[2]/2, box[1]+box[3]/2)
        false_positives = [b for b in candidates if not contains((box[0]-10, box[1]-10, box[2]+20, box[3]+20), (b[0]+b[2]/2, b[1]+b[3]/2))]
        if false_positives:
            yield crop_box(image, false_positives[rng.integers(len(false_positives))], MIN_CROP_PX), False
        for _ in range(20):
            x, y = rng.uniform(0, image.shape[1]-box[2]), rng.uniform(0, image.shape[0]-box[3])
            if math.hypot(x+box[2]/2-center[0], y+box[3]/2-center[1]) > max(box[2:])+MIN_CROP_PX:
                yield crop_box(image, clamp((x, y, box[2], box[3]), image.shape), MIN_CROP_PX), False
                break


def extract_features(root, index, model, cache):
    if Path(cache).exists():
        stored = np.load(cache, allow_pickle=False)
        return {key: stored[key] for key in stored.files}
    verifier = DinoVerifier(model, threads=4)
    rng = np.random.default_rng(7)
    features, labels, splits, names = [], [], [], []
    started = time.perf_counter()
    for number, (name, clip) in enumerate(sorted(index.items())):
        if clip["split"] == "dev":
            continue
        for crop, label in clip_crops(rng, root, name, clip):
            features.append(verifier.embed(crop))
            labels.append(int(label))
            splits.append(clip["split"])
            names.append(name)
        if number % 25 == 0:
            print(f"{number}/{len(index)} clips, {len(features)} crops, {time.perf_counter()-started:.0f} s", flush=True)
    result = {"features": np.stack(features), "labels": np.array(labels), "splits": np.array(splits), "names": np.array(names)}
    Path(cache).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, **result)
    return result


def logistic(features, labels, penalty, iterations=25):
    """Class-balanced logistic regression by Newton steps, with an unpenalized bias."""
    design = np.c_[features, np.ones(len(features))]
    weights = np.where(labels == 1, .5/max(1, labels.sum()), .5/max(1, (1-labels).sum()))*len(labels)
    theta = np.zeros(design.shape[1])
    ridge = np.eye(design.shape[1])*penalty*len(labels)
    ridge[-1, -1] = 1e-9
    for _ in range(iterations):
        probability = 1/(1+np.exp(-np.clip(design @ theta, -30, 30)))
        gradient = design.T @ (weights*(probability-labels))+ridge @ theta
        curvature = np.maximum(probability*(1-probability), 1e-6)
        hessian = (design*(weights*curvature)[:, None]).T @ design+ridge
        step = np.linalg.solve(hessian, gradient)
        theta -= step
        if np.max(np.abs(step)) < 1e-6:
            break
    return theta[:-1], float(theta[-1])


def balanced_accuracy(scores, labels, threshold):
    predicted = scores >= threshold
    return float((predicted[labels == 1].mean()+(~predicted[labels == 0]).mean())/2)


def best_threshold(scores, labels):
    ordered = np.sort(scores)
    candidates = np.r_[ordered[0]-1e-5, (ordered[1:]+ordered[:-1])/2, ordered[-1]+1e-5]
    if len(candidates) > 400:
        candidates = candidates[np.linspace(0, len(candidates)-1, 400).astype(int)]
    values = [balanced_accuracy(scores, labels, value) for value in candidates]
    return float(candidates[int(np.argmax(values))])


def summarize_crops(scores, labels, threshold):
    predicted = scores >= threshold
    return {"balanced_accuracy": balanced_accuracy(scores, labels, threshold),
            "recall": float(predicted[labels == 1].mean()), "false_positive_rate": float(predicted[labels == 0].mean()),
            "crops": int(len(labels)), "positive_crops": int(labels.sum())}


def train(root, model, output):
    index = load_index(root)
    data = extract_features(root, index, model, Path("runs/real/features.npz"))
    part = {split: data["splits"] == split for split in ("train", "val", "test")}
    x, y = data["features"], data["labels"]
    train_x, train_y, val_x, val_y = x[part["train"]], y[part["train"]], x[part["val"]], y[part["val"]]
    positive = train_x[train_y == 1].mean(axis=0)
    negative = train_x[train_y == 0].mean(axis=0)
    positive, negative = positive/np.linalg.norm(positive), negative/np.linalg.norm(negative)
    prototype_threshold = best_threshold(val_x @ (positive-negative), val_y)
    report = {"prototype_val": summarize_crops(val_x @ (positive-negative), val_y, prototype_threshold), "probe_candidates": {}}
    best = None
    for penalty in (1e-4, 1e-3, 1e-2, 1e-1):
        w, b = logistic(train_x, train_y, penalty)
        scores = val_x @ w+b
        threshold = best_threshold(scores, val_y)
        summary = summarize_crops(scores, val_y, threshold)
        report["probe_candidates"][str(penalty)] = summary
        if best is None or summary["balanced_accuracy"] > best[0]:
            best = summary["balanced_accuracy"], penalty, w, b, threshold
    _, penalty, w, b, threshold = best
    # Only now touch test crops; selection above used validation clips only.
    report["selected_penalty"] = penalty
    report["probe_test"] = summarize_crops(x[part["test"]] @ w+b, y[part["test"]], threshold)
    report["prototype_test"] = summarize_crops(x[part["test"]] @ (positive-negative), y[part["test"]], prototype_threshold)
    # The verifier scores f @ positive - f @ negative, which represents a linear probe directly.
    np.savez(Path(output)/"verifier.npz", positive=w, negative=np.zeros_like(w), threshold=np.array(threshold-b))
    (Path(output)/"verifier_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


def track_stream(first, stream, fps, roi, policy, model, verifier_path, negative_limit=2):
    """Run one policy over Frame objects. `first` is only used to build first-frame references."""
    improved = policy in ("target_first_frame", "target_real")
    verifier = None
    if policy != "flow":
        verifier = DinoVerifier(model)
        if policy == "target_real":
            stored = np.load(verifier_path)
            verifier.positive, verifier.negative, verifier.threshold = stored["positive"], stored["negative"], float(stored["threshold"])
        else:
            first_frame_references(verifier, first, roi)
    proposer = TargetProposals(roi) if improved else ForegroundProposals()
    config = Config(fps=fps, negative_limit=negative_limit, reacquire_near_last=improved,
                    min_crop_px=MIN_CROP_PX if policy == "target_real" else 0)
    rows, _ = replay(stream, config, verifier, proposer, initial_box=roi)
    return rows


def track(root, name, clip, policy, model, verifier_path, negative_limit=2):
    frames = list(clip_frames(root/f"{name}.mp4"))
    first_box = clip["boxes"][0] if clip["class"] == "DRONE" and clip["boxes"][0] else None
    roi = None if first_box is None else clamp(first_box, frames[0].shape)
    if clip["class"] == "DRONE" and roi is None:
        return None
    source = (Frame(i, i/30., image) for i, image in enumerate(frames))
    return track_stream(frames[0], source, 30., roi, policy, model, verifier_path, negative_limit)


def score_drone(rows, boxes):
    visible = outputs = within8 = within16 = inside = 0
    for row, box in zip(rows, boxes):
        if box is None:
            continue
        visible += 1
        if row["u_px"] is None:
            continue
        outputs += 1
        error = math.hypot(row["u_px"]-(box[0]+box[2]/2), row["v_px"]-(box[1]+box[3]/2))
        within8 += error <= 8
        within16 += error <= 16
        inside += contains(box, (row["u_px"], row["v_px"]))
    return {"visible": visible, "outputs": outputs, "within8": within8, "within16": within16, "inside": inside}


def evaluate(root, model, output, split, policies, negatives):
    index = load_index(root)
    verifier_path = Path(output)/"verifier.npz"
    results = []
    selected = sorted(name for name, clip in index.items() if clip["split"] == split)
    drones = [name for name in selected if index[name]["class"] == "DRONE"]
    others = [name for name in selected if index[name]["class"] != "DRONE"] if negatives else []
    for name in drones:
        for policy in policies:
            rows = track(root, name, index[name], policy, model, verifier_path)
            if rows is None:
                continue
            results.append({"clip": name, "distance": index[name]["distance"], "class": "DRONE", "policy": policy,
                            **score_drone(rows, index[name]["boxes"])})
            print(json.dumps(results[-1]), flush=True)
    for name in others:
        for policy in ("flow", "target_real"):
            rows = track(root, name, index[name], policy, model, verifier_path)
            results.append({"clip": name, "distance": index[name]["distance"], "class": index[name]["class"], "policy": policy,
                            "frames": len(rows), "false_outputs": sum(row["u_px"] is not None for row in rows)})
            print(json.dumps(results[-1]), flush=True)
    path = Path(output)/f"evaluation_{split}.json"
    path.write_text(json.dumps(results, indent=1), encoding="utf-8")
    print(path)


def read_centers(path):
    """Albl labels: `frame_id x y`, one-indexed. (0, 0) marks a frame without a visible drone."""
    centers = {}
    for line in Path(path).read_text().splitlines():
        parts = line.split()
        if len(parts) == 3 and parts[0][0].isdigit():
            x, y = float(parts[1]), float(parts[2])
            if x > 0 and y > 0:
                centers[int(float(parts[0]))-1] = (x, y)
    return centers


def box_from_center(image, center, window=70):
    """Albl provides centers only, so derive the initial box from local contrast around the label."""
    cx, cy = int(center[0]), int(center[1])
    x0, y0 = max(0, cx-window), max(0, cy-window)
    crop = cv2.cvtColor(image[y0:cy+window, x0:cx+window], cv2.COLOR_BGR2GRAY).astype(np.float32)
    residual = np.abs(crop-cv2.GaussianBlur(crop, (0, 0), 15))
    mask = (residual > max(15., float(residual.mean()+3*residual.std()))).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
    if count > 1:
        distance = [math.hypot(centroids[i][0]+x0-center[0], centroids[i][1]+y0-center[1]) for i in range(1, count)]
        best = 1+int(np.argmin(distance))
        if distance[best-1] < 25:
            x, y, w, h = stats[best][:4]
            return clamp((x+x0-2, y+y0-2, max(12, w+4), max(12, h+4)), image.shape)
    return clamp((center[0]-20, center[1]-20, 40, 40), image.shape)


def albl_stream(path, start, count, fps):
    """Yield Frame objects for one window, decoding lazily because the footage is 1080p."""
    capture = cv2.VideoCapture(str(path))
    try:
        for _ in range(start):
            capture.grab()
        for offset in range(count):
            ok, image = capture.read()
            if not ok:
                return
            yield Frame(offset, offset/fps, image)
    finally:
        capture.release()


def score_centers(rows, centers, start):
    visible = outputs = within8 = within16 = within32 = 0
    for row in rows:
        center = centers.get(start+row["frame"])
        if center is None:
            continue
        visible += 1
        if row["u_px"] is None:
            continue
        outputs += 1
        error = math.hypot(row["u_px"]-center[0], row["v_px"]-center[1])
        within8 += error <= 8
        within16 += error <= 16
        within32 += error <= 32
    return {"visible": visible, "outputs": outputs, "within8": within8, "within16": within16, "within32": within32}


def evaluate_albl(root, model, output, policies, window=300):
    """External test. These clips are never used for fitting; their license is CC BY-NC-SA 4.0."""
    results = []
    tracks = Path("runs/real/tracks")
    tracks.mkdir(parents=True, exist_ok=True)
    for dataset in ("dataset1", "dataset2"):
        for camera in ("cam0", "cam1", "cam3"):
            path = Path(root)/dataset/"video"/camera/f"{camera}.mp4"
            centers = read_centers(Path(root)/dataset/"detections"/f"{camera}.txt")
            capture = cv2.VideoCapture(str(path))
            fps, total = capture.get(cv2.CAP_PROP_FPS), int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            capture.release()
            for fraction in (.25, .5, .75):
                start = next(i for i in range(int(total*fraction), total) if i in centers)
                first = next(albl_stream(path, start, 1, fps)).image
                roi = box_from_center(first, centers[start])
                for policy in policies:
                    rows = track_stream(first, albl_stream(path, start, window, fps), fps, roi, policy, model, Path(output)/"verifier.npz")
                    label = f"{dataset}_{camera}_{start}"
                    (tracks/f"{label}_{policy}.json").write_text(json.dumps([[r["frame"], r["u_px"], r["v_px"], r["status"]] for r in rows]))
                    results.append({"clip": label, "start": start, "roi": list(roi), "policy": policy,
                                    "p95_ms": float(np.percentile([r["processing_ms"] for r in rows], 95)), **score_centers(rows, centers, start)})
                    print(json.dumps(results[-1]), flush=True)
    path = Path(output)/"evaluation_albl.json"
    path.write_text(json.dumps(results, indent=1), encoding="utf-8")
    print(path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["train", "evaluate", "albl"])
    parser.add_argument("--data", type=Path, default=Path("data/svanstrom"))
    parser.add_argument("--model", default="weights/dinov2-small-int8.onnx")
    parser.add_argument("--output", type=Path, default=Path("reports/real"))
    parser.add_argument("--split", default="test", choices=["dev", "test"])
    parser.add_argument("--policies", nargs="+", default=["flow", "first_frame", "target_first_frame", "target_real"])
    parser.add_argument("--no-negatives", action="store_true")
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    cv2.setNumThreads(2)
    if args.command == "train":
        train(args.data, args.model, args.output)
    elif args.command == "albl":
        evaluate_albl(args.data, args.model, args.output, args.policies)
    else:
        evaluate(args.data, args.model, args.output, args.split, args.policies, not args.no_negatives)


if __name__ == "__main__":
    main()
