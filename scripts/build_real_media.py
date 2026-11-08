"""Render README figures for the real-footage evaluation.

Clips are chosen by rule from reports/real, not by appearance: the median clip for
examples and the lowest-scoring clips for failures. Svanstrom examples are re-run
because their tracks are not stored, so their per-frame output can differ slightly
from the recorded evaluation. Albl figures use the stored tracks.
"""
import json
from pathlib import Path
import sys
import cv2
import numpy as np
from PIL import Image
from build_readme_media import GREEN, OUTPUT, RED, captioned, grid, reticle, save_png
from trackingquick.pipeline import Frame
from trackingquick.realdata import albl_stream, box_from_center, clamp, clip_frames, load_index, read_centers, track_stream

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT/"data/svanstrom"
ALBL = ROOT/"data/albl"
REPORT = ROOT/"reports/real"
MODEL = str(ROOT/"weights/dinov2-small-int8.onnx")
VERIFIER = REPORT/"verifier.npz"
WHITE = (255, 255, 255)


def third_party(clip):
    return "youtube" in str(clip["source"]).lower()


def rerun(name, clip, policy):
    frames = list(clip_frames(DATA/f"{name}.mp4"))
    first = clip["boxes"][0] if clip["class"] == "DRONE" else None
    roi = None if first is None else clamp(first, frames[0].shape)
    stream = (Frame(i, i/30., image) for i, image in enumerate(frames))
    rows = track_stream(frames[0], stream, 30., roi, policy, MODEL, VERIFIER)
    return frames, [[r["frame"], r["u_px"], r["v_px"], r["status"]] for r in rows]


def rgb(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def overlay(image, row, box, color, radius=24):
    image = rgb(image)
    if box is not None:
        x, y, w, h = map(round, box)
        cv2.rectangle(image, (x, y), (x+w, y+h), WHITE, 1, cv2.LINE_AA)
    if row[1] is not None:
        reticle(image, (row[1], row[2]), color, radius)
    return image


def status(row):
    return "no output" if row[1] is None else row[3]


def save_gif(images, name, step):
    frames = [Image.fromarray(image).quantize(colors=128, dither=Image.Dither.NONE) for image in images]
    frames[0].save(OUTPUT/name, save_all=True, append_images=frames[1:], duration=round(1000*step/30), loop=0, optimize=True)


def by_rate(entries, numerator, denominator):
    return sorted(entries, key=lambda e: (e[numerator]/max(1, e[denominator]), e["clip"]))


def median_of(entries, numerator, denominator):
    return by_rate(entries, numerator, denominator)[len(entries)//2]


def hero(index, evaluation):
    entries = [e for e in evaluation if e["policy"] == "target_real" and e["class"] == "DRONE" and not third_party(index[e["clip"]])]
    chosen = median_of(entries, "within16", "visible")
    name, clip = chosen["clip"], index[chosen["clip"]]
    frames, rows = rerun(name, clip, "target_real")
    images = []
    for image, row, box in list(zip(frames, rows, clip["boxes"]))[::3]:
        small = cv2.resize(overlay(image, row, box, GREEN), (480, 384), interpolation=cv2.INTER_AREA)
        images.append(captioned(small, f"{row[0]:03d}  Trained verifier  |  {status(row)}", 15))
    save_gif(images, "real_tracking_demo.gif", 3)
    print("hero", name, "recorded within16", chosen["within16"], "of", chosen["visible"])


def bird(index, evaluation):
    entries = [e for e in evaluation if e["policy"] == "target_real" and e["class"] == "BIRD" and not third_party(index[e["clip"]])]
    chosen = median_of(entries, "false_outputs", "frames")
    name, clip = chosen["clip"], index[chosen["clip"]]
    frames, flow = rerun(name, clip, "flow")
    _, trained = rerun(name, clip, "target_real")
    images = []
    for image, a, b in list(zip(frames, flow, trained))[::2]:
        panels = []
        for row, title, color in ((a, "Flow + Kalman", RED), (b, "Trained verifier", GREEN)):
            small = cv2.resize(overlay(image, row, None, color), (400, 320), interpolation=cv2.INTER_AREA)
            panels.append(captioned(small, f"{title}  |  {status(row)}", 14))
        images.append(grid(panels, 2, 6))
    save_gif(images, "real_bird_comparison.gif", 2)
    print("bird", name, "recorded false outputs", chosen["false_outputs"], "of", chosen["frames"],
          "this run", sum(r[1] is not None for r in flow), sum(r[1] is not None for r in trained))


def albl_parts(label):
    dataset, camera, start = label.split("_")
    return dataset, camera, int(start), ALBL/dataset/"video"/camera/f"{camera}.mp4"


def stored(label, policy):
    return json.loads((ROOT/"runs/real/tracks"/f"{label}_{policy}.json").read_text())


def follow_crop(image, center, width, height):
    x0 = int(np.clip(center[0]-width/2, 0, image.shape[1]-width))
    y0 = int(np.clip(center[1]-height/2, 0, image.shape[0]-height))
    return (x0, y0), image[y0:y0+height, x0:x0+width]


def albl_panel(image, center, row, title, color):
    (x0, y0), crop = follow_crop(rgb(image), center, 400, 300)
    crop = crop.copy()
    cv2.circle(crop, (round(center[0]-x0), round(center[1]-y0)), 3, WHITE, 1, cv2.LINE_AA)
    if row[1] is not None:
        reticle(crop, (row[1]-x0, row[2]-y0), color, 26)
    return captioned(crop, f"{title}  |  {status(row)}", 14)


def albl_comparison(evaluation):
    entries = [e for e in evaluation if e["policy"] == "target_real"]
    chosen = median_of(entries, "within16", "visible")
    label = chosen["clip"]
    dataset, camera, start, path = albl_parts(label)
    centers = read_centers(ALBL/dataset/"detections"/f"{camera}.txt")
    first, trained = stored(label, "first_frame"), stored(label, "target_real")
    images, last = [], None
    for packet, a, b in zip(albl_stream(path, start, 300, 30.), first, trained):
        last = centers.get(start+packet.index, last)
        if last is None or packet.index % 4:
            continue
        panels = [albl_panel(packet.image, last, a, "First-frame DINO", GREEN), albl_panel(packet.image, last, b, "Trained verifier", GREEN)]
        images.append(cv2.resize(grid(panels, 2, 6), None, fx=.75, fy=.75, interpolation=cv2.INTER_AREA))
    save_gif(images, "real_albl_comparison.gif", 4)
    print("albl", label, "within16", chosen["within16"], "of", chosen["visible"])


def failures(index, test, albl):
    panels = []
    entries = [e for e in test if e["policy"] == "target_real" and e["class"] == "DRONE" and not third_party(index[e["clip"]])]
    for chosen in by_rate(entries, "within16", "visible")[:2]:
        name, clip = chosen["clip"], index[chosen["clip"]]
        frames, rows = rerun(name, clip, "target_real")
        middle = next(i for i in range(len(frames)//2, len(frames)) if clip["boxes"][i])
        box = clip["boxes"][middle]
        (x0, y0), crop = follow_crop(rgb(frames[middle]), (box[0]+box[2]/2, box[1]+box[3]/2), 320, 240)
        crop = cv2.resize(crop, (640, 480), interpolation=cv2.INTER_CUBIC)
        cv2.rectangle(crop, (round((box[0]-x0)*2), round((box[1]-y0)*2)), (round((box[0]+box[2]-x0)*2), round((box[1]+box[3]-y0)*2)), WHITE, 1, cv2.LINE_AA)
        if rows[middle][1] is not None:
            reticle(crop, ((rows[middle][1]-x0)*2, (rows[middle][2]-y0)*2), GREEN, 30)
        panels.append(captioned(crop, f"Distant drone, clip {name[-3:]}: {status(rows[middle])}"))
        print("failure", name, chosen["within16"], chosen["visible"])
    entries = [e for e in albl if e["policy"] == "target_real" and e["visible"] >= 100]
    for chosen in by_rate(entries, "within16", "visible")[:2]:
        label = chosen["clip"]
        dataset, camera, start, path = albl_parts(label)
        centers = read_centers(ALBL/dataset/"detections"/f"{camera}.txt")
        rows = stored(label, "target_real")
        middle = next(i for i in range(150, len(rows)) if start+i in centers)
        center = centers[start+middle]
        packet = next(p for p in albl_stream(path, start, middle+1, 30.) if p.index == middle)
        (x0, y0), crop = follow_crop(rgb(packet.image), center, 320, 240)
        crop = cv2.resize(crop, (640, 480), interpolation=cv2.INTER_CUBIC)
        cv2.circle(crop, (round((center[0]-x0)*2), round((center[1]-y0)*2)), 6, WHITE, 1, cv2.LINE_AA)
        row = rows[middle]
        if row[1] is not None:
            reticle(crop, ((row[1]-x0)*2, (row[2]-y0)*2), GREEN, 30)
        panels.append(captioned(crop, f"External, {dataset[-1]} {camera}: {status(row)}"))
        print("failure", label, chosen["within16"], chosen["visible"])
    save_png(grid(panels, 2), "real_failures.png")


def main():
    index = load_index(DATA)
    test = json.loads((REPORT/"evaluation_test.json").read_text())
    albl = json.loads((REPORT/"evaluation_albl.json").read_text())
    hero(index, test)
    bird(index, test)
    albl_comparison(albl)
    failures(index, test, albl)
    for path in sorted(OUTPUT.glob("real_*")):
        print(path.name, path.stat().st_size)


if __name__ == "__main__":
    sys.exit(main())
