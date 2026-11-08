"""Download the visible-light clips of the CC0 Svanstrom drone dataset and index their labels.

Source: https://github.com/DroneDetectionThesis/Drone-detection-dataset (CC0-1.0).
Clips are split by block within each class and distance bin, because neighbouring
clip numbers can come from the same recording session.
"""
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import re
import urllib.request
import numpy as np
import openpyxl
from matio import load_from_mat

ROOT = Path(__file__).resolve().parents[1]
REPOSITORY = "DroneDetectionThesis/Drone-detection-dataset"
RAW = f"https://raw.githubusercontent.com/{REPOSITORY}/master/Data"
TREE = f"https://api.github.com/repos/{REPOSITORY}/git/trees/master?recursive=1"
PATTERN = re.compile(r"Data/Video_V/(V_([A-Z]+)_(\d+)(?:\.mp4|_LABELS\.mat))$")
# Inspected while diagnosing failures, so they are kept out of training and the held-out test.
DEVELOPMENT = {f"V_DRONE_{n}" for n in ("001", "023", "046", "069", "092", "114")}


def download(output):
    with urllib.request.urlopen(TREE) as response:
        tree = json.load(response)["tree"]
    names = [m.group(1) for m in (PATTERN.match(item["path"]) for item in tree) if m]
    missing = [name for name in names if not (output/name).exists()]
    def fetch(name):
        urllib.request.urlretrieve(f"{RAW}/Video_V/{name}", output/name)
    with ThreadPoolExecutor(8) as pool:
        list(pool.map(fetch, missing))
    if not (output/"description.xlsx").exists():
        urllib.request.urlretrieve(f"{RAW}/Video_dataset_description.xlsx", output/"description.xlsx")
    print(f"{len(names)} files listed, {len(missing)} downloaded")


def read_description(path):
    sheet = openpyxl.load_workbook(path, read_only=True, data_only=True).worksheets[0]
    described = {}
    for sensor, kind, number, distance, _, source in sheet.iter_rows(min_row=2, values_only=True):
        if sensor == "V":
            described[f"V_{kind}_{int(number):03d}"] = (distance.strip(), source)
    return described


def read_boxes(path, kind):
    data = load_from_mat(path)["gTruth"].properties["any"]["LabelData"][0, 0]
    boxes = []
    for cell in data[kind].ravel():
        values = np.zeros(0) if cell is None else np.asarray(cell)
        if values.dtype.names:  # Some clips store labels as a struct with a Position field.
            values = values["Position"].ravel()[0] if values.size else np.zeros(0)
        values = np.asarray(values, float).ravel()
        # MATLAB boxes are one-indexed.
        boxes.append([values[0]-1, values[1]-1, values[2], values[3]] if values.size >= 4 else None)
    return boxes


def assign_splits(clips):
    groups = defaultdict(list)
    for name, clip in clips.items():
        groups[clip["class"], clip["distance"]].append(name)
    for names in groups.values():
        names.sort(key=lambda name: clips[name]["number"])
        held_out = max(1, round(len(names)*.2)) if len(names) >= 3 else 0
        for index, name in enumerate(names):
            remaining = len(names)-index
            clips[name]["split"] = ("dev" if name in DEVELOPMENT else "test" if remaining <= held_out
                                    else "val" if remaining <= 2*held_out else "train")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT/"data/svanstrom")
    parser.add_argument("--skip-download", action="store_true")
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    if not args.skip_download:
        download(args.output)
    described = read_description(args.output/"description.xlsx")
    clips = {}
    for path in sorted(args.output.glob("V_*.mp4")):
        kind, number = path.stem.split("_")[1:]
        distance, source = described[path.stem]
        clips[path.stem] = {"class": kind, "number": int(number), "distance": distance, "source": source,
                            "boxes": read_boxes(args.output/f"{path.stem}_LABELS.mat", kind)}
    assign_splits(clips)
    (args.output/"index.json").write_text(json.dumps(clips), encoding="utf-8")
    counts = defaultdict(int)
    for clip in clips.values():
        counts[clip["split"], clip["class"]] += 1
    print(dict(sorted(counts.items())))
    print("sources:", sorted({str(clip["source"]) for clip in clips.values()})[:20])


if __name__ == "__main__":
    main()
