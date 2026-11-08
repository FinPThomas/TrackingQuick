"""Download the Albl et al. multi-view drone footage used as an external test.

Source: https://github.com/CenekAlbl/drone-tracking-datasets (CC BY-NC-SA 4.0).
Camera 2 is omitted because it is distributed as a multi-part archive.
"""
import argparse
from pathlib import Path
import shutil
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
RAW = "https://raw.githubusercontent.com/CenekAlbl/drone-tracking-datasets/master"


def fetch(url, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        urllib.request.urlretrieve(url, path)


def extract(archive, output):
    output.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.infolist():
            if member.is_dir():
                continue
            target = output/Path(member.filename).name
            if not target.exists():
                with bundle.open(member) as source, open(target, "wb") as sink:
                    shutil.copyfileobj(source, sink)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT/"data/albl")
    parser.add_argument("--datasets", nargs="+", default=["dataset1", "dataset2"])
    parser.add_argument("--cameras", nargs="+", default=["cam0", "cam1", "cam3"])
    args = parser.parse_args(argv)
    fetch(f"{RAW}/LICENSE", args.output/"LICENSE")
    for dataset in args.datasets:
        for camera in args.cameras:
            fetch(f"{RAW}/{dataset}/detections/{camera}.txt", args.output/dataset/"detections"/f"{camera}.txt")
            archive = args.output/dataset/"video"/f"{camera}.zip"
            fetch(f"{RAW}/{dataset}/{camera}/{camera}.zip", archive)
            extract(archive, args.output/dataset/"video"/camera)
            print(dataset, camera, "ready", flush=True)


if __name__ == "__main__":
    main()
