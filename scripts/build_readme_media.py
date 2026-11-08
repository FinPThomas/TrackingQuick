"""Render README figures by overlaying the published per-frame CSVs on their source clips.

The synthetic clips are deterministic, so these images show exactly the outputs
summarized in reports/v0_2/metrics.json. Nothing is re-tracked here.
"""
import csv
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from trackingquick.synthetic import reference_crop, sequence

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT/"reports/v0_2"
OUTPUT = ROOT/"docs/images"
TWO_NEGATIVE, FLOW = "dino_two_negative", "flow_kf"
GREEN, RED, INK, PAPER = (40, 200, 70), (225, 50, 40), (25, 28, 32), (250, 250, 250)
CONDITION_NAMES = {"clean": "clear sky", "light": "low light", "blur": "motion blur", "occlusion": "full occlusion",
                   "distractor": "bird distractor", "empty": "bird only"}


def load(condition, seed, policy):
    with open(RESULTS/f"{condition}_{seed}_{policy}.csv", newline="") as handle:
        rows = list(csv.DictReader(handle))
    clip = list(sequence(seed, len(rows), condition))
    return [(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), truth, row) for (image, truth), row in zip(clip, rows)]


def output_of(row):
    return None if row["u_px"] == "" else (float(row["u_px"]), float(row["v_px"]))


def error_of(truth, output):
    return None if output is None or not truth["present"] else float(np.hypot(output[0]-truth["u"], output[1]-truth["v"]))


def reticle(image, center, color, radius):
    x, y = round(center[0]), round(center[1])
    cv2.circle(image, (x, y), radius, color, 2, cv2.LINE_AA)
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        cv2.line(image, (x+dx*radius, y+dy*radius), (x+dx*(radius+7), y+dy*(radius+7)), color, 2, cv2.LINE_AA)


def captioned(image, text, size=20):
    """Place a caption bar above an RGB array."""
    font = ImageFont.load_default(size=size)
    bar = size+16
    canvas = Image.new("RGB", (image.shape[1], image.shape[0]+bar), INK)
    canvas.paste(Image.fromarray(image), (0, bar))
    ImageDraw.Draw(canvas).text((10, 7), text, fill=PAPER, font=font)
    return np.asarray(canvas)


def grid(panels, columns, gap=8):
    height, width = panels[0].shape[:2]
    rows = (len(panels)+columns-1)//columns
    sheet = np.full((rows*height+(rows-1)*gap, columns*width+(columns-1)*gap, 3), 255, np.uint8)
    for index, panel in enumerate(panels):
        y, x = (index//columns)*(height+gap), (index % columns)*(width+gap)
        sheet[y:y+height, x:x+width] = panel
    return sheet


def outcome(row, truth):
    output = output_of(row)
    if output is None:
        return f"no output ({row['status']})"
    error = error_of(truth, output)
    return row["status"] if error is None else f"{row['status']}, error {error:.1f} px"


def close_up(frames, index, title):
    """Two-times crop centred on the true target position."""
    image, truth, row = frames[index]
    x0 = int(np.clip(truth["u"]-160, 0, image.shape[1]-320))
    y0 = int(np.clip(truth["v"]-90, 0, image.shape[0]-180))
    crop = cv2.resize(image[y0:y0+180, x0:x0+320], (640, 360), interpolation=cv2.INTER_CUBIC)
    output = output_of(row)
    if output is not None:
        reticle(crop, ((output[0]-x0)*2, (output[1]-y0)*2), GREEN, 38)
    print(f"{title}: frame {index}, {outcome(row, truth)}")
    return captioned(crop, f"{title}: {outcome(row, truth)}")


def save_png(array, name):
    Image.fromarray(array).save(OUTPUT/name, optimize=True)


def demo_gif(frames, name, step=2, scale=.75):
    images = []
    for image, truth, row in frames[::step]:
        image = image.copy()
        output = output_of(row)
        if output is not None:
            reticle(image, output, GREEN, 24)
        size = (int(image.shape[1]*scale), int(image.shape[0]*scale))
        small = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
        labelled = captioned(small, f"{truth['frame']:03d}  {CONDITION_NAMES[truth['condition']]}  |  {outcome(row, truth)}", 15)
        images.append(Image.fromarray(labelled).quantize(colors=96, dither=Image.Dither.NONE))
    images[0].save(OUTPUT/name, save_all=True, append_images=images[1:], duration=round(1000*step/30), loop=0, optimize=True)


def annotated_small(entry, title, color, scale):
    image, truth, row = entry
    image = image.copy()
    output = output_of(row)
    if output is not None:
        reticle(image, output, color, 24)
    small = cv2.resize(image, (int(image.shape[1]*scale), int(image.shape[0]*scale)), interpolation=cv2.INTER_AREA)
    return captioned(small, f"{title}  |  {outcome(row, truth)}", 14)


def comparison_gif(left, right, name, step=2, scale=.625):
    """Same clip, same frames: Flow + Kalman on the left, DINO verification on the right."""
    images = []
    for a, b in zip(left[::step], right[::step]):
        sheet = grid([annotated_small(a, "Flow + Kalman", RED, scale), annotated_small(b, "+ DINO", GREEN, scale)], 2, 6)
        images.append(Image.fromarray(sheet).quantize(colors=96, dither=Image.Dither.NONE))
    images[0].save(OUTPUT/name, save_all=True, append_images=images[1:], duration=round(1000*step/30), loop=0, optimize=True)


def augmentation_sheet(name, seeds=range(100, 108)):
    """The exact DINO reference crops fitted by trackingquick-evaluate."""
    panels = []
    for kind in ("drone", "bird"):
        for seed in seeds:
            crop = cv2.cvtColor(reference_crop(kind, seed), cv2.COLOR_BGR2RGB)
            large = cv2.resize(crop, (192, 192), interpolation=cv2.INTER_NEAREST)
            suffix = ", blur" if seed % 3 == 0 else ""
            panels.append(captioned(large, f"{kind} {seed}{suffix}", 14))
    save_png(grid(panels, len(seeds), 6), name)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    mixed = load("mixed", 42, TWO_NEGATIVE)
    distractor = load("distractor", 42, TWO_NEGATIVE)
    distractor_flow = load("distractor", 42, FLOW)
    empty = load("empty", 42, TWO_NEGATIVE)
    empty_flow = load("empty", 42, FLOW)
    demo_gif(mixed, "tracking_demo.gif")
    save_png(grid([close_up(mixed, 20, "Clear sky"), close_up(mixed, 50, "Low light"),
                   close_up(mixed, 78, "Motion blur"), close_up(mixed, 122, "Full occlusion"),
                   close_up(mixed, 150, "After occlusion"), close_up(mixed, 95, "Motion blur, track dropped")], 3),
             "conditions.png")
    comparison_gif(distractor_flow, distractor, "distractor_comparison.gif")
    comparison_gif(empty_flow, empty, "bird_only_comparison.gif")
    augmentation_sheet("augmentation.png")
    for path in sorted(OUTPUT.iterdir()):
        print(path.name, path.stat().st_size)


if __name__ == "__main__":
    main()
