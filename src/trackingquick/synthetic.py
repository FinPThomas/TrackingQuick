"""Deterministic original synthetic data; no third-party images or labels."""
import cv2
import numpy as np

WIDTH, HEIGHT, FPS, FOCAL = 640, 360, 30, 900.


def draw_target(frame, center, width, angle=0., kind="drone"):
    cx, cy = center
    width = max(8., float(width))
    overlay = np.zeros_like(frame)
    mask = np.zeros(frame.shape[:2], np.uint8)
    color = (35, 38, 42)
    def point(x, y):
        co, si = np.cos(angle), np.sin(angle)
        return int(round(cx + width*(x*co-y*si))), int(round(cy + width*(x*si+y*co)))
    thickness = max(1, int(round(width*.055)))
    if kind == "drone":
        for x in (-.34, .34):
            for y in (-.14, .14):
                cv2.line(mask, point(0, 0), point(x, y), 255, thickness)
                cv2.ellipse(mask, point(x, y), (max(2, int(width*.14)), max(1, int(width*.045))),
                            np.degrees(angle), 0, 360, 255, -1)
        cv2.ellipse(mask, point(0, 0), (max(2, int(width*.11)), max(2, int(width*.09))),
                    np.degrees(angle), 0, 360, 255, -1)
    else:
        pts = np.array([point(-.47, -.15), point(-.17, -.05), point(0, .12),
                        point(.16, -.05), point(.47, -.15)], np.int32)
        cv2.polylines(mask, [pts], False, 255, thickness)
    overlay[mask > 0] = color
    frame[mask > 0] = overlay[mask > 0]
    return mask


def reference_crop(kind, seed, augment=True):
    rng = np.random.default_rng(seed)
    image = np.full((96, 96, 3), rng.integers(140, 225), np.uint8)
    image[:, :, 0] = np.minimum(image[:, :, 0].astype(int)+20, 255)
    draw_target(image, (48+rng.uniform(-4, 4), 48+rng.uniform(-4, 4)),
                rng.uniform(45, 65), rng.uniform(-.25, .25), kind)
    if augment and seed % 3 == 0:
        image = motion_blur(image, 3)
    return image


def motion_blur(image, length):
    kernel = np.zeros((length, length), np.float32)
    kernel[length//2] = 1/length
    return cv2.filter2D(image, -1, kernel)


def sequence(seed=42, frames=180, condition="mixed"):
    """Full frames and truth are returned separately; inference never gets truth."""
    rng = np.random.default_rng(seed)
    sky = np.zeros((HEIGHT, WIDTH, 3), np.uint8)
    for y in range(HEIGHT):
        sky[y, :, :] = (210-int(y*.08), 180+int(y*.04), 145+int(y*.08))
    for _ in range(12):
        cv2.ellipse(sky, tuple(rng.integers([0, 20], [WIDTH, HEIGHT-30])),
                    (int(rng.integers(15, 65)), int(rng.integers(5, 16))), 0, 0, 360, (230, 225, 215), -1)
    for i in range(frames):
        t = i/FPS
        z = 18 + 1.8*t + 1.2*np.sin(t*.9)
        u = 110 + 66*t + 12*np.sin(1.2*t+seed*.1)
        v = 135 + 30*np.sin(t*1.1)
        angle = .18*np.sin(t*2)
        size_m = .6
        width = FOCAL*size_m/z
        frame = sky.copy()
        present = condition != "empty"
        if present:
            draw_target(frame, (u, v), width, angle)
        # Negative test includes a bird large enough to pass the proposal stage.
        draw_target(frame, (520-43*t, 70+10*np.sin(t)), 42 if condition in ("empty", "distractor") else 24, -.12, "bird")
        visible = present
        tag = condition if condition in ("empty", "distractor") else "clean"
        if condition in ("mixed", "light") and 30 <= i < 65:
            frame = np.clip(frame.astype(float)*(.75+.05*np.sin(t)), 0, 255).astype(np.uint8)
            tag = "light"
        if condition in ("mixed", "blur") and 75 <= i < 100:
            frame = motion_blur(frame, 5)
            tag = "blur"
        if condition in ("mixed", "occlusion") and 115 <= i < 130:
            # Full occlusion: remove the rendered target using the background.
            x0, x1 = int(u-width), int(u+width+1)
            y0, y1 = int(v-width), int(v+width+1)
            frame[y0:y1, x0:x1] = sky[y0:y1, x0:x1]
            visible, tag = False, "occlusion"
        yield frame, {"frame": i, "t": t, "u": u, "v": v, "z": z,
                      "visible": visible, "present": present, "condition": tag, "size_m": size_m}
