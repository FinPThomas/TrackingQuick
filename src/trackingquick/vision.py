"""CPU DINOv2 appearance verification and forward/backward point flow."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import time
import cv2
import numpy as np
from PIL import Image
import onnxruntime as ort


def proposals(frame):
    """Cheap dark-object proposals for the synthetic sky. Not a trained YOLO."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mask = (gray < 100).astype(np.uint8)*255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = [cv2.boundingRect(c) for c in contours if cv2.contourArea(c) >= 12]
    return [(x, y, w, h) for x, y, w, h in boxes if 8 <= w <= 100 and 3 <= h <= 60]


def crop_box(image, box, min_side=0):
    x, y, w, h = box
    side = max(max(w, h)*1.65, min_side)
    cx, cy = x+w/2, y+h/2
    x0, y0 = max(0, int(cx-side/2)), max(0, int(cy-side/2))
    x1, y1 = min(image.shape[1], int(cx+side/2)+1), min(image.shape[0], int(cy+side/2)+1)
    return image[y0:y1, x0:x1]


class DinoVerifier:
    def __init__(self, path, threads=2, crop_size=112):
        if not Path(path).is_file():
            raise FileNotFoundError("Run python scripts/fetch_dino.py first")
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        self.session = ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        if crop_size < 28 or crop_size % 14:
            raise ValueError("DINO input size must be a positive multiple of patch size 14")
        self.crop_size = crop_size
        self.times = []
        self.threshold = 0.

    def embed(self, crop):
        start = time.perf_counter()
        if crop.size == 0:
            raise ValueError("Empty DINO crop")
        image = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        w, h = image.size
        # Preserve the published resize/crop ratio; 112 is a deliberate speed ablation.
        scale = (self.crop_size*256/224)/min(w, h)
        image = image.resize((int(w*scale), int(h*scale)), Image.Resampling.BICUBIC)
        w, h = image.size
        side = self.crop_size
        image = image.crop(((w-side)//2, (h-side)//2, (w-side)//2+side, (h-side)//2+side))
        pixels = np.asarray(image).astype(np.float32)/255
        pixels = (pixels-np.array([.485, .456, .406], np.float32))/np.array([.229, .224, .225], np.float32)
        outputs = self.session.run(None, {self.input_name: pixels.transpose(2, 0, 1)[None]})
        # HF DINOv2 export provides final tokens; first token is normalized CLS.
        tokens = next((o for o in outputs if o.ndim == 3), None)
        feature = tokens[0, 0] if tokens is not None else outputs[0][0]
        feature = feature/(np.linalg.norm(feature)+1e-12)
        self.times.append((time.perf_counter()-start)*1000)
        return feature

    def fit(self, train, validation):
        features = [(self.embed(im), int(label)) for im, label in train]
        self.positive = np.stack([f for f, l in features if l]).mean(axis=0)
        self.negative = np.stack([f for f, l in features if not l]).mean(axis=0)
        self.positive /= np.linalg.norm(self.positive)
        self.negative /= np.linalg.norm(self.negative)
        scores = np.array([self.score(im) for im, _ in validation])
        labels = np.array([label for _, label in validation], bool)
        candidates = np.r_[scores.min()-1e-5, (np.sort(scores)[1:]+np.sort(scores)[:-1])/2, scores.max()+1e-5]
        accuracy = [np.mean((scores >= value) == labels) for value in candidates]
        self.threshold = float(candidates[int(np.argmax(accuracy))])
        return {"threshold": self.threshold, "validation_accuracy": float(max(accuracy)),
                "train_count": len(train), "validation_count": len(validation),
                "score_definition": "cosine_to_positive_prototype - cosine_to_negative_prototype; not a probability"}

    def score(self, image):
        feature = self.embed(image)
        return float(feature @ self.positive-feature @ self.negative)


class PointFlow:
    def __init__(self):
        self.gray = self.points = self.center = None
        self.age = 0

    def seed(self, frame, box):
        self.gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        x, y, w, h = map(int, box)
        mask = np.zeros_like(self.gray)
        mask[max(0,y):y+h, max(0,x):x+w] = 255
        self.points = cv2.goodFeaturesToTrack(self.gray, 20, .015, 2, mask=mask, blockSize=3)
        self.center = np.array([x+w/2, y+h/2], float)
        self.age = 0

    def step(self, frame):
        if self.points is None or len(self.points) < 3:
            return None
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        current, valid, error = cv2.calcOpticalFlowPyrLK(self.gray, gray, self.points, None, winSize=(15, 15), maxLevel=2)
        if current is None:
            self.points = None
            return None
        back, valid_back, _ = cv2.calcOpticalFlowPyrLK(gray, self.gray, current, None, winSize=(15, 15), maxLevel=2)
        if back is None:
            self.points = None
            return None
        fb = np.linalg.norm((back-self.points).reshape(-1, 2), axis=1)
        keep = (valid.ravel()>0) & (valid_back.ravel()>0) & (fb < 1.) & (error.ravel() < 25)
        if np.sum(keep) < 3:
            self.points = None
            return None
        displacement = (current-self.points).reshape(-1, 2)[keep]
        delta = np.median(displacement, axis=0)
        scatter = np.median(np.linalg.norm(displacement-delta, axis=1))
        self.center += delta
        self.points, self.gray = current[keep], gray
        self.age += 1
        return self.center.copy(), 2.5+.5*self.age+float(scatter)


class AsyncVerifier:
    """One crop in flight. Stamps and max_age must use the same units.

    The legacy experiment uses frame indices; the v0.2 pipeline uses seconds.
    """
    def __init__(self, verifier, max_age=23):
        self.verifier = verifier
        self.max_age = max_age
        self.pool = ThreadPoolExecutor(max_workers=1)
        self.pending = None
        self.stats = dict(submitted=0, busy=0, completed=0, stale=0, wrong_generation=0)

    def submit(self, crop, frame, generation):
        if self.pending is not None:
            self.stats["busy"] += 1
            return False
        self.pending = (self.pool.submit(self.verifier.score, crop.copy()), frame, generation)
        self.stats["submitted"] += 1
        return True

    def poll(self, frame, generation):
        if self.pending is None or not self.pending[0].done():
            return None
        future, source_frame, source_generation = self.pending
        self.pending = None
        score = future.result()  # Worker errors are reported; never silently ignored.
        self.stats["completed"] += 1
        if source_generation != generation:
            self.stats["wrong_generation"] += 1
            return None
        if frame-source_frame > self.max_age:
            self.stats["stale"] += 1
            return None
        return score, source_frame

    def close(self):
        self.pool.shutdown(wait=True, cancel_futures=True)
