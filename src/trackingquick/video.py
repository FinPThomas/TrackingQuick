"""Inspect or track a short fixed-camera video with an explicitly supplied target ROI."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time
import cv2
import numpy as np
from .pipeline import Config, Frame, replay
from .vision import DinoVerifier, crop_box
from .evaluate import save_csv


class ForegroundProposals:
    def __init__(self, min_area=8, min_width=4, min_height=3, close_size=3):
        self.background = cv2.createBackgroundSubtractorMOG2(history=90,varThreshold=20,detectShadows=False)
        self.calls = 0
        self.min_area, self.min_width, self.min_height, self.close_size = min_area, min_width, min_height, close_size

    def accept(self, box, frame):
        return self.min_width<=box[2]<=frame.shape[1]/4 and self.min_height<=box[3]<=frame.shape[0]/4

    def __call__(self, frame):
        mask = self.background.apply(frame)
        self.calls += 1
        if self.calls < 3:
            return []
        mask = cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((self.close_size,self.close_size),np.uint8))
        contours,_ = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        boxes = [cv2.boundingRect(c) for c in contours if cv2.contourArea(c)>=self.min_area]
        return [b for b in boxes if self.accept(b,frame)]


class TargetProposals(ForegroundProposals):
    """Motion proposals for small targets, plus a motion-free contrast search near the track.

    The contrast search runs only when motion finds nothing near the track, which
    keeps a hovering target alive after background subtraction has absorbed it.
    Size limits and mask closing scale with the initial box, so a large target is
    not split into propeller-sized fragments and a tiny one is not filtered out.
    """
    def __init__(self, roi=None, radius=60, size_ratio=4.):
        smallest = 2 if roi is None else max(2, int(min(roi[2:])/6))
        super().__init__(min_area=smallest, min_width=smallest, min_height=smallest,
                         close_size=3 if roi is None else max(3, int(min(roi[2:])/4)|1))
        self.max_width = None if roi is None else max(24, roi[2]*size_ratio)
        self.max_height = None if roi is None else max(24, roi[3]*size_ratio)
        self.radius = radius
        self.focus = None

    def set_focus(self, center):
        self.focus = None if center is None else (float(center[0]), float(center[1]))

    def accept(self, box, frame):
        if self.max_width is not None and (box[2]>self.max_width or box[3]>self.max_height):
            return False
        return super().accept(box, frame)

    def contrast(self, frame):
        if self.focus is None:
            return []
        x0, y0 = max(0,int(self.focus[0])-self.radius), max(0,int(self.focus[1])-self.radius)
        x1, y1 = min(frame.shape[1],int(self.focus[0])+self.radius), min(frame.shape[0],int(self.focus[1])+self.radius)
        if x1-x0 < 8 or y1-y0 < 8:
            return []
        gray = cv2.cvtColor(frame[y0:y1,x0:x1],cv2.COLOR_BGR2GRAY).astype(np.float32)
        residual = np.abs(gray-cv2.GaussianBlur(gray,(0,0),6))
        mask = (residual>max(10.,float(residual.mean()+5*residual.std()))).astype(np.uint8)*255
        contours,_ = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        boxes = [cv2.boundingRect(c) for c in contours if cv2.contourArea(c)>=1 or len(c)>=3]
        return [(x+x0,y+y0,w,h) for x,y,w,h in boxes if self.accept((x,y,w,h),frame)]

    def __call__(self, frame):
        motion = super().__call__(frame)
        if self.focus is not None and any(np.hypot(b[0]+b[2]/2-self.focus[0], b[1]+b[3]/2-self.focus[1]) <= self.radius for b in motion):
            return motion
        return motion+self.contrast(frame)


def first_frame_references(verifier, first, roi):
    """Fit a target/background reference pair from one frame; the threshold stays uncalibrated."""
    verifier.positive = verifier.embed(crop_box(first,roi))
    side = max(roi[2:])
    side = min(side,first.shape[0]//3,first.shape[1]//3)
    corners = [(0,0),(first.shape[1]-side,0),(0,first.shape[0]-side),(first.shape[1]-side,first.shape[0]-side)]
    x,y,w,h = roi
    negatives = [first[cy:cy+side,cx:cx+side] for cx,cy in corners
                 if cx+side<=x or cx>=x+w or cy+side<=y or cy>=y+h]
    if not negatives:
        raise ValueError("Cannot form a background reference outside the ROI")
    verifier.negative = np.mean([verifier.embed(im) for im in negatives],axis=0)
    verifier.negative /= np.linalg.norm(verifier.negative)
    verifier.threshold = 0.  # Appearance-seed midpoint, explicitly uncalibrated.


def validate_roi(roi, shape):
    x,y,w,h = roi
    if min(x,y)<0 or min(w,h)<4 or x+w>shape[1] or y+h>shape[0]:
        raise ValueError("ROI must be x y width height, fully inside the original frame, at least 4x4 pixels")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input")
    parser.add_argument("--output",default="runs/video")
    parser.add_argument("--roi",nargs=4,type=int,metavar=("X","Y","W","H"))
    parser.add_argument("--mode",choices=["flow","dino"],default="dino")
    parser.add_argument("--model",default="weights/dinov2-small-int8.onnx")
    parser.add_argument("--focal-px",type=float)
    parser.add_argument("--width-m",type=float)
    parser.add_argument("--fps",type=float)
    parser.add_argument("--max-frames",type=int,default=300)
    parser.add_argument("--negative-limit",type=int,choices=[1,2],default=1,
                        help="1: strict rejection (default); 2: retain identity through one negative, with more false-alarm risk")
    parser.add_argument("--write-video",action="store_true")
    args = parser.parse_args(argv)
    if args.max_frames < 1:
        parser.error("--max-frames must be positive")
    if Path(args.output).resolve() == Path(args.input).resolve():
        parser.error("Output must be a folder separate from the input file")
    output = Path(args.output)
    output.mkdir(parents=True,exist_ok=True)
    cv2.setNumThreads(2)
    started = time.perf_counter()
    cap = cv2.VideoCapture(args.input)
    try:
        ok, first = cap.read()
        if not ok:
            raise ValueError("Could not read video")
        fps = args.fps if args.fps is not None else cap.get(cv2.CAP_PROP_FPS)
        config = Config(fps=fps,focal_px=args.focal_px,width_m=args.width_m,negative_limit=args.negative_limit)
        cv2.imwrite(str(output/"first_frame.jpg"),first)
        if args.roi is None:
            (output/"inspection.json").write_text(json.dumps({"width":first.shape[1],"height":first.shape[0],"fps":fps},indent=2))
            print(f"Inspect {output/'first_frame.jpg'}; rerun with --roi X Y WIDTH HEIGHT in original pixels.")
            return
        validate_roi(args.roi,first.shape)
        verifier = None
        if args.mode == "dino":
            verifier = DinoVerifier(args.model)
            first_frame_references(verifier,first,args.roi)
        fallback = {"frames":0}
        def frames():
            previous = -np.inf
            for index in range(args.max_frames):
                if index == 0:
                    image = first
                else:
                    ok,image = cap.read()
                    if not ok:
                        break
                t = index/fps if args.fps is not None else cap.get(cv2.CAP_PROP_POS_MSEC)/1000
                if not np.isfinite(t) or t<=previous:
                    t = 0. if index==0 else previous+1/fps
                    fallback["frames"] += 1
                previous = t
                yield Frame(index,t,image)
        setup_ms = (time.perf_counter()-started)*1000
        rows,diagnostics = replay(frames(),config,verifier,ForegroundProposals(),initial_box=args.roi,
                                  video_path=output/"annotated.mp4" if args.write_video else None)
        save_csv(output/"track.csv",rows)
        manifest = {"config":asdict(config),"roi":args.roi,"mode":args.mode,"setup_ms":setup_ms,
            "timestamp_fallback_frames":fallback["frames"],"frames":len(rows),"diagnostics":diagnostics,
            "reference": "first-frame user-selected appearance; no class calibration or accuracy claim",
            "distance": "disabled" if args.focal_px is None else "conditional on supplied focal length and physical-size prior",
            "evaluation": "no ground-truth labels supplied"}
        (output/"manifest.json").write_text(json.dumps(manifest,indent=2,allow_nan=False))
        print(f"Wrote {len(rows)} rows to {output/'track.csv'}")
    finally:
        cap.release()


if __name__ == "__main__":
    main()
