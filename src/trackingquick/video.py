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
    def __init__(self):
        self.background = cv2.createBackgroundSubtractorMOG2(history=90,varThreshold=20,detectShadows=False)
        self.calls = 0

    def __call__(self, frame):
        mask = self.background.apply(frame)
        self.calls += 1
        if self.calls < 3:
            return []
        mask = cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
        contours,_ = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        boxes = [cv2.boundingRect(c) for c in contours if cv2.contourArea(c)>=8]
        return [b for b in boxes if 4<=b[2]<=frame.shape[1]/4 and 3<=b[3]<=frame.shape[0]/4]


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
            verifier.positive = verifier.embed(crop_box(first,args.roi))
            side = max(args.roi[2:])
            side = min(side,first.shape[0]//3,first.shape[1]//3)
            corners = [(0,0),(first.shape[1]-side,0),(0,first.shape[0]-side),(first.shape[1]-side,first.shape[0]-side)]
            x,y,w,h = args.roi
            negatives = [first[cy:cy+side,cx:cx+side] for cx,cy in corners
                         if cx+side<=x or cx>=x+w or cy+side<=y or cy>=y+h]
            if not negatives:
                raise ValueError("Cannot form a background reference outside the ROI")
            verifier.negative = np.mean([verifier.embed(im) for im in negatives],axis=0)
            verifier.negative /= np.linalg.norm(verifier.negative)
            verifier.threshold = 0.  # Appearance-seed midpoint, explicitly uncalibrated.
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
