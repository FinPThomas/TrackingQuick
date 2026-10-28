"""Preflight or one bounded TAPNext ONNX clip trial; no automatic large download.

The inference branch is unvalidated until the published checkpoint is available.
Input conventions follow Annolid's TAPNext export, not the streaming Torch API.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time
import urllib.request

URL = "https://github.com/healthonrails/annolid/releases/download/v1.6.6/tapnext.onnx"
SHA256 = "4fca0951802f0b745de254930c880938a74bf8b54b10786fc68d0ab4ba5c5300"
SIZE = 984253311
RESERVE = 700_000_000


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(4*1024*1024):
            h.update(chunk)
    return h.hexdigest()


def run_clip(model):
    import cv2
    import numpy as np
    import onnxruntime as ort
    from trackingquick.synthetic import sequence
    options = ort.SessionOptions()
    options.intra_op_num_threads, options.inter_op_num_threads = 2, 1
    start = time.perf_counter()
    session = ort.InferenceSession(str(model),sess_options=options,providers=["CPUExecutionProvider"])
    loaded = (time.perf_counter()-start)*1000
    inputs = session.get_inputs()
    video = next((v for v in inputs if len(v.shape)==5),None)
    query = next((v for v in inputs if len(v.shape)==3 and v.shape[-1]==3),None)
    if len(inputs)!=2 or video is None or query is None or video.shape[-1]!=3:
        raise ValueError("This trial supports only the documented [B,T,H,W,C] video and [B,Q,3] query export")
    t = video.shape[1] if isinstance(video.shape[1],int) else 8
    h = video.shape[2] if isinstance(video.shape[2],int) else 256
    w = video.shape[3] if isinstance(video.shape[3],int) else 256
    q = query.shape[1] if isinstance(query.shape[1],int) else 1
    if not (1<=t<=16 and h<=512 and w<=512 and 1<=q<=256):
        raise ValueError(f"Export exceeds bounded trial budget: T={t}, H={h}, W={w}, Q={q}")
    data = list(sequence(42,t,"clean"))
    source_h,source_w = data[0][0].shape[:2]
    clip = np.stack([cv2.resize(cv2.cvtColor(im,cv2.COLOR_BGR2RGB),(w,h)) for im,_ in data]).astype(np.float32)/255
    first = data[0][1]  # A manually supplied first-frame point is permitted for point tracking.
    queries = np.tile(np.array([0.,first["v"]*h/source_h,first["u"]*w/source_w],np.float32),(1,q,1))
    start = time.perf_counter()
    outputs = session.run(None,{video.name:clip[None],query.name:queries})
    elapsed = (time.perf_counter()-start)*1000
    tracks = next((o for o in outputs if o.ndim==4 and o.shape[-1]==2),None)
    if tracks is None or tracks.shape[1] < t:
        raise ValueError("Unexpected TAPNext track output shape")
    xy = tracks[0,:t,0,::-1]*np.array([source_w/w,source_h/h])
    truth = np.array([[g["u"],g["v"]] for _,g in data])
    return {"session_load_ms":loaded,"cold_clip_inference_ms":elapsed,"frames":t,"query_count":q,
            "center_rmse_px":float(np.sqrt(np.mean(np.sum((xy-truth)**2,axis=1)))),
            "input_shape":list(clip[None].shape),"output_xy":xy.tolist(),
            "scope":"one cold synthetic clip; first-frame point supplied; not streaming latency or drone detection"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model",default="weights/tapnext.onnx")
    parser.add_argument("--output",default="reports/v0_2/tapnext_preflight.json")
    parser.add_argument("--download",action="store_true",help="Explicitly request the 984 MB download if storage permits")
    parser.add_argument("--run",action="store_true")
    args = parser.parse_args()
    model = Path(args.model).resolve()
    parent = model.parent
    while not parent.exists():
        parent = parent.parent
    free = shutil.disk_usage(parent).free
    existing = model.is_file()
    info = {"artifact_url":URL,"artifact_bytes":SIZE,"sha256":SHA256,"free_disk_bytes":free,
            "required_download_free_bytes":SIZE+RESERVE,"model_present":existing,
            "download_attempted":False,"inference_executed":False,"gpu_requirement":"unmeasured",
            "source":"https://annolid.com/portal/tapnext/"}
    status = "model_present" if existing else "missing_model"
    if not existing and free < SIZE+RESERVE:
        status = "blocked_storage"
    try:
        if args.download and not existing and status != "blocked_storage":
            info["download_attempted"] = True
            model.parent.mkdir(parents=True,exist_ok=True)
            part = model.with_suffix(".partial")
            try:
                with urllib.request.urlopen(URL,timeout=60) as response,part.open("wb") as target:
                    count = 0
                    while chunk := response.read(4*1024*1024):
                        count += len(chunk)
                        if count > SIZE:
                            raise ValueError("Unexpected TAPNext artifact size")
                        target.write(chunk)
                if part.stat().st_size != SIZE or digest(part) != SHA256:
                    raise ValueError("TAPNext checksum mismatch")
                part.replace(model)
                existing,status = True,"model_present"
            finally:
                part.unlink(missing_ok=True)
        if args.run and existing:
            if digest(model) != SHA256:
                raise ValueError("Unexpected checkpoint checksum")
            info["trial"] = run_clip(model)
            info["inference_executed"],status = True,"completed"
    except Exception as exc:
        status,info["error"] = "failed",str(exc)
    info["status"] = status
    info["model_present"] = model.is_file()
    output = Path(args.output)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(info,indent=2),encoding="utf-8")
    print(json.dumps(info,indent=2))
    return 0 if status == "completed" or not args.run else 2


if __name__ == "__main__":
    raise SystemExit(main())
