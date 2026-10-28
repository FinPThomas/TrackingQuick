import csv
from concurrent.futures import Future
import cv2
import numpy as np
import pytest
from trackingquick.pipeline import Config, Frame, IdentityState, replay
from trackingquick.vision import AsyncVerifier
from trackingquick.evaluate import summarize
from trackingquick.video import main as video_main
from trackingquick.synthetic import sequence
from trackingquick.vision import proposals


def test_identity_requires_two_failures_without_extending_positive_lifetime():
    identity = IdentityState(.1,ttl_s=1.5,negative_limit=2)
    assert not identity.apply(.2,0.)
    assert not identity.apply(-.2,.5)
    assert identity.valid(1.)
    assert identity.last_positive == 0.
    assert not identity.valid(1.6)
    assert identity.apply(-.1,1.0)
    assert not identity.valid(1.1)


def test_async_queue_is_bounded_and_uses_time_units():
    worker = AsyncVerifier(None,max_age=.75)
    future = Future()
    worker.pending = (future,0.,1)
    assert not worker.submit(np.zeros((4,4,3)),.1,1)
    future.set_result(.5)
    assert worker.poll(.8,1) is None
    assert worker.stats["stale"] == 1 and worker.stats["busy"] == 1
    worker.close()


def test_source_timing_and_no_invented_distance():
    image = np.zeros((60,80,3),np.uint8)
    packets = [Frame(i,t,image) for i,t in enumerate([0.,.1,.3])]
    rows,_ = replay(packets,Config(),proposal_fn=lambda _: [(20,20,10,10)],use_flow=False,paced=False)
    assert rows[-1]["dt_s"] == pytest.approx(.2)
    assert all(r["depth_m"] is None and r["x_m"] is None for r in rows)
    with pytest.raises(ValueError):
        replay([Frame(0,1.,image),Frame(1,1.,image)],Config(),paced=False)
    with pytest.raises(ValueError):
        Config(focal_px=900)


def test_missing_outputs_reduce_all_frame_success():
    rows = [{"u_px":u,"v_px":0 if u is not None else None,"processing_ms":1} for u in (0,None,10)]
    truth = [{"u":0,"v":0,"visible":True,"condition":"clean"} for _ in rows]
    metrics = summarize(rows,truth)
    assert metrics["coverage"] == pytest.approx(2/3)
    assert metrics["success_8px_all_visible"] == pytest.approx(1/3)
    assert metrics["rmse_reported_visible_px"] == pytest.approx(np.sqrt(50))
    for g in truth:
        g.update(visible=False,present=False,condition="empty")
    metrics = summarize(rows,truth)
    assert metrics["empty_false_alarm_fraction"] == pytest.approx(2/3)
    assert metrics["success_8px_all_visible"] is None


def test_target_free_stress_has_a_detector_visible_distractor():
    image,truth = next(sequence(42,1,"empty"))
    assert not truth["present"] and not truth["visible"]
    assert len(proposals(image)) > 0


def test_video_runner_uses_input_dimensions_and_emits_blank_depth(tmp_path):
    path = tmp_path/"input.avi"
    writer = cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*"MJPG"),15,(80,60))
    assert writer.isOpened()
    for i in range(8):
        image = np.full((60,80,3),200,np.uint8)
        cv2.rectangle(image,(20+i,20),(30+i,30),(20,20,20),-1)
        writer.write(image)
    writer.release()
    output = tmp_path/"result"
    video_main([str(path),"--output",str(output),"--mode","flow","--roi","20","20","11","11","--max-frames","8"])
    with (output/"track.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 8
    assert all(r["depth_m"] == "" and r["distance_source"] == "unavailable" for r in rows)
    assert float(rows[-1]["time_s"])-float(rows[0]["time_s"]) == pytest.approx(7/15,abs=.02)
