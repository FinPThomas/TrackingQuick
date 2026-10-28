import numpy as np
import pytest
from trackingquick.filters import PixelKalman, ScaleRangeKalman, triangulate_rectified
from trackingquick.vision import PointFlow, AsyncVerifier
from concurrent.futures import Future


def test_constant_velocity_and_occlusion_uncertainty():
    tracker = PixelKalman([20, 30])
    for i in range(1, 61):
        tracker.predict(1/30)
        tracker.update([20+i, 30+.5*i])
    assert np.linalg.norm(tracker.x[2:]-[30, 15]) < 2
    before = np.trace(tracker.p[:2, :2])
    for _ in range(15):
        tracker.predict(1/30)
    assert np.trace(tracker.p[:2, :2]) > before
    assert np.linalg.norm(tracker.x[:2]-[95, 67.5]) < 2
    assert np.linalg.eigvalsh(tracker.p).min() >= -1e-9


def test_outlier_is_rejected_without_changing_state():
    tracker = PixelKalman([20, 30])
    previous = tracker.x.copy()
    assert not tracker.update([500, -200])
    np.testing.assert_array_equal(previous, tracker.x)


def test_scale_ambiguity_survives_repeated_observations():
    tracker = ScaleRangeKalman(30, 900, width_m=.6, size_log_sigma=.3)
    for _ in range(500):
        tracker.predict(1/30)
        tracker.update(30, pixel_sigma=.1, orientation_sigma=.001)
    z, interval = tracker.estimate()
    assert abs(z-18) < .01
    assert tracker.p[2, 2] == pytest.approx(.09)
    assert interval[1]/interval[0] > np.exp(2*1.96*.299)
    wrong_prior = ScaleRangeKalman(30, 900, width_m=1.2)
    assert wrong_prior.estimate()[0] == pytest.approx(z*2)


def test_triangulation_and_bad_disparity():
    xyz, covariance = triangulate_rectified([365, 198], [320, 198], 900, 1, [320, 180])
    np.testing.assert_allclose(xyz, [1, .4, 20])
    assert np.linalg.eigvalsh(covariance).min() > 0
    with pytest.raises(ValueError):
        triangulate_rectified([365, 198], [364, 198], 900, 1, [320, 180])
    with pytest.raises(ValueError):
        triangulate_rectified([365, 198], [320, 220], 900, 1, [320, 180])


def test_invalid_time_and_featureless_frame():
    tracker = PixelKalman([0, 0])
    with pytest.raises(ValueError):
        tracker.predict(0)
    flow = PointFlow()
    frame = np.full((80, 80, 3), 200, np.uint8)
    flow.seed(frame, [10, 10, 30, 30])
    assert flow.step(frame) is None


def test_async_verdict_rejects_old_tracks_and_expired_frames():
    worker = AsyncVerifier(None, max_age=23)
    for source_generation, now, generation, expected in [(1, 5, 1, (.2, 0)), (1, 5, 2, None), (1, 30, 1, None)]:
        future = Future()
        future.set_result(.2)
        worker.pending = (future, 0, source_generation)
        assert worker.poll(now, generation) == expected
        assert worker.pending is None
    worker.close()
