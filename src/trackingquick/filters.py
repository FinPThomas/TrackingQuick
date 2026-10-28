"""Timestamp-aware filters with innovation gates and explicit scale uncertainty."""
import numpy as np


def correction(x, p, z, h, r, gate):
    residual = np.asarray(z) - h @ x
    s = h @ p @ h.T + r
    distance = float(residual @ np.linalg.solve(s, residual))
    if distance > gate or not np.isfinite(distance):
        return x, p, False
    gain = np.linalg.solve(s, h @ p).T
    updated = x + gain @ residual
    a = np.eye(len(x)) - gain @ h
    # Joseph update retains positive semidefiniteness under roundoff.
    covariance = a @ p @ a.T + gain @ r @ gain.T
    return updated, (covariance + covariance.T) / 2, True


class PixelKalman:
    def __init__(self, center, acceleration_sigma=45.0):
        self.x = np.r_[np.asarray(center, float), 0., 0.]
        self.p = np.diag([9., 9., 900., 900.])
        self.acceleration_sigma = acceleration_sigma

    def predict(self, dt):
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be finite and positive")
        f = np.eye(4)
        f[:2, 2:] = np.eye(2) * dt
        g = np.vstack((np.eye(2) * dt**2 / 2, np.eye(2) * dt))
        self.x = f @ self.x
        self.p = f @ self.p @ f.T + g @ g.T * self.acceleration_sigma**2

    def update(self, center, sigma=2., gate=13.82):
        h = np.eye(4)[:2]
        self.x, self.p, accepted = correction(
            self.x, self.p, center, h, np.eye(2) * sigma**2, gate)
        return accepted

    def distance(self, center, sigma=2.):
        d = np.asarray(center) - self.x[:2]
        return float(d @ np.linalg.solve(self.p[:2, :2] + np.eye(2)*sigma**2, d))


class ScaleRangeKalman:
    """q = log(Z / W), q_rate, L = log(W). L stays an uncertain size prior.

    Image width measures q only. Repeated frames cannot eliminate uncertainty
    in physical size. Z is optical-axis depth, not Euclidean camera range.
    """
    def __init__(self, pixel_width, focal_px, width_m=.6, size_log_sigma=.30):
        if min(pixel_width, focal_px, width_m, size_log_sigma) <= 0:
            raise ValueError("Width, focal length and size uncertainty must be positive")
        self.focal_px = focal_px
        self.x = np.array([np.log(focal_px/pixel_width), 0., np.log(width_m)])
        self.p = np.diag([.08**2, .4**2, size_log_sigma**2])

    def predict(self, dt):
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be finite and positive")
        f = np.eye(3)
        f[0, 1] = dt
        g = np.array([dt**2/2, dt, 0.])
        self.x = f @ self.x
        self.p = f @ self.p @ f.T + np.outer(g, g)*.5**2

    def update(self, pixel_width, pixel_sigma=2., orientation_sigma=.12):
        if not np.isfinite(pixel_width) or pixel_width < 4:
            return False
        h = np.array([[1., 0., 0.]])
        # Viewpoint, blur and width localization contribute observation noise.
        variance = (pixel_sigma/pixel_width)**2 + orientation_sigma**2
        self.x, self.p, accepted = correction(
            self.x, self.p, [np.log(self.focal_px/pixel_width)], h,
            np.array([[variance]]), 10.83)
        return accepted

    def estimate(self):
        a = np.array([1., 0., 1.])
        mean = float(a @ self.x)
        sigma = np.sqrt(float(a @ self.p @ a))
        return float(np.exp(mean)), (float(np.exp(mean-1.96*sigma)), float(np.exp(mean+1.96*sigma)))


def triangulate_rectified(uv_left, uv_right, focal_px, baseline_m, principal, pixel_sigma=1.):
    """Rectified, synchronized pair; x-right, y-down, z-forward. First-order covariance."""
    u, v = np.asarray(uv_left, float)
    ur, vr = np.asarray(uv_right, float)
    disparity = u - ur
    if focal_px <= 0 or baseline_m <= 0 or pixel_sigma <= 0:
        raise ValueError("Invalid calibration/noise")
    if disparity <= 3*np.sqrt(2)*pixel_sigma or abs(v-vr) > 3*np.sqrt(2)*pixel_sigma:
        raise ValueError("Stereo pair is too uncertain or violates epipolar geometry")
    cx, cy = principal
    z = focal_px*baseline_m/disparity
    xyz = np.array([(u-cx)*z/focal_px, (v-cy)*z/focal_px, z])
    # Derivative of XYZ with respect to [u_left, v_left, u_right].
    dz = np.array([-z/disparity, 0., z/disparity])
    j = np.vstack(((u-cx)/focal_px*dz + [z/focal_px, 0., 0.],
                   (v-cy)/focal_px*dz + [0., z/focal_px, 0.], dz))
    return xyz, j @ j.T * pixel_sigma**2
