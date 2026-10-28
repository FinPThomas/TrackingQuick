"""Tiny conditional rectified-flow experiment, independently implemented in NumPy.

This generates motion hypotheses during an occlusion. It is not GenPT/TAPNext,
does not observe images, and is not used as a Kalman measurement.
"""
import time
import numpy as np


def motion_samples(rng, n):
    horizon = rng.uniform(.2, 1.2, n)
    speed = rng.uniform(-1, 1, (n, 2))
    turn = rng.choice([-1., 1.], n)
    c = np.c_[horizon, speed]
    residual = np.c_[turn*1.2*horizon**2, .6*horizon**2 + .2*speed[:, 1]]
    residual += rng.normal(0, .10, (n, 2))
    return c, residual


class ConditionalFlow:
    def __init__(self, seed=11):
        self.rng = np.random.default_rng(seed)
        self.params = [self.rng.normal(0, .25, (6, 48)), np.zeros(48),
                       self.rng.normal(0, .15, (48, 48)), np.zeros(48),
                       self.rng.normal(0, .10, (48, 2)), np.zeros(2)]

    def forward(self, inputs):
        w1, b1, w2, b2, w3, b3 = self.params
        a = np.tanh(inputs @ w1 + b1)
        b = np.tanh(a @ w2 + b2)
        return b @ w3 + b3, (a, b)

    def train(self, steps=1200):
        m, v = [np.zeros_like(p) for p in self.params], [np.zeros_like(p) for p in self.params]
        losses = []
        for step in range(1, steps+1):
            c, x1 = motion_samples(self.rng, 128)
            x0 = self.rng.normal(size=x1.shape)
            t = self.rng.uniform(size=(len(c), 1))
            inputs = np.c_[(1-t)*x0+t*x1, t, c]
            target = x1-x0
            output, (a, b) = self.forward(inputs)
            d = 2*(output-target)/output.size
            w1, b1, w2, b2, w3, b3 = self.params
            gw3, gb3 = b.T @ d, d.sum(axis=0)
            db = (d @ w3.T)*(1-b*b)
            gw2, gb2 = a.T @ db, db.sum(axis=0)
            da = (db @ w2.T)*(1-a*a)
            grads = [inputs.T @ da, da.sum(axis=0), gw2, gb2, gw3, gb3]
            for j, grad in enumerate(grads):
                m[j] = .9*m[j]+.1*grad
                v[j] = .999*v[j]+.001*grad*grad
                self.params[j] -= .002*(m[j]/(1-.9**step))/(np.sqrt(v[j]/(1-.999**step))+1e-8)
            losses.append(float(np.mean((output-target)**2)))
        return {"first_100_loss": float(np.mean(losses[:100])), "last_100_loss": float(np.mean(losses[-100:]))}

    def sample(self, conditions, samples=64, steps=16, seed=99):
        rng = np.random.default_rng(seed)
        c = np.repeat(np.atleast_2d(conditions), samples, axis=0)
        z = rng.normal(size=(len(c), 2))
        for step in range(steps):
            velocity, _ = self.forward(np.c_[z, np.full((len(c), 1), step/steps), c])
            z += velocity/steps
        return z.reshape(-1, samples, 2)


def experiment():
    start = time.perf_counter()
    model = ConditionalFlow()
    loss = model.train()
    c, truth = motion_samples(np.random.default_rng(812), 150)
    t = time.perf_counter()
    samples = model.sample(c)
    elapsed = (time.perf_counter()-t)*1000/len(c)
    mean = samples.mean(axis=1)
    low, high = np.quantile(samples, [.025, .975], axis=1)
    # Energy score rewards a calibrated distribution, not only its mean.
    energy = np.linalg.norm(samples-truth[:, None], axis=-1).mean()
    energy -= .5*np.linalg.norm(samples-np.roll(samples, 1, axis=1), axis=-1).mean()
    info = {**loss, "training_and_evaluation_seconds": time.perf_counter()-start,
            "test_cases": len(c), "samples_per_case": 64, "euler_steps": 16,
            "flow_mean_rmse_px": float(10*np.sqrt(np.mean((mean-truth)**2))),
            "constant_velocity_rmse_px": float(10*np.sqrt(np.mean(truth**2))),
            "marginal_95_coverage": float(np.mean((truth>=low)&(truth<=high))),
            "energy_score_px": float(10*energy), "batched_ms_per_case": elapsed,
            "scope": "independent synthetic residual-motion toy; not an image tracker; not fused into live observations"}
    return info, samples[0]*10, truth[0]*10
