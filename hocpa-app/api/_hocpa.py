"""Core HOCPA simulation (trimmed from the original script, no matplotlib)."""
import numpy as np

Q = 3329
KEY_SPACE = 256
NOISE_STD = 0.18
BETA = 0.18

LAMBDA_INIT, LAMBDA_MAX, LAMBDA_MIN = 0.3, 0.70, 0.20
ADAPTIVE_THRESHOLD, STEP_UP, STEP_DOWN = 0.048, 0.012, 0.018


def hamming_weight(x):
    # vectorized popcount on 16 bits (matches bin(v & 0xFFFF).count('1'))
    x = np.asarray(x).astype(np.int64) & 0xFFFF
    x = x - ((x >> 1) & 0x5555)
    x = (x & 0x3333) + ((x >> 2) & 0x3333)
    x = (x + (x >> 4)) & 0x0F0F
    return (x + (x >> 8)) & 0x1F


def central_reduction(x, q=Q):
    x_mod = np.asarray(x).astype(np.int64) % q
    return np.where(x_mod > q // 2, x_mod - q, x_mod)


def noncentral_reduction(x, q=Q):
    return np.asarray(x).astype(np.int64) % q


def secure_reduction(x, lam, rng):
    x = np.asarray(x).astype(np.int64)
    mask = rng.random(x.shape) < lam
    return np.where(mask, central_reduction(x), noncentral_reduction(x))


def generate_traces(secret, mode, lam, n, rng):
    c = rng.integers(1, Q, size=n)
    s0 = rng.integers(0, Q, size=n)
    s1 = (secret - s0) % Q
    p0, p1 = (s0 * c) % Q, (s1 * c) % Q

    if mode == "central":
        r0, r1 = central_reduction(p0), central_reduction(p1)
    elif mode == "noncentral":
        r0, r1 = noncentral_reduction(p0), noncentral_reduction(p1)
    else:  # secure / adaptive
        r0, r1 = secure_reduction(p0, lam, rng), secure_reduction(p1, lam, rng)

    cross = r0.astype(np.float64) * r1.astype(np.float64)
    leak = lambda r: hamming_weight(r) + rng.normal(0, NOISE_STD, size=n) + BETA * cross
    return leak(r0), leak(r1), c


def hocpa_attack(L0, L1, C):
    comb = (L0 - L0.mean()) * (L1 - L1.mean())
    comb_c = comb - comb.mean()
    guesses = np.arange(KEY_SPACE)[:, None]
    red = central_reduction((guesses * C[None, :]) % Q)
    pred = red.astype(np.float64) ** 2
    pred_c = pred - pred.mean(axis=1, keepdims=True)
    num = pred_c @ comb_c
    den = np.sqrt((pred_c ** 2).sum(axis=1)) * np.sqrt((comb_c ** 2).sum())
    corr = np.abs(np.divide(num, den, out=np.zeros_like(num), where=den > 1e-9))
    return int(np.argmax(corr)), corr


class BoundedAdaptiveController:
    def __init__(self):
        self.lam = LAMBDA_INIT
        self.history = [LAMBDA_INIT]
        self.fails = 0

    def update(self, peak, success):
        if peak > ADAPTIVE_THRESHOLD and success:
            self.lam = min(LAMBDA_MAX, self.lam + STEP_UP)
            self.fails = 0
        elif peak < ADAPTIVE_THRESHOLD * 0.6 or not success:
            self.fails += 1
            if self.fails > 2:
                self.lam = max(LAMBDA_MIN, self.lam - STEP_DOWN)
                self.fails = 0
        else:
            self.fails = 0
        self.history.append(self.lam)


def run_point(mode, traces, trials, lam=0.5, seed=None):
    """One (mode, #traces) point: success rate + mean peak correlation."""
    rng = np.random.default_rng(seed)
    ctrl = BoundedAdaptiveController() if mode == "adaptive" else None
    cur = ctrl.lam if ctrl else lam
    wins, peaks = 0, []
    for _ in range(trials):
        secret = int(rng.integers(0, KEY_SPACE))
        L0, L1, C = generate_traces(secret, mode, cur, traces, rng)
        guess, corr = hocpa_attack(L0, L1, C)
        peaks.append(float(corr.max()))
        ok = guess == secret
        wins += ok
        if ctrl:
            ctrl.update(peaks[-1], ok)
            cur = ctrl.lam
    return {
        "mode": mode, "traces": traces, "trials": trials,
        "success_rate": wins / trials,
        "mean_rmax": float(np.mean(peaks)),
        "lambda": cur,
    }
