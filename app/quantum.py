"""Tiny exact state-vector simulator (pure NumPy, no quantum SDK needed).

Two flavours:
  * Qubit-indexed real state vector (RY / CRY) used by the demo circuit.
  * Batched tensor simulator (RY / CNOT) used to train the variational classifier.
Only real gates are used, so real amplitudes are exact.
"""
import numpy as np


# ---------------------------------------------------------------- demo circuit
def new_state(n: int) -> np.ndarray:
    s = np.zeros(1 << n)
    s[0] = 1.0
    return s


def apply_ry(s: np.ndarray, q: int, theta: float) -> None:
    c, sn, bit = np.cos(theta / 2), np.sin(theta / 2), 1 << q
    idx = np.arange(s.size)
    i0 = idx[(idx & bit) == 0]
    a, b = s[i0].copy(), s[i0 | bit].copy()
    s[i0] = c * a - sn * b
    s[i0 | bit] = sn * a + c * b


def apply_cry(s: np.ndarray, ctrl: int, tgt: int, theta: float) -> None:
    c, sn, cb, tb = np.cos(theta / 2), np.sin(theta / 2), 1 << ctrl, 1 << tgt
    idx = np.arange(s.size)
    i0 = idx[((idx & cb) != 0) & ((idx & tb) == 0)]
    a, b = s[i0].copy(), s[i0 | tb].copy()
    s[i0] = c * a - sn * b
    s[i0 | tb] = sn * a + c * b


def prob_one(s: np.ndarray, q: int) -> float:
    bit = 1 << q
    idx = np.arange(s.size)
    return float(np.sum(s[(idx & bit) != 0] ** 2))


# ------------------------------------------------------ batched VQC (training)
def _ry_axis(state: np.ndarray, axis: int, theta: np.ndarray) -> np.ndarray:
    n = state.ndim - 1
    shape = (-1,) + (1,) * (n - 1)
    c = np.cos(theta / 2).reshape(shape)
    s = np.sin(theta / 2).reshape(shape)
    s0 = np.take(state, 0, axis=axis)
    s1 = np.take(state, 1, axis=axis)
    return np.stack([c * s0 - s * s1, s * s0 + c * s1], axis=axis)


def _cnot(state: np.ndarray, ctrl: int, tgt: int) -> np.ndarray:
    out = state.copy()
    idx = [slice(None)] * state.ndim
    idx[ctrl + 1] = 1
    ax = tgt if tgt > ctrl else tgt + 1  # target axis after ctrl axis is removed
    out[tuple(idx)] = np.flip(state[tuple(idx)], axis=ax)
    return out


def vqc_prob(x: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """x: (B, n) in [0,1]. weights: (L, n). Returns P(qubit0 = 1) per sample.

    Data re-uploading: every layer re-encodes x with RY(pi*x + w), then a CNOT ring.
    """
    B, n = x.shape
    state = np.zeros((B,) + (2,) * n)
    state[(slice(None),) + (0,) * n] = 1.0
    for layer in weights:
        for q in range(n):
            state = _ry_axis(state, q + 1, np.pi * x[:, q] + layer[q])
        for q in range(n):
            state = _cnot(state, q, (q + 1) % n)
    p = np.sum(state[:, 1] ** 2, axis=tuple(range(1, n)))
    return p
