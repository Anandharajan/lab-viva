"""A1: 2PL item response theory in pure numpy.

P(correct | theta, a, b) = 1 / (1 + exp(-a (theta - b)))
Ability: EAP on a 61-point N(0,1) grid. Items: online MAP (Newton steps with a Gaussian prior
on b and a log-normal prior on a), Laplace approximation gives the posterior SE.
Cold start: b from a Bloom-level prior (Remember -1.5 ... Create +1.5), a = 1.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

GRID = np.linspace(-4, 4, 61)
PRIOR = np.exp(-0.5 * GRID**2)
PRIOR /= PRIOR.sum()
BLOOM_B = {"Remember": -1.5, "Understand": -0.9, "Apply": -0.3, "Analyse": 0.3, "Evaluate": 0.9, "Create": 1.5}


def p2pl(theta: np.ndarray | float, a: float, b: float) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(np.clip(-a * (np.asarray(theta) - b), -50, 50)))


def eap(responses: list[tuple[float, float, int]]) -> tuple[float, float]:
    """responses: [(a, b, correct)] -> (theta_hat, posterior SD)."""
    logp = np.log(PRIOR)
    for a, b, u in responses:
        p = np.clip(p2pl(GRID, a, b), 1e-9, 1 - 1e-9)
        logp += np.log(p) if u else np.log(1 - p)
    w = np.exp(logp - logp.max())
    w /= w.sum()
    m = float((GRID * w).sum())
    return m, float(np.sqrt(((GRID - m) ** 2 * w).sum()))


def information(theta: float, a: float, b: float) -> float:
    p = float(p2pl(theta, a, b))
    return a * a * p * (1 - p)


@dataclass
class ItemParams:
    a: float = 1.0
    b: float = 0.0
    b_prior: float = 0.0
    se_b: float = 1.0

    @classmethod
    def cold(cls, bloom: str) -> ItemParams:
        b0 = BLOOM_B.get(bloom, 0.0)
        return cls(1.0, b0, b0, 1.0)


def map_update(item: ItemParams, data: list[tuple[float, int]], sd_b: float = 1.0, sd_loga: float = 0.5,
               iters: int = 20) -> ItemParams:
    """MAP of (log a, b) given [(theta, correct)] for this item; Newton-Raphson with numeric Hessian."""
    if not data:
        return item
    th = np.array([d[0] for d in data])
    u = np.array([d[1] for d in data], float)
    x = np.array([np.log(item.a), item.b])

    def nlp(v: np.ndarray) -> float:
        a, b = np.exp(v[0]), v[1]
        p = np.clip(p2pl(th, a, b), 1e-9, 1 - 1e-9)
        ll = (u * np.log(p) + (1 - u) * np.log(1 - p)).sum()
        return float(-ll + 0.5 * ((b - item.b_prior) / sd_b) ** 2 + 0.5 * (v[0] / sd_loga) ** 2)

    def grad_hess(v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        h = 1e-4
        g = np.zeros(2)
        H = np.zeros((2, 2))
        f0 = nlp(v)
        for i in range(2):
            e = np.zeros(2); e[i] = h
            g[i] = (nlp(v + e) - nlp(v - e)) / (2 * h)
            H[i, i] = (nlp(v + e) - 2 * f0 + nlp(v - e)) / h**2
        e1, e2 = np.array([h, 0]), np.array([0, h])
        H[0, 1] = H[1, 0] = (nlp(v + e1 + e2) - nlp(v + e1 - e2) - nlp(v - e1 + e2) + nlp(v - e1 - e2)) / (4 * h * h)
        return g, H

    for _ in range(iters):
        g, H = grad_hess(x)
        H = H + np.eye(2) * 1e-6
        try:
            step = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            break
        if not np.all(np.isfinite(step)) or np.any(np.linalg.eigvalsh(H) <= 0):
            step = 0.1 * g  # fall back to gradient descent
        t = 1.0
        while nlp(x - t * step) > nlp(x) and t > 1e-4:
            t /= 2
        x = x - t * step
        x[0] = np.clip(x[0], np.log(0.2), np.log(4.0))
        x[1] = np.clip(x[1], -4, 4)
        if np.abs(t * step).max() < 1e-6:
            break
    _, H = grad_hess(x)
    se_b = float(np.sqrt(np.linalg.inv(H + np.eye(2) * 1e-6)[1, 1])) if np.all(np.linalg.eigvalsh(H) > 0) else 1.0
    return ItemParams(float(np.exp(x[0])), float(x[1]), item.b_prior, se_b)


def calibrate(R: np.ndarray, items: list[ItemParams], rounds: int = 5) -> tuple[list[ItemParams], np.ndarray]:
    """Alternating EAP/MAP on a response matrix R (students x items, NaN = not asked)."""
    thetas = np.zeros(R.shape[0])
    for _ in range(rounds):
        for s in range(R.shape[0]):
            obs = [(it.a, it.b, int(R[s, j])) for j, it in enumerate(items) if not np.isnan(R[s, j])]
            thetas[s] = eap(obs)[0]
        items = [map_update(it, [(thetas[s], int(R[s, j])) for s in range(R.shape[0]) if not np.isnan(R[s, j])])
                 for j, it in enumerate(items)]
    return items, thetas
