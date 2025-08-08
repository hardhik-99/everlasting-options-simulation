"""
gbm.py
========

This module provides functionality for generating price paths
using a **Geometric Brownian Motion (GBM)** process. The underlying 
asset price \(S_t\) is assumed to follow a GBM, which ensures 
positive prices and normally distributed log return.

The function `generate_gbm_paths` below implements this update for
multiple independent paths.
"""

from __future__ import annotations

import numpy as np
from typing import Optional


def generate_gbm_paths(
    S0: float,
    mu: float,
    sigma: float,
    dt: float,
    steps: int,
    n_paths: int,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Generate price paths following a geometric Brownian motion.

    Parameters
    ----------
    S0 : float
        Initial asset price at time zero.
    mu : float
        Annualized drift of the underlying asset (expected return).
    sigma : float
        Annualized volatility of the underlying asset.
    dt : float
        Time step size in years (e.g. 1/365 for daily steps).
    steps : int
        Number of time steps to simulate (not counting the initial point).
    n_paths : int
        Number of independent price paths to simulate.
    seed : int, optional
        Random seed to ensure reproducibility.  If ``None`` no seeding is
        performed.

    Returns
    -------
    np.ndarray
        A two‑dimensional array of shape ``(n_paths, steps+1)``.  Each row
        represents a simulated price path, with the first column equal to
        ``S0``.

    Notes
    -----
    The implementation uses vectorized operations for efficiency.
    """
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    # Generate standard normal innovations for all paths and steps
    Z = rng.standard_normal(size=(n_paths, steps))
    # Compute the drift and diffusion components once
    drift = (mu - 0.5 * sigma ** 2) * dt
    diffusion = sigma * np.sqrt(dt)

    # Cumulative log returns
    increments = drift + diffusion * Z
    # Prepend zeros for the initial time point
    log_returns = np.cumsum(increments, axis=1)
    # Include an initial column of zeros to represent t=0
    log_returns = np.hstack([np.zeros((n_paths, 1)), log_returns])

    # Convert to price paths
    paths = S0 * np.exp(log_returns)
    return paths