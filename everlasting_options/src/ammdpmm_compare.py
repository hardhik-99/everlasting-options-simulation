"""
ammdpmm_compare.py
===================

This module contains a utility function for running a side‑by‑side
comparison between a Dynamic Proactive Market Maker (DPMM) and a
constant price market maker (AMM).  It demonstrates how to use the
simulation framework to reproduce the qualitative findings reported in
the paper: namely, that the DPMM exhibits lower volatility and more
stable profits compared with a passive AMM

The `compare_market_makers` function returns summary statistics for
both models, and optionally produces histograms of the final PnL
distribution.  Users can adjust the configuration to explore how
different liquidity levels or hedging ratios affect the outcome.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Dict, Any, Optional

import matplotlib.pyplot as plt
import numpy as np

from .simulation import SimulationConfig, run_simulation


def compare_market_makers(
    base_config: SimulationConfig,
    plot: bool = False,
    bins: int = 50,
    random_seed: Optional[int] = None,
) -> Dict[str, Any]:
    """Run simulations for both DPMM and AMM and return summary stats.

    Parameters
    ----------
    base_config : SimulationConfig
        Base configuration used for both simulations.  The liquidity,
        hedging ratio and other parameters will be shared.
    plot : bool, default=False
        If True, histograms of the final PnL distributions will be
        displayed for each market maker.
    bins : int, default=50
        Number of bins used in the histogram.
    random_seed : int, optional
        Seed to use for both simulations.  When ``None``, the base
        configuration's seed is used.

    Returns
    -------
    dict
        Summary statistics including means, medians and standard
        deviations for each market maker.
    """
    # DPMM configuration
    dpmm_config = replace(base_config, seed=random_seed, use_constant_mm=False)
    ammm_config = replace(base_config, seed=random_seed, use_constant_mm=True)

    dpmm_results = run_simulation(dpmm_config)
    ammm_results = run_simulation(ammm_config)

    def summarize(pnl: np.ndarray) -> Dict[str, float]:
        return {
            "mean": float(np.mean(pnl)),
            "median": float(np.median(pnl)),
            "std": float(np.std(pnl)),
        }

    dpmm_stats = summarize(dpmm_results["final_pnl"])
    ammm_stats = summarize(ammm_results["final_pnl"])

    if plot:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
        axes[0].hist(dpmm_results["final_pnl"], bins=bins, color="skyblue", edgecolor="black")
        axes[0].set_title("DPMM Final PnL Distribution")
        axes[0].set_xlabel("PnL")
        axes[0].set_ylabel("Frequency")
        axes[0].axvline(dpmm_stats["mean"], color="red", linestyle="--", label=f"Mean: {dpmm_stats['mean']:.2f}")
        axes[0].axvline(dpmm_stats["median"], color="green", linestyle="--", label=f"Median: {dpmm_stats['median']:.2f}")
        axes[0].legend()

        axes[1].hist(ammm_results["final_pnl"], bins=bins, color="orange", edgecolor="black")
        axes[1].set_title("AMM Final PnL Distribution")
        axes[1].set_xlabel("PnL")
        axes[1].axvline(ammm_stats["mean"], color="red", linestyle="--", label=f"Mean: {ammm_stats['mean']:.2f}")
        axes[1].axvline(ammm_stats["median"], color="green", linestyle="--", label=f"Median: {ammm_stats['median']:.2f}")
        axes[1].legend()
        plt.tight_layout()
        plt.show()

    return {
        "dpmm": dpmm_stats,
        "ammm": ammm_stats,
        "dpmm_results": dpmm_results,
        "ammm_results": ammm_results,
    }