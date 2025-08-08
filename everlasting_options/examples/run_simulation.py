#!/usr/bin/env python3
"""
run_simulation.py
==================

This example script demonstrates how to use the simulation framework to
reproduce some of the key analyses in the everlasting options paper.  It
allows the user to specify liquidity, strike price, hedging ratio and
other parameters on the command line.  The script prints summary
statistics and optionally displays plots of the PnL distribution and
funding fee time series.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Add the repository root to sys.path so local modules can be imported when
# running the script directly.  This makes the script relocatable.
# Determine the repository root by climbing two directories from this file:
# examples/run_simulation.py -> examples -> everlasting_options -> repo root
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from everlasting_options.src.simulation import SimulationConfig, run_simulation, compute_slippage
from everlasting_options.src.options import bs_price_call


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run everlasting options simulation.")
    parser.add_argument("--liquidity", type=float, default=10_000.0, help="Reference liquidity Q0 (USDC)")
    parser.add_argument("--strike", type=float, default=3200.0, help="Option strike price")
    parser.add_argument("--hedge", type=float, default=0.5, help="Hedging ratio h (0=no hedge, 1=full hedge)")
    parser.add_argument("--shape", type=float, default=0.1, help="DPMM shape parameter k")
    parser.add_argument("--n-paths", type=int, default=100, help="Number of Monte Carlo paths")
    parser.add_argument("--horizon", type=int, default=180, help="Simulation horizon in days")
    parser.add_argument("--use-amm", action="store_true", help="Use constant price market maker (AMM) instead of DPMM")
    parser.add_argument("--plot", action="store_true", help="Display plots of results")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # Build a configuration from command line arguments
    cfg = SimulationConfig(
        liquidity=args.liquidity,
        strike=args.strike,
        hedge_ratio=args.hedge,
        shape_parameter=args.shape,
        n_paths=args.n_paths,
        horizon_days=args.horizon,
        use_constant_mm=args.use_amm,
        seed=args.seed,
    )

    results = run_simulation(cfg)
    final_pnl = results["final_pnl"]

    print("Simulation summary:")
    print(f"  Mean PnL:   {np.mean(final_pnl):.2f}")
    print(f"  Median PnL: {np.median(final_pnl):.2f}")
    print(f"  Std PnL:    {np.std(final_pnl):.2f}")
    print(f"  Positive PnL fraction: {np.mean(final_pnl > 0):.2%}")

    if args.plot:
        # Plot final PnL distribution
        plt.figure(figsize=(8, 4))
        plt.hist(final_pnl, bins=50, color="steelblue", edgecolor="black")
        plt.title("Final PnL Distribution")
        plt.xlabel("PnL")
        plt.ylabel("Frequency")
        plt.axvline(np.mean(final_pnl), color="red", linestyle="--", label="Mean")
        plt.axvline(np.median(final_pnl), color="green", linestyle="--", label="Median")
        plt.legend()
        plt.tight_layout()

        # Plot slippage curve for demonstration
        theo = bs_price_call(cfg.s0, cfg.strike, cfg.r, cfg.sigma, cfg.horizon_days / 365.0)
        maker = cfg.create_market_maker()
        inventories = np.linspace(-100, 100, 201)
        quoted, slippage = compute_slippage(theo, inventories, maker)
        plt.figure(figsize=(8, 4))
        plt.plot(inventories, slippage)
        plt.title("Slippage vs Inventory")
        plt.xlabel("Inventory")
        plt.ylabel("Slippage (USDC)")
        plt.tight_layout()

        plt.show()


if __name__ == "__main__":
    main()