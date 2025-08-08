"""
simulation.py
=============

This module contains a configurable simulation engine for assessing
liquidity provision in everlasting options markets.  The core
components include a geometric Brownian motion price generator (see
:mod:`gbm`), a pricing model for everlasting options (see
:mod:`options`), a market maker curve (see :mod:`market_maker`), and a
delta‑hedging strategy (see :mod:`hedging`).  The simulator stitches
these components together to compute funding fees, hedging gains, and
gas/transaction costs over a multi‑path Monte Carlo experiment.

"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence, Optional, Dict, Any, Tuple
import numpy as np

from .gbm import generate_gbm_paths
from .options import everlasting_option_price, everlasting_option_delta
from .hedging import compute_net_delta, compute_hedge_position
from .market_maker import DynamicProactiveMarketMaker, ConstantPriceMarketMaker


@dataclass
class SimulationConfig:
    """Container for all parameters needed to run a simulation.

    Parameters are annotated with default values corresponding to the
    experiments described in the paper.  Users may override any of
    these values when constructing a config object.
    """

    s0: float = 3000.0  # initial price of ETH
    mu: float = 0.03    # annualized drift
    sigma: float = 0.60  # annualized volatility
    r: float = 0.00      # risk‑free rate (assumed zero for crypto markets)
    horizon_days: int = 180  # length of the simulation in days (≈6 months)
    n_paths: int = 100      # number of Monte Carlo paths
    strike: float = 3200.0  # strike price of the option
    maturities: Sequence[float] = field(default_factory=lambda: np.arange(1, 181) / 365.0)
    decay_factor: float = 1.0  # decay factor for everlasting option weighting
    liquidity: float = 10_000.0  # Q0: reference liquidity for the pool
    shape_parameter: float = 0.1  # k: shape of DPMM curve
    hedge_ratio: float = 0.5       # h: fraction of net delta to hedge
    inventory_volatility: float = 5.0  # std dev of inventory changes per day
    option_trade_fee: float = 0.0  # fixed gas fee per option trade (USDC)
    hedge_trade_fee: float = 0.0   # gas fee per unit of underlying traded
    use_constant_mm: bool = False  # if True, use ConstantPriceMarketMaker instead of DPMM
    seed: Optional[int] = None     # random seed for reproducibility

    def create_market_maker(self) -> Any:
        """Instantiate the appropriate market maker based on config."""
        if self.use_constant_mm:
            return ConstantPriceMarketMaker()
        return DynamicProactiveMarketMaker(k=self.shape_parameter, Q0=self.liquidity)

    @property
    def dt(self) -> float:
        """Return the time step length in years (1/365)."""
        return 1.0 / 365.0

    @property
    def n_steps(self) -> int:
        return self.horizon_days


def run_simulation(config: SimulationConfig) -> Dict[str, Any]:
    """Run a Monte Carlo simulation of the everlasting option market.

    The simulator computes daily funding fees, hedging profits/losses,
    and gas costs for multiple independent price paths.  It returns
    aggregated statistics and the raw per‑path PnL values.

    Parameters
    ----------
    config : SimulationConfig
        Object containing all simulation parameters.

    Returns
    -------
    dict
        Dictionary with aggregated results.  Keys include:

        ``final_pnl`` : ndarray
            Array of length ``n_paths`` containing the final PnL for each
            simulated path.
        ``funding_fees`` : ndarray
            Total funding fees collected along each path.
        ``hedging_pnl`` : ndarray
            Total hedging PnL accrued on underlying trades for each path.
        ``trade_costs`` : ndarray
            Total gas/transaction costs paid along each path.
    """
    # Unpack values
    S0 = config.s0
    mu = config.mu
    sigma = config.sigma
    r = config.r
    dt = config.dt
    steps = config.n_steps
    n_paths = config.n_paths
    strike = config.strike
    maturities = config.maturities
    decay_factor = config.decay_factor
    hedge_ratio = config.hedge_ratio
    inventory_vol = config.inventory_volatility
    option_fee = config.option_trade_fee
    hedge_fee = config.hedge_trade_fee

    # Instantiate market maker
    mm = config.create_market_maker()

    # Generate price paths
    paths = generate_gbm_paths(
        S0=S0,
        mu=mu,
        sigma=sigma,
        dt=dt,
        steps=steps,
        n_paths=n_paths,
        seed=config.seed,
    )

    # Arrays to accumulate metrics
    funding_fees_total = np.zeros(n_paths, dtype=float)
    hedging_pnl_total = np.zeros(n_paths, dtype=float)
    trade_costs_total = np.zeros(n_paths, dtype=float)

    # For each path simulate inventory changes and hedging
    rng = np.random.default_rng(config.seed)
    # loop over paths individually to maintain separate states
    for p_idx in range(n_paths):
        prices = paths[p_idx]
        inventory = 0.0
        hedge_position = 0.0
        for t in range(steps):
            S_t = prices[t]
            S_next = prices[t + 1]

            # Theoretical value and delta of the everlasting option at time t
            ivalue = everlasting_option_price(
                S=S_t,
                K=strike,
                r=r,
                sigma=sigma,
                maturities=maturities,
                decay_factor=decay_factor,
            )
            delta = everlasting_option_delta(
                S=S_t,
                K=strike,
                r=r,
                sigma=sigma,
                maturities=maturities,
                decay_factor=decay_factor,
            )

            # Current mark price given inventory
            mark_price = mm.quote_price(ivalue=ivalue, inventory=inventory)

            # Option payoff at end of day (for a call): max(S_next - K, 0)
            payoff = max(S_next - strike, 0.0)
            # Funding fee is difference between mark price and payoff
            funding_fee = mark_price - payoff

            # Sample random change in inventory to simulate trader activity
            inventory_change = rng.normal(loc=0.0, scale=inventory_vol)
            inventory_new = inventory + inventory_change

            # Option trade costs: charged when inventory changes significantly
            trade_cost = option_fee * abs(inventory_change)

            # Net delta exposure and hedging position
            net_delta = compute_net_delta(inventory_new, delta)
            hedge_target = compute_hedge_position(net_delta, hedge_ratio)
            # Underlying trade required to reach hedge_target
            hedge_trade = hedge_target - hedge_position
            # Cost of hedging trades (gas fee per unit)
            hedge_cost = hedge_fee * abs(hedge_trade)
            # Hedging PnL from previous position
            pnl_hedge = hedge_position * (S_next - S_t)

            # Accumulate totals
            funding_fees_total[p_idx] += funding_fee
            hedging_pnl_total[p_idx] += pnl_hedge
            trade_costs_total[p_idx] += trade_cost + hedge_cost

            # Update state
            inventory = inventory_new
            hedge_position = hedge_target

    final_pnl = funding_fees_total + hedging_pnl_total - trade_costs_total

    return {
        "final_pnl": final_pnl,
        "funding_fees": funding_fees_total,
        "hedging_pnl": hedging_pnl_total,
        "trade_costs": trade_costs_total,
        "price_paths": paths,
    }


def compute_slippage(
    theoretical_price: float,
    inventories: Sequence[float],
    market_maker: DynamicProactiveMarketMaker,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute slippage for a range of inventory changes.

    This helper function is useful for replicating the slippage curve
    shown in Figure 1 of the paper.  It returns the quoted price and
    slippage (difference from the theoretical price) for each provided
    inventory value.  A symmetric range of inventory values should be
    passed to visualize the parabolic shape.

    Parameters
    ----------
    theoretical_price : float
        Fair value of the option.
    inventories : sequence of float
        Inventory values at which to evaluate the mark price.
    market_maker : DynamicProactiveMarketMaker
        The DPMM instance used for quoting.

    Returns
    -------
    tuple (np.ndarray, np.ndarray)
        The quoted prices and corresponding slippages.
    """
    inventories = np.array(inventories, dtype=float)
    quoted_prices = np.array([
        market_maker.quote_price(theoretical_price, inv) for inv in inventories
    ], dtype=float)
    slippage = quoted_prices - theoretical_price
    return quoted_prices, slippage