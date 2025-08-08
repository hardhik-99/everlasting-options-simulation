"""
Top‑level package for the everlasting options simulation framework.

This package exposes convenience functions for generating price paths,
pricing options, simulating market makers and hedging strategies, and
running reproducible simulations.  See the `README.md` for usage
instructions and consult the individual modules for API details.
"""

from .gbm import generate_gbm_paths  # noqa: F401
from .options import (
    bs_price_call,
    bs_delta_call,
    everlasting_option_price,
    everlasting_option_delta,
)  # noqa: F401
from .market_maker import DynamicProactiveMarketMaker, ConstantPriceMarketMaker  # noqa: F401
from .hedging import compute_net_delta, compute_hedge_position  # noqa: F401
from .simulation import SimulationConfig, run_simulation  # noqa: F401