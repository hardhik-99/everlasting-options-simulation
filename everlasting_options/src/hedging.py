"""
hedging.py
==========

Utility functions for computing delta hedges for a liquidity provider.
"""

from __future__ import annotations

import numpy as np


def compute_net_delta(inventory: float, delta: float) -> float:
    """Return the net delta exposure of the LP's inventory.

    Parameters
    ----------
    inventory : float
        The number of option contracts held by the liquidity provider.
        Positive values indicate a long position, negative values a
        short position.
    delta : float
        The option's delta (sensitivity with respect to the underlying).

    Returns
    -------
    float
        The net delta exposure ``inventory * delta``.
    """
    return inventory * delta


def compute_hedge_position(net_delta: float, hedge_ratio: float) -> float:
    """Compute the position in the underlying asset needed to hedge.

    Parameters
    ----------
    net_delta : float
        Net delta exposure of the option inventory.
    hedge_ratio : float
        Fraction of the exposure to offset (between 0 and 1).  A value of
        0 means no hedging; 1 means fully hedged.

    Returns
    -------
    float
        The number of units of the underlying asset to hold to hedge
        the option inventory.
    """
    return -hedge_ratio * net_delta