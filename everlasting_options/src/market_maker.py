"""
market_maker.py
================

This module defines simple market maker abstractions used in the
simulation.  Two implementations are provided:

* **DynamicProactiveMarketMaker**: a dynamic proactive market maker (DPMM)
  that adjusts its quoted price based on inventory.
* **ConstantPriceMarketMaker**: a simplistic market maker that always
  quotes the theoretical value (i.e. ``ivalue``) irrespective of
  inventory.  This serves as a stand‑in for a passive AMM that has
  fixed pricing and exhibits higher volatility, as discussed in the
  experiments.

Both classes expose a common interface via the :meth:`quote_price`
method.  Inventories are not modified internally; external simulation
code should update inventories based on trades.  Pricing is assumed
to be symmetric for buys and sells for simplicity.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DynamicProactiveMarketMaker:
    """Dynamic proactive market maker (DPMM).

    Parameters
    ----------
    k : float
        Shape parameter controlling the curvature of the price adjustment.
        Positive ``k`` results in quadratic slippage; smaller values
        produce a flatter curve.
    Q0 : float
        Reference liquidity in the pool.

    Notes
    -----
    The DPMM does not maintain internal state beyond its parameters.  The
    simulation is responsible for tracking inventory and computing the
    theoretical value (``ivalue``) before calling :meth:`quote_price`.
    """

    k: float
    Q0: float

    def quote_price(self, ivalue: float, inventory: float) -> float:
        """Return the quoted price given the theoretical value and inventory.

        Parameters
        ----------
        ivalue : float
            Theoretical fair value of the derivative (e.g. Black–Scholes or
            everlasting option price).
        inventory : float
            Current inventory of options (positive if long, negative if short).

        Returns
        -------
        float
            The mark price quoted by the DPMM.
        """
        return ivalue * (1.0 + self.k * (inventory / self.Q0)) ** 2


@dataclass
class ConstantPriceMarketMaker:
    """A simplistic market maker that quotes a fixed price.

    This class represents an AMM without inventory sensitivity.  It
    returns the theoretical value passed in to :meth:`quote_price`.  Such
    a model is useful for comparing the performance of DPMMs against
    passive market makers.
    """

    def quote_price(self, ivalue: float, inventory: float) -> float:
        return ivalue