"""
options.py
==========

This module implements option pricing and Greeks for European call
options under the Black–Scholes model, along with helper functions
for approximating the price and delta of *everlasting* options.

According to the paper, the Black–Scholes price of a fixed‑expiration
European call option with underlying price ``S``, strike ``K``, risk
free rate ``r``, volatility ``sigma``, and time to maturity ``T`` is

.. math::

   C(S, K, r, \sigma, T) = S\Phi(d_1) - K e^{-rT}\Phi(d_2),

where \(\Phi(\cdot)\) is the standard normal cumulative distribution
function, and

.. math::

   d_1 = \frac{\ln(S/K) + (r + \tfrac{1}{2}\sigma^2)T}{\sigma\sqrt{T}},
   \qquad
   d_2 = d_1 - \sigma\sqrt{T}.

To price an everlasting option, the paper proposes approximating it as
a decaying weighted sum of call options with consecutive expirations.
Although the exact weighting scheme is not fully
specified in the text, we follow a simple 1/(D + i) decay and
normalize the weights.  Users may substitute their own weighting
scheme by passing explicit weights.
"""

from __future__ import annotations

from typing import Iterable, Sequence, Optional
import numpy as np
from scipy.stats import norm


def bs_price_call(S: float, K: float, r: float, sigma: float, T: float) -> float:
    """Compute the Black–Scholes price of a European call option.

    Parameters
    ----------
    S : float
        Current underlying price.
    K : float
        Strike price of the option.
    r : float
        Risk free rate (annualized, continuously compounded).
    sigma : float
        Volatility of the underlying asset (annualized).
    T : float
        Time to maturity in years.

    Returns
    -------
    float
        The theoretical call price.
    """
    if T <= 0:
        # Option has expired
        return max(S - K, 0.0)
    sqrtT = np.sqrt(T)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT
    price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return price


def bs_delta_call(S: float, K: float, r: float, sigma: float, T: float) -> float:
    """Compute the Black–Scholes delta of a European call option.

    The delta for a European call is the derivative of the option price
    with respect to the underlying price.  It is given by
    \(\Delta = \Phi(d_1)\).

    Parameters and returns are analogous to :func:`bs_price_call`.
    """
    if T <= 0:
        return 0.0 if S < K else 1.0
    sqrtT = np.sqrt(T)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrtT)
    return norm.cdf(d1)


def everlasting_option_price(
    S: float,
    K: float,
    r: float,
    sigma: float,
    maturities: Sequence[float],
    decay_factor: float = 1.0,
    weights: Optional[Sequence[float]] = None,
) -> float:
    """Approximate the price of an everlasting call option.

    The paper suggests pricing an everlasting option as a decreasing
    weighted sum of fixed–expiration calls.  Given a list of
    maturities (in years), we compute call prices for each and form
    a weighted sum.  By default, the weights follow a 1/(D + i)
    pattern controlled by ``decay_factor``.  If explicit weights are
    provided they will be normalized before use.

    Parameters
    ----------
    S, K, r, sigma : float
        Same meanings as in :func:`bs_price_call`.
    maturities : Sequence[float]
        A sequence of maturities (in years) for the fixed‑term options.
    decay_factor : float, default=1.0
        Controls the rate at which weights decay.  The i‑th weight is
        proportional to 1/(decay_factor + i).
    weights : sequence of float, optional
        If provided, these weights will be used instead of the default
        decaying scheme.  They will be normalized to sum to one.

    Returns
    -------
    float
        Approximate price of the everlasting option.
    """
    n = len(maturities)
    # Compute individual call prices
    call_prices = np.array([bs_price_call(S, K, r, sigma, T) for T in maturities], dtype=float)
    # Determine weights
    if weights is not None:
        w = np.array(weights, dtype=float)
    else:
        w = 1.0 / (decay_factor + np.arange(1, n + 1))
    # Normalize weights to sum to one
    w = w / w.sum()
    return float(np.dot(w, call_prices))


def everlasting_option_delta(
    S: float,
    K: float,
    r: float,
    sigma: float,
    maturities: Sequence[float],
    decay_factor: float = 1.0,
    weights: Optional[Sequence[float]] = None,
) -> float:
    """Approximate the delta of an everlasting call option.

    The delta of the everlasting option is approximated as the same
    weighted sum of deltas used to approximate the price.  Delta is
    computed via :func:`bs_delta_call` for each maturity and then
    combined using normalized weights.

    See :func:`everlasting_option_price` for parameter definitions.
    """
    n = len(maturities)
    deltas = np.array([bs_delta_call(S, K, r, sigma, T) for T in maturities], dtype=float)
    if weights is not None:
        w = np.array(weights, dtype=float)
    else:
        w = 1.0 / (decay_factor + np.arange(1, n + 1))
    w = w / w.sum()
    return float(np.dot(w, deltas))