# Everlasting Options Simulation

This repository contains an end‑to‑end, fully‑functional reference implementation of the simulation framework described in the paper **"Proactive Market Making and Liquidity Analysis for Everlasting Options in DeFi Ecosystems"** by Mohanty *et al.* (2024).  The codebase is written in Python and is designed to be easy to reproduce and extend.

The paper introduces *everlasting options* – perpetual options that eliminate the need to roll expiring contracts and evaluates a **Dynamic Proactive Market Maker** (DPMM) to quote prices while accounting for inventory.  Liquidity providers hedge their inventories with delta hedging to achieve positive profitability even in low‑liquidity environments.  Our implementation follows the methodology described in the paper:

* **Geometric Brownian Motion (GBM)** is used to simulate price paths for the underlying asset.  The asset price at the next time step is sampled as \(S_{t+\Delta t}=S_t\exp\{(\mu - \tfrac{1}{2}\sigma^2)\Delta t + \sigma\sqrt{\Delta t}Z\}\) where \(Z\sim\mathcal{N}(0,1)\).
* **Black–Scholes pricing** is used for European call options; the paper defines the call price as \(C(S,K,r,\sigma,T)=S\Phi(d_1)-Ke^{-rT}\Phi(d_2)\) with the usual definitions of \(d_1\) and \(d_2\).  Everlasting options are approximated by a decaying weighted sum of fixed‑maturity calls.
* **Dynamic Proactive Market Making** adjusts the quoted mark price based on inventory.  When the theoretical value of the option is `ivalue`, the DPMM sets the market price as \(P_m = \text{ivalue}\,(1 + k\,\tfrac{V}{Q_0})^2\), where \(V\) is the current inventory, \(Q_0\) is a reference liquidity parameter and \(k\) is a shape parameter.
* **Delta hedging**: The liquidity provider estimates the option's delta \(\Delta\approx\Phi(d_1)\) and maintains a position \(\Pi_t = -h\cdot\Delta_{\text{net},t}\) in the underlying to offset inventory exposure.  The net delta \(\Delta_{\text{net},t}\) is computed as the product of inventory and the option's delta.  A hedging fraction \(h\in[0,1]\) determines how aggressively the LP hedges.
* **Funding fee and PnL**:  Each day the funding fee equals the difference between the mark price and the next–day payoff; transaction costs are subtracted, and hedging gains are added.  The cumulative profit and loss over the simulation horizon is the sum of daily PnL terms.

The simulation parameters are chosen to mirror the paper's experiments: the underlying asset (ETH) starts at \$3000, the annual drift is 3 % and the volatility is 60 %.  The simulation horizon is six months, discretized in daily steps, and multiple independent paths are generated.  Liquidity levels are varied among \(Q_0\in\{1000,10\,000,100\,000\}\) USDC and strikes are set to \$3 100, \$3 200 and \$3 300.  These values can easily be changed via the configuration object.

## Repository structure

```
everlasting-options-simulation/
├── README.md               ← high‑level overview and reproduction instructions
├── requirements.txt        ← Python dependencies
├── everlasting_options/    ← main package
│   ├── src/                ← source code implementing the simulation
│   │   ├── __init__.py
│   │   ├── gbm.py          ← Geometric Brownian Motion path generation
│   │   ├── options.py      ← Black–Scholes pricing, deltas and everlasting option approximation
│   │   ├── market_maker.py ← Dynamic Proactive Market Maker and AMM abstractions
│   │   ├── hedging.py      ← Delta hedging logic for liquidity providers
│   │   ├── simulation.py   ← Driver for running simulations and producing figures
│   │   └── ammdpmm_compare.py ← Example comparison between AMM and DPMM models
│   └── examples/
│       └── run_simulation.py ← Sample script demonstrating the simulation API
```

## Installation

Assuming you have Python 3.10+ installed, create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

You can reproduce the figures from the paper by running the provided example script.  For example, to simulate a funding fee time‑series and a final PnL distribution under a DPMM with moderate liquidity and a hedging ratio of 0.5:

```bash
python everlasting_options/examples/run_simulation.py --liquidity 10000 --strike 3200 --hedge 0.5 --n-paths 100
```

This will generate plots saved to the `outputs/` folder and print summary statistics.  See `--help` for all available options.

## Quick Start

1. **Clone the repository**:
   ```bash
   git clone https://github.com/hardhik-99/everlasting-options-simulation.git
   cd everlasting-options-simulation
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run a simulation**:
   ```bash
   python everlasting_options/examples/run_simulation.py
   ```

4. **View results**:
   The simulation will output summary statistics including mean PnL, median PnL, standard deviation, and the fraction of positive PnL paths.

## Extending the framework

The modular design makes it easy to plug in alternative pricing models, new market maker curves or hedging strategies.  For instance, to switch from a DPMM to a constant product AMM, simply instantiate `ConstantProductAMM` from `market_maker.py` and pass it to the simulator.  Researchers can also adjust the number of simulated paths, time horizon, gas fees, or shape parameter `k` to explore different scenarios.

## References

The code in this repository is based on the methods and notation described in the paper by Mohanty *et al.* (2024).  When in doubt, please consult the paper for a detailed derivation of formulas and further discussion.
