# Decentralized Multi-Asset Volatility Spillover Forecaster

This is a Python implementation of a hybrid Neural GARCH-GNN model for predicting volatility spillovers in DeFi assets (BTC, ETH, SOL, LINK, UNI). It predicts how panic spreads (e.g., "78% chance ETH volatility spikes 42% if BTC drops >3%").

## Features
- Fetches 5-min OHLCV data via Polygon.io.
- Computes log returns, rolling vol, and correlations.
- Hybrid model: LSTM + GAT + Embedded GARCH + Bayesian Dropout.
- Ensemble with meta-learner.
- Outputs JSON risk scores.
- Backtested on 2020-2025 data (simulated via historical fetch).

## Setup
1. Install dependencies: `pip install -r requirements.txt`
2. Set Polygon API key: Export `POLYGON_API_KEY=your_key` (get from polygon.io).
3. Run: `python main.py`

## Usage
- Train: `python main.py --mode train`
- Predict: `python main.py --mode predict --input '{"btc_drop": 0.03}'`
- Output example: `{"eth_vol_spike_prob": 0.78, "spike_pct": 42, "uncertainty": 0.95}`

## Expansion
- Add more assets in `assets` list.
- Integrate with DeFi protocols via APIs.

Let's collaborate—fork and PR!
#DeFi #Crypto #MachineLearning
