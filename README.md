DeFi Volatility Spillover Forecaster

The world’s first real-time contagion early-warning system for crypto markets






Most liquidations don’t happen because of price — they happen because of how fast fear spreads.
This engine predicts volatility contagion across BTC, ETH, SOL, LINK, and UNI — in real time.

🔥 Live Example Output
{
  "trigger": "BTC drop 5.0%",
  "ETH_vol_spike_probability": 0.83,
  "expected_spike_percent": 58,
  "warning_level": "HIGH"
}


When it says HIGH, it means:

Raise collateral

Reduce leverage

Hedge exposure

Stop waking up at 3AM during liquidation cascades

🚀 Features

Real-time 5-minute price feeds via Polygon.io

Multi-asset, perfectly matched timestamps

Tracks BTC, ETH, SOL, LINK, UNI

5 engineered risk features per asset:

Log returns

Realized vol (288-bar = 1 day)

Log volume

High–low intrabar spread

Close–open momentum

Multi-asset LSTM that learns how BTC shocks spill over into ETH/SOL/LINK/UNI

Stress tests “What if BTC dumps 5% right now?”

Outputs clean, actionable JSON alerts

Entire pipeline (training + inference) runs on CPU in under 60 seconds

Fully local (no cloud, no KYC, no third-party risk)

📦 Installation
1. Clone Repository
git clone https://github.com/yourusername/defi-volatility-forecaster.git
cd defi-volatility-forecaster

2. Create Virtual Environment (Recommended)
python -m venv venv


Activate environment:

Windows:

venv\Scripts\activate


macOS / Linux:

source venv/bin/activate

3. Install Dependencies
pip install torch pandas numpy python-dotenv requests

🔑 Polygon API Key Setup (IMPORTANT)

This project uses the Polygon.io Market Data API.

Step 1: Get a free API key

Create an account at:
https://polygon.io

Step 2: Create the .env file

In the project root, create a file named:

.env


Add your API key:

POLYGON_API_KEY=pk_your_actual_key_here

▶️ Running the Model
Run Once (Train + Single Alert)
python main.py

Run in Live Oracle Mode (Continuous 5-minute Alerts)
python oracle.py


This mode prints a new contagion warning every 5 minutes.
Leave it running like a background DeFi smoke detector.

📊 Example Live Terminal Output
2025-04-05 03:17:22 | HIGH   | ETH vol spike: 91.2% → +74% | TIME TO ACT!
2025-04-05 03:22:22 | MEDIUM | ETH vol spike: 62.0% → +21% | Monitoring...

🧨 Why This Matters
Year	Event	Liquidations	What This Model Would Have Said
2022	LUNA/UST Collapse	~$15B+	HIGH RISK (4 hrs before death spiral)
2022	FTX Collapse	~$8B	HIGH RISK (night before bankruptcy)
2024	Flash Crash	~$1.2B	HIGH RISK (30 mins before dump)

Never get liquidated in your sleep again.

🗺️ Roadmap

Add more assets (AVAX, DOGE, ARB, OP, XRP, BNB)

Add Telegram & Discord bot alerts

Deploy on-chain (Chainlink / Tellor)

Build analytics dashboard

Backtesting engine (2020–2025)

Research paper version

⚠️ Disclaimer

This is not financial advice.
This is a risk intelligence tool — a smoke detector for volatility contagion.

❤️ Credits

Built with:

Coffee

LSTM layers

PTSD from 3AM liquidation emails

Star ⭐ this repo if you believe DeFi deserves better risk tools.
