import os
import sys

# --- CONFIGURATION & SETUP ---
# Suppress the specific OpenMP warning on Windows
# Must be set before importing torch
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json
import logging
import warnings
import numpy as np
import pandas as pd
import requests
import time
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from script's directory
# This ensures it works regardless of where the script is called from
env_path = Path(__file__).resolve().parent / '.env'
load_dotenv(dotenv_path=env_path)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

warnings.filterwarnings("ignore")

# Logger setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
ASSETS = ["BTC", "ETH", "SOL", "LINK", "UNI"]
TICKERS = [f"X:{asset}USD" for asset in ASSETS]
TIMESTAMPS = 60  # Lookback window
FEATURES = 5     # LogRet, Vol, NormVol, HL_Spread, CO_Spread
PREDICT_HORIZON = 1
BATCH_SIZE = 32
EPOCHS = 8
LR = 0.001
MODEL_PATH = "model.pth"

# Check for API Key
API_KEY = os.getenv("POLYGON_API_KEY")

# --- DATA PIPELINE ---

def fetch_polygon_data(ticker, limit=5000):
    """Fetches 5-minute bars from Polygon.io."""
    # Polygon API for Aggregates (Bars)
    # Using 2024-01-01 start date to ensure we are within the 2-year lookback limit for standard keys
    url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/5/minute/2024-01-01/2029-01-01"
    params = {
        "adjusted": "true",
        "sort": "desc",
        "limit": limit,
        "apiKey": API_KEY
    }
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        
        if "results" not in data:
            logger.warning(f"No results for {ticker}. Check API key or ticker subscription.")
            return pd.DataFrame()
            
        df = pd.DataFrame(data["results"])
        # Polygon returns 't' (timestamp), 'o', 'h', 'l', 'c', 'v', 'vw', 'n'
        df['datetime'] = pd.to_datetime(df['t'], unit='ms')
        df = df.set_index('datetime').sort_index()
        # Rename columns to standard
        df = df.rename(columns={'o': 'Open', 'h': 'High', 'l': 'Low', 'c': 'Close', 'v': 'Volume'})
        return df[['Open', 'High', 'Low', 'Close', 'Volume']]
        
    except Exception as e:
        logger.error(f"Error fetching {ticker}: {e}")
        return pd.DataFrame()

def prepare_data(assets=ASSETS):
    """Fetches, aligns, and computes features for all assets."""
    logger.info("Fetching and aligning data...")
    combined_dfs = {}
    
    for asset, ticker in zip(assets, TICKERS):
        df = fetch_polygon_data(ticker)
        if df.empty:
            logger.error(f"Failed to load data for {asset}. Exiting.")
            sys.exit(1)
        
        # Rate limit: wait 12 seconds between requests for free tier (5 requests/min)
        time.sleep(12)
        
        # Calculate Features
        # 1. Log Returns
        df['LogRet'] = np.log(df['Close'] / df['Close'].shift(1))
        
        # 2. Rolling Volatility (2 hour window = 24 * 5min bars? No, 1 day = 288 bars)
        # Using 1 day window for meaningful annualization
        window = 288
        df['Vol'] = df['LogRet'].rolling(window=window).std() * np.sqrt(365 * 288) # Annualized
        
        # 3. Normalized Volume (MinMax over rolling window to avoid non-stationarity issues mostly)
        # Simplified: Log Volume
        df['LogVol'] = np.log1p(df['Volume'])
        
        # 4. High-Low Spread (Vol Proxy)
        df['HL_Spread'] = (df['High'] - df['Low']) / df['Close']
        
        # 5. Close-Open Spread (Directional momentum proxy)
        df['CO_Spread'] = (df['Close'] - df['Open']) / df['Close']
        
        # Select Features
        df_feats = df[['LogRet', 'Vol', 'LogVol', 'HL_Spread', 'CO_Spread']].copy()
        
        # Drop NaNs created by rolling windows/shifting
        df_feats = df_feats.dropna()
        
        # Rename columns to prevent collision during concat
        combined_dfs[asset] = df_feats

    # Alignment: Inner Join
    # Find common index
    common_index = None
    for asset in assets:
        if common_index is None:
            common_index = combined_dfs[asset].index
        else:
            common_index = common_index.intersection(combined_dfs[asset].index)
    
    if len(common_index) < TIMESTAMPS + 1:
        logger.error("Not enough aligned data points.")
        sys.exit(1)
        
    aligned_data = [] # Shape will be [Time, Assets, Features]
    
    # We need to stack them: T x A x F
    # Let's create a big Numpy array
    
    # Sort index to be sure
    common_index = common_index.sort_values()
    
    list_of_asset_arrays = []
    for asset in assets:
        df = combined_dfs[asset].loc[common_index]
        list_of_asset_arrays.append(df.values) # Shape (T, F)
        
    # Stack along axis 1 -> (T, A, F)
    # T = time, A = 5 assets, F = 5 features
    data_array = np.stack(list_of_asset_arrays, axis=1)
    
    return data_array, common_index

# --- DATASET ---

class CryptoDataset(Dataset):
    def __init__(self, data_array):
        """
        data_array: (T, A, F)
        x: (60, 5, 5)
        y: (5,) -> Volatility of next step (Feature index 1 is Vol)
        """
        self.data = torch.FloatTensor(data_array)
        self.timestamps = self.data.shape[0]
        self.lookback = TIMESTAMPS
        
    def __len__(self):
        return self.timestamps - self.lookback
        
    def __getitem__(self, idx):
        # x: idx to idx+60
        x = self.data[idx : idx + self.lookback, :, :] # (60, 5, 5)
        # y: Volatility at idx+60 (Target)
        # Feature 1 is Vol
        y = self.data[idx + self.lookback, :, 1] # (5,)
        return x, y

# --- MODEL ---

class VolatilityLSTM(nn.Module):
    def __init__(self, num_assets=5, num_features=5, hidden_size=64, num_layers=2):
        super(VolatilityLSTM, self).__init__()
        # Input to LSTM: Flatten features per timestep -> Input size = Assets * Features = 25
        self.input_size = num_assets * num_features
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(self.input_size, hidden_size, num_layers, batch_first=True)
        # Output head: Predict 5 volatilities
        self.fc = nn.Linear(hidden_size, num_assets)
        
    def forward(self, x):
        # x shape: (Batch, 60, 5, 5)
        B, T, A, F = x.size()
        
        # Flatten Asset and Feature dims
        x_flat = x.reshape(B, T, A * F) # (Batch, 60, 25)
        
        out, _ = self.lstm(x_flat)
        
        # Take last time step hidden state
        last_out = out[:, -1, :] # (Batch, 64)
        
        pred = self.fc(last_out) # (Batch, 5)
        return pred

# --- TRAINING ---

def train_model(data_array):
    logger.info("Starting training...")
    dataset = CryptoDataset(data_array)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    model = VolatilityLSTM()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        for x_batch, y_batch in dataloader:
            optimizer.zero_grad()
            output = model(x_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        logger.info(f"Epoch {epoch+1}/{EPOCHS} - Loss: {avg_loss:.6f}")
        
    torch.save(model.state_dict(), MODEL_PATH)
    logger.info("Model saved.")
    return model

# --- PREDICTION & ALERTS ---

def generate_alert(data_array, model):
    logger.info("Generating alert...")
    model.eval()
    
    # Get last window
    last_window = data_array[-TIMESTAMPS:, :, :] # (60, 5, 5)
    last_window_tensor = torch.FloatTensor(last_window).unsqueeze(0) # (1, 60, 5, 5)
    
    # Original Prediction
    with torch.no_grad():
        base_pred = model(last_window_tensor).numpy()[0] # (5,)
        
    # Scenario: BTC Drop 5.0%
    # BTC is asset 0, LogRet is feature 0
    # Shock: -0.05 log return (approx -5%)
    shock_tensor = last_window_tensor.clone()
    # Modify last timestep, BTC (0), LogRet (0)
    shock_tensor[0, -1, 0, 0] = -0.05
    
    with torch.no_grad():
        shock_pred = model(shock_tensor).numpy()[0] # (5,)
        
    # ETH is asset 1
    eth_vol_pred = shock_pred[1]
    
    # Heuristics for JSON Output
    eth_hist_vol = data_array[:, 1, 1]
    mean_vol = np.mean(eth_hist_vol)
    std_vol = np.std(eth_hist_vol)
    
    z_score = (eth_vol_pred - mean_vol) / (std_vol + 1e-6)
    
    # Map Z-score to probability (0-1)
    prob_spike = 1 / (1 + np.exp(-(z_score - 1))) # Shifted sigmoid
    prob_spike = round(float(prob_spike), 2)
    
    current_vol = data_array[-1, 1, 1]
    spike_pct = int(((eth_vol_pred - current_vol) / current_vol) * 100)
    
    warning_level = "LOW"
    if prob_spike > 0.5: warning_level = "MEDIUM"
    if prob_spike > 0.75: warning_level = "HIGH"
    
    alert = {
        "trigger": "BTC drop 5.0%",
        "ETH_vol_spike_probability": prob_spike,
        "expected_spike_percent": spike_pct,
        "warning_level": warning_level
    }
    
    print(json.dumps(alert, indent=2))
    return alert

# --- MAIN ENTRY ---

def main():
    if not API_KEY:
        logger.error("No API Key. Cannot proceed.")
        logger.error("Ensure .env file exists in: " + str(env_path))
        return

    # 1. Fetch & Process
    data_array, timestamps = prepare_data()
    if data_array.shape[0] == 0:
        logger.error("No data fetched.")
        return
        
    logger.info(f"Data shape: {data_array.shape}")
    
    # 2. Train
    train_model(data_array)
    
    # 3. Reload & Predict (as per requirements)
    model = VolatilityLSTM()
    model.load_state_dict(torch.load(MODEL_PATH))
    
    generate_alert(data_array, model)

if __name__ == "__main__":
    main()
