# -*- coding: utf-8 -*-
"""
Created in 2024

@author: Quant Galore
"""

import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from datetime import datetime, timedelta
from pandas_market_calendars import get_calendar

from sklearn.linear_model import LogisticRegression

# =============================================================================
# Model Building 
# =============================================================================

synthetic_data = pd.read_csv("synth_data.csv")

features = ["divergence", "t-1", "distance_from_high",  "distance_t-1"]
target = "break_new"

X = synthetic_data[features]
Y = synthetic_data[target].values.ravel()

Model = LogisticRegression().fit(X, Y)

# =============================================================================
# Live Predictions
# =============================================================================

polygon_api_key = "KkfCQ7fsZnx0yK4bhX9fD81QplTh0Pf3"
calendar = get_calendar("NYSE")

all_tickers = pd.read_csv("liquid_tickers.csv")["ticker"].values

trading_dates = calendar.schedule(start_date = "2023-05-01", end_date = (datetime.today()-timedelta(days=7))).index.strftime("%Y-%m-%d").values

pred_list = []

# ticker = all_comms["symbol"].values[0]
for ticker in all_tickers:
    
    try:
    
        ticker_data_original = pd.json_normalize(requests.get(f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{trading_dates[0]}/{trading_dates[-1]}?adjusted=true&sort=asc&limit=50000&apiKey={polygon_api_key}").json()["results"]).set_index("t")
        ticker_data_original.index = pd.to_datetime(ticker_data_original.index, unit="ms", utc=True).tz_convert("America/New_York")
        
        ticker_data = ticker_data_original.copy()
        
        ticker_data["fast_avg"] = ticker_data["c"].rolling(window=5).mean()
        ticker_data["slow_avg"] = ticker_data["c"].rolling(window=10).mean()
        
        ticker_data["divergence"] = round(((ticker_data["fast_avg"] - ticker_data["slow_avg"]) / ticker_data["slow_avg"]) * 100, 2)
        ticker_data["t-1"] = ticker_data["divergence"].shift(1)
        
        ticker_data["prior_high"] = ticker_data["c"].rolling(window=252).max()
        
        ticker_data["distance_from_high"] = round(((ticker_data["c"] - ticker_data["prior_high"]) / ticker_data["prior_high"]) * 100, 2)
        ticker_data["distance_t-1"] = ticker_data["distance_from_high"].shift(1)
    
        oos_data = ticker_data.tail(1)
        X_test = oos_data[features]
    
        prediction = Model.predict(X_test)[0]
        prediction_proba = Model.predict_proba(X_test)[:, 1][0]
        
        pred_data = pd.DataFrame([{"date": oos_data.index[-1].strftime("%Y-%m-%d"), "pred": prediction, "1_proba": prediction_proba, "ticker": ticker}])
        
        pred_list.append(pred_data)
        
    except Exception as data_error:
        continue

all_predictions = pd.concat(pred_list)
