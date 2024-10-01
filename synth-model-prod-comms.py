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

fmp_api_key = "be9dfaeee066f09ff52f1504659cffcb"
calendar = get_calendar("NYSE")

all_comms = pd.json_normalize(requests.get(f"https://financialmodelingprep.com/api/v3/symbol/available-commodities?apikey={fmp_api_key}").json())

pred_list = []

# ticker = all_comms["symbol"].values[0]
for ticker in all_comms["symbol"].values:

    ticker_data_original = pd.json_normalize(requests.get(f"https://financialmodelingprep.com/api/v3/historical-price-full/{ticker}?apikey={fmp_api_key}").json()["historical"]).sort_values(by="date", ascending=True)
    
    ticker_data = ticker_data_original.copy()
    
    ticker_data["fast_avg"] = ticker_data["close"].rolling(window=5).mean()
    ticker_data["slow_avg"] = ticker_data["close"].rolling(window=10).mean()
    
    ticker_data["divergence"] = round(((ticker_data["fast_avg"] - ticker_data["slow_avg"]) / ticker_data["slow_avg"]) * 100, 2)
    ticker_data["t-1"] = ticker_data["divergence"].shift(1)
    
    ticker_data["prior_high"] = ticker_data["close"].rolling(window=252).max()
    
    ticker_data["distance_from_high"] = round(((ticker_data["close"] - ticker_data["prior_high"]) / ticker_data["prior_high"]) * 100, 2)
    ticker_data["distance_t-1"] = ticker_data["distance_from_high"].shift(1)

    oos_data = ticker_data.tail(1)
    X_test = oos_data[features]

    prediction = Model.predict(X_test)[0]
    prediction_proba = Model.predict_proba(X_test)[:, 1][0]
    
    pred_data = pd.DataFrame([{"date": oos_data["date"].iloc[0], "pred": prediction, "1_proba": prediction_proba, "ticker": ticker,
                               "name": all_comms[all_comms['symbol']==ticker]['name'].iloc[0]}])
    
    pred_list.append(pred_data)
    
    plt.figure(dpi=600)
    plt.xticks(rotation=45)
    plt.plot(pd.to_datetime(ticker_data["date"][-252:]), ticker_data["close"][-252:])
    plt.title(f"{all_comms[all_comms['symbol']==ticker]['name'].iloc[0]}")
    plt.show()

all_predictions = pd.concat(pred_list)
