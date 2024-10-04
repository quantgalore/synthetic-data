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

synthetic_data = pd.read_csv("synth_trend_data.csv")

features = ["divergence", "t-1", "t-2",  "t-3", "distance_from_high", "distance_t-2", "distance_t-3"]
target = "trend?"

X = synthetic_data[features]
Y = synthetic_data[target].values.ravel()

Model = LogisticRegression().fit(X, Y)

# =============================================================================
# Model Testing
# =============================================================================

fmp_api_key = "be9dfaeee066f09ff52f1504659cffcb"
all_comms = pd.json_normalize(requests.get(f"https://financialmodelingprep.com/api/v3/symbol/available-commodities?apikey={fmp_api_key}").json())

# ['ZQUSD', 'ALIUSD', 'ZBUSD', 'ZLUSX', 'KEUSX', 'ZFUSD', 'PLUSD',
#        'SILUSD', 'HEUSX', 'ZCUSX', 'ZOUSX', 'ESUSD', 'ZMUSD', 'GCUSD',
#        'SBUSX', 'SIUSD', 'CTUSX', 'MGCUSD', 'DXUSD', 'ZSUSX', 'LBUSD',
#        'LEUSX', 'NGUSD', 'CLUSD', 'OJUSX', 'KCUSX', 'HGUSD', 'GFUSX',
#        'ZTUSD', 'ZRUSD', 'PAUSD', 'CCUSD', 'NQUSD', 'ZNUSD', 'RTYUSD',
#        'BZUSD', 'DCUSD', 'YMUSD', 'RBUSD', 'HOUSD']

ticker = "GCUSD"

ticker_data_original = pd.json_normalize(requests.get(f"https://financialmodelingprep.com/api/v3/historical-price-full/{ticker}?apikey={fmp_api_key}").json()["historical"]).sort_values(by="date", ascending=True)

full_oos_data = ticker_data_original.copy()

full_oos_data["c"] = full_oos_data["close"]

full_oos_data["prior_high"] = full_oos_data["c"].rolling(window=60).max()
full_oos_data["slow_avg"] = full_oos_data["c"].rolling(window=20).mean()

full_oos_data["divergence"] = round(((full_oos_data["c"] - full_oos_data["slow_avg"]) / full_oos_data["slow_avg"]) * 100, 2)
full_oos_data["t-1"] = full_oos_data["divergence"].shift(1)
full_oos_data["t-2"] = full_oos_data["divergence"].shift(2)
full_oos_data["t-3"] = full_oos_data["divergence"].shift(3)

full_oos_data["distance_from_high"] = round(((full_oos_data["c"] - full_oos_data["prior_high"]) / full_oos_data["prior_high"]) * 100, 2)
full_oos_data["distance_t-1"] = full_oos_data["divergence"].shift(1)
full_oos_data["distance_t-2"] = full_oos_data["divergence"].shift(2)
full_oos_data["distance_t-3"] = full_oos_data["divergence"].shift(3)

plt.figure(dpi=600)
plt.xticks(rotation=45)
plt.plot(pd.to_datetime(full_oos_data["date"]), full_oos_data["c"])
plt.title(f"{all_comms[all_comms['symbol']==ticker]['name'].iloc[0]}")
plt.show()

full_oos_data = full_oos_data.dropna()

trading_dates = full_oos_data["date"].values

pred_list = []

# date = trading_dates[:-5][0]
for date in trading_dates[:-5]:
    
    oos_data = full_oos_data[full_oos_data["date"] == date].copy()
    
    X_test = oos_data[features]

    prediction = Model.predict(X_test)[0]
    prediction_proba = Model.predict_proba(X_test)[:, 1][0]
    
    pred_data = pd.DataFrame([{"date": date, "pred": prediction, "1_proba": prediction_proba, "price": oos_data["c"].iloc[0]}])
    
    pred_list.append(pred_data)

full_prediction_data = pd.concat(pred_list)

# =============================================================================
# Model Analysis
# =============================================================================

signals = full_prediction_data[full_prediction_data["pred"] == 1]

plt.figure(dpi=600)
plt.xticks(rotation=45)
plt.plot(pd.to_datetime(full_prediction_data["date"]), full_prediction_data["price"])
for _, row in signals.iterrows():
    plt.annotate(text="^", 
                 xy=(pd.to_datetime(row["date"]), row["price"]),  # (date, price)
                 xytext=(0, 10),  # Offset for the annotation
                 textcoords="offset points", 
                 ha='center', color='green'
                 )
plt.title(f"Synthetic Data Performance on {all_comms[all_comms['symbol']==ticker]['name'].iloc[0]}")
plt.show()