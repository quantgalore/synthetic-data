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
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import accuracy_score, classification_report
    
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

ticker = "NGUSD"

ticker_data_original = pd.json_normalize(requests.get(f"https://financialmodelingprep.com/api/v3/historical-price-full/{ticker}?apikey={fmp_api_key}").json()["historical"]).sort_values(by="date", ascending=True)

full_oos_data = ticker_data_original.copy()

full_oos_data["c"] = full_oos_data["close"]

full_oos_data["fast_avg"] = full_oos_data["c"].rolling(window=5).mean()
full_oos_data["slow_avg"] = full_oos_data["c"].rolling(window=10).mean()

full_oos_data["divergence"] = round(((full_oos_data["fast_avg"] - full_oos_data["slow_avg"]) / full_oos_data["slow_avg"]) * 100, 2)
full_oos_data["t-1"] = full_oos_data["divergence"].shift(1)

full_oos_data["prior_high"] = full_oos_data["c"].rolling(window=252).max()

full_oos_data["distance_from_high"] = round(((full_oos_data["c"] - full_oos_data["prior_high"]) / full_oos_data["prior_high"]) * 100, 2)
full_oos_data["distance_t-1"] = full_oos_data["distance_from_high"].shift(1)

full_oos_data = full_oos_data.dropna()

trading_dates = full_oos_data["date"].values

pred_list = []

# date = trading_dates[:-5][0]
for date in trading_dates[:-5]:
    
    oos_data = full_oos_data[full_oos_data["date"] == date].copy()
    
    X_test = oos_data[features]

    prediction = Model.predict(X_test)[0]
    prediction_proba = Model.predict_proba(X_test)[:, 1][0]
    
    forward_data = full_oos_data[full_oos_data["date"] >= date].copy()
    forward_price = forward_data["c"].iloc[5]
    
    if forward_price > oos_data["c"].iloc[0]:
        actual = 1
        forward_pnl = forward_price - oos_data["c"].iloc[0]
    else:
        actual = 0
        forward_pnl = forward_price - oos_data["c"].iloc[0]
        
    real_actual = np.int64(forward_price > oos_data["prior_high"].iloc[0])
        
    pred_data = pd.DataFrame([{"date": date, "pred": prediction, "1_proba": prediction_proba, "real_actual": real_actual,
                               "actual": actual, "price": oos_data["c"].iloc[0], "forward_pnl": forward_pnl}])
    
    pred_list.append(pred_data)

full_prediction_data = pd.concat(pred_list)

# =============================================================================
# Model Analysis
# =============================================================================

preds = full_prediction_data["pred"].values
probas = full_prediction_data["1_proba"].values
actuals = full_prediction_data["real_actual"].values

print("\Accuracy Score:")
print(accuracy_score(actuals, preds))

print("\nClassification Report:")
print(classification_report(actuals, preds))

# Calculate ROC curve and AUC
fpr, tpr, thresholds = roc_curve(actuals, probas)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure(figsize=(10, 8), dpi = 600)
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Future Features –– ROC Curve')
plt.legend(loc="lower right")
plt.show()

x = full_prediction_data.corr(numeric_only=True)

signals = full_prediction_data[full_prediction_data["pred"] == 1]
no_signals = full_prediction_data[full_prediction_data["pred"] == 0]

plt.figure(dpi=600)
plt.xticks(rotation=45)
plt.plot(pd.to_datetime(full_prediction_data["date"]), full_prediction_data["price"])
for _, row in signals.iterrows():
    plt.annotate(text="^", 
                 xy=(pd.to_datetime(row["date"]), row["price"]),  # (date, price)
                 xytext=(0, 10),  # Offset for the annotation
                 textcoords="offset points", 
                 ha='center', color='green')
plt.title(f"Synthetic Features on {all_comms[all_comms['symbol']==ticker]['name'].iloc[0]}")
plt.show()