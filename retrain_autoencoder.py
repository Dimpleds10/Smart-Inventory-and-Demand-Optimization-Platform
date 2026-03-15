import numpy as np
import pandas as pd
import pickle, os
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Input, RepeatVector, TimeDistributed
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

MODEL_DIR = "./models"
DATA_PATH = "./data/retail_store_inventory.csv"
SEQ_LEN   = 14

print("="*50)
print("  Retraining Autoencoder (shape fix)")
print("="*50)

# Load data and build sequences with correct shape (14, 14)
df = pd.read_csv(DATA_PATH)
df['Date'] = pd.to_datetime(df['Date'])

scalers = {
    "units": pickle.load(open(f"{MODEL_DIR}/scaler.pkl","rb")),
    "error": pickle.load(open(f"{MODEL_DIR}/error_scaler.pkl","rb")),
}

X_all = []

for (store, product, category), group in df.groupby(['Store ID','Product ID','Category']):
    group = group.sort_values('Date').reset_index(drop=True)

    group['units_norm']    = scalers["units"].transform(group[['Units Sold']].values)
    group['forecast_norm'] = scalers["units"].transform(group[['Demand Forecast']].values)
    group['error']         = group['Units Sold'] - group['Demand Forecast']
    group['error_norm']    = scalers["error"].transform(group[['error']].values)

    group['err_lag1']  = group['error_norm'].shift(1).fillna(0)
    group['err_lag7']  = group['error_norm'].shift(7).fillna(0)
    group['roll_err7'] = group['error_norm'].shift(1).rolling(7, min_periods=1).mean().fillna(0)
    group['lag1']      = group['units_norm'].shift(1).fillna(0)
    group['lag7']      = group['units_norm'].shift(7).fillna(0)
    group['roll7']     = group['units_norm'].shift(1).rolling(7, min_periods=1).mean().fillna(0)

    inv_sc = RobustScaler()
    group['inv_norm'] = inv_sc.fit_transform(group[['Inventory Level']])

    dates = pd.to_datetime(group['Date'])
    group['dow']     = dates.dt.dayofweek
    group['month']   = dates.dt.month
    group['weekend'] = (group['dow'] >= 5).astype(int)
    group['season']  = group['Seasonality'].map(
        {'Spring':0,'Summer':1,'Autumn':2,'Winter':3}).fillna(0)

    cols = ['units_norm','forecast_norm','err_lag1','err_lag7','roll_err7',
            'lag1','lag7','roll7','inv_norm','Holiday/Promotion',
            'dow','month','weekend','season']

    data = group[cols].values.astype(np.float32)

    for i in range(SEQ_LEN, len(data)):
        X_all.append(data[i-SEQ_LEN:i])

X_all = np.array(X_all, dtype=np.float32)
print(f"  Sequences: {X_all.shape}")  # should be (N, 14, 14)

# Train only on normal samples
sales_vals = X_all[:, -1, 0]
X_normal   = X_all[sales_vals <= np.percentile(sales_vals, 95)]
X_train, X_val = train_test_split(X_normal, test_size=0.1, shuffle=False)
print(f"  Normal samples: {len(X_normal):,}")

seq_len, n_features = X_all.shape[1], X_all.shape[2]
print(f"  Shape: ({seq_len}, {n_features})")

# Build autoencoder with CORRECT shape
inputs  = Input(shape=(seq_len, n_features))
encoded = LSTM(64, return_sequences=False)(inputs)
decoded = RepeatVector(seq_len)(encoded)
decoded = LSTM(64, return_sequences=True)(decoded)
decoded = TimeDistributed(Dense(n_features))(decoded)
ae = Model(inputs, decoded)
ae.compile(optimizer=Adam(0.001), loss='mse')

ae.fit(X_train, X_train,
       validation_data=(X_val, X_val),
       epochs=50, batch_size=64,
       callbacks=[EarlyStopping(patience=8, restore_best_weights=True)],
       verbose=1)

# Save threshold
recon_err = np.mean(np.abs(X_val - ae.predict(X_val, verbose=0)), axis=(1,2))
threshold = float(np.percentile(recon_err, 95))

with open(f"{MODEL_DIR}/anomaly_threshold.pkl","wb") as f:
    pickle.dump(threshold, f)

ae.save(f"{MODEL_DIR}/autoencoder_best.keras")

print(f"\n  ✅ Autoencoder saved with shape ({seq_len}, {n_features})")
print(f"  ✅ Threshold = {threshold:.4f}")
print("="*50)
print("  DONE! Restart flask_app.py now.")
print("="*50)
