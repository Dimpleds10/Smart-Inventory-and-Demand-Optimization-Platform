import numpy as np
import pandas as pd
import pickle, os
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Input, RepeatVector, TimeDistributed
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

MODEL_DIR = "./models"
DATA_PATH = "./data/retail_store_inventory.csv"
SEQ_LEN   = 14
FORECAST_H = 1
os.makedirs(MODEL_DIR, exist_ok=True)
tf.config.threading.set_intra_op_parallelism_threads(4)
tf.config.threading.set_inter_op_parallelism_threads(4)

def build_sequences(df, category, units_scaler, error_scaler):
    cat_df = df[df['Category']==category].copy().sort_values(['Store ID','Product ID','Date']).reset_index(drop=True)
    cat_df['units_norm']    = units_scaler.transform(cat_df[['Units Sold']].values)
    cat_df['forecast_norm'] = units_scaler.transform(cat_df[['Demand Forecast']].values)
    cat_df['error']         = cat_df['Units Sold'] - cat_df['Demand Forecast']
    cat_df['error_norm']    = error_scaler.transform(cat_df[['error']].values)
    X_all, y_all = [], []
    for (store, product), group in cat_df.groupby(['Store ID','Product ID']):
        group = group.sort_values('Date').reset_index(drop=True)
        group['err_lag1']  = group['error_norm'].shift(1).fillna(0)
        group['err_lag7']  = group['error_norm'].shift(7).fillna(0)
        group['roll_err7'] = group['error_norm'].shift(1).rolling(7, min_periods=1).mean().fillna(0)
        group['lag1']      = group['units_norm'].shift(1).fillna(0)
        group['lag7']      = group['units_norm'].shift(7).fillna(0)
        group['roll7']     = group['units_norm'].shift(1).rolling(7, min_periods=1).mean().fillna(0)
        inv_sc = RobustScaler()
        group['inv_norm']  = inv_sc.fit_transform(group[['Inventory Level']])
        dates = pd.to_datetime(group['Date'])
        group['dow']       = dates.dt.dayofweek
        group['month']     = dates.dt.month
        group['weekend']   = (group['dow'] >= 5).astype(int)
        group['season']    = group['Seasonality'].map({'Spring':0,'Summer':1,'Autumn':2,'Winter':3}).fillna(0)
        cols = ['units_norm','forecast_norm','err_lag1','err_lag7','roll_err7',
                'lag1','lag7','roll7','inv_norm','Holiday/Promotion','dow','month','weekend','season']
        data = group[cols].values.astype(np.float32)
        for i in range(SEQ_LEN, len(data)-FORECAST_H):
            X_all.append(data[i-SEQ_LEN:i])
            y_all.append(float(group['error_norm'].iloc[i+FORECAST_H-1]))
    return np.array(X_all, dtype=np.float32), np.array(y_all, dtype=np.float32)

def build_model(seq_len, n_features):
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(seq_len, n_features)),
        BatchNormalization(), Dropout(0.2),
        LSTM(32, return_sequences=False),
        BatchNormalization(), Dropout(0.1),
        Dense(16, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer=Adam(0.001), loss='mse', metrics=['mae'])
    return model

def train_category(df, category, units_scaler, error_scaler):
    print(f"\n{'='*55}\n  Training: {category}\n{'='*55}")
    X, y = build_sequences(df, category, units_scaler, error_scaler)
    print(f"  Sequences: {X.shape[0]:,}  Shape: {X.shape}")
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=42)
    model = build_model(X.shape[1], X.shape[2])
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', patience=7, factor=0.5, verbose=1, min_lr=1e-6),
    ]
    model.fit(X_train, y_train, validation_data=(X_val,y_val),
              epochs=150, batch_size=64, callbacks=callbacks, verbose=1)

    y_pred_err      = model.predict(X_val, verbose=0).flatten()
    y_err_real      = error_scaler.inverse_transform(y_val.reshape(-1,1)).flatten()
    y_pred_err_real = error_scaler.inverse_transform(y_pred_err.reshape(-1,1)).flatten()

    cat_df  = df[df['Category']==category].copy().sort_values(['Store ID','Product ID','Date']).reset_index(drop=True)
    cat_val = cat_df[['Units Sold','Demand Forecast']].tail(len(y_val))
    actual  = cat_val['Units Sold'].values
    base_fc = cat_val['Demand Forecast'].values
    corrected = np.clip(base_fc + y_pred_err_real[:len(base_fc)], 0, None)

    mask           = actual > 10
    mape_base      = float(np.mean(np.abs((actual[mask]-base_fc[mask])/actual[mask]))*100)
    mape_corrected = float(np.mean(np.abs((actual[mask]-corrected[mask])/actual[mask]))*100) if mask.sum()>0 else 999

    print(f"\n  Results for {category}:")
    print(f"    Base forecast MAPE:   {mape_base:.2f}%")
    print(f"    LSTM corrected MAPE:  {mape_corrected:.2f}%")
    print(f"    Improvement:          {mape_base-mape_corrected:+.2f}%")
    print(f"\n  Sample (Actual | Base | LSTM Corrected):")
    for i in range(8):
        print(f"    Actual:{actual[i]:>5.0f}  Base:{base_fc[i]:>6.1f}  Corrected:{corrected[i]:>6.1f}")

    model.save(f"{MODEL_DIR}/lstm_{category.lower()}.keras")
    print(f"  Saved -> {MODEL_DIR}/lstm_{category.lower()}.keras")
    return {"mape_base": mape_base, "mape_corrected": mape_corrected}

def train_autoencoder():
    print(f"\n{'='*55}\n  Training: Autoencoder\n{'='*55}")
    X_all = np.load(f"{MODEL_DIR}/X_train.npy")
    seq_len, n_features = X_all.shape[1], X_all.shape[2]
    X_normal = X_all[X_all[:,-1,0] <= np.percentile(X_all[:,-1,0], 95)]
    X_train, X_val = train_test_split(X_normal, test_size=0.1, shuffle=False)
    inputs  = Input(shape=(seq_len, n_features))
    encoded = LSTM(64, return_sequences=False)(inputs)
    decoded = RepeatVector(seq_len)(encoded)
    decoded = LSTM(64, return_sequences=True)(decoded)
    decoded = TimeDistributed(Dense(n_features))(decoded)
    ae = Model(inputs, decoded)
    ae.compile(optimizer=Adam(0.001), loss='mse')
    ae.fit(X_train, X_train, validation_data=(X_val,X_val), epochs=50, batch_size=64,
           callbacks=[EarlyStopping(patience=8, restore_best_weights=True)], verbose=1)
    recon_err = np.mean(np.abs(X_val - ae.predict(X_val,verbose=0)), axis=(1,2))
    threshold = float(np.percentile(recon_err, 95))
    with open(f"{MODEL_DIR}/anomaly_threshold.pkl","wb") as f:
        pickle.dump(threshold, f)
    ae.save(f"{MODEL_DIR}/autoencoder_best.keras")
    print(f"  Threshold={threshold:.4f}  Saved!")

if __name__ == "__main__":
    print("="*55)
    print("  RESIDUAL LEARNING — LSTM Corrects Existing Forecast")
    print("="*55)
    df = pd.read_csv(DATA_PATH)
    df['Date'] = pd.to_datetime(df['Date'])
    units_scaler = RobustScaler()
    error_scaler = RobustScaler()
    units_scaler.fit(df[['Units Sold']])
    df['error_temp'] = df['Units Sold'] - df['Demand Forecast']
    error_scaler.fit(df[['error_temp']])
    with open(f"{MODEL_DIR}/scaler.pkl","wb") as f:
        pickle.dump(units_scaler, f)
    with open(f"{MODEL_DIR}/error_scaler.pkl","wb") as f:
        pickle.dump(error_scaler, f)
    print("  Scalers saved")
    categories  = ['Clothing','Electronics','Furniture','Groceries','Toys']
    all_metrics = {}
    for cat in categories:
        all_metrics[cat] = train_category(df, cat, units_scaler, error_scaler)
    train_autoencoder()
    print("\n"+"="*55)
    print("  ALL MODELS TRAINED!")
    print(f"  {'Category':<15} {'Base MAPE':>10} {'Our MAPE':>10} {'Improvement':>12}")
    print(f"  {'-'*52}")
    for cat, m in all_metrics.items():
        imp = m['mape_base'] - m['mape_corrected']
        print(f"  {cat:<15} {m['mape_base']:>9.1f}%  {m['mape_corrected']:>9.1f}%  {imp:>+11.1f}%")
    print("="*55)
