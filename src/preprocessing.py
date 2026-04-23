# =============================================================
# preprocessing.py — Retail Store Inventory Dataset
# =============================================================
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler, LabelEncoder
import pickle, os

DATA_PATH  = "./data/retail_store_inventory.csv"
MODEL_DIR  = "./models"
SEQ_LEN    = 30
FORECAST_H = 7
os.makedirs(MODEL_DIR, exist_ok=True)

def load_data():
    print("\n📂 STEP 1: Loading dataset...")
    df = pd.read_csv(DATA_PATH)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(['Store ID', 'Product ID', 'Date']).reset_index(drop=True)
    print(f"   ✅ Loaded {len(df):,} rows")
    print(f"   Date range: {df['Date'].min().date()} to {df['Date'].max().date()}")
    print(f"   Stores: {df['Store ID'].nunique()} | Products: {df['Product ID'].nunique()} | Categories: {df['Category'].nunique()}")
    return df

def clean_data(df):
    print("\n🧹 STEP 2: Cleaning data...")
    before = len(df)
    df = df.dropna().copy()
    print(f"   ✅ Removed {before - len(df)} null rows")
    print(f"   ✅ Clean dataset: {len(df):,} rows")
    return df

def encode_categories(df):
    print("\n🔤 STEP 3: Encoding categorical columns...")
    encoders = {}
    for col in ['Category', 'Region', 'Weather Condition', 'Seasonality', 'Store ID']:
        le = LabelEncoder()
        df[f'{col}_enc'] = le.fit_transform(df[col])
        encoders[col] = le
        print(f"   ✅ {col}: {list(le.classes_)}")
    with open(f"{MODEL_DIR}/encoders.pkl", "wb") as f:
        pickle.dump(encoders, f)
    return df, encoders

def engineer_features(df):
    print("\n⚙️  STEP 4: Engineering features...")
    df['day_of_week']  = df['Date'].dt.dayofweek
    df['month']        = df['Date'].dt.month
    df['week_of_year'] = df['Date'].dt.isocalendar().week.astype(int)
    df['is_weekend']   = (df['day_of_week'] >= 5).astype(int)
    df['quarter']      = df['Date'].dt.quarter

    group_key = ['Store ID', 'Product ID']
    for lag in [1, 7, 14, 30]:
        df[f'lag_{lag}'] = df.groupby(group_key)['Units Sold'].shift(lag).fillna(0)

    df['rolling_mean_7']  = df.groupby(group_key)['Units Sold'].transform(lambda x: x.shift(1).rolling(7,  min_periods=1).mean()).fillna(0)
    df['rolling_mean_30'] = df.groupby(group_key)['Units Sold'].transform(lambda x: x.shift(1).rolling(30, min_periods=1).mean()).fillna(0)
    df['rolling_std_7']   = df.groupby(group_key)['Units Sold'].transform(lambda x: x.shift(1).rolling(7,  min_periods=1).std()).fillna(0)

    df['discount_factor'] = df['Discount'] / 100
    df['price_diff']      = df['Price'] - df['Competitor Pricing']
    df['effective_price'] = df['Price'] * (1 - df['discount_factor'])

    print(f"   ✅ Features engineered. Shape: {df.shape}")
    return df

def normalize(df):
    print("\n📐 STEP 5: Normalizing...")
    scaler = RobustScaler()
    df['units_sold_normalized'] = scaler.fit_transform(df[['Units Sold']])
    with open(f"{MODEL_DIR}/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    print(f"   ✅ Scaler saved")
    return df, scaler

def build_sequences(df):
    print("\n🔢 STEP 6: Building LSTM sequences...")
    feature_cols = [
        'units_sold_normalized',
        'lag_1', 'lag_7', 'lag_14', 'lag_30',
        'rolling_mean_7', 'rolling_mean_30', 'rolling_std_7',
        'Price', 'discount_factor', 'effective_price', 'price_diff',
        'Inventory Level', 'Holiday/Promotion',
        'day_of_week', 'month', 'week_of_year', 'is_weekend', 'quarter',
        'Category_enc', 'Region_enc', 'Weather Condition_enc', 'Seasonality_enc',
    ]
    X_all, y_all = [], []
    for (store, product), group in df.groupby(['Store ID', 'Product ID']):
        group = group.sort_values('Date').reset_index(drop=True)
        data  = group[feature_cols].values
        for i in range(SEQ_LEN, len(data) - FORECAST_H):
            X_all.append(data[i - SEQ_LEN : i])
            y_all.append(data[i + FORECAST_H - 1, 0])

    X = np.array(X_all, dtype=np.float32)
    y = np.array(y_all, dtype=np.float32)
    np.save(f"{MODEL_DIR}/X_train.npy", X)
    np.save(f"{MODEL_DIR}/y_train.npy", y)
    with open(f"{MODEL_DIR}/feature_cols.pkl", "wb") as f:
        pickle.dump(feature_cols, f)
    print(f"   ✅ X shape: {X.shape}")
    print(f"   ✅ y shape: {y.shape}")
    return X, y, feature_cols

def save_processed(df):
    print("\n💾 STEP 7: Saving processed CSV...")
    df.to_csv("./data/processed_inventory.csv", index=False)
    categories = df['Category'].unique().tolist()
    stores     = df['Store ID'].unique().tolist()
    with open(f"{MODEL_DIR}/categories.pkl", "wb") as f:
        pickle.dump({'categories': categories, 'stores': stores}, f)
    print(f"   ✅ Categories: {categories}")
    print(f"   ✅ Stores: {stores}")

if __name__ == "__main__":
    print("=" * 60)
    print("  PREPROCESSING — Retail Store Inventory Dataset")
    print("=" * 60)
    df           = load_data()
    df           = clean_data(df)
    df, encoders = encode_categories(df)
    df           = engineer_features(df)
    df, scaler   = normalize(df)
    X, y, cols   = build_sequences(df)
    save_processed(df)
    print("\n" + "=" * 60)
    print("  ✅ PREPROCESSING COMPLETE!")
    print(f"  Ready to train on {X.shape[0]:,} sequences")
    print(f"  Each = {X.shape[1]} days x {X.shape[2]} features")
    print("=" * 60)
