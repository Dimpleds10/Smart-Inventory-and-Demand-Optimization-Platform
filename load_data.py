# =============================================================
# load_data.py
# Loads CSV data into MySQL database
# =============================================================
# Run this AFTER schema.sql has been executed
# It will:
#   1. Load raw sales data into the sales table
#   2. Load daily aggregated data into daily_sales table
# =============================================================

import pandas as pd
import pymysql
from sqlalchemy import create_engine
import os

# ─── DATABASE SETTINGS ───────────────────────────────────────
# Change password to whatever you set during MySQL installation
DB_HOST     = "localhost"
DB_USER     = "root"
DB_PASSWORD = "dds2005"       # ← change this if different
DB_NAME     = "smart_inventory"

# ─── FILE PATHS ──────────────────────────────────────────────
RAW_CSV       = "./data/fashion_boutique_dataset.csv"
PROCESSED_CSV = "./data/processed_daily.csv"

# =============================================================
# CONNECT TO DATABASE
# =============================================================
def get_engine():
    url = f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}/{DB_NAME}"
    engine = create_engine(url)
    return engine

# =============================================================
# STEP 1 — LOAD RAW SALES DATA
# =============================================================
def load_sales(engine):
    print("\n📂 STEP 1: Loading raw sales data into MySQL...")
    df = pd.read_csv(RAW_CSV)
    df['purchase_date'] = pd.to_datetime(df['purchase_date']).dt.date
    df['is_returned']   = df['is_returned'].astype(int)
    df['return_reason'].fillna('None', inplace=True)
    df['size'].fillna('Unknown', inplace=True)
    df['customer_rating'].fillna(df['customer_rating'].mean(), inplace=True)

    df.to_sql('sales', engine, if_exists='replace', index=False, chunksize=500)
    print(f"   ✅ Loaded {len(df):,} rows into sales table")

# =============================================================
# STEP 2 — LOAD DAILY AGGREGATED DATA
# =============================================================
def load_daily_sales(engine):
    print("\n📅 STEP 2: Loading daily aggregated data into MySQL...")
    df = pd.read_csv(PROCESSED_CSV)
    df['purchase_date'] = pd.to_datetime(df['purchase_date']).dt.date

    # Select only the columns that match our daily_sales table
    daily = df[['purchase_date', 'category', 'daily_revenue',
                'units_sold', 'avg_price', 'avg_markdown', 'stock_quantity']].copy()
    daily = daily.rename(columns={'purchase_date': 'sale_date'})

    daily.to_sql('daily_sales', engine, if_exists='replace', index=False, chunksize=500)
    print(f"   ✅ Loaded {len(daily):,} rows into daily_sales table")

# =============================================================
# STEP 3 — VERIFY DATA
# =============================================================
def verify(engine):
    print("\n🔍 STEP 3: Verifying data in MySQL...")
    with engine.connect() as conn:
        from sqlalchemy import text

        sales_count = conn.execute(text("SELECT COUNT(*) FROM sales")).scalar()
        daily_count = conn.execute(text("SELECT COUNT(*) FROM daily_sales")).scalar()
        categories  = conn.execute(text("SELECT DISTINCT category FROM daily_sales")).fetchall()
        date_range  = conn.execute(text("SELECT MIN(sale_date), MAX(sale_date) FROM daily_sales")).fetchone()

        print(f"   ✅ sales table:       {sales_count:,} rows")
        print(f"   ✅ daily_sales table: {daily_count:,} rows")
        print(f"   ✅ Categories: {[r[0] for r in categories]}")
        print(f"   ✅ Date range: {date_range[0]} → {date_range[1]}")

# =============================================================
# MAIN
# =============================================================
if __name__ == "__main__":
    print("=" * 55)
    print("  LOADING DATA INTO MYSQL")
    print("=" * 55)

    try:
        engine = get_engine()
        print("   ✅ Connected to MySQL successfully!")

        load_sales(engine)
        load_daily_sales(engine)
        verify(engine)

        print("\n" + "=" * 55)
        print("  ✅ ALL DATA LOADED INTO MYSQL!")
        print("=" * 55)

    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\n💡 Check:")
        print("   1. Is MySQL running?")
        print("   2. Is your password correct in this file?")
        print("   3. Did you run schema.sql first?")
