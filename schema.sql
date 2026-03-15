-- ============================================================
-- schema.sql
-- AI-Driven Smart Inventory & Demand Optimization Platform
-- Fashion Boutique Dataset
-- ============================================================

USE smart_inventory;

CREATE TABLE IF NOT EXISTS sales (
    id                  BIGINT AUTO_INCREMENT PRIMARY KEY,
    product_id          VARCHAR(20),
    category            VARCHAR(50),
    brand               VARCHAR(50),
    season              VARCHAR(20),
    size                VARCHAR(20),
    color               VARCHAR(30),
    original_price      FLOAT,
    markdown_percentage FLOAT,
    current_price       FLOAT,
    purchase_date       DATE,
    stock_quantity      INT,
    customer_rating     FLOAT,
    is_returned         TINYINT(1),
    return_reason       VARCHAR(100),
    INDEX idx_date     (purchase_date),
    INDEX idx_category (category)
);

CREATE TABLE IF NOT EXISTS daily_sales (
    id              BIGINT AUTO_INCREMENT PRIMARY KEY,
    sale_date       DATE NOT NULL,
    category        VARCHAR(50) NOT NULL,
    daily_revenue   FLOAT,
    units_sold      INT,
    avg_price       FLOAT,
    avg_markdown    FLOAT,
    stock_quantity  FLOAT,
    INDEX idx_date_cat (sale_date, category)
);

CREATE TABLE IF NOT EXISTS demand_forecasts (
    id              BIGINT AUTO_INCREMENT PRIMARY KEY,
    forecast_date   DATE NOT NULL,
    category        VARCHAR(50) NOT NULL,
    predicted_sales FLOAT NOT NULL,
    actual_sales    FLOAT,
    confidence_low  FLOAT,
    confidence_high FLOAT,
    model_version   VARCHAR(50) DEFAULT 'lstm_v1',
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_date_cat (forecast_date, category)
);

CREATE TABLE IF NOT EXISTS anomalies (
    id                   BIGINT AUTO_INCREMENT PRIMARY KEY,
    anomaly_date         DATE NOT NULL,
    category             VARCHAR(50) NOT NULL,
    actual_revenue       FLOAT,
    reconstruction_error FLOAT,
    threshold            FLOAT,
    anomaly_type         ENUM('spike', 'drop', 'unknown') DEFAULT 'unknown',
    severity             ENUM('low', 'medium', 'high') DEFAULT 'medium',
    is_reviewed          TINYINT(1) DEFAULT 0,
    detected_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_date_cat (anomaly_date, category)
);

CREATE TABLE IF NOT EXISTS inventory_insights (
    id                BIGINT AUTO_INCREMENT PRIMARY KEY,
    insight_date      DATE NOT NULL,
    category          VARCHAR(50) NOT NULL,
    current_stock     FLOAT,
    recommended_stock FLOAT,
    reorder_flag      TINYINT(1) DEFAULT 0,
    overstock_flag    TINYINT(1) DEFAULT 0,
    insight_notes     TEXT,
    created_at        TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS model_registry (
    id           INT AUTO_INCREMENT PRIMARY KEY,
    model_name   VARCHAR(100),
    model_type   ENUM('lstm', 'autoencoder'),
    version      VARCHAR(20),
    mae          FLOAT,
    rmse         FLOAT,
    mape         FLOAT,
    is_active    TINYINT(1) DEFAULT 0,
    trained_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    notes        TEXT
);

INSERT INTO model_registry (model_name, model_type, version, is_active, notes)
VALUES
('LSTM Demand Forecaster',       'lstm',        'v1.0', 1, 'Trained on Fashion Boutique 2025 dataset'),
('Autoencoder Anomaly Detector', 'autoencoder', 'v1.0', 1, 'Reconstruction error threshold based');

CREATE OR REPLACE VIEW vw_recent_forecasts AS
SELECT category, forecast_date, predicted_sales, actual_sales, confidence_low, confidence_high
FROM demand_forecasts ORDER BY forecast_date DESC LIMIT 100;

CREATE OR REPLACE VIEW vw_open_anomalies AS
SELECT category, anomaly_date, actual_revenue, anomaly_type, severity, reconstruction_error
FROM anomalies WHERE is_reviewed = 0
ORDER BY severity DESC, anomaly_date DESC;
