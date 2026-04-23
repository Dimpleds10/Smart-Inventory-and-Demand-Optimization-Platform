# AI-Driven Smart Inventory & Demand Optimization Platform

## Overview

This project presents an **AI-driven Smart Inventory System** that forecasts product demand and detects anomalies in retail sales using deep learning techniques.

Using a real-world **Retail Store Inventory dataset (73,100 records)**, the system combines:

* **LSTM (Long Short-Term Memory)** for demand forecasting
* **Autoencoder Neural Networks** for anomaly detection

The system helps retailers minimize **overstocking and stockouts** by generating accurate predictions and intelligent inventory insights. 

---
## Team Members
1. Dimple Sachanandani
2. Niyati Yele
3. Pranav Dighole
Branch:- Electronics & Computer Science

## Objectives

* Build an AI-based system for demand forecasting and anomaly detection
* Transform raw retail data into structured time-series sequences
* Improve existing demand forecasts using **Residual Learning with LSTM**
* Detect abnormal sales patterns using Autoencoder
* Generate reorder and overstock alerts
* Provide decision support via dashboard

---

## Dataset

* **Source:** Kaggle (Retail Store Inventory Dataset) https://www.kaggle.com/datasets/anirudhchauhan/retail-store-inventory-forecasting-dataset
* **Size:** 73,100 rows
* **Duration:** 2022 – 2024
* **Entities:**

  * 5 Stores
  * 5 Product Categories

---

## Methodology

---

### 🔹 1. Data Preprocessing

* Time-series windowing (14-day sequence)
* Feature Engineering:

  * Lag features (t−1, t−7)
  * Rolling averages
  * Calendar features (day, month, weekend, season)
  * Inventory normalization
* Scaling using **RobustScaler**

---

### 🔹 2. LSTM Demand Forecasting (Residual Learning)

Instead of predicting demand directly, the model learns to **correct existing forecasts**.

### Mathematical Formulation

Residual:

$$
e_t = y_t - \hat{y}_t^{base}
$$

LSTM Prediction:

$$
\hat{e}*t = f*{LSTM}(X_{t-n:t})
$$

Final Forecast:

$$
\hat{y}_t = \hat{y}_t^{base} + \hat{e}_t
$$

Where:

* (y_t): Actual demand
* (\hat{y}_t^{base}): Existing forecast
* (\hat{e}_t): Predicted correction

### Loss Function (MSE)

$$
L = \frac{1}{n} \sum (y_{true} - y_{pred})^2
$$

---

### 🔹 3. Autoencoder for Anomaly Detection

The Autoencoder is used to identify unusual patterns in sales data.

#### Concept:

* Learns **normal sales behavior**
* Compresses input → reconstructs it
* High reconstruction error = anomaly

#### Mathematical Formulation

Encoding:

$$
z = f(x)
$$

Decoding:

$$
\hat{x} = g(z)
$$

Reconstruction Error:

$$
Error = ||x - \hat{x}||^2
$$

#### Anomaly Condition:

$$
\text{If } Error > Threshold \Rightarrow \text{Anomaly}
$$

#### What it detects:

* Sudden sales spikes
* Unexpected drops
* Irregular patterns

This makes it highly effective for **real-world retail monitoring**

---

## Results

* Baseline Forecast: ~12% MAPE
* LSTM Forecast: ~10–11% MAPE
* Improvement: **~1.2 – 1.4%**

✔ Improved accuracy
✔ Reliable anomaly detection
✔ Better inventory decisions

---

## Tech Stack

| Category      | Tools               |
| ------------- | ------------------- |
| Language      | Python              |
| ML/DL         | TensorFlow, Keras   |
| Data          | Pandas, NumPy       |
| Visualization | Matplotlib, Seaborn |
| Backend       | Flask               |
| Frontend      | React               |

---

## Project Structure

```id="projstruct"
Smart-Inventory/
│
├── app/
├── dashboard/
├── src/
├── models/
├── data/
├── screenshots/
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Running the Project

```bash id="runproj"
python app/flask_app.py
```

---

## Features

* 📈 Demand forecasting
* ⚠️ Anomaly detection
* 📦 Inventory alerts
* 📊 Dashboard visualization
* 📉 Trend analysis

---

## Challenges Faced

* Low autocorrelation in sales data
* Time-series feature engineering
* Model generalization across categories
* Handling large datasets

---

## Future Scope

* 🌐 Cloud deployment (AWS / Render)
* 📡 Real-time data streaming
* 🤖 Advanced models (Transformers, Attention)
* 📱 Mobile-friendly dashboard
* 🔄 Automated retraining pipelines
* 📊 Integration with ERP systems

---

## Conclusion

This project successfully demonstrates how **Deep Learning can enhance retail inventory management**.

By combining:

* **LSTM-based residual forecasting**
* **Autoencoder-based anomaly detection**

the system provides:
✔ Improved prediction accuracy
✔ Early detection of unusual patterns
✔ Data-driven decision support

Overall, it shows strong potential for **real-world retail applications**, enabling smarter inventory planning and operational efficiency.

---

## Author

Dimple Sachanandani
Niyati Yele
Pranav Dighole
B.Tech Electronics & Computer Science

---

## 📌 Note

* Large `.npy` training files are excluded due to GitHub limits
* All trained models and code are included

---
