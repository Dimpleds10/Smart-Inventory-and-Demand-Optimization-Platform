import numpy as np
import pickle
from tensorflow.keras.models import load_model

print("Loading model and data...")
X      = np.load('./models/X_train.npy')
y      = np.load('./models/y_train.npy')
scaler = pickle.load(open('./models/scaler.pkl', 'rb'))
model  = load_model('./models/lstm_best.keras')

# Test on last 500 samples
X_test = X[-500:]
y_test = y[-500:]
y_pred = model.predict(X_test, verbose=0).flatten()

# Convert back to real units
y_real      = scaler.inverse_transform(y_test.reshape(-1,1)).flatten()
y_pred_real = scaler.inverse_transform(y_pred.reshape(-1,1)).flatten()

# Calculate real metrics
mask = y_real > 5
mape = float(np.mean(np.abs((y_real[mask] - y_pred_real[mask]) / y_real[mask])) * 100)
mae  = float(np.mean(np.abs(y_real - y_pred_real)))
rmse = float(np.sqrt(np.mean((y_real - y_pred_real) ** 2)))

print(f'\n📊 REAL MODEL ACCURACY:')
print(f'   MAE  = {mae:.2f} units')
print(f'   RMSE = {rmse:.2f} units')
print(f'   MAPE = {mape:.2f}%')
print(f'\n🔍 Sample Predictions vs Actual:')
print(f'   {"Actual":>10} | {"Predicted":>10} | {"Diff":>10}')
print(f'   {"-"*36}')
for i in range(10):
    diff = y_real[i] - y_pred_real[i]
    print(f'   {y_real[i]:>10.0f} | {y_pred_real[i]:>10.0f} | {diff:>+10.0f}')
