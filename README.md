# Stock Price Prediction with LSTM
 
Stock Price Prediction with LSTM is a hands-on deep learning project that demonstrates how sequential models can be applied to real-world financial data. Using historical OHLCV (Open, High, Low, Close, Volume) data, the project builds and trains an LSTM network to capture time-dependent patterns in stock movements.

The pipeline handles everything from preprocessing and sliding-window dataset creation to model training with early stopping and evaluation. The results are presented with intuitive visualizations — training and validation loss curves, predicted vs. actual stock prices, and short-horizon forecasts into the future. Metrics such as RMSE, MAE, and MAPE provide quantitative insight into performance.

This project serves as both a learning tool and a portfolio-ready showcase of time-series forecasting, deep learning, and financial modeling with PyTorch.

---

## Features
- Load stock data from CSV or fetch with Yahoo Finance (via `yfinance`)
- Preprocessing: scaling & sliding window dataset creation
- LSTM model with dropout and Adam optimizer
- Metrics: **RMSE, MAE, MAPE**
- Plots:
  - Training & validation curves
  - Predicted vs actual prices
  - Short-horizon future forecast
- Saved artifacts: `best_lstm.pt`, `scaler.pkl`, `metrics.json`

---

## Project Structure
```
stock-lstm-forecasting/
├─ README.md
├─ LICENSE
├─ requirements.txt
├─ data/
│  ├─ fetch_yfinance.py      # Fetch data from Yahoo Finance
│  └─ aapl.csv               # Stock dataset (real or synthetic)
├─ src/
│  ├─ train_lstm_stock.py    # Training script
│  ├─ evaluate.py            # Evaluation script
│  └─ utils.py               # Helpers (scaling, metrics, windowing)
└─ outputs/
   ├─ best_lstm.pt
   ├─ scaler.pkl
   ├─ metrics.json
   ├─ training_curves.png
   ├─ predicted_vs_actual.png
   └─ future_forecast.png
```

---

## Setup
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate

pip install -r requirements.txt
```

---

## Fetch Data (optional)
```bash
# downloads daily OHLCV for AAPL (Jan 2015 → today)
python data/fetch_yfinance.py --ticker AAPL --start 2015-01-01 --out data/aapl.csv
```

Or use the included **synthetic dataset** (`data/aapl.csv`).

---

## Train the Model
```bash
python src/train_lstm_stock.py --input data/aapl.csv --column close     --lookback 60 --epochs 25 --batch-size 64 --outdir outputs --horizon 1 --seed 42
```

---

## Evaluate the Model
```bash
python src/evaluate.py --input data/aapl.csv --model outputs/best_lstm.pt     --column close --lookback 60 --horizon 1 --outdir outputs
```

---

## Results


### Training & Validation Loss

<img width="1120" height="800" alt="training_curves" src="https://github.com/user-attachments/assets/25fdcfc9-e28a-476e-997f-c79ea0ac5349" />

---

### Predicted vs Actual

<img width="1600" height="800" alt="predicted_vs_actual" src="https://github.com/user-attachments/assets/f01ef615-476f-4f25-ad49-49367a1d76e4" />

---

### Short-Horizon Forecast

<img width="1600" height="800" alt="future_forecast" src="https://github.com/user-attachments/assets/1836d7e4-2942-4e2d-b101-2aa9b3894d3d" />

---

**Metrics (`metrics.json`):**
```json
{
  "rmse": 3.59,
  "mae": 3.59,
  "mape": 1.97
}
```

---

## Recommendations
- Train longer (50–100 epochs) for improved stability  
- Try multi-step forecasts (`--horizon 5` or `--horizon 30`)  
- Experiment with other assets (e.g., MSFT, GOOGL, TSLA)  
- Add more features (Volume, technical indicators)  
