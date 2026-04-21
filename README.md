# Stock Forecast & Backtest

XGBoost and LSTM stock forecasting with walk-forward backtesting. Built to measure model performance honestly, since a common failure point in finance ML projects is data leakage that inflates backtest results.

## What it does

- Fetches 1-5 years of OHLCV data via yfinance
- Engineers 34 features (lag returns, rolling MAs, RSI, MACD, volatility, volume ratio)
- Trains XGBoost and a PyTorch LSTM on **walk-forward temporal splits** (no random k-fold, no lookahead)
- Compares both models side-by-side on identical fold boundaries (MAPE, RMSE, directional accuracy)
- Computes SHAP feature attribution (mean |SHAP value|) for XGBoost
- Finds similar stocks via PCA on 34-feature behavioural profiles + cosine similarity across a 20-ticker universe
- Serves predictions through a FastAPI layer with in-memory model caching (~85ms p50)
- Stores forecast history in MongoDB (indexed for O(log n) history queries)

## Models

**XGBoost** — sees one row at a time (today's feature snapshot). Scale-invariant, no preprocessing needed, fast to train. Good at picking up non-linear feature interactions.

**LSTM** — sees the last 40 trading days (~2 months) as a sequence. Can learn momentum, trend exhaustion, and regime patterns that single-row models miss. Features are StandardScaled per fold (fit on training data only). 2-layer architecture with hidden size 64, dropout 0.2, trained with Adam/MSE.

Both models use identical fold boundaries for a fair comparison.

## Leakage fix

Standard k-fold assigns rows randomly to folds, so a row from day 300 can appear in training while day 250 is in validation. This is invalid for time series data.

Walk-forward helps, but there's a subtler issue: the last `horizon` rows of each training fold have targets (`close[t+horizon]/close[t] - 1`) computed from prices that fall inside the test window. The fix is to end training at `train_end - horizon`, so no training label touches a test-period price.

With this fix applied, directional accuracy dropped from ~64% to ~57% and RMSE increased slightly. The difference was leakage.

## Stack

| Layer | Tech |
|-------|------|
| Frontend | Vue 3, Tailwind CSS, Chart.js |
| Backend | Node.js, Express |
| ML service | FastAPI, XGBoost, PyTorch, SHAP, scikit-learn, yfinance, pandas |
| Database | MongoDB (Mongoose) |

## Running locally

**1. ML service**
```bash
cd ml-service
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

**2. Backend**
```bash
cd backend
npm install
cp .env.example .env   # fill in MongoDB URI
npm run dev
```

**3. Frontend**
```bash
cd frontend
npm install
npm run dev            # http://localhost:5173
```

**Smoke test (no auth needed):**
```bash
curl -X POST http://localhost:8000/train \
  -H "Content-Type: application/json" \
  -d '{"ticker":"AAPL","start_date":"2020-01-01","end_date":"2025-01-01","horizon":5}'
```

## Metrics

All metrics are computed on **returns**, not prices. Price-based RMSE conflates model skill with price scale (AAPL at $190 vs a $10 stock). Return-based RMSE is scale-invariant and cross-ticker comparable.

Directional accuracy is computed on "conviction" predictions only — days where `|predicted return| >= median(|predicted returns|)`, i.e. the top 50% by model confidence. Near-zero predictions are noise; their sign is arbitrary. The sample size (N) is shown alongside the metric. The `+Xpp vs random` label shows the margin above the 50% coin-flip baseline.

## Limitations

- No transaction cost simulation; directional accuracy alone does not imply profitability
- In-memory model cache resets on service restart; no persistence layer for trained models
- Regime changes (e.g. 2020 crash) can invalidate a model trained on bull-market data
- For production you'd move to PostgreSQL and add model versioning
