# Stock Forecast & Backtest

XGBoost stock forecasting with walk-forward backtesting. Built to measure model performance honestly, since a common failure point in finance ML projects is data leakage that inflates backtest results.

## What it does

- Fetches 1-5 years of OHLCV data via yfinance
- Engineers 34 features (lag returns, rolling MAs, RSI, MACD, volatility, volume ratio)
- Trains XGBoost with **walk-forward temporal splitting** (no random k-fold, no lookahead)
- Reports MAPE, RMSE (return basis), and directional accuracy vs 50% random baseline
- Serves predictions through a FastAPI layer with in-memory model caching (~85ms p50)
- Stores forecast history per user in MongoDB (indexed for O(log n) history queries)
- Auth via Google OAuth

## Leakage fix

Standard k-fold assigns rows randomly to folds, so a row from day 300 can appear in training while day 250 is in validation. This is invalid for time series data.

Walk-forward helps, but there's a subtler issue: the last `horizon` rows of each training fold have targets (`close[t+horizon]/close[t] - 1`) computed from prices that fall inside the test window. The fix is to end training at `train_end - horizon`, so no training label touches a test-period price.

With this fix applied, directional accuracy dropped from ~64% to ~57% and RMSE increased slightly. The difference was leakage.

## Stack

| Layer | Tech |
|-------|------|
| Frontend | Vue 3, Tailwind CSS, Chart.js |
| Backend | Node.js, Express, Passport.js |
| ML service | FastAPI, XGBoost, yfinance, pandas |
| Database | MongoDB (Mongoose) |
| Auth | Google OAuth 2.0 |

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
cp .env.example .env   # fill in MongoDB URI + Google OAuth credentials
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

Directional accuracy is shown with a `+Xpp vs random` label relative to the 50% random baseline. The number is smaller after fixing leakage, which is expected.

## Limitations

- No transaction cost simulation; directional accuracy alone does not imply profitability
- In-memory model cache resets on service restart; no persistence layer for trained models
- Regime changes (e.g. 2020 crash) can invalidate a model trained on bull-market data
- For production you'd move to PostgreSQL and add model versioning
