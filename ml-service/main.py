"""
Stock Forecast ML Service
FastAPI + XGBoost + yfinance
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import yfinance as yf
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

app = FastAPI(title="Stock Forecast ML Service")


# ─── Request/Response Models ───────────────────────────────────────────────────

class TrainRequest(BaseModel):
    ticker: str
    start_date: str
    end_date: str
    horizon: int  # days ahead to predict (1, 5, 20)


# ─── Feature Engineering ───────────────────────────────────────────────────────

def build_features(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """
    Build lag features, rolling stats, and technical indicators.
    Target: horizon-day forward return (close[t+horizon] / close[t] - 1)
    """
    df = df.copy()
    close = df['Close']

    # Lag features
    for lag in [1, 2, 3, 5, 10, 20]:
        df[f'lag_{lag}'] = close.shift(lag)
        df[f'return_{lag}d'] = close.pct_change(lag)

    # Rolling statistics
    for window in [5, 10, 20, 60]:
        df[f'ma_{window}'] = close.rolling(window).mean()
        df[f'std_{window}'] = close.rolling(window).std()
        df[f'ma_ratio_{window}'] = close / df[f'ma_{window}']

    # Volatility
    df['daily_return'] = close.pct_change()
    df['volatility_10'] = df['daily_return'].rolling(10).std()
    df['volatility_20'] = df['daily_return'].rolling(20).std()

    # Volume features
    df['vol_ma_5'] = df['Volume'].rolling(5).mean()
    df['vol_ratio'] = df['Volume'] / df['vol_ma_5']

    # RSI (14-day)
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / (loss + 1e-10)
    df['rsi_14'] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = close.ewm(span=12).mean()
    ema26 = close.ewm(span=26).mean()
    df['macd'] = ema12 - ema26
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']

    # Calendar features
    df['day_of_week'] = df.index.dayofweek
    df['month'] = df.index.month

    # Target: forward return over horizon
    df['target'] = close.shift(-horizon) / close - 1

    df.dropna(inplace=True)
    return df


# ─── Walk-Forward Backtest ─────────────────────────────────────────────────────

def walk_forward_backtest(df: pd.DataFrame, horizon: int, n_splits: int = 5):
    """
    Time-series walk-forward cross-validation with strict temporal partitioning.

    Partitioning happens before training: the last `horizon` rows of each training
    window are excluded because their target labels (close[t+horizon]/close[t]-1)
    are computed using prices from the test window. Including them would allow the
    model to implicitly train on future data — the canonical time-series leakage bug.

    Standard k-fold cross-validation randomly assigns rows to folds, so a row from
    day 300 can appear in training while day 250 is in validation. For time series
    this is invalid. Walk-forward enforces strict temporal ordering.
    """
    feature_cols = [c for c in df.columns if c not in ['Open', 'High', 'Low', 'Close', 'Volume', 'target']]
    X = df[feature_cols]
    y = df['target']
    close = df['Close']

    n = len(df)
    min_train = int(n * 0.6)
    fold_size = (n - min_train) // n_splits

    predictions = []
    model = None

    for i in range(n_splits):
        train_end = min_train + i * fold_size
        test_end = min(train_end + fold_size, n)

        # Exclude last `horizon` rows from training: their target values use
        # close prices from [train_end, train_end+horizon), which overlaps the
        # test window. This is the exact fix for target-label leakage.
        safe_train_end = max(1, train_end - horizon)
        X_train, y_train = X.iloc[:safe_train_end], y.iloc[:safe_train_end]
        X_test = X.iloc[train_end:test_end]
        close_test = close.iloc[train_end:test_end]

        model = XGBRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=5,
            random_state=42,
            n_jobs=-1,
            verbosity=0
        )
        model.fit(X_train, y_train, eval_set=[(X_train, y_train)], verbose=False)

        y_pred_returns = model.predict(X_test)

        for j, (idx, pred_return) in enumerate(zip(X_test.index, y_pred_returns)):
            true_return = y.iloc[train_end + j] if train_end + j < len(y) else None
            predictions.append({
                'date': idx.strftime('%Y-%m-%d'),
                'y_true': float(close_test.iloc[j] * (1 + true_return)) if true_return is not None else None,
                'y_pred': float(close_test.iloc[j] * (1 + pred_return)),
                'close': float(close_test.iloc[j]),
                'pred_return': float(pred_return),
                'true_return': float(true_return) if true_return is not None else None
            })

    return predictions, model, feature_cols


# ─── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(predictions: list) -> dict:
    valid = [p for p in predictions if p['y_true'] is not None and p['y_pred'] is not None]
    if len(valid) < 5:
        raise ValueError("Insufficient predictions for metrics")

    y_true = np.array([p['y_true'] for p in valid])
    y_pred = np.array([p['y_pred'] for p in valid])
    true_returns = np.array([p['true_return'] for p in valid])
    pred_returns = np.array([p['pred_return'] for p in valid])

    mape = float(mean_absolute_percentage_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))

    # Directional accuracy: did we predict up/down correctly?
    correct_direction = np.sign(true_returns) == np.sign(pred_returns)
    dir_acc = float(correct_direction.mean())

    return {
        'mape': round(mape, 4),
        'rmse': round(rmse, 4),
        'dirAcc': round(dir_acc, 4)
    }


# ─── Routes ────────────────────────────────────────────────────────────────────

@app.get("/prices/{ticker}")
def get_prices(ticker: str, start: str, end: str):
    """Fetch and return OHLCV data via yfinance."""
    try:
        df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
        if df.empty:
            raise HTTPException(status_code=404, detail=f"No data found for {ticker}")

        records = []
        for date, row in df.iterrows():
            records.append({
                'date': date.strftime('%Y-%m-%d'),
                'open': round(float(row['Open']), 4),
                'high': round(float(row['High']), 4),
                'low': round(float(row['Low']), 4),
                'close': round(float(row['Close']), 4),
                'volume': int(row['Volume'])
            })
        return {'ticker': ticker, 'data': records}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/train")
def train_model(req: TrainRequest):
    """
    Train XGBoost model with walk-forward backtesting.
    Returns metrics, backtest predictions, and next-horizon forecast.
    """
    try:
        # 1. Fetch data
        df = yf.download(
            req.ticker,
            start=req.start_date,
            end=req.end_date,
            auto_adjust=True,
            progress=False
        )
        if df.empty or len(df) < 100:
            raise HTTPException(status_code=400, detail=f"Insufficient data for {req.ticker}")

        # Flatten MultiIndex columns if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # 2. Feature engineering
        df_features = build_features(df, req.horizon)
        if len(df_features) < 50:
            raise HTTPException(status_code=400, detail="Not enough data after feature engineering")

        # 3. Walk-forward backtest
        predictions, final_model, feature_cols = walk_forward_backtest(
            df_features, req.horizon, n_splits=5
        )

        # 4. Compute metrics
        metrics = compute_metrics(predictions)

        # 5. Next forecast: train on ALL data, predict future
        X_all = df_features[feature_cols]
        y_all = df_features['target']
        final_model.fit(X_all, y_all, verbose=False)

        # Generate next-horizon predictions
        last_close = float(df['Close'].iloc[-1])
        last_features = X_all.iloc[[-1]]
        pred_return = float(final_model.predict(last_features)[0])

        next_forecast = []
        current_date = datetime.strptime(req.end_date, '%Y-%m-%d')
        # Simple linear interpolation for intermediate days
        for day in range(1, req.horizon + 1):
            current_date += timedelta(days=1)
            while current_date.weekday() >= 5:  # skip weekends
                current_date += timedelta(days=1)
            interpolated_return = pred_return * (day / req.horizon)
            next_forecast.append({
                'date': current_date.strftime('%Y-%m-%d'),
                'y_pred': round(last_close * (1 + interpolated_return), 2)
            })

        # Clean predictions for response (remove internal fields)
        clean_predictions = [
            {'date': p['date'], 'y_true': p['y_true'], 'y_pred': p['y_pred']}
            for p in predictions if p['y_true'] is not None
        ]

        return {
            'metrics': metrics,
            'predictions': clean_predictions,
            'next_forecast': next_forecast,
            'data_points': len(df_features),
            'feature_count': len(feature_cols)
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health():
    return {"status": "ok"}
