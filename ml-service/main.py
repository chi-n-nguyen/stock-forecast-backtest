"""
Stock Forecast ML Service
FastAPI + XGBoost + PyTorch LSTM + yfinance
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import yfinance as yf
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import shap
import torch
import torch.nn as nn
import time
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

torch.manual_seed(42)

app = FastAPI(title="Stock Forecast ML Service")

# ─── In-Memory Caches ──────────────────────────────────────────────────────────
# _model_cache: (ticker, horizon) -> (model, feature_cols, last_close)
# Populated after /train. Avoids model deserialisation on subsequent /predict calls.
_model_cache: dict = {}

# _feature_cache: ticker -> raw OHLCV DataFrame (last 90 trading days).
# 90 rows covers the longest rolling window (60-day MA) with margin.
# Avoids a yfinance round-trip when the model is already cached.
_feature_cache: dict = {}

# _similarity_cache: ticker -> (timestamp, result). TTL = 24 hours.
_similarity_cache: dict = {}
_SIMILARITY_TTL = 86400

UNIVERSE = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA', 'NFLX',
    'AMD', 'INTC', 'JPM', 'BAC', 'WMT', 'DIS', 'PYPL', 'CRM', 'UBER',
    'SPOT', 'SQ', 'COIN'
]


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

    true_returns = np.array([p['true_return'] for p in valid])
    pred_returns = np.array([p['pred_return'] for p in valid])

    # Metrics computed on returns (dimensionless), not prices.
    # Price-based RMSE conflates model skill with price scale (AAPL at $190
    # vs a $10 stock), making cross-ticker comparison meaningless.
    # Return-based RMSE (e.g. 0.024 = 2.4% average error) is scale-invariant.
    # MAPE on returns uses mean(|true|) as denominator to avoid division by
    # zero on flat days where true_return ≈ 0.
    abs_true = np.abs(true_returns)
    mape = float(np.mean(np.abs(true_returns - pred_returns)) / (np.mean(abs_true) + 1e-8))
    rmse = float(np.sqrt(mean_squared_error(true_returns, pred_returns)))

    # Directional accuracy: restrict to "conviction" predictions where the model
    # predicted a move >= the median |predicted return|. Near-zero predictions
    # are essentially noise — the sign is arbitrary at that magnitude, so including
    # them deflates dirAcc without reflecting useful model behaviour.
    threshold = float(np.median(np.abs(pred_returns)))
    mask = np.abs(pred_returns) >= threshold
    if mask.sum() >= 5:
        dir_acc = float((np.sign(true_returns[mask]) == np.sign(pred_returns[mask])).mean())
        dir_acc_n = int(mask.sum())
    else:
        dir_acc = float((np.sign(true_returns) == np.sign(pred_returns)).mean())
        dir_acc_n = len(pred_returns)

    return {
        'mape': round(mape, 4),
        'rmse': round(rmse, 4),
        'dirAcc': round(dir_acc, 4),
        'dirAccN': dir_acc_n
    }


# ─── LSTM Model ────────────────────────────────────────────────────────────────

_LSTM_SEQ_LEN   = 40   # trading days per input sequence (~2 months)
_LSTM_HIDDEN    = 64
_LSTM_LAYERS    = 2
_LSTM_DROPOUT   = 0.2
_LSTM_EPOCHS    = 30
_LSTM_BATCH     = 64
_LSTM_LR        = 1e-3


class LSTMForecaster(nn.Module):
    def __init__(self, n_features: int):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=_LSTM_HIDDEN,
            num_layers=_LSTM_LAYERS,
            batch_first=True,
            dropout=_LSTM_DROPOUT
        )
        self.head = nn.Linear(_LSTM_HIDDEN, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])  # use last timestep hidden state


def walk_forward_lstm(df: pd.DataFrame, horizon: int, n_splits: int = 5) -> list:
    """
    Walk-forward LSTM backtesting using identical fold boundaries as XGBoost.
    Each fold: fit StandardScaler on training features only (trees are scale-invariant;
    LSTMs are not), build overlapping sequences of length _LSTM_SEQ_LEN (40 days, ~2 months),
    train with Adam/MSE for _LSTM_EPOCHS epochs, then predict across the test window.
    """
    feature_cols = [c for c in df.columns
                    if c not in ['Open', 'High', 'Low', 'Close', 'Volume', 'target']]
    X_raw = df[feature_cols].values.astype(np.float32)
    y_raw = df['target'].values.astype(np.float32)
    close  = df['Close'].values

    n         = len(X_raw)
    min_train = int(n * 0.6)
    fold_size = (n - min_train) // n_splits
    seq_len   = _LSTM_SEQ_LEN

    predictions = []

    for i in range(n_splits):
        train_end     = min_train + i * fold_size
        test_end      = min(train_end + fold_size, n)
        safe_train_end = max(seq_len + 1, train_end - horizon)

        if safe_train_end <= seq_len:
            continue

        # ── Scale features on training window only ──────────────────────────
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_raw[:safe_train_end]).astype(np.float32)
        y_train = y_raw[:safe_train_end]

        # ── Build overlapping sequences for training ─────────────────────────
        # Sequence ending at row j predicts y[j].
        X_seq = np.array([X_train[j:j + seq_len] for j in range(len(X_train) - seq_len)])
        y_seq = np.array([y_train[j + seq_len - 1] for j in range(len(X_train) - seq_len)])

        if len(X_seq) < 10:
            continue

        X_t = torch.from_numpy(X_seq)
        y_t = torch.from_numpy(y_seq)

        # ── Train ────────────────────────────────────────────────────────────
        model   = LSTMForecaster(n_features=X_raw.shape[1])
        opt     = torch.optim.Adam(model.parameters(), lr=_LSTM_LR)
        loss_fn = nn.MSELoss()
        dataset = torch.utils.data.TensorDataset(X_t, y_t)
        loader  = torch.utils.data.DataLoader(dataset, batch_size=_LSTM_BATCH, shuffle=True)

        model.train()
        for _ in range(_LSTM_EPOCHS):
            for xb, yb in loader:
                opt.zero_grad()
                loss_fn(model(xb).squeeze(-1), yb).backward()
                opt.step()

        # ── Predict on test window ───────────────────────────────────────────
        # Context window: seq_len rows before the first test row so every
        # test position can form a complete input sequence.
        ctx_start  = train_end - seq_len
        X_ctx      = scaler.transform(X_raw[ctx_start:test_end]).astype(np.float32)

        model.eval()
        with torch.no_grad():
            for k in range(test_end - train_end):
                seq    = torch.from_numpy(X_ctx[k: k + seq_len]).unsqueeze(0)
                p_ret  = float(model(seq).squeeze())
                t_ret  = float(y_raw[train_end + k])
                c_val  = float(close[train_end + k])
                predictions.append({
                    'date':        df.index[train_end + k].strftime('%Y-%m-%d'),
                    'y_true':      c_val * (1 + t_ret),
                    'y_pred':      c_val * (1 + p_ret),
                    'pred_return': p_ret,
                    'true_return': t_ret
                })

    return predictions


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

        # 6. SHAP feature importance (mean |SHAP value| across all training rows)
        explainer = shap.TreeExplainer(final_model)
        shap_values = explainer.shap_values(X_all)
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        total = mean_abs_shap.sum() + 1e-10
        feature_importance = [
            {'feature': col, 'importance': round(float(v / total), 4)}
            for col, v in sorted(zip(feature_cols, mean_abs_shap), key=lambda x: x[1], reverse=True)
        ]

        # Generate next-horizon predictions
        last_close = float(df['Close'].iloc[-1])
        last_features = X_all.iloc[[-1]]
        pred_return = float(final_model.predict(last_features)[0])

        # Populate caches so /predict can serve instantly without retraining.
        # last 90 rows of raw OHLCV is enough to recompute any rolling feature.
        _model_cache[(req.ticker.upper(), req.horizon)] = (final_model, feature_cols, last_close)
        _feature_cache[req.ticker.upper()] = df.iloc[-90:].copy()

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

        # 7. LSTM walk-forward backtest (same fold boundaries, normalised features)
        lstm_predictions = walk_forward_lstm(df_features, req.horizon, n_splits=5)
        lstm_metrics = compute_metrics(lstm_predictions) if len(lstm_predictions) >= 5 else None

        # Clean predictions for response (strip internal return fields)
        def _clean(preds):
            return [{'date': p['date'], 'y_true': p['y_true'], 'y_pred': p['y_pred']}
                    for p in preds if p.get('y_true') is not None]

        return {
            'metrics':           metrics,
            'predictions':       _clean(predictions),
            'feature_importance': feature_importance,
            'lstm_metrics':      lstm_metrics,
            'lstm_predictions':  _clean(lstm_predictions),
            'next_forecast':     next_forecast,
            'data_points':       len(df_features),
            'feature_count':     len(feature_cols)
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/predict/{ticker}")
def predict(ticker: str, horizon: int = 5):
    """
    Low-latency prediction using the in-memory cached model.
    Requires a prior /train call for the same (ticker, horizon) pair.
    Skips model training and full feature recomputation — only builds
    features on the last 90 days of cached OHLCV data, then runs
    model.predict() on the final row. Typical latency: <100ms.
    """
    key = (ticker.upper(), horizon)
    if key not in _model_cache:
        raise HTTPException(
            status_code=404,
            detail=f"No cached model for {ticker} horizon={horizon}. Call /train first."
        )

    model, feature_cols, last_close = _model_cache[key]

    try:
        # Use cached raw OHLCV if available; otherwise fetch a short window
        if ticker.upper() in _feature_cache:
            raw = _feature_cache[ticker.upper()]
        else:
            end = datetime.utcnow().strftime('%Y-%m-%d')
            start = (datetime.utcnow() - timedelta(days=120)).strftime('%Y-%m-%d')
            raw = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)

        df_feat = build_features(raw, horizon)
        last_row = df_feat[feature_cols].iloc[[-1]]
        last_close = float(df_feat['Close'].iloc[-1])
        pred_return = float(model.predict(last_row)[0])

        next_forecast = []
        current_date = datetime.utcnow()
        for day in range(1, horizon + 1):
            current_date += timedelta(days=1)
            while current_date.weekday() >= 5:
                current_date += timedelta(days=1)
            interpolated_return = pred_return * (day / horizon)
            next_forecast.append({
                'date': current_date.strftime('%Y-%m-%d'),
                'y_pred': round(last_close * (1 + interpolated_return), 2)
            })

        return {'ticker': ticker.upper(), 'horizon': horizon, 'next_forecast': next_forecast}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/similarity/{ticker}")
def get_similarity(ticker: str):
    """
    Cluster the 20-ticker universe by technical feature profiles and return
    the tickers most similar to `ticker` using cosine similarity.

    Pipeline:
      1. Batch-download 1 year of OHLCV for all tickers via yfinance.
      2. Build the same 34 features used for forecasting.
      3. Represent each ticker as the mean of its feature vector over the
         last 60 trading days — a stable behavioural "fingerprint".
      4. StandardScale across tickers so no feature dominates by magnitude.
      5. PCA to 2D for scatter-plot visualisation.
      6. Cosine similarity in the scaled space to rank neighbours.

    Results are cached in memory for 24 hours to avoid redundant fetches.
    """
    ticker = ticker.upper()

    cached = _similarity_cache.get(ticker)
    if cached and time.time() - cached[0] < _SIMILARITY_TTL:
        return cached[1]

    end = datetime.utcnow()
    start = end - timedelta(days=400)  # extra buffer for rolling-window warmup

    raw = yf.download(
        UNIVERSE,
        start=start.strftime('%Y-%m-%d'),
        end=end.strftime('%Y-%m-%d'),
        auto_adjust=True,
        progress=False
    )

    profiles = {}
    for t in UNIVERSE:
        try:
            if isinstance(raw.columns, pd.MultiIndex):
                df_t = raw.xs(t, axis=1, level=1).dropna()
            else:
                df_t = raw.copy().dropna()

            if len(df_t) < 80:
                continue

            df_feat = build_features(df_t.copy(), horizon=5)
            feature_cols = [c for c in df_feat.columns
                            if c not in ['Open', 'High', 'Low', 'Close', 'Volume', 'target']]
            # Mean over last 60 rows = stable behavioural fingerprint
            profiles[t] = df_feat[feature_cols].iloc[-60:].mean().values
        except Exception:
            continue

    if ticker not in profiles:
        raise HTTPException(status_code=404, detail=f"Could not build profile for {ticker}")

    tickers_list = list(profiles.keys())
    X = np.array([profiles[t] for t in tickers_list])

    X_scaled = StandardScaler().fit_transform(X)

    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X_scaled)

    query_idx = tickers_list.index(ticker)
    query_vec = X_scaled[query_idx]

    norms = np.linalg.norm(X_scaled, axis=1) + 1e-10
    cos_sims = X_scaled @ query_vec / (norms * np.linalg.norm(query_vec) + 1e-10)

    similar = []
    for i, t in enumerate(tickers_list):
        if t == ticker:
            continue
        similar.append({
            'ticker': t,
            'similarity': round(float(cos_sims[i]), 4),
            'x': round(float(X_2d[i][0]), 4),
            'y': round(float(X_2d[i][1]), 4)
        })
    similar.sort(key=lambda s: s['similarity'], reverse=True)

    result = {
        'ticker': ticker,
        'query': {
            'ticker': ticker,
            'x': round(float(X_2d[query_idx][0]), 4),
            'y': round(float(X_2d[query_idx][1]), 4)
        },
        'similar': similar[:5],
        'all_tickers': [
            {'ticker': t, 'x': round(float(X_2d[i][0]), 4), 'y': round(float(X_2d[i][1]), 4)}
            for i, t in enumerate(tickers_list)
        ]
    }

    _similarity_cache[ticker] = (time.time(), result)
    return result


@app.get("/health")
def health():
    return {"status": "ok"}
