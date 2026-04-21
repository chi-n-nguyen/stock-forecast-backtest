const express = require('express');
const axios = require('axios');
const Forecast = require('../models/Forecast');
const router = express.Router();

const ML_URL = process.env.ML_SERVICE_URL || 'http://localhost:8000';

// POST /ml/train
router.post('/train', async (req, res) => {
  const { ticker, startDate, endDate, horizon } = req.body;

  if (!ticker || !startDate || !endDate || !horizon) {
    return res.status(400).json({ error: 'ticker, startDate, endDate, horizon required' });
  }

  const validHorizons = [1, 5, 20];
  if (!validHorizons.includes(Number(horizon))) {
    return res.status(400).json({ error: 'horizon must be 1, 5, or 20' });
  }

  try {
    const response = await axios.post(`${ML_URL}/train`, {
      ticker: ticker.toUpperCase(),
      start_date: startDate,
      end_date: endDate,
      horizon: Number(horizon)
    }, { timeout: 300000 }); // 5 min timeout for training (LSTM adds ~2 min)

    const {
      metrics, predictions, next_forecast, feature_importance,
      lstm_metrics, lstm_predictions,
      data_points, feature_count
    } = response.data;

    const forecast = await Forecast.create({
      userId: req.user._id,
      ticker: ticker.toUpperCase(),
      startDate,
      endDate,
      horizon: Number(horizon),
      metrics,
      predictions,
      nextForecast: next_forecast,
      featureImportance: feature_importance,
      lstmMetrics: lstm_metrics,
      lstmPredictions: lstm_predictions,
      dataPoints: data_points,
      featureCount: feature_count
    });

    res.json({
      forecastId: forecast._id,
      metrics,
      predictions,
      nextForecast: next_forecast,
      featureImportance: feature_importance,
      lstmMetrics: lstm_metrics,
      lstmPredictions: lstm_predictions,
      dataPoints: data_points,
      featureCount: feature_count
    });
  } catch (err) {
    const detail = err.response?.data?.detail || err.message;
    res.status(500).json({ error: detail });
  }
});

module.exports = router;
