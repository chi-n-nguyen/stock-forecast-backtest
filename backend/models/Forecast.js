const mongoose = require('mongoose');

const forecastSchema = new mongoose.Schema({
  userId: { type: mongoose.Schema.Types.ObjectId, ref: 'User', required: true },
  ticker: { type: String, required: true, uppercase: true },
  startDate: { type: String, required: true },
  endDate: { type: String, required: true },
  horizon: { type: Number, required: true },
  modelType: { type: String, default: 'xgboost' },
  metrics: {
    mape: Number,
    rmse: Number,
    dirAcc: Number
  },
  predictions: [{ date: String, y_true: Number, y_pred: Number }],
  nextForecast: [{ date: String, y_pred: Number }],
  dataPoints: Number,
  featureCount: Number
}, { timestamps: true });

// Compound index on the history query path: GET /forecasts filters by userId
// and sorts by createdAt DESC. Without this, MongoDB does a full collection scan
// O(n) regardless of how many documents exist. With it, the query is O(log n).
forecastSchema.index({ userId: 1, createdAt: -1 });

// Sparse index on ticker for any future cross-user analytics queries.
forecastSchema.index({ ticker: 1 });

module.exports = mongoose.model('Forecast', forecastSchema);
