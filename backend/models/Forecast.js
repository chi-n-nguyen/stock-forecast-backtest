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

module.exports = mongoose.model('Forecast', forecastSchema);
