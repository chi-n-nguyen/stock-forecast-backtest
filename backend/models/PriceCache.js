const mongoose = require('mongoose');

const priceCacheSchema = new mongoose.Schema({
  ticker: { type: String, required: true, uppercase: true },
  date: { type: String, required: true },
  open: Number,
  high: Number,
  low: Number,
  close: Number,
  volume: Number
}, { timestamps: true });

priceCacheSchema.index({ ticker: 1, date: 1 }, { unique: true });

module.exports = mongoose.model('PriceCache', priceCacheSchema);
