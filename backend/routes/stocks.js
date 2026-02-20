const express = require('express');
const axios = require('axios');
const PriceCache = require('../models/PriceCache');
const { requireAuth } = require('../middleware/auth');
const router = express.Router();

const ML_URL = process.env.ML_SERVICE_URL || 'http://localhost:8000';

const POPULAR_TICKERS = [
  { ticker: 'AAPL', name: 'Apple Inc.' },
  { ticker: 'MSFT', name: 'Microsoft Corporation' },
  { ticker: 'GOOGL', name: 'Alphabet Inc.' },
  { ticker: 'AMZN', name: 'Amazon.com Inc.' },
  { ticker: 'META', name: 'Meta Platforms Inc.' },
  { ticker: 'TSLA', name: 'Tesla Inc.' },
  { ticker: 'NVDA', name: 'NVIDIA Corporation' },
  { ticker: 'NFLX', name: 'Netflix Inc.' },
  { ticker: 'AMD', name: 'Advanced Micro Devices' },
  { ticker: 'INTC', name: 'Intel Corporation' },
  { ticker: 'JPM', name: 'JPMorgan Chase & Co.' },
  { ticker: 'BAC', name: 'Bank of America Corp.' },
  { ticker: 'WMT', name: 'Walmart Inc.' },
  { ticker: 'DIS', name: 'The Walt Disney Company' },
  { ticker: 'PYPL', name: 'PayPal Holdings Inc.' },
  { ticker: 'CRM', name: 'Salesforce Inc.' },
  { ticker: 'UBER', name: 'Uber Technologies Inc.' },
  { ticker: 'SPOT', name: 'Spotify Technology S.A.' },
  { ticker: 'SQ', name: 'Block Inc.' },
  { ticker: 'COIN', name: 'Coinbase Global Inc.' }
];

router.get('/search', requireAuth, (req, res) => {
  const q = (req.query.q || '').toUpperCase();
  if (!q) return res.json([]);
  const results = POPULAR_TICKERS.filter(
    t => t.ticker.includes(q) || t.name.toUpperCase().includes(q)
  ).slice(0, 8);
  res.json(results);
});

// GET /stocks/:ticker/prices?start=YYYY-MM-DD&end=YYYY-MM-DD
router.get('/:ticker/prices', requireAuth, async (req, res) => {
  const ticker = req.params.ticker.toUpperCase();
  const { start, end } = req.query;

  if (!start || !end) {
    return res.status(400).json({ error: 'start and end query params required' });
  }

  try {
    // Check cache first
    const cached = await PriceCache.find({
      ticker,
      date: { $gte: start, $lte: end }
    }).sort({ date: 1 }).lean();

    if (cached.length > 50) {
      return res.json({ ticker, data: cached, source: 'cache' });
    }

    // Fetch from ML service (which uses yfinance)
    const response = await axios.get(`${ML_URL}/prices/${ticker}`, {
      params: { start, end },
      timeout: 30000
    });

    const records = response.data.data;

    // Bulk upsert into cache
    const ops = records.map(r => ({
      updateOne: {
        filter: { ticker, date: r.date },
        update: { $set: { ...r, ticker } },
        upsert: true
      }
    }));
    await PriceCache.bulkWrite(ops);

    res.json({ ticker, data: records, source: 'live' });
  } catch (err) {
    const detail = err.response?.data?.detail || err.message;
    res.status(500).json({ error: detail });
  }
});

module.exports = router;
