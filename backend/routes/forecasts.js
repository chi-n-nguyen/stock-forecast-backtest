const express = require('express');
const Forecast = require('../models/Forecast');
const { requireAuth } = require('../middleware/auth');
const router = express.Router();

// GET /forecasts — list user's forecast history
router.get('/', requireAuth, async (req, res) => {
  try {
    const forecasts = await Forecast.find({ userId: req.user._id })
      .sort({ createdAt: -1 })
      .select('ticker startDate endDate horizon metrics createdAt')
      .lean();
    res.json(forecasts);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// GET /forecasts/:id — get full forecast detail
router.get('/:id', requireAuth, async (req, res) => {
  try {
    const forecast = await Forecast.findOne({
      _id: req.params.id,
      userId: req.user._id
    }).lean();

    if (!forecast) return res.status(404).json({ error: 'Forecast not found' });
    res.json(forecast);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

module.exports = router;
