require('dotenv').config();
const express = require('express');
const cors = require('cors');
const mongoose = require('mongoose');

const stockRoutes = require('./routes/stocks');
const mlRoutes = require('./routes/ml');
const forecastRoutes = require('./routes/forecasts');

const app = express();

app.use(cors({ origin: process.env.FRONTEND_URL || 'http://localhost:5173', credentials: true }));
app.use(express.json());

// Attach a static user to every request — no auth needed
app.use((req, _res, next) => {
  req.user = { _id: '000000000000000000000001', name: 'Demo User', email: 'demo@local' };
  next();
});

app.use('/stocks', stockRoutes);
app.use('/ml', mlRoutes);
app.use('/forecasts', forecastRoutes);

app.get('/me', (req, res) => res.json(req.user));

mongoose.connect(process.env.MONGODB_URI || 'mongodb://localhost:27017/stockforecast')
  .then(() => console.log('MongoDB connected'))
  .catch(err => console.error('MongoDB error:', err));

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => console.log(`Server running on port ${PORT}`));
