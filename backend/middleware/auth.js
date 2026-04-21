function requireAuth(req, res, next) {
  if (process.env.DEV_BYPASS_AUTH === 'true') {
    req.user = { _id: '000000000000000000000001', name: 'Dev User', email: 'dev@local' };
    return next();
  }
  if (req.isAuthenticated()) return next();
  res.status(401).json({ error: 'Not authenticated' });
}

module.exports = { requireAuth };
