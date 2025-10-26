/**
 * Rate limiting middleware for API Gateway
 */

const config = require('../config');
const logger = require('../utils/logger');

// In-memory store for rate limiting (in production, use Redis)
const rateLimitStore = new Map();

/**
 * Clean up expired entries from rate limit store
 */
const cleanupExpiredEntries = () => {
  const now = Date.now();
  for (const [key, data] of rateLimitStore.entries()) {
    if (now > data.resetTime) {
      rateLimitStore.delete(key);
    }
  }
};

/**
 * Rate limiting middleware
 */
const rateLimit = (options = {}) => {
  const {
    windowMs = config.security.rateLimit.windowMs,
    max = config.security.rateLimit.max,
    message = config.security.rateLimit.message,
    skipSuccessfulRequests = false,
    skipFailedRequests = false,
    keyGenerator = (req) => req.ip
  } = options;

  return (req, res, next) => {
    try {
      // Clean up expired entries periodically
      if (Math.random() < 0.01) { // 1% chance to cleanup
        cleanupExpiredEntries();
      }

      // Generate key for rate limiting
      const key = keyGenerator(req);
      const now = Date.now();
      const windowStart = now - windowMs;

      // Get or create rate limit data for this key
      let rateLimitData = rateLimitStore.get(key);
      if (!rateLimitData || now > rateLimitData.resetTime) {
        rateLimitData = {
          count: 0,
          resetTime: now + windowMs,
          windowStart: now
        };
      }

      // Check if request should be skipped
      if (skipSuccessfulRequests && res.statusCode < 400) {
        rateLimitStore.set(key, rateLimitData);
        return next();
      }

      if (skipFailedRequests && res.statusCode >= 400) {
        rateLimitStore.set(key, rateLimitData);
        return next();
      }

      // Increment counter
      rateLimitData.count++;

      // Check if limit exceeded
      if (rateLimitData.count > max) {
        const resetTime = new Date(rateLimitData.resetTime);
        const retryAfter = Math.ceil((rateLimitData.resetTime - now) / 1000);

        logger.warn('Rate limit exceeded', {
          key,
          count: rateLimitData.count,
          max,
          retryAfter
        });

        res.set({
          'X-RateLimit-Limit': max,
          'X-RateLimit-Remaining': 0,
          'X-RateLimit-Reset': resetTime.toISOString(),
          'Retry-After': retryAfter
        });

        return res.status(429).json({
          error: 'Too many requests',
          message,
          retryAfter,
          resetTime: resetTime.toISOString()
        });
      }

      // Update store
      rateLimitStore.set(key, rateLimitData);

      // Add rate limit headers
      const remaining = Math.max(0, max - rateLimitData.count);
      const resetTime = new Date(rateLimitData.resetTime);

      res.set({
        'X-RateLimit-Limit': max,
        'X-RateLimit-Remaining': remaining,
        'X-RateLimit-Reset': resetTime.toISOString()
      });

      next();

    } catch (error) {
      logger.error('Rate limiting error', { error: error.message });
      // Continue without rate limiting on error
      next();
    }
  };
};

/**
 * Stricter rate limiting for pricing endpoints
 */
const strictRateLimit = rateLimit({
  windowMs: 60 * 1000, // 1 minute
  max: 10, // 10 requests per minute
  message: 'Pricing API rate limit exceeded. Please wait before making another request.',
  keyGenerator: (req) => `${req.ip}:pricing`
});

/**
 * Lenient rate limiting for health checks
 */
const healthRateLimit = rateLimit({
  windowMs: 60 * 1000, // 1 minute
  max: 60, // 60 requests per minute
  skipFailedRequests: true,
  keyGenerator: (req) => `${req.ip}:health`
});

/**
 * User-based rate limiting (requires authentication)
 */
const userRateLimit = (options = {}) => {
  const baseOptions = {
    windowMs: 15 * 60 * 1000, // 15 minutes
    max: 100, // 100 requests per 15 minutes
    keyGenerator: (req) => req.user ? `user:${req.user.id}` : req.ip,
    ...options
  };

  return rateLimit(baseOptions);
};

/**
 * API key-based rate limiting
 */
const apiKeyRateLimit = (options = {}) => {
  const baseOptions = {
    windowMs: 60 * 60 * 1000, // 1 hour
    max: 1000, // 1000 requests per hour
    keyGenerator: (req) => req.apiKey ? `apikey:${req.apiKey.substring(0, 8)}` : req.ip,
    ...options
  };

  return rateLimit(baseOptions);
};

/**
 * Burst rate limiting for sudden traffic spikes
 */
const burstRateLimit = rateLimit({
  windowMs: 10 * 1000, // 10 seconds
  max: 20, // 20 requests per 10 seconds
  message: 'Too many requests in a short time. Please slow down.',
  keyGenerator: (req) => `${req.ip}:burst`
});

/**
 * Get rate limit status for a key
 */
const getRateLimitStatus = (key) => {
  const data = rateLimitStore.get(key);
  if (!data) {
    return null;
  }

  const now = Date.now();
  if (now > data.resetTime) {
    rateLimitStore.delete(key);
    return null;
  }

  return {
    count: data.count,
    resetTime: new Date(data.resetTime),
    remaining: Math.max(0, config.security.rateLimit.max - data.count)
  };
};

/**
 * Reset rate limit for a key
 */
const resetRateLimit = (key) => {
  rateLimitStore.delete(key);
  logger.info('Rate limit reset', { key });
};

/**
 * Get rate limit statistics
 */
const getRateLimitStats = () => {
  const now = Date.now();
  const activeKeys = Array.from(rateLimitStore.entries())
    .filter(([, data]) => now <= data.resetTime)
    .map(([key, data]) => ({
      key,
      count: data.count,
      resetTime: new Date(data.resetTime),
      timeToReset: Math.max(0, data.resetTime - now)
    }));

  return {
    totalKeys: rateLimitStore.size,
    activeKeys: activeKeys.length,
    keysByCount: activeKeys.reduce((acc, item) => {
      const count = item.count;
      acc[count] = (acc[count] || 0) + 1;
      return acc;
    }, {})
  };
};

module.exports = {
  rateLimit,
  strictRateLimit,
  healthRateLimit,
  userRateLimit,
  apiKeyRateLimit,
  burstRateLimit,
  getRateLimitStatus,
  resetRateLimit,
  getRateLimitStats
};