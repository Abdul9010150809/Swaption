/**
 * Caching middleware for API Gateway
 */

const config = require('../config');
const logger = require('../utils/logger');

// In-memory cache store (in production, use Redis)
const cacheStore = new Map();

/**
 * Clean up expired cache entries
 */
const cleanupExpiredEntries = () => {
  const now = Date.now();
  for (const [key, data] of cacheStore.entries()) {
    if (now > data.expiry) {
      cacheStore.delete(key);
    }
  }
};

/**
 * Cache middleware
 */
const cache = (ttl = config.cache.ttl) => {
  return (req, res, next) => {
    if (!config.cache.enabled) {
      return next();
    }

    // Clean up expired entries periodically
    if (Math.random() < 0.01) { // 1% chance to cleanup
      cleanupExpiredEntries();
    }

    // Generate cache key
    const cacheKey = generateCacheKey(req);

    // Check if response is cached
    const cachedResponse = cacheStore.get(cacheKey);
    if (cachedResponse && Date.now() < cachedResponse.expiry) {
      logger.debug('Cache hit', { key: cacheKey });

      // Add cache headers
      res.set({
        'X-Cache': 'HIT',
        'X-Cache-Expires': new Date(cachedResponse.expiry).toISOString()
      });

      return res.json(cachedResponse.data);
    }

    // Cache miss - intercept response
    const originalJson = res.json;
    res.json = function(data) {
      // Only cache successful responses
      if (res.statusCode >= 200 && res.statusCode < 300) {
        const expiry = Date.now() + (ttl * 1000);
        cacheStore.set(cacheKey, {
          data,
          expiry,
          timestamp: Date.now()
        });

        logger.debug('Cache stored', { key: cacheKey, ttl });
      }

      // Add cache headers
      res.set({
        'X-Cache': 'MISS',
        'Cache-Control': `public, max-age=${ttl}`
      });

      // Call original json method
      return originalJson.call(this, data);
    };

    next();
  };
};

/**
 * Generate cache key from request
 */
function generateCacheKey(req) {
  const keyParts = [
    req.method,
    req.originalUrl,
    JSON.stringify(req.body || {}),
    JSON.stringify(req.query || {}),
    req.user?.id || 'anonymous',
    req.apiKey || 'no-key'
  ];

  return require('crypto').createHash('md5')
    .update(keyParts.join('|'))
    .digest('hex');
}

/**
 * Get cached response
 */
const get = (key) => {
  const cached = cacheStore.get(key);
  if (cached && Date.now() < cached.expiry) {
    return cached.data;
  }
  return null;
};

/**
 * Set cache entry
 */
const set = (key, data, ttl = config.cache.ttl) => {
  if (!config.cache.enabled) return;

  const expiry = Date.now() + (ttl * 1000);
  cacheStore.set(key, {
    data,
    expiry,
    timestamp: Date.now()
  });

  logger.debug('Cache manually set', { key, ttl });
};

/**
 * Delete cache entry
 */
const del = (key) => {
  const deleted = cacheStore.delete(key);
  if (deleted) {
    logger.debug('Cache entry deleted', { key });
  }
  return deleted;
};

/**
 * Clear all cache entries
 */
const clear = () => {
  const size = cacheStore.size;
  cacheStore.clear();
  logger.info('Cache cleared', { entriesCleared: size });
};

/**
 * Get cache statistics
 */
const getStats = () => {
  const now = Date.now();
  const totalEntries = cacheStore.size;
  const validEntries = Array.from(cacheStore.values())
    .filter(entry => now < entry.expiry).length;
  const expiredEntries = totalEntries - validEntries;

  // Calculate memory usage (rough estimate)
  let totalSize = 0;
  for (const [key, entry] of cacheStore.entries()) {
    totalSize += JSON.stringify(entry.data).length;
    totalSize += key.length;
  }

  return {
    enabled: config.cache.enabled,
    totalEntries,
    validEntries,
    expiredEntries,
    memoryUsage: Math.round(totalSize / 1024), // KB
    hitRate: totalEntries > 0 ? (validEntries / totalEntries * 100).toFixed(2) : 0
  };
};

/**
 * Middleware to conditionally cache based on request
 */
const conditionalCache = (conditionFn, ttl = config.cache.ttl) => {
  return (req, res, next) => {
    if (conditionFn(req)) {
      return cache(ttl)(req, res, next);
    }
    next();
  };
};

/**
 * Cache pricing requests (idempotent requests)
 */
const cachePricingRequests = cache(config.cache.ttl);

/**
 * Cache market data (can be cached longer)
 */
const cacheMarketData = cache(config.cache.ttl * 2);

/**
 * Don't cache sensitive operations
 */
const noCache = (req, res, next) => {
  res.set('Cache-Control', 'no-cache, no-store, must-revalidate');
  next();
};

module.exports = {
  cache,
  get,
  set,
  del,
  clear,
  getStats,
  conditionalCache,
  cachePricingRequests,
  cacheMarketData,
  noCache
};