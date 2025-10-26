/**
 * Authentication middleware for API Gateway
 */

const jwt = require('jsonwebtoken');
const config = require('../config');
const logger = require('../utils/logger');

/**
 * JWT authentication middleware
 */
const authenticateToken = (req, res, next) => {
  try {
    // Get token from header
    const authHeader = req.headers['authorization'];
    const token = authHeader && authHeader.split(' ')[1]; // Bearer TOKEN

    if (!token) {
      return res.status(401).json({
        error: 'Access token required',
        message: 'Please provide a valid JWT token in the Authorization header'
      });
    }

    // Verify token
    jwt.verify(token, config.security.jwtSecret, (err, user) => {
      if (err) {
        logger.warn('JWT verification failed', { error: err.message });
        return res.status(403).json({
          error: 'Invalid token',
          message: 'The provided token is invalid or expired'
        });
      }

      // Add user info to request
      req.user = user;
      next();
    });

  } catch (error) {
    logger.error('Authentication middleware error', { error: error.message });
    res.status(500).json({
      error: 'Authentication error',
      message: 'An error occurred during authentication'
    });
  }
};

/**
 * Optional authentication middleware (doesn't fail if no token)
 */
const optionalAuth = (req, res, next) => {
  try {
    const authHeader = req.headers['authorization'];
    const token = authHeader && authHeader.split(' ')[1];

    if (token) {
      jwt.verify(token, config.security.jwtSecret, (err, user) => {
        if (!err) {
          req.user = user;
        }
      });
    }

    next();
  } catch (error) {
    // Continue without authentication
    next();
  }
};

/**
 * Role-based authorization middleware
 */
const requireRole = (requiredRole) => {
  return (req, res, next) => {
    if (!req.user) {
      return res.status(401).json({
        error: 'Authentication required',
        message: 'You must be authenticated to access this resource'
      });
    }

    if (!req.user.role || req.user.role !== requiredRole) {
      return res.status(403).json({
        error: 'Insufficient permissions',
        message: `This resource requires ${requiredRole} role`
      });
    }

    next();
  };
};

/**
 * Admin role authorization middleware
 */
const requireAdmin = requireRole('admin');

/**
 * API key authentication middleware
 */
const authenticateApiKey = (req, res, next) => {
  try {
    const apiKey = req.headers['x-api-key'];

    if (!apiKey) {
      return res.status(401).json({
        error: 'API key required',
        message: 'Please provide a valid API key in the X-API-Key header'
      });
    }

    // In production, validate against database
    // For now, accept any non-empty key
    if (apiKey.length < 10) {
      return res.status(401).json({
        error: 'Invalid API key',
        message: 'The provided API key is invalid'
      });
    }

    // Add API key info to request
    req.apiKey = apiKey;
    next();

  } catch (error) {
    logger.error('API key authentication error', { error: error.message });
    res.status(500).json({
      error: 'Authentication error',
      message: 'An error occurred during API key validation'
    });
  }
};

/**
 * Combined authentication middleware (JWT or API key)
 */
const authenticateFlexible = (req, res, next) => {
  try {
    // Try JWT first
    const authHeader = req.headers['authorization'];
    const token = authHeader && authHeader.split(' ')[1];

    if (token) {
      jwt.verify(token, config.security.jwtSecret, (err, user) => {
        if (!err) {
          req.user = user;
          req.authMethod = 'jwt';
          return next();
        }
      });
    }

    // Try API key
    const apiKey = req.headers['x-api-key'];
    if (apiKey && apiKey.length >= 10) {
      req.apiKey = apiKey;
      req.authMethod = 'api_key';
      return next();
    }

    // No valid authentication
    return res.status(401).json({
      error: 'Authentication required',
      message: 'Please provide either a JWT token or API key'
    });

  } catch (error) {
    logger.error('Flexible authentication error', { error: error.message });
    res.status(500).json({
      error: 'Authentication error',
      message: 'An error occurred during authentication'
    });
  }
};

module.exports = {
  authenticateToken,
  optionalAuth,
  requireRole,
  requireAdmin,
  authenticateApiKey,
  authenticateFlexible
};