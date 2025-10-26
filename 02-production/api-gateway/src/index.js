/**
 * Price Matrix API Gateway
 *
 * Main entry point for the API Gateway service that routes requests
 * to appropriate microservices and handles authentication, rate limiting,
 * and monitoring.
 */

const express = require('express');
const cors = require('cors');
const helmet = require('helmet');
const compression = require('compression');
const config = require('./config');
const { logger, requestLogger, errorLogger, requestId } = require('./utils/logger');
const { authenticateFlexible } = require('./middleware/auth');
const rateLimit = require('./middleware/rateLimit');
const cache = require('./middleware/cache');
const circuitBreaker = require('./middleware/circuitBreaker');

// Import routes
const healthRoutes = require('./routes/health');
const pricingRoutes = require('./routes/pricing');

// Initialize Express app
const app = express();

// Trust proxy for accurate IP addresses
app.set('trust proxy', 1);

// Security middleware
app.use(helmet({
  contentSecurityPolicy: {
    directives: {
      defaultSrc: ["'self'"],
      styleSrc: ["'self'", "'unsafe-inline'"],
      scriptSrc: ["'self'"],
      imgSrc: ["'self'", "data:", "https:"],
    },
  },
}));

// CORS configuration
app.use(cors(config.security.cors));

// Compression middleware
app.use(compression());

// Body parsing middleware
app.use(express.json({ limit: config.api.limits.json }));
app.use(express.urlencoded({ extended: true, limit: config.api.limits.urlencoded }));

// Request logging and ID middleware
app.use(requestId);
app.use(requestLogger);

// Rate limiting
if (config.features.rateLimiting) {
  app.use(rateLimit.globalLimit());
}

// Health check endpoint (no auth required)
app.use('/health', healthRoutes);

// API documentation
if (config.api.docs.enabled) {
  const swaggerJsdoc = require('swagger-jsdoc');
  const swaggerUi = require('swagger-ui-express');

  const specs = swaggerJsdoc(config.api.docs.swaggerDefinition);
  app.use(config.api.docs.path, swaggerUi.serve, swaggerUi.setup(specs));
}

// API routes
app.use(config.api.prefix, pricingRoutes);

// 404 handler
app.use('*', (req, res) => {
  res.status(404).json({
    error: 'Not Found',
    message: `Route ${req.originalUrl} not found`,
    timestamp: new Date().toISOString()
  });
});

// Error logging middleware
app.use(errorLogger);

// Global error handler
app.use((error, req, res, next) => {
  logger.error('Unhandled application error', {
    error: error.message,
    stack: error.stack,
    url: req.url,
    method: req.method,
    requestId: req.requestId
  });

  // Don't leak error details in production
  const isDevelopment = config.server.env === 'development';
  const errorResponse = {
    error: 'Internal Server Error',
    message: isDevelopment ? error.message : 'An unexpected error occurred',
    timestamp: new Date().toISOString(),
    requestId: req.requestId
  };

  if (isDevelopment) {
    errorResponse.stack = error.stack;
  }

  res.status(500).json(errorResponse);
});

// Graceful shutdown handling
process.on('SIGTERM', () => {
  logger.info('SIGTERM received, shutting down gracefully');
  process.exit(0);
});

process.on('SIGINT', () => {
  logger.info('SIGINT received, shutting down gracefully');
  process.exit(0);
});

// Start server
const PORT = config.server.port;
const HOST = config.server.host;

if (require.main === module) {
  app.listen(PORT, HOST, () => {
    logger.info(`Price Matrix API Gateway listening on ${HOST}:${PORT}`, {
      environment: config.server.env,
      apiVersion: config.server.apiVersion,
      features: config.features
    });

    // Log configuration summary
    logger.info('Server configuration loaded', {
      corsOrigins: config.security.cors.origin,
      rateLimitEnabled: config.features.rateLimiting,
      cacheEnabled: config.features.caching,
      monitoringEnabled: config.features.monitoring,
      circuitBreakerEnabled: config.features.circuitBreaker
    });
  });
}

module.exports = app;