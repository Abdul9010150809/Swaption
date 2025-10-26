/**
 * Logging utilities for API Gateway
 */

const winston = require('winston');
const path = require('path');
const config = require('../config');

// Define log levels
const levels = {
  error: 0,
  warn: 1,
  info: 2,
  http: 3,
  debug: 4
};

// Define colors for each level
const colors = {
  error: 'red',
  warn: 'yellow',
  info: 'green',
  http: 'magenta',
  debug: 'white'
};

winston.addColors(colors);

// Custom format for console logging
const consoleFormat = winston.format.combine(
  winston.format.timestamp({ format: 'YYYY-MM-DD HH:mm:ss' }),
  winston.format.errors({ stack: true }),
  winston.format.colorize({ all: true }),
  winston.format.printf(({ timestamp, level, message, ...meta }) => {
    let metaStr = Object.keys(meta).length ? `\n${JSON.stringify(meta, null, 2)}` : '';
    return `${timestamp} ${level}: ${message}${metaStr}`;
  })
);

// JSON format for file logging
const fileFormat = winston.format.combine(
  winston.format.timestamp(),
  winston.format.errors({ stack: true }),
  winston.format.json()
);

// Create winston logger
const logger = winston.createLogger({
  level: config.logging.level,
  levels,
  format: fileFormat,
  transports: []
});

// Add console transport if enabled
if (config.logging.logToConsole) {
  logger.add(new winston.transports.Console({
    format: consoleFormat
  }));
}

// Add file transport if enabled
if (config.logging.logToFile) {
  const logDir = config.logging.logDir;
  const filename = config.logging.rotation.filename;

  logger.add(new winston.transports.File({
    filename: path.join(logDir, filename),
    format: fileFormat,
    maxsize: config.logging.rotation.maxSize,
    maxFiles: config.logging.rotation.maxFiles,
    zippedArchive: config.logging.rotation.zippedArchive
  }));
}

// Request logging middleware
const requestLogger = (req, res, next) => {
  const start = Date.now();
  req.startTime = start;

  // Log request
  logger.http('Request received', {
    method: req.method,
    url: req.url,
    ip: req.ip,
    userAgent: req.get('User-Agent'),
    userId: req.user?.id,
    apiKey: req.apiKey ? req.apiKey.substring(0, 8) + '...' : undefined,
    requestId: req.requestId
  });

  // Log response
  res.on('finish', () => {
    const duration = Date.now() - start;
    const level = res.statusCode >= 400 ? 'warn' : 'http';

    logger.log(level, 'Request completed', {
      method: req.method,
      url: req.url,
      statusCode: res.statusCode,
      duration,
      ip: req.ip,
      userId: req.user?.id,
      apiKey: req.apiKey ? req.apiKey.substring(0, 8) + '...' : undefined,
      requestId: req.requestId
    });
  });

  next();
};

// Error logging middleware
const errorLogger = (error, req, res, next) => {
  logger.error('Unhandled error', {
    error: error.message,
    stack: error.stack,
    method: req.method,
    url: req.url,
    ip: req.ip,
    userId: req.user?.id,
    requestId: req.requestId,
    body: req.method !== 'GET' ? req.body : undefined,
    query: req.query,
    params: req.params
  });

  next(error);
};

// Performance logging
const performanceLogger = {
  logTiming: (operation, duration, metadata = {}) => {
    logger.info('Performance timing', {
      operation,
      duration_ms: duration,
      ...metadata
    });
  },

  logMemoryUsage: (operation, memoryMB, metadata = {}) => {
    logger.info('Memory usage', {
      operation,
      memory_mb: memoryMB,
      ...metadata
    });
  }
};

// Request ID middleware
const requestId = (req, res, next) => {
  req.requestId = req.headers['x-request-id'] ||
                  req.headers['x-correlation-id'] ||
                  generateRequestId();
  res.setHeader('x-request-id', req.requestId);
  next();
};

// Generate unique request ID
function generateRequestId() {
  return Date.now().toString(36) + Math.random().toString(36).substr(2);
}

module.exports = {
  logger,
  requestLogger,
  errorLogger,
  performanceLogger,
  requestId,
  generateRequestId
};