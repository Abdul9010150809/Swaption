/**
 * Configuration management for API Gateway
 */

const path = require('path');

// Environment variables with defaults
const config = {
  // Server configuration
  server: {
    port: process.env.PORT || 3000,
    host: process.env.HOST || '0.0.0.0',
    env: process.env.NODE_ENV || 'development',
    apiVersion: process.env.API_VERSION || 'v1'
  },

  // Database configuration
  database: {
    type: process.env.DB_TYPE || 'redis',
    host: process.env.DB_HOST || 'localhost',
    port: process.env.DB_PORT || 6379,
    password: process.env.DB_PASSWORD,
    database: process.env.DB_NAME || 0,
    ttl: parseInt(process.env.DB_TTL) || 3600, // 1 hour default
    connectionTimeout: parseInt(process.env.DB_CONNECTION_TIMEOUT) || 5000
  },

  // External services
  services: {
    mlService: {
      url: process.env.ML_SERVICE_URL || 'http://localhost:8000',
      timeout: parseInt(process.env.ML_SERVICE_TIMEOUT) || 30000,
      retries: parseInt(process.env.ML_SERVICE_RETRIES) || 3,
      retryDelay: parseInt(process.env.ML_SERVICE_RETRY_DELAY) || 1000
    },
    frontend: {
      url: process.env.FRONTEND_URL || 'http://localhost:3001',
      corsOrigins: process.env.CORS_ORIGINS ? process.env.CORS_ORIGINS.split(',') : ['http://localhost:3001']
    }
  },

  // Security configuration
  security: {
    jwtSecret: process.env.JWT_SECRET || 'your-super-secret-jwt-key-change-in-production',
    jwtExpiration: process.env.JWT_EXPIRATION || '24h',
    bcryptRounds: parseInt(process.env.BCRYPT_ROUNDS) || 12,

    // Rate limiting
    rateLimit: {
      windowMs: parseInt(process.env.RATE_LIMIT_WINDOW_MS) || 15 * 60 * 1000, // 15 minutes
      max: parseInt(process.env.RATE_LIMIT_MAX) || 100, // limit each IP to 100 requests per windowMs
      message: 'Too many requests from this IP, please try again later.',
      standardHeaders: true,
      legacyHeaders: false
    },

    // CORS configuration
    cors: {
      origin: process.env.CORS_ORIGINS ? process.env.CORS_ORIGINS.split(',') : ['http://localhost:3001'],
      credentials: true,
      methods: ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'],
      allowedHeaders: ['Content-Type', 'Authorization', 'X-Requested-With']
    }
  },

  // Logging configuration
  logging: {
    level: process.env.LOG_LEVEL || 'info',
    format: process.env.LOG_FORMAT || 'json',
    logDir: process.env.LOG_DIR || path.join(__dirname, '../../logs'),

    // Winston daily rotate file configuration
    rotation: {
      filename: 'api-gateway-%DATE%.log',
      datePattern: 'YYYY-MM-DD',
      maxSize: '20m',
      maxFiles: '14d',
      zippedArchive: true
    }
  },

  // API configuration
  api: {
    prefix: '/api',
    docs: {
      enabled: process.env.API_DOCS_ENABLED !== 'false',
      path: '/api-docs',
      swaggerDefinition: {
        openapi: '3.0.0',
        info: {
          title: 'Price Matrix API',
          version: '1.0.0',
          description: 'Financial derivative pricing API',
          contact: {
            name: 'Price Matrix Team',
            email: 'support@pricematrix.com'
          }
        },
        servers: [
          {
            url: `http://localhost:${process.env.PORT || 3000}`,
            description: 'Development server'
          }
        ],
        components: {
          securitySchemes: {
            bearerAuth: {
              type: 'http',
              scheme: 'bearer',
              bearerFormat: 'JWT'
            }
          }
        },
        security: [
          {
            bearerAuth: []
          }
        ]
      }
    },

    // Request/response limits
    limits: {
      json: '10mb',
      urlencoded: '10mb',
      files: '10mb'
    }
  },

  // Monitoring and metrics
  monitoring: {
    enabled: process.env.MONITORING_ENABLED !== 'false',
    metrics: {
      prefix: 'price_matrix_api_',
      collectDefaultMetrics: true,
      requestDurationBuckets: [0.1, 0.5, 1, 2, 5, 10]
    },

    // Health check configuration
    health: {
      checks: {
        database: true,
        mlService: true,
        memory: true,
        disk: true
      },
      thresholds: {
        memoryUsage: 0.9, // 90% memory usage threshold
        diskUsage: 0.9,   // 90% disk usage threshold
        responseTime: 5000 // 5 second response time threshold
      }
    }
  },

  // Caching configuration
  cache: {
    enabled: process.env.CACHE_ENABLED !== 'false',
    ttl: parseInt(process.env.CACHE_TTL) || 300, // 5 minutes default
    maxKeys: parseInt(process.env.CACHE_MAX_KEYS) || 10000,

    // Cache keys for different endpoints
    keys: {
      pricing: 'pricing:{params}',
      marketData: 'market:{symbol}:{date}',
      risk: 'risk:{portfolio}:{date}'
    }
  },

  // Circuit breaker configuration
  circuitBreaker: {
    enabled: process.env.CIRCUIT_BREAKER_ENABLED !== 'false',
    timeout: parseInt(process.env.CIRCUIT_BREAKER_TIMEOUT) || 5000,
    errorThresholdPercentage: parseFloat(process.env.CIRCUIT_BREAKER_ERROR_THRESHOLD) || 50,
    resetTimeout: parseInt(process.env.CIRCUIT_BREAKER_RESET_TIMEOUT) || 30000,
    monitoringPeriod: parseInt(process.env.CIRCUIT_BREAKER_MONITORING_PERIOD) || 10000
  },

  // Feature flags
  features: {
    authentication: process.env.FEATURE_AUTH !== 'false',
    rateLimiting: process.env.FEATURE_RATE_LIMIT !== 'false',
    caching: process.env.FEATURE_CACHING !== 'false',
    monitoring: process.env.FEATURE_MONITORING !== 'false',
    circuitBreaker: process.env.FEATURE_CIRCUIT_BREAKER !== 'false',
    swaggerDocs: process.env.FEATURE_SWAGGER_DOCS !== 'false'
  }
};

// Environment-specific overrides
if (config.server.env === 'production') {
  // Production-specific configuration
  config.logging.level = 'warn';
  config.monitoring.enabled = true;
  config.cache.enabled = true;
  config.circuitBreaker.enabled = true;

  // Stricter rate limiting in production
  config.security.rateLimit.max = 50;

} else if (config.server.env === 'test') {
  // Test-specific configuration
  config.logging.level = 'error';
  config.monitoring.enabled = false;
  config.cache.enabled = false;
  config.circuitBreaker.enabled = false;
}

// Validate critical configuration
function validateConfig() {
  const required = ['security.jwtSecret'];

  for (const key of required) {
    const keys = key.split('.');
    let value = config;

    for (const k of keys) {
      value = value[k];
    }

    if (!value) {
      throw new Error(`Required configuration missing: ${key}`);
    }
  }
}

// Validate configuration on load
validateConfig();

module.exports = config;