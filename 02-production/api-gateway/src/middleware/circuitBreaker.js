/**
 * Circuit breaker middleware for API Gateway
 */

const config = require('../config');
const logger = require('../utils/logger');

// Circuit breaker states
const STATES = {
  CLOSED: 'closed',
  OPEN: 'open',
  HALF_OPEN: 'half_open'
};

// Circuit breaker store
const circuitBreakers = new Map();

/**
 * Circuit breaker class
 */
class CircuitBreaker {
  constructor(name, options = {}) {
    this.name = name;
    this.failureThreshold = options.failureThreshold || config.circuitBreaker.errorThresholdPercentage;
    this.resetTimeout = options.resetTimeout || config.circuitBreaker.resetTimeout;
    this.monitoringPeriod = options.monitoringPeriod || config.circuitBreaker.monitoringPeriod;
    this.successThreshold = options.successThreshold || 3; // For half-open state

    this.state = STATES.CLOSED;
    this.failures = 0;
    this.successes = 0;
    this.lastFailureTime = null;
    this.nextAttemptTime = null;
  }

  async execute(operation) {
    if (!config.circuitBreaker.enabled) {
      return await operation();
    }

    if (this.state === STATES.OPEN) {
      if (Date.now() < this.nextAttemptTime) {
        throw new Error(`Circuit breaker ${this.name} is OPEN`);
      } else {
        this.state = STATES.HALF_OPEN;
        this.successes = 0;
        logger.info(`Circuit breaker ${this.name} moved to HALF_OPEN`);
      }
    }

    try {
      const result = await operation();
      this.onSuccess();
      return result;
    } catch (error) {
      this.onFailure();
      throw error;
    }
  }

  onSuccess() {
    this.failures = 0;

    if (this.state === STATES.HALF_OPEN) {
      this.successes++;
      if (this.successes >= this.successThreshold) {
        this.state = STATES.CLOSED;
        logger.info(`Circuit breaker ${this.name} moved to CLOSED`);
      }
    }
  }

  onFailure() {
    this.failures++;
    this.lastFailureTime = Date.now();

    if (this.state === STATES.HALF_OPEN) {
      this.state = STATES.OPEN;
      this.nextAttemptTime = Date.now() + this.resetTimeout;
      logger.warn(`Circuit breaker ${this.name} moved to OPEN (half-open failure)`);
    } else if (this.state === STATES.CLOSED && this.failures >= this.failureThreshold) {
      this.state = STATES.OPEN;
      this.nextAttemptTime = Date.now() + this.resetTimeout;
      logger.warn(`Circuit breaker ${this.name} moved to OPEN (threshold exceeded)`);
    }
  }

  getState() {
    return {
      name: this.name,
      state: this.state,
      failures: this.failures,
      successes: this.successes,
      lastFailureTime: this.lastFailureTime,
      nextAttemptTime: this.nextAttemptTime
    };
  }
}

/**
 * Get or create circuit breaker
 */
function getCircuitBreaker(name, options = {}) {
  if (!circuitBreakers.has(name)) {
    circuitBreakers.set(name, new CircuitBreaker(name, options));
  }
  return circuitBreakers.get(name);
}

/**
 * Circuit breaker middleware
 */
const circuitBreaker = (name, options = {}) => {
  return async (req, res, next) => {
    if (!config.circuitBreaker.enabled) {
      return next();
    }

    const breaker = getCircuitBreaker(name, options);

    try {
      await breaker.execute(async () => {
        // Add circuit breaker state to request
        req.circuitBreaker = breaker.getState();
        next();
      });
    } catch (error) {
      logger.error(`Circuit breaker ${name} blocked request`, {
        state: breaker.getState(),
        error: error.message
      });

      return res.status(503).json({
        error: 'Service temporarily unavailable',
        message: 'The requested service is currently experiencing issues. Please try again later.',
        circuitBreaker: {
          name,
          state: breaker.state,
          retryAfter: breaker.nextAttemptTime ?
            Math.ceil((breaker.nextAttemptTime - Date.now()) / 1000) : null
        },
        timestamp: new Date().toISOString()
      });
    }
  };
};

/**
 * HTTP request circuit breaker
 */
const httpCircuitBreaker = (url, options = {}) => {
  return async (req, res, next) => {
    if (!config.circuitBreaker.enabled) {
      return next();
    }

    const breakerName = `http:${url}`;
    const breaker = getCircuitBreaker(breakerName, options);

    try {
      await breaker.execute(async () => {
        next();
      });
    } catch (error) {
      // Don't proceed with the request
      logger.error(`HTTP circuit breaker blocked request to ${url}`, {
        state: breaker.getState()
      });

      return res.status(503).json({
        error: 'External service unavailable',
        message: `The service at ${url} is temporarily unavailable.`,
        circuitBreaker: {
          name: breakerName,
          state: breaker.state,
          retryAfter: breaker.nextAttemptTime ?
            Math.ceil((breaker.nextAttemptTime - Date.now()) / 1000) : null
        },
        timestamp: new Date().toISOString()
      });
    }
  };
};

/**
 * Service-specific circuit breakers
 */
const mlServiceBreaker = circuitBreaker('ml-service', {
  failureThreshold: 5,
  resetTimeout: 30000
});

const databaseBreaker = circuitBreaker('database', {
  failureThreshold: 3,
  resetTimeout: 15000
});

/**
 * Get all circuit breaker states
 */
const getCircuitBreakerStates = () => {
  const states = {};
  for (const [name, breaker] of circuitBreakers.entries()) {
    states[name] = breaker.getState();
  }
  return states;
};

/**
 * Reset circuit breaker
 */
const resetCircuitBreaker = (name) => {
  const breaker = circuitBreakers.get(name);
  if (breaker) {
    breaker.state = STATES.CLOSED;
    breaker.failures = 0;
    breaker.successes = 0;
    breaker.lastFailureTime = null;
    breaker.nextAttemptTime = null;
    logger.info(`Circuit breaker ${name} manually reset`);
    return true;
  }
  return false;
};

/**
 * Reset all circuit breakers
 */
const resetAllCircuitBreakers = () => {
  for (const [name, breaker] of circuitBreakers.entries()) {
    resetCircuitBreaker(name);
  }
  logger.info('All circuit breakers reset');
};

module.exports = {
  circuitBreaker,
  httpCircuitBreaker,
  mlServiceBreaker,
  databaseBreaker,
  getCircuitBreakerStates,
  resetCircuitBreaker,
  resetAllCircuitBreakers,
  STATES
};