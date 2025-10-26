/**
 * Pricing routes for API Gateway
 */

const express = require('express');
const router = express.Router();
const axios = require('axios');
const config = require('../config');
const logger = require('../utils/logger');
const cache = require('../middleware/cache');
const circuitBreaker = require('../middleware/circuitBreaker');
const { authenticateFlexible, userRateLimit } = require('../middleware/auth');
const { rateLimit } = require('../middleware/rateLimit');
const {
  optionPricingValidation,
  swaptionPricingValidation,
  batchPricingValidation
} = require('../middleware/validation');

/**
 * @swagger
 * /pricing/options:
 *   post:
 *     summary: Price European options
 *     description: Calculate prices for European call/put options using various models
 *     tags: [Pricing]
 *     security:
 *       - bearerAuth: []
 *     requestBody:
 *       required: true
 *       content:
 *         application/json:
 *           schema:
 *             type: object
 *             required:
 *               - spot_price
 *               - strike_price
 *               - time_to_expiry
 *               - risk_free_rate
 *               - volatility
 *               - option_type
 *             properties:
 *               spot_price:
 *                 type: number
 *                 example: 100.0
 *               strike_price:
 *                 type: number
 *                 example: 105.0
 *               time_to_expiry:
 *                 type: number
 *                 example: 1.0
 *               risk_free_rate:
 *                 type: number
 *                 example: 0.05
 *               volatility:
 *                 type: number
 *                 example: 0.2
 *               option_type:
 *                 type: string
 *                 enum: [call, put]
 *                 example: call
 *               model:
 *                 type: string
 *                 enum: [black_scholes, monte_carlo, ml]
 *                 default: black_scholes
 *               dividend_yield:
 *                 type: number
 *                 default: 0.0
 *     responses:
 *       200:
 *         description: Option price calculated successfully
 *         content:
 *           application/json:
 *             schema:
 *               type: object
 *               properties:
 *                 price:
 *                   type: number
 *                   example: 8.02
 *                 greeks:
 *                   type: object
 *                   properties:
 *                     delta:
 *                       type: number
 *                     gamma:
 *                       type: number
 *                     theta:
 *                       type: number
 *                     vega:
 *                       type: number
 *                     rho:
 *                       type: number
 *                 model:
 *                   type: string
 *                 timestamp:
 *                   type: string
 *                   format: date-time
 *       400:
 *         description: Invalid input parameters
 *       503:
 *         description: ML service unavailable
 */
router.post('/options',
  authenticateFlexible,
  userRateLimit(),
  optionPricingValidation,
  async (req, res) => {
  try {

    const {
      spot_price,
      strike_price,
      time_to_expiry,
      risk_free_rate,
      volatility,
      option_type,
      model = 'black_scholes',
      dividend_yield = 0.0
    } = req.body;

    logger.info('Option pricing request', {
      spot_price,
      strike_price,
      option_type,
      model,
      user_id: req.user?.id
    });

    let result;

    if (model === 'black_scholes') {
      // Use internal Black-Scholes calculation
      const bsResult = await priceWithBlackScholes(
        spot_price, strike_price, time_to_expiry,
        risk_free_rate, volatility, option_type, dividend_yield
      );
      result = bsResult;

    } else if (model === 'monte_carlo') {
      // Use Monte Carlo service
      const mcResult = await priceWithMonteCarlo(req.body);
      result = mcResult;

    } else if (model === 'ml') {
      // Use ML service
      const mlResult = await priceWithML(req.body);
      result = mlResult;
    }

    // Cache successful results
    const cacheKey = `option:${JSON.stringify(req.body)}`;
    cache.set(cacheKey, result, config.cache.ttl);

    logger.info('Option pricing completed', {
      price: result.price,
      model,
      response_time: Date.now() - req.startTime
    });

    res.json({
      ...result,
      timestamp: new Date().toISOString()
    });

  } catch (error) {
    logger.error('Option pricing failed', {
      error: error.message,
      stack: error.stack,
      body: req.body
    });

    res.status(500).json({
      error: 'Pricing calculation failed',
      message: error.message,
      timestamp: new Date().toISOString()
    });
  }
});

/**
 * @swagger
 * /pricing/swaptions:
 *   post:
 *     summary: Price European swaptions
 *     description: Calculate prices for European swaptions using various models
 *     tags: [Pricing]
 *     security:
 *       - bearerAuth: []
 *     requestBody:
 *       required: true
 *       content:
 *         application/json:
 *           schema:
 *             type: object
 *             required:
 *               - swap_rate
 *               - strike_rate
 *               - option_tenor
 *               - swap_tenor
 *               - volatility
 *             properties:
 *               swap_rate:
 *                 type: number
 *                 example: 0.03
 *               strike_rate:
 *                 type: number
 *                 example: 0.035
 *               option_tenor:
 *                 type: number
 *                 example: 1.0
 *               swap_tenor:
 *                 type: number
 *                 example: 5.0
 *               volatility:
 *                 type: number
 *                 example: 0.15
 *               model:
 *                 type: string
 *                 enum: [black, monte_carlo, ml]
 *                 default: black
 *     responses:
 *       200:
 *         description: Swaption price calculated successfully
 *         content:
 *           application/json:
 *             schema:
 *               type: object
 *               properties:
 *                 price:
 *                   type: number
 *                   example: 0.0025
 *                 greeks:
 *                   type: object
 *                   properties:
 *                     delta:
 *                       type: number
 *                     vega:
 *                       type: number
 *                 model:
 *                   type: string
 *                 timestamp:
 *                   type: string
 *                   format: date-time
 */
router.post('/swaptions',
  authenticateFlexible,
  userRateLimit(),
  swaptionPricingValidation,
  async (req, res) => {
  try {

    const {
      swap_rate,
      strike_rate,
      option_tenor,
      swap_tenor,
      volatility,
      model = 'black'
    } = req.body;

    logger.info('Swaption pricing request', {
      swap_rate,
      strike_rate,
      option_tenor,
      swap_tenor,
      model,
      user_id: req.user?.id
    });

    let result;

    if (model === 'black') {
      // Use internal Black model
      const blackResult = await priceWithBlackSwaption(
        swap_rate, strike_rate, option_tenor, swap_tenor, volatility
      );
      result = blackResult;

    } else if (model === 'monte_carlo') {
      // Use Monte Carlo service
      const mcResult = await priceWithMonteCarlo(req.body);
      result = mcResult;

    } else if (model === 'ml') {
      // Use ML service
      const mlResult = await priceWithML(req.body);
      result = mlResult;
    }

    const cacheKey = `swaption:${JSON.stringify(req.body)}`;
    cache.set(cacheKey, result, config.cache.ttl);

    logger.info('Swaption pricing completed', {
      price: result.price,
      model,
      response_time: Date.now() - req.startTime
    });

    res.json({
      ...result,
      timestamp: new Date().toISOString()
    });

  } catch (error) {
    logger.error('Swaption pricing failed', {
      error: error.message,
      stack: error.stack,
      body: req.body
    });

    res.status(500).json({
      error: 'Swaption pricing calculation failed',
      message: error.message,
      timestamp: new Date().toISOString()
    });
  }
});

/**
 * @swagger
 * /pricing/batch:
 *   post:
 *     summary: Batch pricing request
 *     description: Calculate prices for multiple instruments in a single request
 *     tags: [Pricing]
 *     security:
 *       - bearerAuth: []
 *     requestBody:
 *       required: true
 *       content:
 *         application/json:
 *           schema:
 *             type: object
 *             required:
 *               - instruments
 *             properties:
 *               instruments:
 *                 type: array
 *                 items:
 *                   type: object
 *                   properties:
 *                     type:
 *                       type: string
 *                       enum: [option, swaption]
 *                     parameters:
 *                       type: object
 *     responses:
 *       200:
 *         description: Batch pricing completed
 *         content:
 *           application/json:
 *             schema:
 *               type: object
 *               properties:
 *                 results:
 *                   type: array
 *                   items:
 *                     type: object
 *                     properties:
 *                       instrument_id:
 *                         type: string
 *                       price:
 *                         type: number
 *                       error:
 *                         type: string
 */
router.post('/batch',
  authenticateFlexible,
  rateLimit({ windowMs: 60000, max: 5 }), // Stricter limit for batch requests
  batchPricingValidation,
  async (req, res) => {
  try {

    const { instruments } = req.body;
    const results = [];

    logger.info('Batch pricing request', {
      count: instruments.length,
      user_id: req.user?.id
    });

    // Process instruments in parallel with concurrency limit
    const concurrencyLimit = 10;
    for (let i = 0; i < instruments.length; i += concurrencyLimit) {
      const batch = instruments.slice(i, i + concurrencyLimit);
      const batchPromises = batch.map(async (instrument, index) => {
        const instrumentId = instrument.id || `instrument_${i + index}`;

        try {
          let result;
          if (instrument.type === 'option') {
            result = await priceWithBlackScholes(
              instrument.parameters.spot_price,
              instrument.parameters.strike_price,
              instrument.parameters.time_to_expiry,
              instrument.parameters.risk_free_rate,
              instrument.parameters.volatility,
              instrument.parameters.option_type,
              instrument.parameters.dividend_yield || 0
            );
          } else if (instrument.type === 'swaption') {
            result = await priceWithBlackSwaption(
              instrument.parameters.swap_rate,
              instrument.parameters.strike_rate,
              instrument.parameters.option_tenor,
              instrument.parameters.swap_tenor,
              instrument.parameters.volatility
            );
          }

          return {
            instrument_id: instrumentId,
            ...result,
            success: true
          };

        } catch (error) {
          logger.warn('Batch pricing item failed', {
            instrument_id: instrumentId,
            error: error.message
          });

          return {
            instrument_id: instrumentId,
            error: error.message,
            success: false
          };
        }
      });

      const batchResults = await Promise.all(batchPromises);
      results.push(...batchResults);
    }

    const successCount = results.filter(r => r.success).length;

    logger.info('Batch pricing completed', {
      total: instruments.length,
      successful: successCount,
      failed: instruments.length - successCount,
      response_time: Date.now() - req.startTime
    });

    res.json({
      results,
      summary: {
        total: instruments.length,
        successful: successCount,
        failed: instruments.length - successCount
      },
      timestamp: new Date().toISOString()
    });

  } catch (error) {
    logger.error('Batch pricing failed', {
      error: error.message,
      stack: error.stack
    });

    res.status(500).json({
      error: 'Batch pricing failed',
      message: error.message,
      timestamp: new Date().toISOString()
    });
  }
});

// Helper functions for pricing calculations

async function priceWithBlackScholes(spot, strike, time, rate, vol, type, dividend = 0) {
  // Internal Black-Scholes calculation
  const d1 = (Math.log(spot/strike) + (rate - dividend + vol*vol/2) * time) / (vol * Math.sqrt(time));
  const d2 = d1 - vol * Math.sqrt(time);

  // CDF approximation using error function
  const cdf = (x) => 0.5 * (1 + erf(x / Math.sqrt(2)));

  let price, delta;
  if (type === 'call') {
    price = spot * Math.exp(-dividend * time) * cdf(d1) - strike * Math.exp(-rate * time) * cdf(d2);
    delta = Math.exp(-dividend * time) * cdf(d1);
  } else {
    price = strike * Math.exp(-rate * time) * cdf(-d2) - spot * Math.exp(-dividend * time) * cdf(-d1);
    delta = -Math.exp(-dividend * time) * cdf(-d1);
  }

  return {
    price: Math.round(price * 10000) / 10000, // Round to 4 decimal places
    greeks: {
      delta: Math.round(delta * 10000) / 10000
    },
    model: 'black_scholes'
  };
}

async function priceWithBlackSwaption(swapRate, strikeRate, optionTenor, swapTenor, volatility) {
  // Simplified Black model for swaptions
  const forwardRate = swapRate;
  const time = optionTenor;

  const d1 = (Math.log(forwardRate/strikeRate) + (volatility**2/2) * time) / (volatility * Math.sqrt(time));
  const d2 = d1 - volatility * Math.sqrt(time);

  // Simplified annuity factor
  const annuityFactor = swapTenor * 0.95; // Approximation

  const price = annuityFactor * (forwardRate * normCdf(d1) - strikeRate * normCdf(d2));

  return {
    price: Math.round(price * 10000) / 10000,
    greeks: {
      delta: Math.round(annuityFactor * normCdf(d1) * 10000) / 10000
    },
    model: 'black'
  };
}

async function priceWithMonteCarlo(params) {
  // Forward to Monte Carlo service
  const response = await axios.post(
    `${config.services.mlService.url}/pricing/monte-carlo`,
    params,
    {
      timeout: config.services.mlService.timeout,
      headers: {
        'X-API-Key': process.env.ML_SERVICE_API_KEY
      }
    }
  );

  return {
    ...response.data,
    model: 'monte_carlo'
  };
}

async function priceWithML(params) {
  // Forward to ML service
  const response = await axios.post(
    `${config.services.mlService.url}/pricing/ml`,
    params,
    {
      timeout: config.services.mlService.timeout,
      headers: {
        'X-API-Key': process.env.ML_SERVICE_API_KEY
      }
    }
  );

  return {
    ...response.data,
    model: 'ml'
  };
}

// Utility functions
function erf(x) {
  // Abramowitz and Stegun approximation
  const a1 =  0.254829592;
  const a2 = -0.284496736;
  const a3 =  1.421413741;
  const a4 = -1.453152027;
  const a5 =  1.061405429;
  const p  =  0.3275911;

  const sign = x < 0 ? -1 : 1;
  x = Math.abs(x);

  const t = 1.0 / (1.0 + p * x);
  const y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * Math.exp(-x * x);

  return sign * y;
}

function normCdf(x) {
  return 0.5 * (1 + erf(x / Math.sqrt(2)));
}

module.exports = router;