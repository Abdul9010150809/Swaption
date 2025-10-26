/**
 * Request validation middleware for API Gateway
 */

const { body, param, query, validationResult } = require('express-validator');
const logger = require('../utils/logger');

/**
 * Handle validation errors
 */
const handleValidationErrors = (req, res, next) => {
  const errors = validationResult(req);
  if (!errors.isEmpty()) {
    logger.warn('Validation failed', {
      errors: errors.array(),
      body: req.body,
      query: req.query,
      params: req.params
    });

    return res.status(400).json({
      error: 'Validation failed',
      message: 'One or more fields failed validation',
      details: errors.array().map(err => ({
        field: err.path,
        message: err.msg,
        value: err.value
      }))
    });
  }
  next();
};

/**
 * Custom validation rules
 */

// Check if value is a positive number
const isPositiveNumber = (value) => {
  const num = parseFloat(value);
  return !isNaN(num) && num > 0;
};

// Check if value is a valid percentage (0-100)
const isValidPercentage = (value) => {
  const num = parseFloat(value);
  return !isNaN(num) && num >= 0 && num <= 100;
};

// Check if value is a valid rate (-1 to 1)
const isValidRate = (value) => {
  const num = parseFloat(value);
  return !isNaN(num) && num >= -1 && num <= 1;
};

// Check if value is a reasonable volatility (0-5)
const isValidVolatility = (value) => {
  const num = parseFloat(value);
  return !isNaN(num) && num >= 0 && num <= 5;
};

// Check if value is a reasonable time to expiry (0-10 years)
const isValidTimeToExpiry = (value) => {
  const num = parseFloat(value);
  return !isNaN(num) && num >= 0 && num <= 10;
};

// Check if value is a reasonable interest rate (-0.1 to 0.2)
const isValidInterestRate = (value) => {
  const num = parseFloat(value);
  return !isNaN(num) && num >= -0.1 && num <= 0.2;
};

/**
 * Validation chains for different endpoints
 */

// Option pricing validation
const optionPricingValidation = [
  body('spot_price')
    .exists().withMessage('Spot price is required')
    .custom(isPositiveNumber).withMessage('Spot price must be a positive number'),

  body('strike_price')
    .exists().withMessage('Strike price is required')
    .custom(isPositiveNumber).withMessage('Strike price must be a positive number'),

  body('time_to_expiry')
    .exists().withMessage('Time to expiry is required')
    .custom(isValidTimeToExpiry).withMessage('Time to expiry must be between 0 and 10 years'),

  body('risk_free_rate')
    .exists().withMessage('Risk-free rate is required')
    .custom(isValidInterestRate).withMessage('Risk-free rate must be between -10% and 20%'),

  body('volatility')
    .exists().withMessage('Volatility is required')
    .custom(isValidVolatility).withMessage('Volatility must be between 0% and 500%'),

  body('option_type')
    .exists().withMessage('Option type is required')
    .isIn(['call', 'put']).withMessage('Option type must be either "call" or "put"'),

  body('model')
    .optional()
    .isIn(['black_scholes', 'monte_carlo', 'ml']).withMessage('Model must be one of: black_scholes, monte_carlo, ml'),

  body('dividend_yield')
    .optional()
    .custom(isValidRate).withMessage('Dividend yield must be between -100% and 100%'),

  handleValidationErrors
];

// Swaption pricing validation
const swaptionPricingValidation = [
  body('swap_rate')
    .exists().withMessage('Swap rate is required')
    .custom(isValidInterestRate).withMessage('Swap rate must be between -10% and 20%'),

  body('strike_rate')
    .exists().withMessage('Strike rate is required')
    .custom(isValidInterestRate).withMessage('Strike rate must be between -10% and 20%'),

  body('option_tenor')
    .exists().withMessage('Option tenor is required')
    .custom(isValidTimeToExpiry).withMessage('Option tenor must be between 0 and 10 years'),

  body('swap_tenor')
    .exists().withMessage('Swap tenor is required')
    .custom((value) => {
      const num = parseFloat(value);
      return !isNaN(num) && num >= 0.5 && num <= 50;
    }).withMessage('Swap tenor must be between 0.5 and 50 years'),

  body('volatility')
    .exists().withMessage('Volatility is required')
    .custom((value) => {
      const num = parseFloat(value);
      return !isNaN(num) && num >= 0.001 && num <= 2;
    }).withMessage('Volatility must be between 0.1% and 200%'),

  body('model')
    .optional()
    .isIn(['black', 'monte_carlo', 'ml']).withMessage('Model must be one of: black, monte_carlo, ml'),

  handleValidationErrors
];

// Bond pricing validation
const bondPricingValidation = [
  body('face_value')
    .exists().withMessage('Face value is required')
    .custom(isPositiveNumber).withMessage('Face value must be positive'),

  body('coupon_rate')
    .exists().withMessage('Coupon rate is required')
    .custom(isValidRate).withMessage('Coupon rate must be between -100% and 100%'),

  body('maturity')
    .exists().withMessage('Maturity is required')
    .custom((value) => {
      const num = parseFloat(value);
      return !isNaN(num) && num > 0 && num <= 100;
    }).withMessage('Maturity must be between 0 and 100 years'),

  body('yield_to_maturity')
    .exists().withMessage('Yield to maturity is required')
    .custom(isValidInterestRate).withMessage('Yield to maturity must be between -10% and 20%'),

  body('frequency')
    .optional()
    .isInt({ min: 1, max: 12 }).withMessage('Payment frequency must be between 1 and 12'),

  handleValidationErrors
];

// Risk metrics validation
const riskMetricsValidation = [
  body('returns')
    .exists().withMessage('Returns data is required')
    .isArray().withMessage('Returns must be an array')
    .custom((value) => value.length > 0).withMessage('Returns array cannot be empty'),

  body('confidence_level')
    .optional()
    .custom(isValidPercentage).withMessage('Confidence level must be between 0% and 100%'),

  body('time_horizon')
    .optional()
    .isInt({ min: 1, max: 252 }).withMessage('Time horizon must be between 1 and 252 days'),

  body('method')
    .optional()
    .isIn(['historical', 'parametric', 'monte_carlo']).withMessage('Method must be one of: historical, parametric, monte_carlo'),

  handleValidationErrors
];

// Portfolio analysis validation
const portfolioAnalysisValidation = [
  body('weights')
    .exists().withMessage('Portfolio weights are required')
    .isArray().withMessage('Weights must be an array')
    .custom((value) => {
      const sum = value.reduce((acc, w) => acc + parseFloat(w || 0), 0);
      return Math.abs(sum - 1.0) < 0.01; // Allow small tolerance
    }).withMessage('Portfolio weights must sum to 1'),

  body('returns')
    .exists().withMessage('Asset returns are required')
    .isArray().withMessage('Returns must be an array'),

  body('portfolio_value')
    .optional()
    .custom(isPositiveNumber).withMessage('Portfolio value must be positive'),

  handleValidationErrors
];

// Batch pricing validation
const batchPricingValidation = [
  body('instruments')
    .exists().withMessage('Instruments array is required')
    .isArray({ min: 1, max: 50 }).withMessage('Must provide 1-50 instruments'),

  body('instruments.*.type')
    .exists().withMessage('Instrument type is required')
    .isIn(['option', 'swaption', 'bond']).withMessage('Instrument type must be option, swaption, or bond'),

  body('instruments.*.parameters')
    .exists().withMessage('Instrument parameters are required')
    .isObject().withMessage('Parameters must be an object'),

  handleValidationErrors
];

// Market data validation
const marketDataValidation = [
  param('symbol')
    .exists().withMessage('Symbol parameter is required')
    .isLength({ min: 1, max: 10 }).withMessage('Symbol must be 1-10 characters'),

  query('start_date')
    .optional()
    .isISO8601().withMessage('Start date must be in ISO 8601 format'),

  query('end_date')
    .optional()
    .isISO8601().withMessage('End date must be in ISO 8601 format'),

  handleValidationErrors
];

// User registration validation
const userRegistrationValidation = [
  body('email')
    .exists().withMessage('Email is required')
    .isEmail().withMessage('Must be a valid email address')
    .normalizeEmail(),

  body('password')
    .exists().withMessage('Password is required')
    .isLength({ min: 8 }).withMessage('Password must be at least 8 characters long')
    .matches(/^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)/).withMessage('Password must contain at least one lowercase letter, one uppercase letter, and one number'),

  body('first_name')
    .optional()
    .isLength({ min: 1, max: 50 }).withMessage('First name must be 1-50 characters'),

  body('last_name')
    .optional()
    .isLength({ min: 1, max: 50 }).withMessage('Last name must be 1-50 characters'),

  body('organization')
    .optional()
    .isLength({ min: 1, max: 100 }).withMessage('Organization must be 1-100 characters'),

  handleValidationErrors
];

// Login validation
const loginValidation = [
  body('email')
    .exists().withMessage('Email is required')
    .isEmail().withMessage('Must be a valid email address')
    .normalizeEmail(),

  body('password')
    .exists().withMessage('Password is required')
    .notEmpty().withMessage('Password cannot be empty'),

  handleValidationErrors
];

module.exports = {
  handleValidationErrors,
  optionPricingValidation,
  swaptionPricingValidation,
  bondPricingValidation,
  riskMetricsValidation,
  portfolioAnalysisValidation,
  batchPricingValidation,
  marketDataValidation,
  userRegistrationValidation,
  loginValidation
};