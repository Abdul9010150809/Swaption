/**
 * Validation utilities for the frontend
 */

/**
 * Validate option pricing parameters
 */
export const validateOptionParams = (params) => {
  const errors = {};

  // Spot price validation
  if (!params.spot_price || params.spot_price <= 0) {
    errors.spot_price = 'Spot price must be a positive number';
  }

  // Strike price validation
  if (!params.strike_price || params.strike_price <= 0) {
    errors.strike_price = 'Strike price must be a positive number';
  }

  // Time to expiry validation
  if (!params.time_to_expiry || params.time_to_expiry < 0) {
    errors.time_to_expiry = 'Time to expiry must be non-negative';
  } else if (params.time_to_expiry > 10) {
    errors.time_to_expiry = 'Time to expiry cannot exceed 10 years';
  }

  // Risk-free rate validation
  if (params.risk_free_rate === undefined || params.risk_free_rate === null) {
    errors.risk_free_rate = 'Risk-free rate is required';
  } else if (params.risk_free_rate < -0.1 || params.risk_free_rate > 0.2) {
    errors.risk_free_rate = 'Risk-free rate must be between -10% and 20%';
  }

  // Volatility validation
  if (!params.volatility || params.volatility <= 0) {
    errors.volatility = 'Volatility must be positive';
  } else if (params.volatility > 5) {
    errors.volatility = 'Volatility cannot exceed 500%';
  }

  // Option type validation
  if (!params.option_type || !['call', 'put'].includes(params.option_type)) {
    errors.option_type = 'Option type must be either "call" or "put"';
  }

  // Dividend yield validation (optional)
  if (params.dividend_yield !== undefined && params.dividend_yield !== null) {
    if (params.dividend_yield < 0 || params.dividend_yield > 1) {
      errors.dividend_yield = 'Dividend yield must be between 0% and 100%';
    }
  }

  return {
    isValid: Object.keys(errors).length === 0,
    errors
  };
};

/**
 * Validate swaption pricing parameters
 */
export const validateSwaptionParams = (params) => {
  const errors = {};

  // Swap rate validation
  if (!params.swap_rate || params.swap_rate <= 0) {
    errors.swap_rate = 'Swap rate must be positive';
  } else if (params.swap_rate > 0.2) {
    errors.swap_rate = 'Swap rate cannot exceed 20%';
  }

  // Strike rate validation
  if (!params.strike_rate || params.strike_rate <= 0) {
    errors.strike_rate = 'Strike rate must be positive';
  } else if (params.strike_rate > 0.2) {
    errors.strike_rate = 'Strike rate cannot exceed 20%';
  }

  // Option tenor validation
  if (!params.option_tenor || params.option_tenor <= 0) {
    errors.option_tenor = 'Option tenor must be positive';
  } else if (params.option_tenor > 10) {
    errors.option_tenor = 'Option tenor cannot exceed 10 years';
  }

  // Swap tenor validation
  if (!params.swap_tenor || params.swap_tenor < 0.5) {
    errors.swap_tenor = 'Swap tenor must be at least 0.5 years';
  } else if (params.swap_tenor > 50) {
    errors.swap_tenor = 'Swap tenor cannot exceed 50 years';
  }

  // Volatility validation
  if (!params.volatility || params.volatility <= 0) {
    errors.volatility = 'Volatility must be positive';
  } else if (params.volatility > 2) {
    errors.volatility = 'Volatility cannot exceed 200%';
  }

  return {
    isValid: Object.keys(errors).length === 0,
    errors
  };
};

/**
 * Validate batch pricing parameters
 */
export const validateBatchParams = (instruments) => {
  const errors = {};

  if (!Array.isArray(instruments)) {
    errors.instruments = 'Instruments must be an array';
    return { isValid: false, errors };
  }

  if (instruments.length === 0) {
    errors.instruments = 'At least one instrument must be provided';
    return { isValid: false, errors };
  }

  if (instruments.length > 50) {
    errors.instruments = 'Cannot process more than 50 instruments at once';
    return { isValid: false, errors };
  }

  // Validate each instrument
  instruments.forEach((instrument, index) => {
    if (!instrument.type || !['option', 'swaption'].includes(instrument.type)) {
      errors[`instrument_${index}_type`] = `Instrument ${index + 1}: type must be "option" or "swaption"`;
    }

    if (!instrument.parameters) {
      errors[`instrument_${index}_parameters`] = `Instrument ${index + 1}: parameters are required`;
    } else {
      let paramValidation;
      if (instrument.type === 'option') {
        paramValidation = validateOptionParams(instrument.parameters);
      } else if (instrument.type === 'swaption') {
        paramValidation = validateSwaptionParams(instrument.parameters);
      }

      if (!paramValidation.isValid) {
        Object.entries(paramValidation.errors).forEach(([field, message]) => {
          errors[`instrument_${index}_${field}`] = `Instrument ${index + 1}: ${message}`;
        });
      }
    }
  });

  return {
    isValid: Object.keys(errors).length === 0,
    errors
  };
};

/**
 * Validate risk metrics parameters
 */
export const validateRiskParams = (params) => {
  const errors = {};

  // Returns validation
  if (!params.returns || !Array.isArray(params.returns)) {
    errors.returns = 'Returns must be an array of numbers';
  } else if (params.returns.length < 30) {
    errors.returns = 'At least 30 return observations are required';
  }

  // Confidence level validation
  if (params.confidence_level !== undefined) {
    if (params.confidence_level <= 0 || params.confidence_level >= 1) {
      errors.confidence_level = 'Confidence level must be between 0 and 1';
    }
  }

  // Time horizon validation
  if (params.time_horizon !== undefined) {
    if (params.time_horizon < 1 || params.time_horizon > 252) {
      errors.time_horizon = 'Time horizon must be between 1 and 252 days';
    }
  }

  return {
    isValid: Object.keys(errors).length === 0,
    errors
  };
};

/**
 * Validate portfolio analysis parameters
 */
export const validatePortfolioParams = (params) => {
  const errors = {};

  // Weights validation
  if (!params.weights || !Array.isArray(params.weights)) {
    errors.weights = 'Portfolio weights must be an array';
  } else {
    const sum = params.weights.reduce((acc, w) => acc + parseFloat(w || 0), 0);
    if (Math.abs(sum - 1.0) > 0.01) {
      errors.weights = 'Portfolio weights must sum to 1.0';
    }
  }

  // Returns validation
  if (!params.returns || !Array.isArray(params.returns)) {
    errors.returns = 'Asset returns must be an array';
  }

  // Portfolio value validation
  if (params.portfolio_value !== undefined && params.portfolio_value <= 0) {
    errors.portfolio_value = 'Portfolio value must be positive';
  }

  return {
    isValid: Object.keys(errors).length === 0,
    errors
  };
};

/**
 * Format validation errors for display
 */
export const formatValidationErrors = (errors) => {
  return Object.entries(errors).map(([field, message]) => ({
    field: field.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase()),
    message
  }));
};

/**
 * Check if a value is a valid number
 */
export const isValidNumber = (value, min = -Infinity, max = Infinity) => {
  const num = parseFloat(value);
  return !isNaN(num) && num >= min && num <= max;
};

/**
 * Check if a value is a valid percentage
 */
export const isValidPercentage = (value, min = 0, max = 100) => {
  const num = parseFloat(value);
  return !isNaN(num) && num >= min && num <= max;
};

/**
 * Sanitize numeric input
 */
export const sanitizeNumericInput = (value, decimals = 4) => {
  const num = parseFloat(value);
  if (isNaN(num)) return '';
  return num.toFixed(decimals);
};