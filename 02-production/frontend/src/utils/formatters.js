/**
 * Utility functions for formatting data in the frontend
 */

/**
 * Format currency values
 * @param {number} value - The numeric value to format
 * @param {string} currency - Currency code (default: 'USD')
 * @param {number} decimals - Number of decimal places (default: 2)
 * @returns {string} Formatted currency string
 */
export const formatCurrency = (value, currency = 'USD', decimals = 2) => {
  if (value === null || value === undefined || isNaN(value)) {
    return 'N/A';
  }

  try {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: currency,
      minimumFractionDigits: decimals,
      maximumFractionDigits: decimals,
    }).format(value);
  } catch (error) {
    console.warn('Currency formatting error:', error);
    return `${currency} ${value.toFixed(decimals)}`;
  }
};

/**
 * Format percentage values
 * @param {number} value - The numeric value to format (as decimal)
 * @param {number} decimals - Number of decimal places (default: 2)
 * @returns {string} Formatted percentage string
 */
export const formatPercentage = (value, decimals = 2) => {
  if (value === null || value === undefined || isNaN(value)) {
    return 'N/A';
  }

  try {
    return new Intl.NumberFormat('en-US', {
      style: 'percent',
      minimumFractionDigits: decimals,
      maximumFractionDigits: decimals,
    }).format(value);
  } catch (error) {
    console.warn('Percentage formatting error:', error);
    return `${(value * 100).toFixed(decimals)}%`;
  }
};

/**
 * Format large numbers with appropriate suffixes
 * @param {number} value - The numeric value to format
 * @param {number} decimals - Number of decimal places (default: 2)
 * @returns {string} Formatted number string
 */
export const formatNumber = (value, decimals = 2) => {
  if (value === null || value === undefined || isNaN(value)) {
    return 'N/A';
  }

  const absValue = Math.abs(value);
  let suffix = '';
  let formattedValue = value;

  if (absValue >= 1e12) {
    suffix = 'T';
    formattedValue = value / 1e12;
  } else if (absValue >= 1e9) {
    suffix = 'B';
    formattedValue = value / 1e9;
  } else if (absValue >= 1e6) {
    suffix = 'M';
    formattedValue = value / 1e6;
  } else if (absValue >= 1e3) {
    suffix = 'K';
    formattedValue = value / 1e3;
  }

  return `${formattedValue.toFixed(decimals)}${suffix}`;
};

/**
 * Format dates consistently
 * @param {Date|string|number} date - Date to format
 * @param {string} format - Format type ('short', 'medium', 'long', 'iso')
 * @returns {string} Formatted date string
 */
export const formatDate = (date, format = 'medium') => {
  if (!date) return 'N/A';

  try {
    const dateObj = new Date(date);

    if (isNaN(dateObj.getTime())) {
      return 'Invalid Date';
    }

    switch (format) {
      case 'short':
        return dateObj.toLocaleDateString('en-US', {
          month: 'short',
          day: 'numeric',
          year: 'numeric'
        });
      case 'medium':
        return dateObj.toLocaleDateString('en-US', {
          month: 'short',
          day: 'numeric',
          year: 'numeric',
          hour: '2-digit',
          minute: '2-digit'
        });
      case 'long':
        return dateObj.toLocaleDateString('en-US', {
          weekday: 'long',
          year: 'numeric',
          month: 'long',
          day: 'numeric',
          hour: '2-digit',
          minute: '2-digit'
        });
      case 'iso':
        return dateObj.toISOString();
      default:
        return dateObj.toLocaleString();
    }
  } catch (error) {
    console.warn('Date formatting error:', error);
    return 'Invalid Date';
  }
};

/**
 * Format time duration
 * @param {number} seconds - Duration in seconds
 * @returns {string} Formatted duration string
 */
export const formatDuration = (seconds) => {
  if (seconds === null || seconds === undefined || isNaN(seconds)) {
    return 'N/A';
  }

  if (seconds < 1) {
    return `${(seconds * 1000).toFixed(0)}ms`;
  } else if (seconds < 60) {
    return `${seconds.toFixed(2)}s`;
  } else if (seconds < 3600) {
    const minutes = Math.floor(seconds / 60);
    const remainingSeconds = seconds % 60;
    return `${minutes}m ${remainingSeconds.toFixed(0)}s`;
  } else {
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    return `${hours}h ${minutes}m`;
  }
};

/**
 * Format volatility values
 * @param {number} volatility - Volatility as decimal (e.g., 0.20 for 20%)
 * @param {boolean} asPercentage - Whether to show as percentage (default: true)
 * @returns {string} Formatted volatility string
 */
export const formatVolatility = (volatility, asPercentage = true) => {
  if (volatility === null || volatility === undefined || isNaN(volatility)) {
    return 'N/A';
  }

  const value = asPercentage ? volatility * 100 : volatility;
  const suffix = asPercentage ? '%' : '';
  return `${value.toFixed(2)}${suffix}`;
};

/**
 * Format option Greeks
 * @param {number} greek - Greek value
 * @param {string} greekType - Type of Greek ('delta', 'gamma', 'theta', 'vega', 'rho')
 * @returns {string} Formatted Greek string
 */
export const formatGreek = (greek, greekType) => {
  if (greek === null || greek === undefined || isNaN(greek)) {
    return 'N/A';
  }

  let decimals = 4;
  let prefix = '';

  switch (greekType.toLowerCase()) {
    case 'delta':
      decimals = 4;
      break;
    case 'gamma':
      decimals = 6;
      break;
    case 'theta':
      decimals = 4;
      prefix = greek < 0 ? '' : '+';
      break;
    case 'vega':
      decimals = 4;
      break;
    case 'rho':
      decimals = 4;
      break;
    default:
      decimals = 4;
  }

  return `${prefix}${greek.toFixed(decimals)}`;
};

/**
 * Format confidence intervals
 * @param {number} lower - Lower bound
 * @param {number} upper - Upper bound
 * @param {string} currency - Currency for monetary values
 * @returns {string} Formatted confidence interval string
 */
export const formatConfidenceInterval = (lower, upper, currency = 'USD') => {
  if (lower === null || lower === undefined || isNaN(lower) ||
      upper === null || upper === undefined || isNaN(upper)) {
    return 'N/A';
  }

  const formattedLower = formatCurrency(lower, currency, 2);
  const formattedUpper = formatCurrency(upper, currency, 2);

  return `${formattedLower} - ${formattedUpper}`;
};

/**
 * Truncate text with ellipsis
 * @param {string} text - Text to truncate
 * @param {number} maxLength - Maximum length
 * @returns {string} Truncated text
 */
export const truncateText = (text, maxLength = 50) => {
  if (!text || text.length <= maxLength) {
    return text;
  }

  return `${text.substring(0, maxLength - 3)}...`;
};

/**
 * Format file sizes
 * @param {number} bytes - Size in bytes
 * @returns {string} Formatted file size
 */
export const formatFileSize = (bytes) => {
  if (bytes === null || bytes === undefined || isNaN(bytes)) {
    return 'N/A';
  }

  const units = ['B', 'KB', 'MB', 'GB', 'TB'];
  let size = bytes;
  let unitIndex = 0;

  while (size >= 1024 && unitIndex < units.length - 1) {
    size /= 1024;
    unitIndex++;
  }

  return `${size.toFixed(1)} ${units[unitIndex]}`;
};