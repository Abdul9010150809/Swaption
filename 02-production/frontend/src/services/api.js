/**
 * API service for communicating with the backend
 */

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:3000';

class ApiService {
  constructor() {
    this.baseURL = API_BASE_URL;
    this.defaultHeaders = {
      'Content-Type': 'application/json',
    };
  }

  /**
   * Make an HTTP request
   */
  async request(endpoint, options = {}) {
    const url = `${this.baseURL}${endpoint}`;
    const config = {
      headers: { ...this.defaultHeaders, ...options.headers },
      ...options,
    };

    // Add authorization header if token exists
    const token = localStorage.getItem('authToken');
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }

    try {
      const response = await fetch(url, config);

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.message || `HTTP ${response.status}: ${response.statusText}`);
      }

      return await response.json();
    } catch (error) {
      console.error(`API request failed: ${endpoint}`, error);
      throw error;
    }
  }

  /**
   * GET request
   */
  async get(endpoint, params = {}) {
    const url = new URL(`${this.baseURL}${endpoint}`);
    Object.keys(params).forEach(key => {
      if (params[key] !== null && params[key] !== undefined) {
        url.searchParams.append(key, params[key]);
      }
    });

    return this.request(url.pathname + url.search, {
      method: 'GET',
    });
  }

  /**
   * POST request
   */
  async post(endpoint, data = {}) {
    return this.request(endpoint, {
      method: 'POST',
      body: JSON.stringify(data),
    });
  }

  /**
   * PUT request
   */
  async put(endpoint, data = {}) {
    return this.request(endpoint, {
      method: 'PUT',
      body: JSON.stringify(data),
    });
  }

  /**
   * DELETE request
   */
  async delete(endpoint) {
    return this.request(endpoint, {
      method: 'DELETE',
    });
  }
}

// Create singleton instance
const apiService = new ApiService();

// Pricing API methods
export const pricingAPI = {
  /**
   * Price European options
   */
  async priceOption(params) {
    return apiService.post('/api/pricing/options', params);
  },

  /**
   * Price European swaptions
   */
  async priceSwaption(params) {
    return apiService.post('/api/pricing/swaptions', params);
  },

  /**
   * Batch pricing request
   */
  async priceBatch(instruments) {
    return apiService.post('/api/pricing/batch', { instruments });
  },
};

// Health check API methods
export const healthAPI = {
  /**
   * Basic health check
   */
  async getHealth() {
    return apiService.get('/health');
  },

  /**
   * Detailed health check
   */
  async getDetailedHealth() {
    return apiService.get('/health/detailed');
  },

  /**
   * Get metrics
   */
  async getMetrics() {
    return apiService.get('/health/metrics');
  },
};

// Market data API methods (if implemented)
export const marketDataAPI = {
  /**
   * Get current market data
   */
  async getMarketData() {
    return apiService.get('/api/market-data/current');
  },

  /**
   * Get yield curve
   */
  async getYieldCurve() {
    return apiService.get('/api/market-data/yield-curve');
  },

  /**
   * Get volatility surface
   */
  async getVolatilitySurface() {
    return apiService.get('/api/market-data/volatility-surface');
  },
};

// Authentication API methods (if implemented)
export const authAPI = {
  /**
   * Login
   */
  async login(credentials) {
    const response = await apiService.post('/auth/login', credentials);
    if (response.token) {
      localStorage.setItem('authToken', response.token);
    }
    return response;
  },

  /**
   * Logout
   */
  async logout() {
    localStorage.removeItem('authToken');
    return apiService.post('/auth/logout');
  },

  /**
   * Get current user
   */
  async getCurrentUser() {
    return apiService.get('/auth/me');
  },
};

// Risk metrics API methods (if implemented)
export const riskAPI = {
  /**
   * Calculate VaR
   */
  async calculateVaR(params) {
    return apiService.post('/api/risk/var', params);
  },

  /**
   * Calculate risk metrics
   */
  async calculateRiskMetrics(params) {
    return apiService.post('/api/risk/metrics', params);
  },

  /**
   * Stress testing
   */
  async stressTest(params) {
    return apiService.post('/api/risk/stress-test', params);
  },
};

export default apiService;