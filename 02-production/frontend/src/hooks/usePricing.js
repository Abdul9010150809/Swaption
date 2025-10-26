import { createContext, useContext, useState, useCallback } from 'react';
import axios from 'axios';

const PricingContext = createContext();

export const usePricing = () => {
  const context = useContext(PricingContext);
  if (!context) {
    throw new Error('usePricing must be used within a PricingProvider');
  }
  return context;
};

export const PricingProvider = ({ children }) => {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [result, setResult] = useState(null);
  const [history, setHistory] = useState([]);

  const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:3000';

  const calculatePrice = useCallback(async (instrumentType, params) => {
    setLoading(true);
    setError(null);

    try {
      const endpoint = instrumentType === 'option' ? '/api/pricing/options' : '/api/pricing/swaptions';
      const response = await axios.post(`${API_BASE_URL}${endpoint}`, params, {
        headers: {
          'Content-Type': 'application/json',
        },
        timeout: 30000, // 30 second timeout
      });

      const pricingResult = response.data;
      setResult(pricingResult);

      // Add to history
      const historyItem = {
        id: Date.now(),
        timestamp: new Date().toISOString(),
        instrumentType,
        params,
        result: pricingResult,
      };
      setHistory(prev => [historyItem, ...prev.slice(0, 9)]); // Keep last 10 items

      return pricingResult;
    } catch (err) {
      const errorMessage = err.response?.data?.message ||
                          err.response?.data?.error ||
                          err.message ||
                          'An error occurred during pricing calculation';

      setError(errorMessage);
      throw err;
    } finally {
      setLoading(false);
    }
  }, [API_BASE_URL]);

  const calculateBatchPrices = useCallback(async (instruments) => {
    setLoading(true);
    setError(null);

    try {
      const response = await axios.post(`${API_BASE_URL}/api/pricing/batch`, {
        instruments
      }, {
        headers: {
          'Content-Type': 'application/json',
        },
        timeout: 60000, // 60 second timeout for batch
      });

      const batchResult = response.data;
      setResult(batchResult);

      // Add to history
      const historyItem = {
        id: Date.now(),
        timestamp: new Date().toISOString(),
        instrumentType: 'batch',
        params: { instruments },
        result: batchResult,
      };
      setHistory(prev => [historyItem, ...prev.slice(0, 9)]);

      return batchResult;
    } catch (err) {
      const errorMessage = err.response?.data?.message ||
                          err.response?.data?.error ||
                          err.message ||
                          'An error occurred during batch pricing calculation';

      setError(errorMessage);
      throw err;
    } finally {
      setLoading(false);
    }
  }, [API_BASE_URL]);

  const clearResult = useCallback(() => {
    setResult(null);
    setError(null);
  }, []);

  const clearHistory = useCallback(() => {
    setHistory([]);
  }, []);

  const retryCalculation = useCallback(async () => {
    if (history.length > 0) {
      const lastCalculation = history[0];
      return calculatePrice(lastCalculation.instrumentType, lastCalculation.params);
    }
  }, [history, calculatePrice]);

  const value = {
    // State
    loading,
    error,
    result,
    history,

    // Actions
    calculatePrice,
    calculateBatchPrices,
    clearResult,
    clearHistory,
    retryCalculation,
  };

  return (
    <PricingContext.Provider value={value}>
      {children}
    </PricingContext.Provider>
  );
};