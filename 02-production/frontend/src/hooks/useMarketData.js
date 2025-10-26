import { createContext, useContext, useState, useEffect, useCallback } from 'react';
import axios from 'axios';

const MarketDataContext = createContext();

export const useMarketData = () => {
  const context = useContext(MarketDataContext);
  if (!context) {
    throw new Error('useMarketData must be used within a MarketDataProvider');
  }
  return context;
};

export const MarketDataProvider = ({ children }) => {
  const [marketData, setMarketData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [lastUpdated, setLastUpdated] = useState(null);

  const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:3000';

  const fetchMarketData = useCallback(async () => {
    setLoading(true);
    setError(null);

    try {
      // In a real application, this would fetch from market data APIs
      // For now, we'll simulate market data
      const simulatedData = {
        spotPrice: 100 + (Math.random() - 0.5) * 10, // Around 100
        riskFreeRate: 0.03 + (Math.random() - 0.5) * 0.01, // Around 3%
        volatility: 0.20 + (Math.random() - 0.5) * 0.05, // Around 20%
        timestamp: new Date().toISOString()
      };

      setMarketData(simulatedData);
      setLastUpdated(new Date());

      return simulatedData;
    } catch (err) {
      const errorMessage = err.message || 'Failed to fetch market data';
      setError(errorMessage);
      throw err;
    } finally {
      setLoading(false);
    }
  }, []);

  const fetchYieldCurve = useCallback(async () => {
    try {
      // Simulate yield curve data
      const tenors = [0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30];
      const rates = tenors.map(tenor => {
        const baseRate = 0.03;
        const termPremium = 0.001 * Math.log(tenor + 1);
        const noise = (Math.random() - 0.5) * 0.002;
        return Math.max(0, baseRate + termPremium + noise);
      });

      return {
        tenors,
        rates,
        timestamp: new Date().toISOString()
      };
    } catch (err) {
      console.error('Failed to fetch yield curve:', err);
      throw err;
    }
  }, []);

  const fetchVolatilitySurface = useCallback(async () => {
    try {
      // Simulate volatility surface data
      const expiries = [0.25, 0.5, 1, 2, 5];
      const strikes = [0.8, 0.9, 1.0, 1.1, 1.2];
      const spot = marketData?.spotPrice || 100;

      const surface = expiries.map(expiry =>
        strikes.map(strike => {
          const moneyness = Math.log(strike);
          const timeFactor = Math.sqrt(expiry);
          const baseVol = 0.20;
          const skew = 0.05 * moneyness;
          const termStructure = 0.02 * Math.exp(-expiry / 2);
          const noise = (Math.random() - 0.5) * 0.02;

          return Math.max(0.05, baseVol - skew + termStructure + noise);
        })
      );

      return {
        expiries,
        strikes: strikes.map(s => s * spot),
        surface,
        timestamp: new Date().toISOString()
      };
    } catch (err) {
      console.error('Failed to fetch volatility surface:', err);
      throw err;
    }
  }, [marketData]);

  // Auto-refresh market data every 5 minutes
  useEffect(() => {
    fetchMarketData();

    const interval = setInterval(() => {
      fetchMarketData();
    }, 5 * 60 * 1000); // 5 minutes

    return () => clearInterval(interval);
  }, [fetchMarketData]);

  const refreshMarketData = useCallback(async () => {
    return fetchMarketData();
  }, [fetchMarketData]);

  const value = {
    // State
    marketData,
    loading,
    error,
    lastUpdated,

    // Actions
    refreshMarketData,
    fetchYieldCurve,
    fetchVolatilitySurface,
  };

  return (
    <MarketDataContext.Provider value={value}>
      {children}
    </MarketDataContext.Provider>
  );
};