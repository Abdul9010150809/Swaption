import React, { useState, useEffect } from 'react';
import {
  Box, Paper, Typography, Grid, TextField, Button, FormControl,
  InputLabel, Select, MenuItem, Alert, CircularProgress,
  Accordion, AccordionSummary, AccordionDetails, Chip
} from '@mui/material';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import { usePricing } from '../../hooks/usePricing';
import { useMarketData } from '../../hooks/useMarketData';

const PricingForm = () => {
  const { calculatePrice, loading, error, result } = usePricing();
  const { marketData, loading: marketLoading } = useMarketData();

  const [formData, setFormData] = useState({
    // Option parameters
    spot_price: '',
    strike_price: '',
    time_to_expiry: '',
    risk_free_rate: '',
    volatility: '',
    option_type: 'call',
    dividend_yield: '0',

    // Swaption parameters
    swap_rate: '',
    strike_rate: '',
    option_tenor: '',
    swap_tenor: '',
    model: 'black_scholes',

    // Instrument type
    instrument_type: 'option'
  });

  const [validationErrors, setValidationErrors] = useState({});

  useEffect(() => {
    // Pre-populate with market data if available
    if (marketData) {
      setFormData(prev => ({
        ...prev,
        risk_free_rate: marketData.riskFreeRate || prev.risk_free_rate,
        volatility: marketData.volatility || prev.volatility,
        spot_price: marketData.spotPrice || prev.spot_price
      }));
    }
  }, [marketData]);

  const handleInputChange = (field, value) => {
    setFormData(prev => ({
      ...prev,
      [field]: value
    }));

    // Clear validation error for this field
    if (validationErrors[field]) {
      setValidationErrors(prev => ({
        ...prev,
        [field]: null
      }));
    }
  };

  const validateForm = () => {
    const errors = {};

    if (formData.instrument_type === 'option') {
      if (!formData.spot_price || parseFloat(formData.spot_price) <= 0) {
        errors.spot_price = 'Spot price must be positive';
      }
      if (!formData.strike_price || parseFloat(formData.strike_price) <= 0) {
        errors.strike_price = 'Strike price must be positive';
      }
      if (!formData.time_to_expiry || parseFloat(formData.time_to_expiry) < 0) {
        errors.time_to_expiry = 'Time to expiry must be non-negative';
      }
      if (!formData.risk_free_rate || parseFloat(formData.risk_free_rate) < -0.1 || parseFloat(formData.risk_free_rate) > 0.2) {
        errors.risk_free_rate = 'Risk-free rate must be between -10% and 20%';
      }
      if (!formData.volatility || parseFloat(formData.volatility) <= 0 || parseFloat(formData.volatility) > 5) {
        errors.volatility = 'Volatility must be between 0% and 500%';
      }
    } else if (formData.instrument_type === 'swaption') {
      if (!formData.swap_rate || parseFloat(formData.swap_rate) <= 0) {
        errors.swap_rate = 'Swap rate must be positive';
      }
      if (!formData.strike_rate || parseFloat(formData.strike_rate) <= 0) {
        errors.strike_rate = 'Strike rate must be positive';
      }
      if (!formData.option_tenor || parseFloat(formData.option_tenor) <= 0) {
        errors.option_tenor = 'Option tenor must be positive';
      }
      if (!formData.swap_tenor || parseFloat(formData.swap_tenor) < 0.5) {
        errors.swap_tenor = 'Swap tenor must be at least 0.5 years';
      }
      if (!formData.volatility || parseFloat(formData.volatility) <= 0 || parseFloat(formData.volatility) > 2) {
        errors.volatility = 'Volatility must be between 0% and 200%';
      }
    }

    setValidationErrors(errors);
    return Object.keys(errors).length === 0;
  };

  const handleSubmit = async (e) => {
    e.preventDefault();

    if (!validateForm()) {
      return;
    }

    try {
      const pricingData = formData.instrument_type === 'option' ? {
        spot_price: parseFloat(formData.spot_price),
        strike_price: parseFloat(formData.strike_price),
        time_to_expiry: parseFloat(formData.time_to_expiry),
        risk_free_rate: parseFloat(formData.risk_free_rate),
        volatility: parseFloat(formData.volatility),
        option_type: formData.option_type,
        dividend_yield: parseFloat(formData.dividend_yield),
        model: formData.model
      } : {
        swap_rate: parseFloat(formData.swap_rate),
        strike_rate: parseFloat(formData.strike_rate),
        option_tenor: parseFloat(formData.option_tenor),
        swap_tenor: parseFloat(formData.swap_tenor),
        volatility: parseFloat(formData.volatility),
        model: formData.model
      };

      await calculatePrice(formData.instrument_type, pricingData);
    } catch (err) {
      console.error('Pricing calculation failed:', err);
    }
  };

  const resetForm = () => {
    setFormData({
      spot_price: '',
      strike_price: '',
      time_to_expiry: '',
      risk_free_rate: '',
      volatility: '',
      option_type: 'call',
      dividend_yield: '0',
      swap_rate: '',
      strike_rate: '',
      option_tenor: '',
      swap_tenor: '',
      model: 'black_scholes',
      instrument_type: 'option'
    });
    setValidationErrors({});
  };

  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        Derivative Pricing Calculator
      </Typography>

      <Grid container spacing={3}>
        <Grid item xs={12} md={8}>
          <Paper elevation={3} sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              Pricing Parameters
            </Typography>

            <Box component="form" onSubmit={handleSubmit} sx={{ mt: 2 }}>
              {/* Instrument Type Selection */}
              <FormControl fullWidth margin="normal">
                <InputLabel>Instrument Type</InputLabel>
                <Select
                  value={formData.instrument_type}
                  onChange={(e) => handleInputChange('instrument_type', e.target.value)}
                  label="Instrument Type"
                >
                  <MenuItem value="option">European Option</MenuItem>
                  <MenuItem value="swaption">European Swaption</MenuItem>
                </Select>
              </FormControl>

              {formData.instrument_type === 'option' ? (
                /* Option Parameters */
                <Box>
                  <Grid container spacing={2}>
                    <Grid item xs={12} sm={6}>
                      <TextField
                        fullWidth
                        label="Spot Price"
                        type="number"
                        value={formData.spot_price}
                        onChange={(e) => handleInputChange('spot_price', e.target.value)}
                        error={!!validationErrors.spot_price}
                        helperText={validationErrors.spot_price}
                        InputProps={{ inputProps: { min: 0, step: 0.01 } }}
                      />
                    </Grid>
                    <Grid item xs={12} sm={6}>
                      <TextField
                        fullWidth
                        label="Strike Price"
                        type="number"
                        value={formData.strike_price}
                        onChange={(e) => handleInputChange('strike_price', e.target.value)}
                        error={!!validationErrors.strike_price}
                        helperText={validationErrors.strike_price}
                        InputProps={{ inputProps: { min: 0, step: 0.01 } }}
                      />
                    </Grid>
                    <Grid item xs={12} sm={6}>
                      <TextField
                        fullWidth
                        label="Time to Expiry (years)"
                        type="number"
                        value={formData.time_to_expiry}
                        onChange={(e) => handleInputChange('time_to_expiry', e.target.value)}
                        error={!!validationErrors.time_to_expiry}
                        helperText={validationErrors.time_to_expiry}
                        InputProps={{ inputProps: { min: 0, step: 0.01 } }}
                      />
                    </Grid>
                    <Grid item xs={12} sm={6}>
                      <TextField
                        fullWidth
                        label="Risk-Free Rate (%)"
                        type="number"
                        value={formData.risk_free_rate}
                        onChange={(e) => handleInputChange('risk_free_rate', e.target.value)}
                        error={!!validationErrors.risk_free_rate}
                        helperText={validationErrors.risk_free_rate}
                        InputProps={{ inputProps: { min: -10, max: 20, step: 0.01 } }}
                      />
                    </Grid>
                    <Grid item xs={12} sm={6}>
                      <TextField
                        fullWidth
                        label="Volatility (%)"
                        type="number"
                        value={formData.volatility}
                        onChange={(e) => handleInputChange('volatility', e.target.value)}
                        error={!!validationErrors.volatility}
                        helperText={validationErrors.volatility}
                        InputProps={{ inputProps: { min: 0, max: 500, step: 0.01 } }}
                      />
                    </Grid>
                    <Grid item xs={12} sm={6}>
                      <FormControl fullWidth>
                        <InputLabel>Option Type</InputLabel>
                        <Select
                          value={formData.option_type}
                          onChange={(e) => handleInputChange('option_type', e.target.value)}
                          label="Option Type"
                        >
                          <MenuItem value="call">Call</MenuItem>
                          <MenuItem value="put">Put</MenuItem>
                        </Select>
                      </FormControl>
                    </Grid>
                  </Grid>

                  <Accordion sx={{ mt: 2 }}>
                    <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                      <Typography>Advanced Options</Typography>
                    </AccordionSummary>
                    <AccordionDetails>
                      <Grid container spacing={2}>
                        <Grid item xs={12} sm={6}>
                          <TextField
                            fullWidth
                            label="Dividend Yield (%)"
                            type="number"
                            value={formData.dividend_yield}
                            onChange={(e) => handleInputChange('dividend_yield', e.target.value)}
                            InputProps={{ inputProps: { min: 0, step: 0.01 } }}
                          />
                        </Grid>
                        <Grid item xs={12} sm={6}>
                          <FormControl fullWidth>
                            <InputLabel>Pricing Model</InputLabel>
                            <Select
                              value={formData.model}
                              onChange={(e) => handleInputChange('model', e.target.value)}
                              label="Pricing Model"
                            >
                              <MenuItem value="black_scholes">Black-Scholes</MenuItem>
                              <MenuItem value="monte_carlo">Monte Carlo</MenuItem>
                              <MenuItem value="ml">Machine Learning</MenuItem>
                            </Select>
                          </FormControl>
                        </Grid>
                      </Grid>
                    </AccordionDetails>
                  </Accordion>
                </Box>
              ) : (
                /* Swaption Parameters */
                <Box>
                  <Grid container spacing={2}>
                    <Grid item xs={12} sm={6}>
                      <TextField
                        fullWidth
                        label="Swap Rate (%)"
                        type="number"
                        value={formData.swap_rate}
                        onChange={(e) => handleInputChange('swap_rate', e.target.value)}
                        error={!!validationErrors.swap_rate}
                        helperText={validationErrors.swap_rate}
                        InputProps={{ inputProps: { min: 0, step: 0.01 } }}
                      />
                    </Grid>
                    <Grid item xs={12} sm={6}>
                      <TextField
                        fullWidth
                        label="Strike Rate (%)"
                        type="number"
                        value={formData.strike_rate}
                        onChange={(e) => handleInputChange('strike_rate', e.target.value)}
                        error={!!validationErrors.strike_rate}
                        helperText={validationErrors.strike_rate}
                        InputProps={{ inputProps: { min: 0, step: 0.01 } }}
                      />
                    </Grid>
                    <Grid item xs={12} sm={6}>
                      <TextField
                        fullWidth
                        label="Option Tenor (years)"
                        type="number"
                        value={formData.option_tenor}
                        onChange={(e) => handleInputChange('option_tenor', e.target.value)}
                        error={!!validationErrors.option_tenor}
                        helperText={validationErrors.option_tenor}
                        InputProps={{ inputProps: { min: 0, step: 0.01 } }}
                      />
                    </Grid>
                    <Grid item xs={12} sm={6}>
                      <TextField
                        fullWidth
                        label="Swap Tenor (years)"
                        type="number"
                        value={formData.swap_tenor}
                        onChange={(e) => handleInputChange('swap_tenor', e.target.value)}
                        error={!!validationErrors.swap_tenor}
                        helperText={validationErrors.swap_tenor}
                        InputProps={{ inputProps: { min: 0.5, step: 0.5 } }}
                      />
                    </Grid>
                    <Grid item xs={12} sm={6}>
                      <TextField
                        fullWidth
                        label="Volatility (%)"
                        type="number"
                        value={formData.volatility}
                        onChange={(e) => handleInputChange('volatility', e.target.value)}
                        error={!!validationErrors.volatility}
                        helperText={validationErrors.volatility}
                        InputProps={{ inputProps: { min: 0, max: 200, step: 0.01 } }}
                      />
                    </Grid>
                    <Grid item xs={12} sm={6}>
                      <FormControl fullWidth>
                        <InputLabel>Pricing Model</InputLabel>
                        <Select
                          value={formData.model}
                          onChange={(e) => handleInputChange('model', e.target.value)}
                          label="Pricing Model"
                        >
                          <MenuItem value="black">Black Model</MenuItem>
                          <MenuItem value="monte_carlo">Monte Carlo</MenuItem>
                          <MenuItem value="ml">Machine Learning</MenuItem>
                          <MenuItem value="quantum_monte_carlo">Quantum Monte Carlo</MenuItem>
                          <MenuItem value="quantum_amplitude_estimation">Quantum Amplitude Estimation</MenuItem>
                          <MenuItem value="quantum_hybrid">Quantum Hybrid</MenuItem>
                        </Select>
                      </FormControl>
                    </Grid>
                  </Grid>
                </Box>
              )}

              <Box sx={{ mt: 3, display: 'flex', gap: 2 }}>
                <Button
                  type="submit"
                  variant="contained"
                  color="primary"
                  size="large"
                  disabled={loading}
                  startIcon={loading ? <CircularProgress size={20} /> : null}
                >
                  {loading ? 'Calculating...' : 'Calculate Price'}
                </Button>
                <Button
                  type="button"
                  variant="outlined"
                  onClick={resetForm}
                  disabled={loading}
                >
                  Reset
                </Button>
              </Box>
            </Box>
          </Paper>
        </Grid>

        <Grid item xs={12} md={4}>
          {/* Results Panel */}
          {error && (
            <Alert severity="error" sx={{ mb: 2 }}>
              {error}
            </Alert>
          )}

          {result && (
            <Paper elevation={3} sx={{ p: 3, mb: 2 }}>
              <Typography variant="h6" gutterBottom>
                Pricing Results
              </Typography>

              <Box sx={{ mb: 2 }}>
                <Typography variant="body2" color="text.secondary">
                  Price
                </Typography>
                <Typography variant="h5" color="primary">
                  ${result.price?.toFixed(4)}
                </Typography>
              </Box>

              {result.greeks && (
                <Box sx={{ mb: 2 }}>
                  <Typography variant="body2" color="text.secondary" gutterBottom>
                    Greeks
                  </Typography>
                  <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
                    {Object.entries(result.greeks).map(([greek, value]) => (
                      <Chip
                        key={greek}
                        label={`${greek.toUpperCase()}: ${value?.toFixed(4)}`}
                        size="small"
                        variant="outlined"
                      />
                    ))}
                  </Box>
                </Box>
              )}

              <Box sx={{ mb: 2 }}>
                <Typography variant="body2" color="text.secondary">
                  Model
                </Typography>
                <Chip label={result.model?.toUpperCase()} color="secondary" size="small" />
              </Box>

              {result.timestamp && (
                <Typography variant="caption" color="text.secondary">
                  Calculated at: {new Date(result.timestamp).toLocaleString()}
                </Typography>
              )}
            </Paper>
          )}

          {/* Market Data Panel */}
          {marketData && (
            <Paper elevation={3} sx={{ p: 3 }}>
              <Typography variant="h6" gutterBottom>
                Market Data
              </Typography>

              {marketLoading ? (
                <CircularProgress size={24} />
              ) : (
                <Box>
                  {marketData.spotPrice && (
                    <Box sx={{ mb: 1 }}>
                      <Typography variant="body2" color="text.secondary">
                        Spot Price
                      </Typography>
                      <Typography variant="body1">
                        ${marketData.spotPrice.toFixed(2)}
                      </Typography>
                    </Box>
                  )}

                  {marketData.riskFreeRate && (
                    <Box sx={{ mb: 1 }}>
                      <Typography variant="body2" color="text.secondary">
                        Risk-Free Rate
                      </Typography>
                      <Typography variant="body1">
                        {(marketData.riskFreeRate * 100).toFixed(2)}%
                      </Typography>
                    </Box>
                  )}

                  {marketData.volatility && (
                    <Box sx={{ mb: 1 }}>
                      <Typography variant="body2" color="text.secondary">
                        Implied Volatility
                      </Typography>
                      <Typography variant="body1">
                        {(marketData.volatility * 100).toFixed(2)}%
                      </Typography>
                    </Box>
                  )}
                </Box>
              )}
            </Paper>
          )}
        </Grid>
      </Grid>
    </Box>
  );
};

export default PricingForm;