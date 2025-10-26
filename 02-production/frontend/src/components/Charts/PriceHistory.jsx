import React, { useState, useEffect } from 'react';
import {
  Box, Paper, Typography, Grid, TextField, Button,
  FormControl, InputLabel, Select, MenuItem, Alert,
  CircularProgress, Table, TableBody, TableCell,
  TableContainer, TableHead, TableRow
} from '@mui/material';
import { usePricing } from '../../hooks/usePricing';

const PriceHistory = () => {
  const { history, loading } = usePricing();
  const [filteredHistory, setFilteredHistory] = useState([]);
  const [filters, setFilters] = useState({
    instrumentType: 'all',
    dateRange: 'all',
    minPrice: '',
    maxPrice: ''
  });

  useEffect(() => {
    applyFilters();
  }, [history, filters]);

  const applyFilters = () => {
    let filtered = [...history];

    // Filter by instrument type
    if (filters.instrumentType !== 'all') {
      filtered = filtered.filter(item => item.instrumentType === filters.instrumentType);
    }

    // Filter by date range
    if (filters.dateRange !== 'all') {
      const now = new Date();
      const days = parseInt(filters.dateRange);
      const cutoffDate = new Date(now.getTime() - days * 24 * 60 * 60 * 1000);

      filtered = filtered.filter(item => new Date(item.timestamp) >= cutoffDate);
    }

    // Filter by price range
    if (filters.minPrice) {
      const minPrice = parseFloat(filters.minPrice);
      filtered = filtered.filter(item =>
        item.result && item.result.price && item.result.price >= minPrice
      );
    }

    if (filters.maxPrice) {
      const maxPrice = parseFloat(filters.maxPrice);
      filtered = filtered.filter(item =>
        item.result && item.result.price && item.result.price <= maxPrice
      );
    }

    setFilteredHistory(filtered);
  };

  const handleFilterChange = (field, value) => {
    setFilters(prev => ({
      ...prev,
      [field]: value
    }));
  };

  const clearFilters = () => {
    setFilters({
      instrumentType: 'all',
      dateRange: 'all',
      minPrice: '',
      maxPrice: ''
    });
  };

  const formatCurrency = (value) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 4,
      maximumFractionDigits: 4
    }).format(value);
  };

  const formatDateTime = (timestamp) => {
    return new Date(timestamp).toLocaleString();
  };

  const getInstrumentTypeLabel = (type) => {
    switch (type) {
      case 'option': return 'European Option';
      case 'swaption': return 'European Swaption';
      case 'batch': return 'Batch Calculation';
      default: return type;
    }
  };

  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        Pricing History
      </Typography>

      <Grid container spacing={3}>
        {/* Filters */}
        <Grid item xs={12}>
          <Paper elevation={3} sx={{ p: 3, mb: 2 }}>
            <Typography variant="h6" gutterBottom>
              Filters
            </Typography>

            <Grid container spacing={2} alignItems="center">
              <Grid item xs={12} sm={6} md={3}>
                <FormControl fullWidth>
                  <InputLabel>Instrument Type</InputLabel>
                  <Select
                    value={filters.instrumentType}
                    onChange={(e) => handleFilterChange('instrumentType', e.target.value)}
                    label="Instrument Type"
                  >
                    <MenuItem value="all">All Types</MenuItem>
                    <MenuItem value="option">European Option</MenuItem>
                    <MenuItem value="swaption">European Swaption</MenuItem>
                    <MenuItem value="batch">Batch Calculation</MenuItem>
                  </Select>
                </FormControl>
              </Grid>

              <Grid item xs={12} sm={6} md={3}>
                <FormControl fullWidth>
                  <InputLabel>Date Range</InputLabel>
                  <Select
                    value={filters.dateRange}
                    onChange={(e) => handleFilterChange('dateRange', e.target.value)}
                    label="Date Range"
                  >
                    <MenuItem value="all">All Time</MenuItem>
                    <MenuItem value="1">Last 24 Hours</MenuItem>
                    <MenuItem value="7">Last 7 Days</MenuItem>
                    <MenuItem value="30">Last 30 Days</MenuItem>
                  </Select>
                </FormControl>
              </Grid>

              <Grid item xs={12} sm={6} md={2}>
                <TextField
                  fullWidth
                  label="Min Price"
                  type="number"
                  value={filters.minPrice}
                  onChange={(e) => handleFilterChange('minPrice', e.target.value)}
                  InputProps={{ inputProps: { min: 0, step: 0.01 } }}
                />
              </Grid>

              <Grid item xs={12} sm={6} md={2}>
                <TextField
                  fullWidth
                  label="Max Price"
                  type="number"
                  value={filters.maxPrice}
                  onChange={(e) => handleFilterChange('maxPrice', e.target.value)}
                  InputProps={{ inputProps: { min: 0, step: 0.01 } }}
                />
              </Grid>

              <Grid item xs={12} sm={6} md={2}>
                <Button
                  fullWidth
                  variant="outlined"
                  onClick={clearFilters}
                  sx={{ height: '56px' }}
                >
                  Clear Filters
                </Button>
              </Grid>
            </Grid>
          </Paper>
        </Grid>

        {/* History Table */}
        <Grid item xs={12}>
          <Paper elevation={3} sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              Calculation History ({filteredHistory.length} results)
            </Typography>

            {loading ? (
              <Box sx={{ display: 'flex', justifyContent: 'center', p: 3 }}>
                <CircularProgress />
              </Box>
            ) : filteredHistory.length === 0 ? (
              <Alert severity="info">
                No pricing calculations found matching the current filters.
              </Alert>
            ) : (
              <TableContainer>
                <Table>
                  <TableHead>
                    <TableRow>
                      <TableCell><strong>Timestamp</strong></TableCell>
                      <TableCell><strong>Instrument Type</strong></TableCell>
                      <TableCell><strong>Price</strong></TableCell>
                      <TableCell><strong>Model</strong></TableCell>
                      <TableCell><strong>Parameters</strong></TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {filteredHistory.map((item) => (
                      <TableRow key={item.id} hover>
                        <TableCell>
                          {formatDateTime(item.timestamp)}
                        </TableCell>
                        <TableCell>
                          {getInstrumentTypeLabel(item.instrumentType)}
                        </TableCell>
                        <TableCell>
                          {item.result?.price ? formatCurrency(item.result.price) : 'N/A'}
                        </TableCell>
                        <TableCell>
                          {item.result?.model?.toUpperCase() || 'N/A'}
                        </TableCell>
                        <TableCell>
                          <Box sx={{ maxWidth: 300 }}>
                            {item.instrumentType === 'option' && item.params && (
                              <Typography variant="body2" noWrap>
                                S: {item.params.spot_price}, K: {item.params.strike_price},
                                T: {item.params.time_to_expiry}Y, r: {(item.params.risk_free_rate * 100).toFixed(1)}%,
                                σ: {(item.params.volatility * 100).toFixed(1)}%
                              </Typography>
                            )}
                            {item.instrumentType === 'swaption' && item.params && (
                              <Typography variant="body2" noWrap>
                                Swap: {(item.params.swap_rate * 100).toFixed(2)}%,
                                Strike: {(item.params.strike_rate * 100).toFixed(2)}%,
                                Opt: {item.params.option_tenor}Y, Swap: {item.params.swap_tenor}Y
                              </Typography>
                            )}
                            {item.instrumentType === 'batch' && (
                              <Typography variant="body2">
                                {item.params?.instruments?.length || 0} instruments
                              </Typography>
                            )}
                          </Box>
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </TableContainer>
            )}
          </Paper>
        </Grid>

        {/* Summary Statistics */}
        {filteredHistory.length > 0 && (
          <Grid item xs={12}>
            <Paper elevation={3} sx={{ p: 3 }}>
              <Typography variant="h6" gutterBottom>
                Summary Statistics
              </Typography>

              <Grid container spacing={2}>
                <Grid item xs={12} sm={6} md={3}>
                  <Box sx={{ textAlign: 'center' }}>
                    <Typography variant="body2" color="text.secondary">
                      Total Calculations
                    </Typography>
                    <Typography variant="h4">
                      {filteredHistory.length}
                    </Typography>
                  </Box>
                </Grid>

                <Grid item xs={12} sm={6} md={3}>
                  <Box sx={{ textAlign: 'center' }}>
                    <Typography variant="body2" color="text.secondary">
                      Average Price
                    </Typography>
                    <Typography variant="h4">
                      {formatCurrency(
                        filteredHistory
                          .filter(item => item.result?.price)
                          .reduce((sum, item) => sum + item.result.price, 0) /
                        filteredHistory.filter(item => item.result?.price).length || 0
                      )}
                    </Typography>
                  </Box>
                </Grid>

                <Grid item xs={12} sm={6} md={3}>
                  <Box sx={{ textAlign: 'center' }}>
                    <Typography variant="body2" color="text.secondary">
                      Price Range
                    </Typography>
                    <Typography variant="h4">
                      {filteredHistory.filter(item => item.result?.price).length > 0 ?
                        `${formatCurrency(Math.min(...filteredHistory.filter(item => item.result?.price).map(item => item.result.price)))} - ${formatCurrency(Math.max(...filteredHistory.filter(item => item.result?.price).map(item => item.result.price)))}`
                        : 'N/A'
                      }
                    </Typography>
                  </Box>
                </Grid>

                <Grid item xs={12} sm={6} md={3}>
                  <Box sx={{ textAlign: 'center' }}>
                    <Typography variant="body2" color="text.secondary">
                      Most Used Model
                    </Typography>
                    <Typography variant="h4">
                      {(() => {
                        const models = filteredHistory
                          .filter(item => item.result?.model)
                          .map(item => item.result.model);
                        const modelCounts = models.reduce((acc, model) => {
                          acc[model] = (acc[model] || 0) + 1;
                          return acc;
                        }, {});
                        const mostUsed = Object.entries(modelCounts)
                          .sort(([,a], [,b]) => b - a)[0];
                        return mostUsed ? mostUsed[0].toUpperCase() : 'N/A';
                      })()}
                    </Typography>
                  </Box>
                </Grid>
              </Grid>
            </Paper>
          </Grid>
        )}
      </Grid>
    </Box>
  );
};

export default PriceHistory;