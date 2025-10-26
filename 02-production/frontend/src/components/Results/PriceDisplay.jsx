import React from 'react';
import {
  Box, Paper, Typography, Grid, Chip, Divider,
  Table, TableBody, TableCell, TableContainer, TableHead, TableRow
} from '@mui/material';

const PriceDisplay = ({ result, instrumentType }) => {
  if (!result) {
    return (
      <Paper elevation={3} sx={{ p: 3 }}>
        <Typography variant="h6" gutterBottom>
          Pricing Results
        </Typography>
        <Typography variant="body2" color="text.secondary">
          No pricing results available. Please calculate a price first.
        </Typography>
      </Paper>
    );
  }

  const formatCurrency = (value) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 4,
      maximumFractionDigits: 4
    }).format(value);
  };

  const formatPercentage = (value) => {
    return `${(value * 100).toFixed(2)}%`;
  };

  const formatNumber = (value, decimals = 4) => {
    return Number(value).toFixed(decimals);
  };

  return (
    <Paper elevation={3} sx={{ p: 3 }}>
      <Typography variant="h6" gutterBottom>
        Pricing Results - {instrumentType.charAt(0).toUpperCase() + instrumentType.slice(1)}
      </Typography>

      <Grid container spacing={3}>
        {/* Main Price Display */}
        <Grid item xs={12} md={6}>
          <Box sx={{ textAlign: 'center', mb: 3 }}>
            <Typography variant="body2" color="text.secondary" gutterBottom>
              Calculated Price
            </Typography>
            <Typography variant="h3" color="primary" sx={{ fontWeight: 'bold' }}>
              {formatCurrency(result.price)}
            </Typography>
            <Chip
              label={`Model: ${result.model?.toUpperCase()}`}
              color="secondary"
              size="small"
              sx={{ mt: 1 }}
            />
          </Box>
        </Grid>

        {/* Greeks Table */}
        {result.greeks && Object.keys(result.greeks).length > 0 && (
          <Grid item xs={12} md={6}>
            <Typography variant="h6" gutterBottom>
              Greeks
            </Typography>
            <TableContainer component={Paper} variant="outlined">
              <Table size="small">
                <TableHead>
                  <TableRow>
                    <TableCell><strong>Greek</strong></TableCell>
                    <TableCell align="right"><strong>Value</strong></TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {Object.entries(result.greeks).map(([greek, value]) => (
                    <TableRow key={greek}>
                      <TableCell component="th" scope="row">
                        {greek.toUpperCase()}
                      </TableCell>
                      <TableCell align="right">
                        {greek.toLowerCase().includes('rho') || greek.toLowerCase().includes('rate')
                          ? formatPercentage(value)
                          : formatNumber(value)}
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </TableContainer>
          </Grid>
        )}

        {/* Additional Metrics */}
        <Grid item xs={12}>
          <Divider sx={{ my: 2 }} />
          <Typography variant="h6" gutterBottom>
            Additional Information
          </Typography>

          <Grid container spacing={2}>
            {result.standard_error && (
              <Grid item xs={12} sm={6} md={3}>
                <Box sx={{ textAlign: 'center' }}>
                  <Typography variant="body2" color="text.secondary">
                    Standard Error
                  </Typography>
                  <Typography variant="h6">
                    {formatCurrency(result.standard_error)}
                  </Typography>
                </Box>
              </Grid>
            )}

            {result.confidence_interval && (
              <Grid item xs={12} sm={6} md={3}>
                <Box sx={{ textAlign: 'center' }}>
                  <Typography variant="body2" color="text.secondary">
                    95% CI
                  </Typography>
                  <Typography variant="h6">
                    ±{formatCurrency(result.confidence_interval)}
                  </Typography>
                </Box>
              </Grid>
            )}

            {result.lower_bound && result.upper_bound && (
              <>
                <Grid item xs={12} sm={6} md={3}>
                  <Box sx={{ textAlign: 'center' }}>
                    <Typography variant="body2" color="text.secondary">
                      Lower Bound
                    </Typography>
                    <Typography variant="h6">
                      {formatCurrency(result.lower_bound)}
                    </Typography>
                  </Box>
                </Grid>
                <Grid item xs={12} sm={6} md={3}>
                  <Box sx={{ textAlign: 'center' }}>
                    <Typography variant="body2" color="text.secondary">
                      Upper Bound
                    </Typography>
                    <Typography variant="h6">
                      {formatCurrency(result.upper_bound)}
                    </Typography>
                  </Box>
                </Grid>
              </>
            )}

            {result.n_simulations && (
              <Grid item xs={12} sm={6} md={3}>
                <Box sx={{ textAlign: 'center' }}>
                  <Typography variant="body2" color="text.secondary">
                    Simulations
                  </Typography>
                  <Typography variant="h6">
                    {result.n_simulations.toLocaleString()}
                  </Typography>
                </Box>
              </Grid>
            )}

            {result.exercise_probability !== undefined && (
              <Grid item xs={12} sm={6} md={3}>
                <Box sx={{ textAlign: 'center' }}>
                  <Typography variant="body2" color="text.secondary">
                    Exercise Probability
                  </Typography>
                  <Typography variant="h6">
                    {formatPercentage(result.exercise_probability)}
                  </Typography>
                </Box>
              </Grid>
            )}
          </Grid>
        </Grid>

        {/* Timestamp */}
        {result.timestamp && (
          <Grid item xs={12}>
            <Divider sx={{ my: 2 }} />
            <Typography variant="caption" color="text.secondary" sx={{ display: 'block', textAlign: 'center' }}>
              Calculated on {new Date(result.timestamp).toLocaleString()}
            </Typography>
          </Grid>
        )}
      </Grid>
    </Paper>
  );
};

export default PriceDisplay;