import React from 'react';
import {
  AppBar, Toolbar, Typography, IconButton, Box, Chip
} from '@mui/material';
import MenuIcon from '@mui/icons-material/Menu';
import AccountCircle from '@mui/icons-material/AccountCircle';
import { useMarketData } from '../../hooks/useMarketData';

const Header = ({ onMenuClick }) => {
  const { marketData, lastUpdated } = useMarketData();

  const formatLastUpdated = (timestamp) => {
    if (!timestamp) return 'Never';
    const now = new Date();
    const updated = new Date(timestamp);
    const diffMinutes = Math.floor((now - updated) / (1000 * 60));

    if (diffMinutes < 1) return 'Just now';
    if (diffMinutes < 60) return `${diffMinutes}m ago`;
    const diffHours = Math.floor(diffMinutes / 60);
    if (diffHours < 24) return `${diffHours}h ago`;
    return updated.toLocaleDateString();
  };

  return (
    <AppBar position="fixed" sx={{ zIndex: (theme) => theme.zIndex.drawer + 1 }}>
      <Toolbar>
        <IconButton
          color="inherit"
          aria-label="open drawer"
          edge="start"
          onClick={onMenuClick}
          sx={{ mr: 2 }}
        >
          <MenuIcon />
        </IconButton>

        <Typography variant="h6" noWrap component="div" sx={{ flexGrow: 1 }}>
          Price Matrix - Financial Derivatives Pricing
        </Typography>

        <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
          {/* Market Data Status */}
          {marketData && (
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
              <Chip
                label={`Spot: $${marketData.spotPrice?.toFixed(2)}`}
                size="small"
                color="primary"
                variant="outlined"
              />
              <Chip
                label={`Rate: ${(marketData.riskFreeRate * 100)?.toFixed(2)}%`}
                size="small"
                color="secondary"
                variant="outlined"
              />
              <Chip
                label={`Vol: ${(marketData.volatility * 100)?.toFixed(1)}%`}
                size="small"
                color="info"
                variant="outlined"
              />
            </Box>
          )}

          {/* Last Updated */}
          <Typography variant="caption" color="inherit" sx={{ opacity: 0.7 }}>
            Updated: {formatLastUpdated(lastUpdated)}
          </Typography>

          {/* User Account */}
          <IconButton color="inherit">
            <AccountCircle />
          </IconButton>
        </Box>
      </Toolbar>
    </AppBar>
  );
};

export default Header;