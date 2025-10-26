import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import { CssBaseline, Box, AppBar, Toolbar, Typography, Container } from '@mui/material';
import PricingForm from './components/PricingForm/PricingForm';
import PriceHistory from './components/Charts/PriceHistory';
import YieldCurveChart from './components/Charts/YieldCurveChart';
import Header from './components/Layout/Header';
import Sidebar from './components/Layout/Sidebar';
import ResultsDisplay from './components/Results/PriceDisplay';
import RiskMetrics from './components/Results/RiskMetrics';
import { MarketDataProvider } from './hooks/useMarketData';
import { PricingProvider } from './hooks/usePricing';
import './App.css';

// Create theme
const theme = createTheme({
  palette: {
    mode: 'light',
    primary: {
      main: '#1976d2',
    },
    secondary: {
      main: '#dc004e',
    },
  },
  typography: {
    h4: {
      fontWeight: 600,
    },
  },
});

function App() {
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [currentView, setCurrentView] = useState('pricing');

  const toggleSidebar = () => {
    setSidebarOpen(!sidebarOpen);
  };

  const handleViewChange = (view) => {
    setCurrentView(view);
  };

  const renderCurrentView = () => {
    switch (currentView) {
      case 'pricing':
        return <PricingForm />;
      case 'history':
        return <PriceHistory />;
      case 'yield-curve':
        return <YieldCurveChart />;
      case 'results':
        return <ResultsDisplay />;
      case 'risk':
        return <RiskMetrics />;
      default:
        return <PricingForm />;
    }
  };

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <MarketDataProvider>
        <PricingProvider>
          <Router>
            <Box sx={{ display: 'flex', minHeight: '100vh' }}>
              <Header onMenuClick={toggleSidebar} />
              <Sidebar
                open={sidebarOpen}
                onClose={toggleSidebar}
                currentView={currentView}
                onViewChange={handleViewChange}
              />
              <Box
                component="main"
                sx={{
                  flexGrow: 1,
                  p: 3,
                  marginTop: '64px', // Account for AppBar height
                  marginLeft: sidebarOpen ? '240px' : 0,
                  transition: 'margin-left 0.3s ease',
                }}
              >
                <Container maxWidth="lg">
                  {renderCurrentView()}
                </Container>
              </Box>
            </Box>
          </Router>
        </PricingProvider>
      </MarketDataProvider>
    </ThemeProvider>
  );
}

export default App;