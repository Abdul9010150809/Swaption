import React from 'react';
import {
  Drawer, List, ListItem, ListItemButton, ListItemIcon, ListItemText,
  Divider, Box, Typography
} from '@mui/material';
import CalculateIcon from '@mui/icons-material/Calculate';
import ShowChartIcon from '@mui/icons-material/ShowChart';
import TimelineIcon from '@mui/icons-material/Timeline';
import AssessmentIcon from '@mui/icons-material/Assessment';
import WarningIcon from '@mui/icons-material/Warning';

const drawerWidth = 240;

const Sidebar = ({ open, onClose, currentView, onViewChange }) => {
  const menuItems = [
    {
      id: 'pricing',
      label: 'Pricing Calculator',
      icon: <CalculateIcon />,
      description: 'Calculate derivative prices'
    },
    {
      id: 'history',
      label: 'Price History',
      icon: <TimelineIcon />,
      description: 'View historical prices'
    },
    {
      id: 'yield-curve',
      label: 'Yield Curve',
      icon: <ShowChartIcon />,
      description: 'Analyze yield curves'
    },
    {
      id: 'results',
      label: 'Results',
      icon: <AssessmentIcon />,
      description: 'View calculation results'
    },
    {
      id: 'risk',
      label: 'Risk Metrics',
      icon: <WarningIcon />,
      description: 'Risk analysis tools'
    }
  ];

  const handleItemClick = (viewId) => {
    onViewChange(viewId);
    // Close sidebar on mobile after selection
    if (window.innerWidth < 768) {
      onClose();
    }
  };

  const drawer = (
    <Box sx={{ width: drawerWidth }}>
      <Box sx={{ p: 2, borderBottom: 1, borderColor: 'divider' }}>
        <Typography variant="h6" color="primary">
          Navigation
        </Typography>
        <Typography variant="body2" color="text.secondary">
          Financial Tools
        </Typography>
      </Box>

      <List>
        {menuItems.map((item) => (
          <ListItem key={item.id} disablePadding>
            <ListItemButton
              selected={currentView === item.id}
              onClick={() => handleItemClick(item.id)}
              sx={{
                '&.Mui-selected': {
                  backgroundColor: 'primary.light',
                  '&:hover': {
                    backgroundColor: 'primary.main',
                  },
                },
              }}
            >
              <ListItemIcon
                sx={{
                  color: currentView === item.id ? 'primary.contrastText' : 'inherit'
                }}
              >
                {item.icon}
              </ListItemIcon>
              <Box>
                <ListItemText
                  primary={item.label}
                  primaryTypographyProps={{
                    variant: 'body1',
                    color: currentView === item.id ? 'primary.contrastText' : 'inherit'
                  }}
                />
                <Typography
                  variant="caption"
                  color={currentView === item.id ? 'primary.contrastText' : 'text.secondary'}
                  sx={{ display: 'block', mt: 0.5 }}
                >
                  {item.description}
                </Typography>
              </Box>
            </ListItemButton>
          </ListItem>
        ))}
      </List>

      <Divider sx={{ my: 2 }} />

      <Box sx={{ p: 2 }}>
        <Typography variant="body2" color="text.secondary">
          Version 1.0.0
        </Typography>
        <Typography variant="caption" color="text.secondary">
          Price Matrix System
        </Typography>
      </Box>
    </Box>
  );

  return (
    <Drawer
      variant="persistent"
      anchor="left"
      open={open}
      sx={{
        width: open ? drawerWidth : 0,
        flexShrink: 0,
        '& .MuiDrawer-paper': {
          width: drawerWidth,
          boxSizing: 'border-box',
          top: '64px', // Account for AppBar height
          height: 'calc(100% - 64px)',
        },
      }}
    >
      {drawer}
    </Drawer>
  );
};

export default Sidebar;