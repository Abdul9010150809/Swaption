# Price Matrix User Guide

## Welcome to Price Matrix

Price Matrix is a comprehensive financial pricing system for derivatives, providing fast and accurate pricing for European options and swaptions using a combination of machine learning and analytical methods.

## Getting Started

### System Requirements
- Modern web browser (Chrome, Firefox, Safari, Edge)
- Internet connection
- JavaScript enabled

### Accessing the System
1. Open your web browser
2. Navigate to the Price Matrix application URL
3. The system will load the main pricing interface

## Main Interface

### Navigation
The application features a sidebar navigation with the following sections:
- **Pricing Calculator**: Main pricing interface
- **Price History**: View previous calculations
- **Yield Curve**: Analyze interest rate curves
- **Results**: View detailed calculation results
- **Risk Metrics**: Risk analysis tools

### Header Information
The header displays:
- Current market data (spot price, rates, volatility)
- Last data update timestamp
- User account access

## Pricing Calculator

### European Option Pricing

#### Input Parameters
- **Spot Price**: Current price of the underlying asset
- **Strike Price**: Option strike price
- **Time to Expiry**: Time remaining until option expires (in years)
- **Risk-Free Rate**: Current risk-free interest rate (annual)
- **Volatility**: Implied volatility of the underlying asset (annual)
- **Option Type**: Call or Put
- **Dividend Yield**: Dividend yield of the underlying asset (optional)

#### How to Price an Option
1. Select "European Option" from the instrument type
2. Enter all required parameters
3. Click "Calculate Price"
4. Review the results including price, confidence intervals, and Greeks

#### Understanding Results
- **Price**: The calculated fair value of the option
- **Confidence Interval**: Range of possible prices based on model uncertainty
- **Greeks**: Risk sensitivities (Delta, Gamma, Theta, Vega, Rho)
- **Model Used**: Which pricing model was employed

### European Swaption Pricing

#### Input Parameters
- **Swap Rate**: Current underlying swap rate
- **Strike Rate**: Strike swap rate
- **Option Tenor**: Time to swaption expiry (in years)
- **Swap Tenor**: Underlying swap maturity (in years)
- **Volatility**: Swaption volatility

#### How to Price a Swaption
1. Select "European Swaption" from the instrument type
2. Enter swap and option parameters
3. Click "Calculate Price"
4. Review pricing results and risk metrics

## Batch Pricing

### Overview
Batch pricing allows you to price multiple instruments simultaneously for efficient portfolio valuation.

### Creating a Batch
1. Navigate to the Pricing Calculator
2. Select "Batch Pricing" mode
3. Add instruments one by one or upload a CSV file
4. Configure parameters for each instrument
5. Click "Calculate Batch"

### Batch Results
- Individual pricing results for each instrument
- Summary statistics (total value, success rate)
- Export options for results

## Price History

### Viewing History
1. Click "Price History" in the sidebar
2. Use filters to narrow down results:
   - Instrument type
   - Date range
   - Price range
3. Click on any calculation for detailed view

### History Features
- **Filtering**: Filter by date, instrument type, or price range
- **Search**: Search through calculation history
- **Export**: Export history to CSV or PDF
- **Details**: View full parameters and results for any calculation

## Risk Analysis

### Value at Risk (VaR)
1. Navigate to "Risk Metrics"
2. Select "VaR Calculation"
3. Input portfolio returns data
4. Configure confidence level and time horizon
5. Choose calculation method (Historical, Parametric, Monte Carlo)

### Stress Testing
1. Select "Stress Testing"
2. Define stress scenarios (market crash, rate changes, etc.)
3. Input portfolio composition
4. Run stress test analysis
5. Review potential losses under different scenarios

## Yield Curve Analysis

### Viewing Yield Curves
1. Click "Yield Curve" in the sidebar
2. View current yield curve data
3. Analyze term structure
4. Export curve data

### Interactive Features
- Zoom and pan on the curve
- Compare multiple curves
- View historical curves
- Export visualizations

## Settings and Preferences

### User Preferences
- **Theme**: Light/dark mode toggle
- **Currency**: Default currency display
- **Decimal Places**: Number formatting precision
- **Notifications**: Alert preferences

### API Access
- Generate API keys for programmatic access
- View API documentation
- Test API endpoints

## Troubleshooting

### Common Issues

#### Pricing Calculation Errors
- **Issue**: "Invalid parameters" error
- **Solution**: Check that all required fields are filled and values are within reasonable ranges
- **Prevention**: Use the built-in parameter validation

#### Slow Performance
- **Issue**: Calculations taking too long
- **Solution**: Check internet connection, try again later, or contact support
- **Prevention**: Ensure stable internet connection

#### Login Issues
- **Issue**: Unable to access the system
- **Solution**: Clear browser cache, check credentials, contact administrator

### Error Messages
- **"Model not available"**: ML service is temporarily down, analytical pricing is used as fallback
- **"Rate limit exceeded"**: Too many requests, wait before retrying
- **"Invalid input format"**: Check parameter formats and ranges

## API Usage

### Authentication
```bash
# Include API key in headers
curl -H "X-API-Key: your-api-key" https://api.pricematrix.com/api/pricing/options
```

### Option Pricing Example
```bash
curl -X POST https://api.pricematrix.com/api/pricing/options \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{
    "spot_price": 100.0,
    "strike_price": 105.0,
    "time_to_expiry": 1.0,
    "risk_free_rate": 0.05,
    "volatility": 0.20,
    "option_type": "call"
  }'
```

### Batch Pricing Example
```bash
curl -X POST https://api.pricematrix.com/api/pricing/batch \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{
    "instruments": [
      {
        "type": "option",
        "parameters": {
          "spot_price": 100.0,
          "strike_price": 105.0,
          "time_to_expiry": 1.0,
          "risk_free_rate": 0.05,
          "volatility": 0.20,
          "option_type": "call"
        }
      }
    ]
  }'
```

## Best Practices

### Pricing Accuracy
1. **Use Current Data**: Ensure market data is up to date
2. **Validate Inputs**: Double-check all parameters before calculation
3. **Consider Confidence Intervals**: Always review uncertainty estimates
4. **Cross-Validate**: Compare results with other pricing sources when possible

### Performance Optimization
1. **Batch Processing**: Use batch pricing for multiple instruments
2. **Caching**: System automatically caches frequent calculations
3. **Off-Peak Usage**: Schedule large calculations during off-peak hours

### Risk Management
1. **Diversify Models**: Don't rely on a single pricing model
2. **Monitor Changes**: Regularly review pricing changes
3. **Set Limits**: Implement position limits based on risk metrics
4. **Stress Test**: Regularly perform stress testing

## Security and Compliance

### Data Security
- All calculations are encrypted in transit and at rest
- No personal data is stored without explicit consent
- Regular security audits and updates

### Regulatory Compliance
- System designed to comply with financial regulations
- All calculations are logged for audit purposes
- Clear disclaimers about model limitations

## Support and Resources

### Getting Help
- **Documentation**: Comprehensive API and user documentation
- **Support Portal**: Submit tickets for technical issues
- **Community Forum**: Connect with other users
- **Training**: Online tutorials and webinars

### Contact Information
- **General Support**: support@pricematrix.com
- **Technical Issues**: tech@pricematrix.com
- **Sales**: sales@pricematrix.com
- **Phone**: 1-800-PRICE-MATRIX

### Additional Resources
- [API Documentation](https://docs.pricematrix.com/api)
- [Model Documentation](https://docs.pricematrix.com/models)
- [Video Tutorials](https://learn.pricematrix.com)
- [Blog and Updates](https://blog.pricematrix.com)

---

*This user guide was last updated on October 24, 2025.*