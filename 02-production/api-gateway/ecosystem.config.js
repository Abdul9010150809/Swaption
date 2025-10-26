/**
 * PM2 ecosystem configuration for API Gateway
 */

module.exports = {
  apps: [
    {
      name: 'price-matrix-api-gateway',
      script: 'src/index.js',
      instances: process.env.NODE_ENV === 'production' ? 'max' : 1,
      exec_mode: process.env.NODE_ENV === 'production' ? 'cluster' : 'fork',
      env: {
        NODE_ENV: 'development',
        PORT: 3000
      },
      env_production: {
        NODE_ENV: 'production',
        PORT: 3000,
        // Production environment variables would be set here
        // or loaded from .env file
      },
      env_staging: {
        NODE_ENV: 'staging',
        PORT: 3000
      },
      // Logging configuration
      log_file: 'logs/pm2/combined.log',
      out_file: 'logs/pm2/out.log',
      error_file: 'logs/pm2/error.log',
      log_date_format: 'YYYY-MM-DD HH:mm:ss Z',
      // Restart configuration
      max_restarts: 10,
      min_uptime: '10s',
      // Memory management
      max_memory_restart: '1G',
      // Graceful shutdown
      kill_timeout: 5000,
      wait_ready: true,
      listen_timeout: 10000,
      // Health check
      health_check: {
        enabled: true,
        url: 'http://localhost:3000/health',
        interval: 30000, // 30 seconds
        timeout: 5000,
        retries: 3
      }
    }
  ]
};