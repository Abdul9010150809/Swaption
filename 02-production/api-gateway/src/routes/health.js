/**
 * Health check routes for API Gateway
 */

const express = require('express');
const router = express.Router();
const axios = require('axios');
const config = require('../config');
const logger = require('../utils/logger');

/**
 * @swagger
 * /health:
 *   get:
 *     summary: Basic health check
 *     description: Returns basic health status of the API Gateway
 *     tags: [Health]
 *     responses:
 *       200:
 *         description: Service is healthy
 *         content:
 *           application/json:
 *             schema:
 *               type: object
 *               properties:
 *                 status:
 *                   type: string
 *                   example: healthy
 *                 timestamp:
 *                   type: string
 *                   format: date-time
 *                 uptime:
 *                   type: number
 *                   description: Server uptime in seconds
 *                 version:
 *                   type: string
 *                   example: 1.0.0
 */
router.get('/', async (req, res) => {
  try {
    const healthStatus = {
      status: 'healthy',
      timestamp: new Date().toISOString(),
      uptime: process.uptime(),
      version: process.env.npm_package_version || '1.0.0',
      environment: config.server.env
    };

    logger.info('Health check requested', { status: 'healthy' });
    res.status(200).json(healthStatus);
  } catch (error) {
    logger.error('Health check failed', { error: error.message });
    res.status(500).json({
      status: 'unhealthy',
      timestamp: new Date().toISOString(),
      error: 'Internal server error'
    });
  }
});

/**
 * @swagger
 * /health/detailed:
 *   get:
 *     summary: Detailed health check
 *     description: Returns detailed health status including dependencies
 *     tags: [Health]
 *     responses:
 *       200:
 *         description: All services are healthy
 *         content:
 *           application/json:
 *             schema:
 *               type: object
 *               properties:
 *                 status:
 *                   type: string
 *                   example: healthy
 *                 timestamp:
 *                   type: string
 *                   format: date-time
 *                 services:
 *                   type: object
 *                   properties:
 *                     database:
 *                       type: object
 *                       properties:
 *                         status:
 *                           type: string
 *                           example: healthy
 *                         response_time:
 *                           type: number
 *                     ml_service:
 *                       type: object
 *                       properties:
 *                         status:
 *                           type: string
 *                           example: healthy
 *                         response_time:
 *                           type: number
 *                     memory:
 *                       type: object
 *                       properties:
 *                         usage:
 *                           type: number
 *                         status:
 *                           type: string
 *                     disk:
 *                       type: object
 *                       properties:
 *                         usage:
 *                           type: number
 *                         status:
 *                           type: string
 */
router.get('/detailed', async (req, res) => {
  const startTime = Date.now();
  const healthChecks = {};

  try {
    // Database health check
    if (config.monitoring.health.checks.database) {
      try {
        const dbStart = Date.now();
        // Add database connectivity check here
        // For Redis, you would check connection
        const dbResponseTime = Date.now() - dbStart;

        healthChecks.database = {
          status: 'healthy',
          response_time: dbResponseTime
        };
      } catch (error) {
        healthChecks.database = {
          status: 'unhealthy',
          error: error.message
        };
      }
    }

    // ML Service health check
    if (config.monitoring.health.checks.mlService) {
      try {
        const mlStart = Date.now();
        const mlResponse = await axios.get(`${config.services.mlService.url}/health`, {
          timeout: 5000
        });
        const mlResponseTime = Date.now() - mlStart;

        healthChecks.ml_service = {
          status: mlResponse.status === 200 ? 'healthy' : 'unhealthy',
          response_time: mlResponseTime,
          version: mlResponse.data?.version
        };
      } catch (error) {
        healthChecks.ml_service = {
          status: 'unhealthy',
          error: error.message
        };
      }
    }

    // Memory usage check
    if (config.monitoring.health.checks.memory) {
      const memUsage = process.memoryUsage();
      const memUsagePercent = memUsage.heapUsed / memUsage.heapTotal;

      healthChecks.memory = {
        usage: memUsagePercent,
        status: memUsagePercent < config.monitoring.health.thresholds.memoryUsage ? 'healthy' : 'warning',
        details: {
          used: Math.round(memUsage.heapUsed / 1024 / 1024), // MB
          total: Math.round(memUsage.heapTotal / 1024 / 1024), // MB
          external: Math.round(memUsage.external / 1024 / 1024) // MB
        }
      };
    }

    // Disk usage check (simplified)
    if (config.monitoring.health.checks.disk) {
      try {
        const fs = require('fs').promises;
        const stats = await fs.statvfs ? await fs.statvfs('/') : null;

        if (stats) {
          const diskUsage = 1 - (stats.f_bavail / stats.f_blocks);
          healthChecks.disk = {
            usage: diskUsage,
            status: diskUsage < config.monitoring.health.thresholds.diskUsage ? 'healthy' : 'warning'
          };
        } else {
          healthChecks.disk = {
            status: 'not_available',
            note: 'statvfs not supported on this platform'
          };
        }
      } catch (error) {
        healthChecks.disk = {
          status: 'error',
          error: error.message
        };
      }
    }

    // Overall status
    const hasUnhealthy = Object.values(healthChecks).some(check =>
      check.status === 'unhealthy'
    );

    const overallStatus = hasUnhealthy ? 'unhealthy' : 'healthy';
    const responseTime = Date.now() - startTime;

    const detailedHealth = {
      status: overallStatus,
      timestamp: new Date().toISOString(),
      response_time: responseTime,
      services: healthChecks,
      version: process.env.npm_package_version || '1.0.0',
      environment: config.server.env
    };

    const statusCode = overallStatus === 'healthy' ? 200 : 503;

    logger.info('Detailed health check completed', {
      status: overallStatus,
      response_time: responseTime
    });

    res.status(statusCode).json(detailedHealth);

  } catch (error) {
    logger.error('Detailed health check failed', { error: error.message });

    res.status(503).json({
      status: 'unhealthy',
      timestamp: new Date().toISOString(),
      error: 'Health check failed',
      details: error.message
    });
  }
});

/**
 * @swagger
 * /health/ping:
 *   get:
 *     summary: Simple ping endpoint
 *     description: Returns pong for connectivity testing
 *     tags: [Health]
 *     responses:
 *       200:
 *         description: Pong response
 *         content:
 *           application/json:
 *             schema:
 *               type: object
 *               properties:
 *                 message:
 *                   type: string
 *                   example: pong
 *                 timestamp:
 *                   type: string
 *                   format: date-time
 */
router.get('/ping', (req, res) => {
  res.status(200).json({
    message: 'pong',
    timestamp: new Date().toISOString()
  });
});

/**
 * @swagger
 * /health/metrics:
 *   get:
 *     summary: Application metrics
 *     description: Returns application performance metrics
 *     tags: [Health]
 *     responses:
 *       200:
 *         description: Metrics data
 *         content:
 *           application/json:
 *             schema:
 *               type: object
 *               properties:
 *                 uptime:
 *                   type: number
 *                   description: Server uptime in seconds
 *                 memory:
 *                   type: object
 *                   properties:
 *                     used:
 *                       type: number
 *                     total:
 *                       type: number
 *                     percentage:
 *                       type: number
 *                 requests:
 *                   type: object
 *                   properties:
 *                     total:
 *                       type: number
 *                     per_second:
 *                       type: number
 */
router.get('/metrics', (req, res) => {
  try {
    const memUsage = process.memoryUsage();

    const metrics = {
      uptime: process.uptime(),
      memory: {
        used: Math.round(memUsage.heapUsed / 1024 / 1024), // MB
        total: Math.round(memUsage.heapTotal / 1024 / 1024), // MB
        percentage: Math.round((memUsage.heapUsed / memUsage.heapTotal) * 100)
      },
      timestamp: new Date().toISOString(),
      version: process.env.npm_package_version || '1.0.0'
    };

    // Add request metrics if available (would be populated by middleware)
    if (global.requestMetrics) {
      metrics.requests = global.requestMetrics;
    }

    res.status(200).json(metrics);
  } catch (error) {
    logger.error('Metrics endpoint failed', { error: error.message });
    res.status(500).json({
      error: 'Failed to retrieve metrics',
      timestamp: new Date().toISOString()
    });
  }
});

module.exports = router;