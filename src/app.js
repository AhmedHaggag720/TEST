// src/app.js
/**
 * Express application setup
 */

const express = require('express');
const config = require('./config');
const initializeRoutes = require('./routes/faceRoutes');
const { errorHandler, notFoundHandler } = require('./middleware/errorHandler');
const Logger = require('./utils/logger');

const logger = new Logger('App');

/**
 * Create and configure Express application
 */
function createApp(detectionService, recognitionService, comparisonService) {
  const app = express();

  // Middleware
  app.use(express.json());
  app.use(express.urlencoded({ extended: true }));

  // Request logging in development
  if (config.server.env === 'development') {
    app.use((req, res, next) => {
      logger.debug(`${req.method} ${req.path}`);
      next();
    });
  }

  // Routes
  const faceRoutes = initializeRoutes(detectionService, recognitionService, comparisonService);
  app.use('/api', faceRoutes);

  // Root endpoint
  app.get('/', (req, res) => {
    res.json({
      success: true,
      message: 'Face Verification API',
      version: '1.0.0',
      endpoints: {
        health: 'GET /api/health',
        cropFace: 'POST /api/cropface',
        encode: 'POST /api/encode',
        compare: 'POST /api/compare',
      },
    });
  });

  // Error handlers (must be last)
  app.use(notFoundHandler);
  app.use(errorHandler);

  return app;
}

module.exports = createApp;
