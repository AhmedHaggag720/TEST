// src/middleware/errorHandler.js
/**
 * Global error handling middleware
 */

const { AppError } = require('../constants/errors');
const Logger = require('../utils/logger');

const logger = new Logger('ErrorHandler');

/**
 * Error handling middleware
 */
function errorHandler(err, req, res, next) {
  // Log error
  logger.error(`Error in ${req.method} ${req.path}:`, {
    message: err.message,
    stack: err.stack,
  });

  // Handle operational errors
  if (err instanceof AppError && err.isOperational) {
    return res.status(err.statusCode).json({
      success: false,
      error: err.message,
      type: err.name,
    });
  }

  // Handle multer errors
  if (err.name === 'MulterError') {
    return res.status(400).json({
      success: false,
      error: `File upload error: ${err.message}`,
      type: 'UploadError',
    });
  }

  // Handle validation errors
  if (err.name === 'ValidationError') {
    return res.status(400).json({
      success: false,
      error: err.message,
      type: 'ValidationError',
    });
  }

  // Handle unknown errors
  res.status(500).json({
    success: false,
    error: 'Internal server error',
    type: 'InternalError',
  });
}

/**
 * Async route handler wrapper to catch errors
 */
function asyncHandler(fn) {
  return (req, res, next) => {
    Promise.resolve(fn(req, res, next)).catch(next);
  };
}

/**
 * 404 handler
 */
function notFoundHandler(req, res) {
  res.status(404).json({
    success: false,
    error: `Route ${req.method} ${req.path} not found`,
    type: 'NotFound',
  });
}

module.exports = {
  errorHandler,
  asyncHandler,
  notFoundHandler,
};
