// src/index.js
/**
 * Application entry point
 */

const ort = require('onnxruntime-node');
const config = require('./config');
const createApp = require('./app');
const FaceDetectionService = require('./services/faceDetectionService');
const FaceRecognitionService = require('./services/faceRecognitionService');
const FaceComparisonService = require('./services/faceComparisonService');
const Logger = require('./utils/logger');

const logger = new Logger('Server');

// Suppress ONNX warnings
ort.env.logLevel = 'error';

/**
 * Initialize services and start server
 */
async function startServer() {
  try {
    logger.info('Starting Face Verification Server...');

    // Initialize services
    const detectionService = new FaceDetectionService();
    const recognitionService = new FaceRecognitionService();
    const comparisonService = FaceComparisonService;

    // Load models
    logger.info('Loading AI models...');
    await Promise.all([
      detectionService.loadModel(),
      recognitionService.loadModel(),
    ]);
    logger.info('✅ All models loaded successfully');

    // Create Express app
    const app = createApp(detectionService, recognitionService, comparisonService);

    // Start server
    const server = app.listen(config.server.port, () => {
      logger.info('=================================');
      logger.info(`✅ Server running on port ${config.server.port}`);
      logger.info(`📊 Environment: ${config.server.env}`);
      logger.info(`📏 Similarity threshold: ${config.faceRecognition.similarityThreshold}`);
      logger.info(`📐 Face expand ratio: ${config.faceRecognition.faceExpandRatio}`);
      logger.info(`🔲 Square crop: ${config.faceRecognition.useSquareCrop}`);
      logger.info('=================================');
    });

    // Graceful shutdown
    process.on('SIGTERM', () => {
      logger.info('SIGTERM received, shutting down gracefully...');
      server.close(() => {
        logger.info('Server closed');
        process.exit(0);
      });
    });

    process.on('SIGINT', () => {
      logger.info('SIGINT received, shutting down gracefully...');
      server.close(() => {
        logger.info('Server closed');
        process.exit(0);
      });
    });

  } catch (error) {
    logger.error('❌ Failed to start server:', error);
    process.exit(1);
  }
}

// Start the server
startServer();
