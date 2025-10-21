// src/routes/faceRoutes.js
/**
 * Face verification API routes
 */

const express = require('express');
const upload = require('../middleware/uploadMiddleware');
const { asyncHandler } = require('../middleware/errorHandler');
const { validateImageUpload, validateEmbedding } = require('../utils/validation');
const { AppError, ErrorMessages, ErrorTypes } = require('../constants/errors');
const FaceProcessingService = require('../services/faceProcessingService');
const Logger = require('../utils/logger');

const router = express.Router();
const logger = new Logger('Routes');

/**
 * Initialize routes with services
 */
function initializeRoutes(detectionService, recognitionService, comparisonService) {
  /**
   * POST /cropface
   * Extract and return cropped face image for visual inspection
   */
  router.post(
    '/cropface',
    upload.single('image'),
    asyncHandler(async (req, res) => {
      validateImageUpload(req.file);

      const faceBox = await FaceProcessingService.detectAndValidateFace(
        req.file.buffer,
        detectionService
      );

      const faceCrop = await FaceProcessingService.extractFaceCrop(
        req.file.buffer,
        faceBox
      );

      res.set('Content-Type', 'image/png');
      res.send(faceCrop);
    })
  );

  /**
   * POST /encode
   * Generate face embedding from image
   */
  router.post(
    '/encode',
    upload.single('image'),
    asyncHandler(async (req, res) => {
      validateImageUpload(req.file);

      const faceBox = await FaceProcessingService.detectAndValidateFace(
        req.file.buffer,
        detectionService
      );

      const embedding = await recognitionService.encodeface(
        req.file.buffer,
        faceBox
      );

      logger.info('Successfully generated face embedding');

      res.json({
        success: true,
        embedding,
      });
    })
  );

  /**
   * POST /compare
   * Compare face against stored embedding
   */
  router.post(
    '/compare',
    upload.single('image'),
    asyncHandler(async (req, res) => {
      validateImageUpload(req.file);

      if (!req.body.storedEmbedding) {
        throw new AppError(
          ErrorMessages.MISSING_STORED_EMBEDDING,
          ErrorTypes.VALIDATION,
          400
        );
      }

      // Validate and parse stored embedding
      const storedEmbedding = validateEmbedding(req.body.storedEmbedding);

      // Detect face and generate embedding
      const faceBox = await FaceProcessingService.detectAndValidateFace(
        req.file.buffer,
        detectionService
      );

      const newEmbedding = await recognitionService.encodeface(
        req.file.buffer,
        faceBox
      );

      // Compare embeddings
      const result = comparisonService.detailedComparison(
        newEmbedding,
        storedEmbedding
      );

      res.json({
        success: true,
        isMatch: result.isMatch,
        similarity: result.similarity,
      });
    })
  );

  /**
   * GET /health
   * Health check endpoint
   */
  router.get('/health', (req, res) => {
    res.json({
      success: true,
      status: 'healthy',
      timestamp: new Date().toISOString(),
    });
  });

  return router;
}

module.exports = initializeRoutes;
