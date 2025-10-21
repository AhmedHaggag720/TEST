// src/utils/validation.js
/**
 * Input validation utilities
 */

const { AppError, ErrorMessages, ErrorTypes } = require('../constants/errors');
const config = require('../config');

/**
 * Validate image file upload
 */
function validateImageUpload(file) {
  if (!file) {
    throw new AppError(
      ErrorMessages.NO_IMAGE_PROVIDED,
      ErrorTypes.VALIDATION,
      400
    );
  }

  if (!file.buffer || file.buffer.length === 0) {
    throw new AppError(
      ErrorMessages.INVALID_FILE_TYPE,
      ErrorTypes.VALIDATION,
      400
    );
  }

  if (file.size > config.server.maxFileSize) {
    throw new AppError(
      ErrorMessages.FILE_TOO_LARGE,
      ErrorTypes.VALIDATION,
      400
    );
  }

  return true;
}

/**
 * Validate and parse embedding array
 */
function validateEmbedding(embeddingString) {
  let embedding;

  try {
    embedding = JSON.parse(embeddingString);
  } catch (e) {
    throw new AppError(
      `Invalid embedding format: ${e.message}`,
      ErrorTypes.VALIDATION,
      400
    );
  }

  if (!Array.isArray(embedding)) {
    throw new AppError(
      ErrorMessages.INVALID_EMBEDDING_FORMAT,
      ErrorTypes.VALIDATION,
      400
    );
  }

  if (embedding.length !== config.faceRecognition.embeddingDimension) {
    throw new AppError(
      ErrorMessages.INVALID_EMBEDDING_FORMAT,
      ErrorTypes.VALIDATION,
      400
    );
  }

  if (!embedding.every(val => typeof val === 'number' && !isNaN(val))) {
    throw new AppError(
      ErrorMessages.INVALID_EMBEDDING_VALUES,
      ErrorTypes.VALIDATION,
      400
    );
  }

  return embedding;
}

/**
 * Validate bounding box
 */
function validateBoundingBox(box, imageWidth, imageHeight) {
  if (!box || typeof box !== 'object') {
    return false;
  }

  const { x, y, width, height } = box;

  if (
    typeof x !== 'number' || typeof y !== 'number' ||
    typeof width !== 'number' || typeof height !== 'number'
  ) {
    return false;
  }

  if (x < 0 || y < 0 || width <= 0 || height <= 0) {
    return false;
  }

  if (x + width > imageWidth || y + height > imageHeight) {
    return false;
  }

  return true;
}

module.exports = {
  validateImageUpload,
  validateEmbedding,
  validateBoundingBox,
};
