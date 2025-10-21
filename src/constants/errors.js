// src/constants/errors.js
/**
 * Error messages and codes for consistent error handling
 */

const ErrorMessages = {
  // Input Validation Errors
  NO_IMAGE_PROVIDED: 'No image provided',
  INVALID_FILE_TYPE: 'Invalid file type. Please provide an image file.',
  FILE_TOO_LARGE: 'File size exceeds maximum limit',
  
  // Face Detection Errors
  NO_FACE_DETECTED: 'No face detected. Please provide a clear image with a visible face.',
  FACE_TOO_SMALL: 'Detected face is too small. Please provide a clearer image with a larger face.',
  MULTIPLE_FACES_DETECTED: 'Multiple faces detected. Please provide an image with a single face.',
  
  // Embedding Errors
  INVALID_EMBEDDING: 'Failed to generate valid face embedding. Please try with a different image.',
  MISSING_STORED_EMBEDDING: 'Missing storedEmbedding parameter',
  INVALID_EMBEDDING_FORMAT: 'storedEmbedding must be an array of 512 numbers',
  INVALID_EMBEDDING_VALUES: 'storedEmbedding must contain only valid numbers',
  
  // Comparison Errors
  SIMILARITY_CALCULATION_FAILED: 'Failed to calculate similarity. Embeddings may be invalid.',
  
  // Processing Errors
  IMAGE_PROCESSING_FAILED: 'Failed to process image. Please try with a different image.',
  MODEL_INFERENCE_FAILED: 'Model inference failed. Please try again.',
  
  // Generic Errors
  INTERNAL_SERVER_ERROR: 'Internal server error. Please try again later.',
};

const ErrorTypes = {
  VALIDATION: 'ValidationError',
  DETECTION: 'DetectionError',
  PROCESSING: 'ProcessingError',
  MODEL: 'ModelError',
  INTERNAL: 'InternalError',
};

class AppError extends Error {
  constructor(message, type = ErrorTypes.INTERNAL, statusCode = 500) {
    super(message);
    this.name = type;
    this.statusCode = statusCode;
    this.isOperational = true;
  }
}

module.exports = {
  ErrorMessages,
  ErrorTypes,
  AppError,
};
