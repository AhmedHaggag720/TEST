// src/services/faceComparisonService.js
/**
 * Face comparison service
 * Handles similarity calculation and matching logic
 */

const config = require('../config');
const { cosineSimilarity, euclideanDistance } = require('../utils/math');
const { AppError, ErrorMessages, ErrorTypes } = require('../constants/errors');
const Logger = require('../utils/logger');

const logger = new Logger('FaceComparison');

class FaceComparisonService {
  /**
   * Compare two face embeddings
   */
  static compareEmbeddings(embedding1, embedding2) {
    const similarity = cosineSimilarity(embedding1, embedding2);

    if (similarity === null) {
      throw new AppError(
        ErrorMessages.SIMILARITY_CALCULATION_FAILED,
        ErrorTypes.PROCESSING,
        500
      );
    }

    const threshold = config.faceRecognition.similarityThreshold;
    const isMatch = similarity >= threshold;

    logger.debug(`Similarity: ${similarity.toFixed(4)}, Threshold: ${threshold}, Match: ${isMatch}`);

    return {
      similarity: parseFloat(similarity.toFixed(4)),
      isMatch,
      threshold,
    };
  }

  /**
   * Calculate multiple similarity metrics
   */
  static calculateMetrics(embedding1, embedding2) {
    const cosine = cosineSimilarity(embedding1, embedding2);
    const euclidean = euclideanDistance(embedding1, embedding2);

    return {
      cosineSimilarity: cosine !== null ? parseFloat(cosine.toFixed(4)) : null,
      euclideanDistance: euclidean !== null ? parseFloat(euclidean.toFixed(4)) : null,
    };
  }

  /**
   * Detailed comparison with diagnostics
   */
  static detailedComparison(embedding1, embedding2) {
    const comparison = this.compareEmbeddings(embedding1, embedding2);
    const metrics = this.calculateMetrics(embedding1, embedding2);

    // Calculate embedding statistics
    const stats1 = this._calculateStats(embedding1);
    const stats2 = this._calculateStats(embedding2);

    logger.info('=== COMPARISON RESULT ===');
    logger.info(`Similarity: ${comparison.similarity}`);
    logger.info(`Threshold: ${comparison.threshold}`);
    logger.info(`Is Match: ${comparison.isMatch}`);
    logger.info(`Embedding 1: range=[${stats1.min.toFixed(4)}, ${stats1.max.toFixed(4)}], mean=${stats1.mean.toFixed(4)}`);
    logger.info(`Embedding 2: range=[${stats2.min.toFixed(4)}, ${stats2.max.toFixed(4)}], mean=${stats2.mean.toFixed(4)}`);
    logger.info('========================');

    return {
      ...comparison,
      metrics,
      stats: { embedding1: stats1, embedding2: stats2 },
    };
  }

  /**
   * Calculate embedding statistics
   */
  static _calculateStats(embedding) {
    const min = Math.min(...embedding);
    const max = Math.max(...embedding);
    const sum = embedding.reduce((a, b) => a + b, 0);
    const mean = sum / embedding.length;

    return { min, max, mean };
  }
}

module.exports = FaceComparisonService;
