// src/services/faceRecognitionService.js
/**
 * Face recognition service using ArcFace model
 */

const ort = require('onnxruntime-node');
const config = require('../config');
const { convertToRGB, normalizePixels, convertNHWCtoNCHW, flipHorizontal, extractRegion } = require('../utils/image');
const { l2Normalize } = require('../utils/math');
const Logger = require('../utils/logger');

const logger = new Logger('FaceRecognition');

class FaceRecognitionService {
  constructor() {
    this.session = null;
  }

  /**
   * Load ArcFace model
   */
  async loadModel() {
    this.session = await ort.InferenceSession.create(config.models.arcface);
    logger.info('ArcFace model loaded successfully');
  }

  /**
   * Preprocess face crop for ArcFace inference
   */
  async preprocessFace(buffer, boundingBox) {
    const SIZE = config.arcface.inputSize;

    // Extract and resize face region
    const processed = await extractRegion(buffer, boundingBox, SIZE, SIZE);
    const { data, info } = processed;

    // Convert to RGB
    const rgbData = convertToRGB(data, info.channels, SIZE, SIZE);

    // Normalize pixels
    const normalizedData = normalizePixels(rgbData, config.arcface.normMode);

    // Get channel order
    const channelOrder = config.arcface.channelOrder === 'BGR' ? [2, 1, 0] : [0, 1, 2];

    // Convert to NCHW format
    const nchwData = convertNHWCtoNCHW(normalizedData, SIZE, SIZE, channelOrder);

    return {
      nchw: nchwData,
      nhwc: normalizedData,
      size: SIZE,
    };
  }

  /**
   * Run ArcFace inference on preprocessed data
   */
  async _runInference(nchwData) {
    const SIZE = config.arcface.inputSize;
    const tensor = new ort.Tensor('float32', nchwData, [1, 3, SIZE, SIZE]);

    logger.debug(
      `ArcFace input: shape=[${tensor.dims}], range=[${Math.min(...nchwData).toFixed(4)}, ${Math.max(...nchwData).toFixed(4)}]`
    );

    const output = await this.session.run({ [this.session.inputNames[0]]: tensor });
    const embeddingTensor = output[this.session.outputNames[0]];

    return Array.from(embeddingTensor.data || embeddingTensor);
  }

  /**
   * Generate face embedding with optional Test-Time Augmentation (TTA)
   */
  async generateEmbedding(preprocessedData) {
    const { nchw, nhwc, size } = preprocessedData;

    // Run inference on original image
    let embedding = await this._runInference(nchw);

    // Apply horizontal flip TTA if enabled
    if (config.arcface.useTTAFlip && nhwc) {
      const flipped = flipHorizontal(nhwc, size, size);
      const channelOrder = config.arcface.channelOrder === 'BGR' ? [2, 1, 0] : [0, 1, 2];
      const flippedNCHW = convertNHWCtoNCHW(flipped, size, size, channelOrder);
      const flippedEmbedding = await this._runInference(flippedNCHW);

      // Average original and flipped embeddings
      for (let i = 0; i < embedding.length && i < flippedEmbedding.length; i++) {
        embedding[i] = 0.5 * (embedding[i] + flippedEmbedding[i]);
      }

      logger.debug('Applied TTA flip augmentation');
    }

    // L2 normalize
    const normalized = l2Normalize(embedding);

    // Verify normalization
    let norm = 0;
    for (const v of normalized) norm += v * v;
    norm = Math.sqrt(norm);

    logger.debug(
      `Generated embedding: dim=${normalized.length}, norm=${norm.toFixed(6)}, first5=[${normalized.slice(0, 5).map(v => v.toFixed(4)).join(', ')}]`
    );

    return normalized;
  }

  /**
   * Full pipeline: preprocess and generate embedding
   */
  async encodeface(buffer, boundingBox) {
    const preprocessed = await this.preprocessFace(buffer, boundingBox);
    const embedding = await this.generateEmbedding(preprocessed);

    if (!embedding || embedding.length !== config.faceRecognition.embeddingDimension) {
      throw new Error('Failed to generate valid embedding');
    }

    return embedding;
  }
}

module.exports = FaceRecognitionService;
