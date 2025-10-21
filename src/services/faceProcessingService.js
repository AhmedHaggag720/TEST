// src/services/faceProcessingService.js
/**
 * High-level face processing service
 * Combines detection, selection, and cropping logic
 */

const sharp = require('sharp');
const config = require('../config');
const { AppError, ErrorMessages, ErrorTypes } = require('../constants/errors');
const Logger = require('../utils/logger');
const { clamp } = require('../utils/math');

const logger = new Logger('FaceProcessing');

class FaceProcessingService {
  /**
   * Choose the best face from detection results
   * Filters by size, position, and aspect ratio, then scores by confidence and centrality
   */
  static selectBestFace(boxes, imageWidth, imageHeight) {
    if (!boxes || boxes.length === 0) {
      return null;
    }

    const cfg = config.faceDetection;
    const minSize = Math.max(cfg.minPixelSize, Math.min(imageWidth, imageHeight) * 0.08);
    const maxSize = Math.min(imageWidth, imageHeight) * 0.65;
    const edgeMargin = {
      x: imageWidth * cfg.edgeMarginRatio,
      y: imageHeight * cfg.edgeMarginRatio,
    };

    const candidates = boxes
      .map(box => {
        // Convert normalized coordinates to pixels
        const x = Math.max(0, Math.round(box.x * imageWidth));
        const y = Math.max(0, Math.round(box.y * imageHeight));
        let width = Math.max(1, Math.round(box.width * imageWidth));
        let height = Math.max(1, Math.round(box.height * imageHeight));

        // Clamp to image bounds
        if (x + width > imageWidth) width = imageWidth - x;
        if (y + height > imageHeight) height = imageHeight - y;

        return { x, y, width, height, score: box.score ?? 0.5 };
      })
      .filter(box => {
        // Size filters
        if (box.width < minSize || box.height < minSize) return false;
        if (box.width > maxSize || box.height > maxSize) return false;

        // Edge filter - avoid faces touching edges
        if (box.x < edgeMargin.x || box.y < edgeMargin.y) return false;
        if (box.x + box.width > imageWidth - edgeMargin.x) return false;
        if (box.y + box.height > imageHeight - edgeMargin.y) return false;

        // Aspect ratio filter
        const aspectRatio = box.width / box.height;
        return aspectRatio >= cfg.aspectRatioMin && aspectRatio <= cfg.aspectRatioMax;
      })
      .map(box => {
        // Calculate composite score
        const centerX = box.x + box.width / 2;
        const centerY = box.y + box.height / 2;
        const dx = Math.abs(centerX - imageWidth / 2) / (imageWidth / 2);
        const dy = Math.abs(centerY - imageHeight / 2) / (imageHeight / 2);
        const centrality = 1 - Math.min(1, Math.sqrt(dx * dx + dy * dy));
        const sizeRatio = Math.min(1, (box.width * box.height) / (imageWidth * imageHeight));

        const weights = config.faceScoring;
        const compositeScore =
          weights.detectionConfidence * box.score +
          weights.centrality * centrality +
          weights.sizeRatio * sizeRatio;

        return { ...box, compositeScore };
      });

    if (candidates.length === 0) {
      return null;
    }

    // Sort by composite score and return best
    candidates.sort((a, b) => b.compositeScore - a.compositeScore);
    return candidates[0];
  }

  /**
   * Expand face bounding box by a ratio
   */
  static expandBoundingBox(box, imageWidth, imageHeight, expandRatio) {
    const expandX = Math.round(box.width * expandRatio / 2);
    const expandY = Math.round(box.height * expandRatio / 2);

    return {
      x: clamp(box.x - expandX, 0, imageWidth),
      y: clamp(box.y - expandY, 0, imageHeight),
      width: Math.min(imageWidth - (box.x - expandX), box.width + 2 * expandX),
      height: Math.min(imageHeight - (box.y - expandY), box.height + 2 * expandY),
    };
  }

  /**
   * Make bounding box square (center-preserving)
   */
  static makeSquareBoundingBox(box, imageWidth, imageHeight) {
    const centerX = box.x + box.width / 2;
    const centerY = box.y + box.height / 2;
    const side = Math.round(Math.max(box.width, box.height));

    let x = Math.round(centerX - side / 2);
    let y = Math.round(centerY - side / 2);

    // Clamp to image bounds
    x = clamp(x, 0, imageWidth - side);
    y = clamp(y, 0, imageHeight - side);

    return {
      x: Math.max(0, x),
      y: Math.max(0, y),
      width: Math.min(side, imageWidth),
      height: Math.min(side, imageHeight),
    };
  }

  /**
   * Process bounding box (expand and optionally make square)
   */
  static processBoundingBox(box, imageWidth, imageHeight) {
    const cfg = config.faceRecognition;
    
    // Cap expand ratio
    const expandRatio = Math.min(cfg.maxExpandRatio, cfg.faceExpandRatio);
    
    // Expand box
    let processedBox = this.expandBoundingBox(box, imageWidth, imageHeight, expandRatio);

    // Make square if enabled
    if (cfg.useSquareCrop) {
      processedBox = this.makeSquareBoundingBox(processedBox, imageWidth, imageHeight);
    }

    // Validate minimum size
    if (processedBox.width < cfg.minPixelSize || processedBox.height < cfg.minPixelSize) {
      throw new AppError(
        ErrorMessages.FACE_TOO_SMALL,
        ErrorTypes.DETECTION,
        400
      );
    }

    logger.debug(
      `Processed box: (${processedBox.x}, ${processedBox.y}) ${processedBox.width}x${processedBox.height}`
    );

    return processedBox;
  }

  /**
   * Extract face crop from image as PNG
   */
  static async extractFaceCrop(buffer, box) {
    const decoded = await sharp(buffer).rotate().raw().toBuffer({ resolveWithObject: true });
    let { data, info } = decoded;

    // Ensure RGB (strip alpha if present)
    if (info.channels === 4) {
      const rgb = Buffer.alloc(info.width * info.height * 3);
      for (let i = 0, j = 0; i < data.length; i += 4, j += 3) {
        rgb[j] = data[i];
        rgb[j + 1] = data[i + 1];
        rgb[j + 2] = data[i + 2];
      }
      data = rgb;
    }

    return sharp(data, { raw: { width: info.width, height: info.height, channels: 3 } })
      .extract({ left: box.x, top: box.y, width: box.width, height: box.height })
      .resize(112, 112)
      .png({ compressionLevel: 9 })
      .toBuffer();
  }

  /**
   * Detect and validate face in image
   * Returns processed bounding box ready for recognition
   */
  static async detectAndValidateFace(buffer, detectionService) {
    const { boxes, metadata } = await detectionService.detectFaces(buffer);

    if (!boxes || boxes.length === 0) {
      throw new AppError(
        ErrorMessages.NO_FACE_DETECTED,
        ErrorTypes.DETECTION,
        400
      );
    }

    const bestFace = this.selectBestFace(boxes, metadata.width, metadata.height);
    
    if (!bestFace) {
      throw new AppError(
        ErrorMessages.NO_FACE_DETECTED,
        ErrorTypes.DETECTION,
        400
      );
    }

    logger.info(
      `Face detected: score=${bestFace.score.toFixed(3)}, position=(${bestFace.x}, ${bestFace.y}), size=${bestFace.width}x${bestFace.height}`
    );

    // Process bounding box
    const processedBox = this.processBoundingBox(bestFace, metadata.width, metadata.height);

    return processedBox;
  }
}

module.exports = FaceProcessingService;
