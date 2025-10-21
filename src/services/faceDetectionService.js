// src/services/faceDetectionService.js
/**
 * Face detection service using BlazeFace model
 */

const ort = require('onnxruntime-node');
const config = require('../config');
const { convertToRGB, normalizePixels, getImageMetadata } = require('../utils/image');
const { calculateIOU } = require('../utils/math');
const Logger = require('../utils/logger');
const sharp = require('sharp');

const logger = new Logger('FaceDetection');

class FaceDetectionService {
  constructor() {
    this.session = null;
    this.anchors = null;
  }

  /**
   * Load BlazeFace model
   */
  async loadModel() {
    this.session = await ort.InferenceSession.create(config.models.blazeface);
    logger.info('BlazeFace model loaded successfully');
  }

  /**
   * Generate anchor boxes for detection
   */
  _generateAnchors() {
    if (this.anchors) return this.anchors;

    const anchors = [];
    const targetSize = config.faceDetection.targetSize;
    const featureMaps = [
      { size: 16, stride: 8, anchorsPerCell: 2 },
      { size: 8, stride: 16, anchorsPerCell: 6 },
    ];

    for (const fm of featureMaps) {
      for (let y = 0; y < fm.size; y++) {
        for (let x = 0; x < fm.size; x++) {
          for (let a = 0; a < fm.anchorsPerCell; a++) {
            anchors.push({
              cx: ((x + 0.5) * fm.stride) / targetSize,
              cy: ((y + 0.5) * fm.stride) / targetSize,
              w: fm.stride / targetSize,
              h: fm.stride / targetSize,
            });
          }
        }
      }
    }

    this.anchors = anchors;
    return anchors;
  }

  /**
   * Apply Non-Maximum Suppression to filter overlapping boxes
   */
  _applyNMS(boxes, threshold = config.faceDetection.nmsThreshold) {
    const kept = [];
    const sorted = boxes.slice().sort((a, b) => b.score - a.score);

    while (sorted.length) {
      const current = sorted.shift();
      kept.push(current);

      // Remove overlapping boxes
      for (let i = sorted.length - 1; i >= 0; i--) {
        if (calculateIOU(current, sorted[i]) > threshold) {
          sorted.splice(i, 1);
        }
      }
    }

    return kept;
  }

  /**
   * Map coordinates from letterboxed space to original image
   */
  _mapFromLetterbox(bx, by, bw, bh, targetSize, leftPad, topPad, scale, origWidth, origHeight) {
    const px = bx * targetSize;
    const py = by * targetSize;
    const pw = bw * targetSize;
    const ph = bh * targetSize;

    const xTop = Math.max(0, ((px - pw / 2) - leftPad) / (scale * origWidth));
    const yTop = Math.max(0, ((py - ph / 2) - topPad) / (scale * origHeight));
    const wNorm = Math.max(0, (pw / scale) / origWidth);
    const hNorm = Math.max(0, (ph / scale) / origHeight);

    return {
      x: Math.min(1, xTop),
      y: Math.min(1, yTop),
      width: Math.min(1, wNorm),
      height: Math.min(1, hNorm),
    };
  }

  /**
   * Decode anchor-based predictions
   */
  _decodeAnchors(regressors, classificators, targetSize, mapFunc) {
    const anchors = this._generateAnchors();
    const regStride = 16;
    const clsStride = classificators.length / anchors.length;
    const detections = [];

    for (let i = 0; i < anchors.length; i++) {
      // Calculate confidence score
      let score = 0;
      if (clsStride === 1) {
        const logit = classificators[i];
        score = 1 / (1 + Math.exp(-logit));
      } else {
        const l0 = classificators[i * 2];
        const l1 = classificators[i * 2 + 1];
        const m = Math.max(l0, l1);
        const e0 = Math.exp(l0 - m);
        const e1 = Math.exp(l1 - m);
        score = e1 / (e0 + e1);
      }

      if (score < config.faceDetection.confidenceThreshold) continue;

      // Decode box coordinates
      const offset = i * regStride;
      if (offset + 3 >= regressors.length) continue;

      const dx = regressors[offset];
      const dy = regressors[offset + 1];
      const dw = regressors[offset + 2];
      const dh = regressors[offset + 3];

      const cx = anchors[i].cx + dx / targetSize;
      const cy = anchors[i].cy + dy / targetSize;
      const w = Math.abs(dw) / targetSize;
      const h = Math.abs(dh) / targetSize;

      const mapped = mapFunc(cx, cy, w, h);

      // Filter implausible sizes
      if (mapped.width <= 0 || mapped.height <= 0) continue;
      if (mapped.width < config.faceDetection.minFaceSize || mapped.width > config.faceDetection.maxFaceSize) continue;
      if (mapped.height < config.faceDetection.minFaceSize || mapped.height > config.faceDetection.maxFaceSize) continue;

      detections.push({ ...mapped, score });
    }

    return detections;
  }

  /**
   * Prepare letterboxed input for detection
   */
  async _prepareInput(buffer) {
    const TARGET = config.faceDetection.targetSize;
    const sharpImg = sharp(buffer).rotate();
    const meta = await sharpImg.metadata();

    const scale = Math.min(TARGET / meta.width, TARGET / meta.height);
    const newWidth = Math.round(meta.width * scale);
    const newHeight = Math.round(meta.height * scale);
    const leftPad = Math.floor((TARGET - newWidth) / 2);
    const topPad = Math.floor((TARGET - newHeight) / 2);

    const resized = await sharpImg
      .resize(newWidth, newHeight)
      .extend({
        top: topPad,
        bottom: TARGET - newHeight - topPad,
        left: leftPad,
        right: TARGET - newWidth - leftPad,
        background: { r: 0, g: 0, b: 0 },
      })
      .removeAlpha()
      .raw()
      .toBuffer({ resolveWithObject: true });

    const pixelData = convertToRGB(resized.data, resized.info.channels, TARGET, TARGET);
    const floatData = normalizePixels(pixelData, 'INSIGHT');

    return {
      tensor: new ort.Tensor('float32', floatData, [1, TARGET, TARGET, 3]),
      metadata: meta,
      transform: { scale, leftPad, topPad, targetSize: TARGET },
    };
  }

  /**
   * Detect faces in an image buffer
   */
  async detectFaces(buffer) {
    const { tensor, metadata, transform } = await this._prepareInput(buffer);

    // Prepare feeds
    const feeds = { [this.session.inputNames[0]]: tensor };
    if (this.session.inputNames.includes('conf_threshold')) {
      feeds['conf_threshold'] = new ort.Tensor('float32', new Float32Array([0.2]), [1]);
    }
    if (this.session.inputNames.includes('iou_threshold')) {
      feeds['iou_threshold'] = new ort.Tensor('float32', new Float32Array([0.45]), [1]);
    }

    // Run inference
    const output = await this.session.run(feeds);

    // Mapping function
    const mapFunc = (cx, cy, w, h) =>
      this._mapFromLetterbox(
        cx, cy, w, h,
        transform.targetSize,
        transform.leftPad,
        transform.topPad,
        transform.scale,
        metadata.width,
        metadata.height
      );

    // Try different output formats
    const boxesOutput = Object.keys(output).find(k => k.toLowerCase().includes('box'));
    const scoresOutput = Object.keys(output).find(k => k.toLowerCase().includes('score') || k.toLowerCase().includes('conf'));

    let detections = [];

    if (boxesOutput && scoresOutput) {
      // Direct boxes and scores format
      const boxData = Array.from(output[boxesOutput].data);
      const scoreData = Array.from(output[scoresOutput].data);
      const numBoxes = Math.floor(boxData.length / 4);

      for (let i = 0; i < numBoxes; i++) {
        const x1 = boxData[i * 4];
        const y1 = boxData[i * 4 + 1];
        const x2 = boxData[i * 4 + 2];
        const y2 = boxData[i * 4 + 3];
        const cx = (x1 + x2) / 2;
        const cy = (y1 + y2) / 2;
        const w = Math.max(0, x2 - x1);
        const h = Math.max(0, y2 - y1);

        const mapped = mapFunc(cx, cy, w, h);
        detections.push({ ...mapped, score: scoreData[i] ?? 1.0 });
      }
    } else if (output['regressors'] && output['classificators']) {
      // Anchor-based format
      const regressors = Array.from(output['regressors'].data);
      const classificators = Array.from(output['classificators'].data);
      detections = this._decodeAnchors(regressors, classificators, transform.targetSize, mapFunc);
    } else {
      logger.warn('No recognizable BlazeFace output format');
      return { boxes: [], metadata };
    }

    const kept = this._applyNMS(detections);
    logger.debug(`Detected ${detections.length} faces, kept ${kept.length} after NMS`);

    return {
      boxes: kept.sort((a, b) => b.score - a.score),
      metadata,
    };
  }
}

module.exports = FaceDetectionService;
