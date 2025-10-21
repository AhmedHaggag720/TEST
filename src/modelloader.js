// src/modelLoader.js
const ort = require('onnxruntime-node');
const sharp = require('sharp');
const path = require('path');

// ==================== Configuration ====================

ort.env.logLevel = 'error'; // Suppress ONNX warnings

const MODEL_PATHS = {
  arcface: path.resolve(__dirname, '../models/arcface.onnx'),
  blazeface: path.resolve(__dirname, '../models/blazeface.onnx')
};

const DETECTION_CONFIG = {
  targetSize: 128,
  confidenceThreshold: 0.5,
  nmsThreshold: 0.5,
  minFaceSize: 0.02,  // 2% of image
  maxFaceSize: 0.8    // 80% of image
};

const ARCFACE_CONFIG = {
  inputSize: 112,
  channelOrder: (process.env.ARC_CHANNEL_ORDER || 'RGB').toUpperCase(),
  normMode: (process.env.ARC_NORM_MODE || 'INSIGHT').toUpperCase() // INSIGHT | ZERO_ONE | TORCH
};

// ==================== Model Sessions ====================

let arcSession = null;
let blazeSession = null;

/**
 * Load ONNX models into memory
 */
async function loadModels() {
  arcSession = await ort.InferenceSession.create(MODEL_PATHS.arcface);
  blazeSession = await ort.InferenceSession.create(MODEL_PATHS.blazeface);
  console.log('✅ ONNX models loaded successfully');
}

// ==================== Helper Functions ====================

/**
 * Convert buffer to RGB (strip alpha or expand grayscale)
 */
function normalizeToRGB(data, channels, width, height) {
  if (channels === 3) return data;
  
  const rgbSize = width * height * 3;
  const rgb = Buffer.alloc(rgbSize);
  
  if (channels === 4) {
    // Strip alpha channel
    for (let i = 0, j = 0; i < data.length; i += 4, j += 3) {
      rgb[j] = data[i];
      rgb[j + 1] = data[i + 1];
      rgb[j + 2] = data[i + 2];
    }
  } else if (channels === 1) {
    // Expand grayscale to RGB
    for (let i = 0, j = 0; i < data.length; i++, j += 3) {
      const val = data[i];
      rgb[j] = val;
      rgb[j + 1] = val;
      rgb[j + 2] = val;
    }
  }
  
  return rgb;
}

/**
 * Normalize pixel values to [-1, 1] range
 */
function normalizePixels(pixelData) {
  const floatData = new Float32Array(pixelData.length);
  for (let i = 0; i < pixelData.length; i++) {
    floatData[i] = (pixelData[i] - 127.5) / 128.0;
  }
  return floatData;
}

/**
 * Map coordinates from letterboxed space to original image normalized coords
 */
function mapFromLetterbox(bx, by, bw, bh, targetSize, leftPad, topPad, scale, origWidth, origHeight) {
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
    height: Math.min(1, hNorm)
  };
}

/**
 * Non-Maximum Suppression to filter overlapping boxes
 */
function applyNMS(boxes, threshold = DETECTION_CONFIG.nmsThreshold) {
  const kept = [];
  const sorted = boxes.slice().sort((a, b) => b.score - a.score);
  
  const calculateIOU = (a, b) => {
    const x1 = Math.max(a.x, b.x);
    const y1 = Math.max(a.y, b.y);
    const x2 = Math.min(a.x + a.width, b.x + b.width);
    const y2 = Math.min(a.y + a.height, b.y + b.height);
    const w = Math.max(0, x2 - x1);
    const h = Math.max(0, y2 - y1);
    const intersection = w * h;
    const union = a.width * a.height + b.width * b.height - intersection;
    return union <= 0 ? 0 : intersection / union;
  };
  
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
 * Generate anchor boxes for BlazeFace detection
 */
function generateAnchors(targetSize) {
  const anchors = [];
  const featureMaps = [
    { size: 16, stride: 8, anchorsPerCell: 2 },
    { size: 8, stride: 16, anchorsPerCell: 6 }
  ];
  
  for (const fm of featureMaps) {
    for (let y = 0; y < fm.size; y++) {
      for (let x = 0; x < fm.size; x++) {
        for (let a = 0; a < fm.anchorsPerCell; a++) {
          anchors.push({
            cx: ((x + 0.5) * fm.stride) / targetSize,
            cy: ((y + 0.5) * fm.stride) / targetSize,
            w: fm.stride / targetSize,
            h: fm.stride / targetSize
          });
        }
      }
    }
  }
  
  return anchors;
}

/**
 * Decode BlazeFace anchor-based predictions
 */
function decodeAnchors(regressors, classificators, anchors, targetSize, mapping) {
  const regStride = 16;  // 4 box coords + 12 keypoints
  const clsStride = classificators.length / anchors.length;
  const detections = [];
  
  for (let i = 0; i < anchors.length; i++) {
    // Calculate confidence score
    let score = 0;
    if (clsStride === 1) {
      // Single-class sigmoid
      const logit = classificators[i];
      score = 1 / (1 + Math.exp(-logit));
    } else {
      // Two-class softmax
      const l0 = classificators[i * 2];
      const l1 = classificators[i * 2 + 1];
      const m = Math.max(l0, l1);
      const e0 = Math.exp(l0 - m);
      const e1 = Math.exp(l1 - m);
      score = e1 / (e0 + e1);
    }
    
    if (score < DETECTION_CONFIG.confidenceThreshold) continue;
    
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
    
    // Map to original image coordinates
    const mapped = mapping(cx, cy, w, h);
    
    // Filter implausible sizes
    if (mapped.width <= 0 || mapped.height <= 0) continue;
    if (mapped.width < DETECTION_CONFIG.minFaceSize || mapped.width > DETECTION_CONFIG.maxFaceSize) continue;
    if (mapped.height < DETECTION_CONFIG.minFaceSize || mapped.height > DETECTION_CONFIG.maxFaceSize) continue;
    
    detections.push({ ...mapped, score });
  }
  
  return detections;
}

// ==================== Face Detection ====================

/**
 * Detect faces in an image buffer
 * Returns normalized bounding boxes [0,1] relative to original image
 */
async function detectFaces(buffer) {
  const TARGET = DETECTION_CONFIG.targetSize;
  
  // Prepare letterboxed input
  const sharpImg = sharp(buffer).rotate();
  const meta = await sharpImg.metadata();
  const origWidth = meta.width;
  const origHeight = meta.height;
  
  const scale = Math.min(TARGET / origWidth, TARGET / origHeight);
  const newWidth = Math.round(origWidth * scale);
  const newHeight = Math.round(origHeight * scale);
  const leftPad = Math.floor((TARGET - newWidth) / 2);
  const topPad = Math.floor((TARGET - newHeight) / 2);
  
  // Create letterboxed image
  const resized = await sharpImg
    .resize(newWidth, newHeight)
    .extend({
      top: topPad,
      bottom: TARGET - newHeight - topPad,
      left: leftPad,
      right: TARGET - newWidth - leftPad,
      background: { r: 0, g: 0, b: 0 }
    })
    .removeAlpha()
    .raw()
    .toBuffer({ resolveWithObject: true });
  
  // Normalize to RGB
  let pixelData = normalizeToRGB(resized.data, resized.info.channels, TARGET, TARGET);
  
  // Normalize pixel values and create tensor
  const floatData = normalizePixels(pixelData);
  const inputTensor = new ort.Tensor('float32', floatData, [1, TARGET, TARGET, 3]);
  
  // Prepare feeds with optional inputs
  const feeds = { [blazeSession.inputNames[0]]: inputTensor };
  if (blazeSession.inputNames.includes('conf_threshold')) {
    feeds['conf_threshold'] = new ort.Tensor('float32', new Float32Array([0.2]), [1]);
  }
  if (blazeSession.inputNames.includes('iou_threshold')) {
    feeds['iou_threshold'] = new ort.Tensor('float32', new Float32Array([0.45]), [1]);
  }
  
  // Run inference
  const output = await blazeSession.run(feeds);
  
  // Mapping function for coordinate conversion
  const mapCoords = (bx, by, bw, bh) => 
    mapFromLetterbox(bx, by, bw, bh, TARGET, leftPad, topPad, scale, origWidth, origHeight);
  
  // Try different output formats
  // Format 1: Direct boxes and scores
  const boxesOutput = Object.keys(output).find(k => k.toLowerCase().includes('box'));
  const scoresOutput = Object.keys(output).find(k => k.toLowerCase().includes('score') || k.toLowerCase().includes('conf'));
  
  if (boxesOutput && scoresOutput) {
    const boxData = Array.from(output[boxesOutput].data);
    const scoreData = Array.from(output[scoresOutput].data);
    const numBoxes = Math.floor(boxData.length / 4);
    
    const detections = [];
    for (let i = 0; i < numBoxes; i++) {
      const x1 = boxData[i * 4];
      const y1 = boxData[i * 4 + 1];
      const x2 = boxData[i * 4 + 2];
      const y2 = boxData[i * 4 + 3];
      const cx = (x1 + x2) / 2;
      const cy = (y1 + y2) / 2;
      const w = Math.max(0, x2 - x1);
      const h = Math.max(0, y2 - y1);
      
      const mapped = mapCoords(cx, cy, w, h);
      detections.push({ ...mapped, score: scoreData[i] ?? 1.0 });
    }
    
    return { boxes: detections.sort((a, b) => b.score - a.score) };
  }
  
  // Format 2: Anchor-based (regressors + classificators)
  if (output['regressors'] && output['classificators']) {
    const regressors = Array.from(output['regressors'].data);
    const classificators = Array.from(output['classificators'].data);
    const anchors = generateAnchors(TARGET);
    
    const detections = decodeAnchors(regressors, classificators, anchors, TARGET, mapCoords);
    const kept = applyNMS(detections);
    
    console.log(`Detected ${detections.length} faces -> kept ${kept.length} after NMS`);
    return { boxes: kept.sort((a, b) => b.score - a.score) };
  }
  
  console.log('⚠️ No recognizable BlazeFace output format');
  return { boxes: [] };
}

// ==================== Face Preprocessing ====================

/**
 * Preprocess face crop for ArcFace model
 * Expects pixel coordinates: {x, y, width, height}
 */
async function preprocessFaceCrop(buffer, box) {
  const SIZE = ARCFACE_CONFIG.inputSize;
  
  // Extract and resize face region
  const processed = await sharp(buffer)
    .rotate()
    .extract({
      left: Math.max(Math.round(box.x), 0),
      top: Math.max(Math.round(box.y), 0),
      width: Math.max(Math.round(box.width), 1),
      height: Math.max(Math.round(box.height), 1)
    })
    .resize(SIZE, SIZE)
    .normalize() // improve local contrast for low-light/washed images
    .raw()
    .toBuffer({ resolveWithObject: true });
  
  // Normalize to RGB
  let data = normalizeToRGB(processed.data, processed.info.channels, SIZE, SIZE);
  
  // Normalize pixel values according to selected mode
  let floatData;
  switch (ARCFACE_CONFIG.normMode) {
    case 'ZERO_ONE':
      floatData = new Float32Array(SIZE * SIZE * 3);
      for (let i = 0; i < floatData.length; i++) floatData[i] = data[i] / 255.0;
      break;
    case 'TORCH': {
      // x/255 then (x-mean)/std with mean=0.5, std=0.5 per channel
      const tmp = new Float32Array(SIZE * SIZE * 3);
      for (let i = 0; i < tmp.length; i++) tmp[i] = data[i] / 255.0;
      // (x-0.5)/0.5 = 2x - 1
      floatData = new Float32Array(SIZE * SIZE * 3);
      for (let i = 0; i < tmp.length; i++) floatData[i] = 2.0 * tmp[i] - 1.0;
      break;
    }
    case 'INSIGHT':
    default:
      floatData = normalizePixels(data); // (x - 127.5)/128.0
      break;
  }
  
  // Convert NHWC to NCHW format
  const channelOrder = ARCFACE_CONFIG.channelOrder === 'BGR' ? [2, 1, 0] : [0, 1, 2];
  const nchw = new Float32Array(3 * SIZE * SIZE);
  
  let idx = 0;
  for (const c of channelOrder) {
    for (let y = 0; y < SIZE; y++) {
      for (let x = 0; x < SIZE; x++) {
        nchw[idx++] = floatData[(y * SIZE + x) * 3 + c];
      }
    }
  }
  
  return { 
    tensorDataNCHW: nchw, 
    tensorDataNHWC: floatData, 
    width: SIZE, 
    height: SIZE, 
    channels: 3 
  };
}

// ==================== Face Encoding ====================

/**
 * Generate face embedding using ArcFace model
 * Returns L2-normalized 512-dimensional vector
 */
async function runArcFace(tensorDataNCHW, tensorDataNHWC) {
  const SIZE = ARCFACE_CONFIG.inputSize;
  
  // Create input tensor [1, 3, 112, 112]
  const tensor = new ort.Tensor('float32', tensorDataNCHW, [1, 3, SIZE, SIZE]);
  
  console.log(`🔍 ArcFace input tensor: shape=[${tensor.dims}], data range=[${Math.min(...tensorDataNCHW).toFixed(4)}, ${Math.max(...tensorDataNCHW).toFixed(4)}]`);
  
  // Run inference (with optional horizontal flip TTA)
  const runOnce = async (nchw) => {
    const t = new ort.Tensor('float32', nchw, [1, 3, SIZE, SIZE]);
    const out = await arcSession.run({ [arcSession.inputNames[0]]: t });
    const embT = out[arcSession.outputNames[0]];
    return Array.from(embT.data || embT);
  };

  let emb = await runOnce(tensorDataNCHW);

  const useFlip = String(process.env.ARC_TTA_FLIP || 'true').toLowerCase() === 'true';
  if (useFlip && tensorDataNHWC && tensorDataNHWC.length === SIZE * SIZE * 3) {
    // Build horizontally flipped NHWC then convert to NCHW with channel order
    const flippedNHWC = new Float32Array(SIZE * SIZE * 3);
    for (let y = 0; y < SIZE; y++) {
      for (let x = 0; x < SIZE; x++) {
        for (let c = 0; c < 3; c++) {
          const srcIdx = (y * SIZE + x) * 3 + c;
          const dstIdx = (y * SIZE + (SIZE - 1 - x)) * 3 + c;
          flippedNHWC[dstIdx] = tensorDataNHWC[srcIdx];
        }
      }
    }
    const channelOrder = ARCFACE_CONFIG.channelOrder === 'BGR' ? [2, 1, 0] : [0, 1, 2];
    const flippedNCHW = new Float32Array(3 * SIZE * SIZE);
    let idx = 0;
    for (const c of channelOrder) {
      for (let y = 0; y < SIZE; y++) {
        for (let x = 0; x < SIZE; x++) {
          flippedNCHW[idx++] = flippedNHWC[(y * SIZE + x) * 3 + c];
        }
      }
    }
    const embFlip = await runOnce(flippedNCHW);
    // Average original and flipped embeddings
    for (let i = 0; i < emb.length && i < embFlip.length; i++) {
      emb[i] = 0.5 * (emb[i] + embFlip[i]);
    }
  }

  // L2-normalize averaged embedding
  let norm2 = 0;
  for (const val of emb) norm2 += val * val;
  norm2 = Math.sqrt(norm2) || 1.0;
  let embedding = emb.map(v => v / norm2);

  // Diagnostics
  console.log(`📊 Raw embedding stats: length=${emb.length}, pre-norm=${norm2.toFixed(4)}`);
  let check = 0; for (const v of embedding) check += v * v; check = Math.sqrt(check);
  console.log(`✅ Generated embedding: post-norm=${check.toFixed(6)}, first5=[${embedding.slice(0,5).map(v=>v.toFixed(4)).join(', ')}]`);

  return embedding;
}

// ==================== Exports ====================

module.exports = { 
  loadModels, 
  detectFaces, 
  preprocessFaceCrop, 
  runArcFace 
};
