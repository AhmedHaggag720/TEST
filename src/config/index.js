// src/config/index.js
/**
 * Centralized configuration management
 * All environment variables and constants in one place
 */

const path = require('path');

const config = {
  // Server Configuration
  server: {
    port: parseInt(process.env.PORT || '3000', 10),
    env: process.env.NODE_ENV || 'development',
    maxFileSize: 5 * 1024 * 1024, // 5MB
  },

  // Face Detection Configuration
  faceDetection: {
    targetSize: 128,
    confidenceThreshold: 0.5,
    nmsThreshold: 0.5,
    minFaceSize: 0.02, // 2% of image
    maxFaceSize: 0.8,  // 80% of image
    minPixelSize: 20,
    edgeMarginRatio: 0.05,
    aspectRatioMin: 0.6,
    aspectRatioMax: 1.8,
  },

  // Face Recognition Configuration
  faceRecognition: {
    similarityThreshold: parseFloat(process.env.SIM_THRESHOLD || '0.45'),
    faceExpandRatio: parseFloat(process.env.FACE_EXPAND_RATIO || '0.10'),
    maxExpandRatio: 0.20,
    useSquareCrop: String(process.env.SQUARE_CROP || 'true').toLowerCase() !== 'false',
    embeddingDimension: 512,
  },

  // ArcFace Model Configuration
  arcface: {
    inputSize: 112,
    channelOrder: (process.env.ARC_CHANNEL_ORDER || 'RGB').toUpperCase(),
    normMode: (process.env.ARC_NORM_MODE || 'INSIGHT').toUpperCase(),
    useTTAFlip: String(process.env.ARC_TTA_FLIP || 'true').toLowerCase() === 'true',
  },

  // Model Paths
  models: {
    arcface: path.resolve(__dirname, '../../models/arcface.onnx'),
    blazeface: path.resolve(__dirname, '../../models/blazeface.onnx'),
  },

  // Database Configuration
  database: {
    connectionString: process.env.DATABASE_URL || 'postgresql://localhost/face_verif_db',
  },

  // Scoring Weights for Face Selection
  faceScoring: {
    detectionConfidence: 0.65,
    centrality: 0.25,
    sizeRatio: 0.10,
  },
};

module.exports = config;
