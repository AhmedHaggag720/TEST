// src/utils/image.js
/**
 * Image processing utilities
 */

const sharp = require('sharp');

/**
 * Convert buffer to RGB format (strip alpha or expand grayscale)
 * @param {Buffer} data - Raw pixel data
 * @param {number} channels - Number of channels
 * @param {number} width - Image width
 * @param {number} height - Image height
 * @returns {Buffer} RGB buffer
 */
function convertToRGB(data, channels, width, height) {
  if (channels === 3) {
    return data;
  }

  const rgbSize = width * height * 3;
  const rgb = Buffer.alloc(rgbSize);

  if (channels === 4) {
    // Strip alpha channel (RGBA -> RGB)
    for (let i = 0, j = 0; i < data.length; i += 4, j += 3) {
      rgb[j] = data[i];
      rgb[j + 1] = data[i + 1];
      rgb[j + 2] = data[i + 2];
    }
  } else if (channels === 1) {
    // Expand grayscale to RGB
    for (let i = 0, j = 0; i < data.length; i++, j += 3) {
      const value = data[i];
      rgb[j] = value;
      rgb[j + 1] = value;
      rgb[j + 2] = value;
    }
  }

  return rgb;
}

/**
 * Normalize pixel values to specified range
 * @param {Buffer} pixelData - Raw pixel data (0-255)
 * @param {string} mode - Normalization mode: 'INSIGHT', 'ZERO_ONE', 'TORCH'
 * @returns {Float32Array} Normalized pixel values
 */
function normalizePixels(pixelData, mode = 'INSIGHT') {
  const floatData = new Float32Array(pixelData.length);

  switch (mode) {
    case 'ZERO_ONE':
      // Normalize to [0, 1]
      for (let i = 0; i < pixelData.length; i++) {
        floatData[i] = pixelData[i] / 255.0;
      }
      break;

    case 'TORCH':
      // PyTorch style: (x/255 - 0.5) / 0.5 = 2*(x/255) - 1
      for (let i = 0; i < pixelData.length; i++) {
        floatData[i] = 2.0 * (pixelData[i] / 255.0) - 1.0;
      }
      break;

    case 'INSIGHT':
    default:
      // InsightFace style: (x - 127.5) / 128.0
      for (let i = 0; i < pixelData.length; i++) {
        floatData[i] = (pixelData[i] - 127.5) / 128.0;
      }
      break;
  }

  return floatData;
}

/**
 * Convert NHWC (Height, Width, Channels) to NCHW (Channels, Height, Width) format
 * @param {Float32Array} nhwcData - Data in NHWC format
 * @param {number} height - Image height
 * @param {number} width - Image width
 * @param {Array<number>} channelOrder - Channel order [R, G, B] or [B, G, R]
 * @returns {Float32Array} Data in NCHW format
 */
function convertNHWCtoNCHW(nhwcData, height, width, channelOrder = [0, 1, 2]) {
  const nchwData = new Float32Array(3 * height * width);
  let index = 0;

  for (const channel of channelOrder) {
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        nchwData[index++] = nhwcData[(y * width + x) * 3 + channel];
      }
    }
  }

  return nchwData;
}

/**
 * Flip image horizontally in NHWC format
 * @param {Float32Array} nhwcData - Data in NHWC format
 * @param {number} height - Image height
 * @param {number} width - Image width
 * @returns {Float32Array} Horizontally flipped data
 */
function flipHorizontal(nhwcData, height, width) {
  const flipped = new Float32Array(height * width * 3);

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      for (let c = 0; c < 3; c++) {
        const srcIdx = (y * width + x) * 3 + c;
        const dstIdx = (y * width + (width - 1 - x)) * 3 + c;
        flipped[dstIdx] = nhwcData[srcIdx];
      }
    }
  }

  return flipped;
}

/**
 * Get image metadata
 * @param {Buffer} buffer - Image buffer
 * @returns {Promise<Object>} Image metadata
 */
async function getImageMetadata(buffer) {
  return sharp(buffer).metadata();
}

/**
 * Extract region from image
 * @param {Buffer} buffer - Image buffer
 * @param {Object} region - Region to extract {left, top, width, height}
 * @param {number} resizeWidth - Optional resize width
 * @param {number} resizeHeight - Optional resize height
 * @returns {Promise<Buffer>} Extracted and optionally resized image
 */
async function extractRegion(buffer, region, resizeWidth = null, resizeHeight = null) {
  let pipeline = sharp(buffer)
    .rotate()
    .extract({
      left: Math.max(Math.round(region.left || region.x), 0),
      top: Math.max(Math.round(region.top || region.y), 0),
      width: Math.max(Math.round(region.width), 1),
      height: Math.max(Math.round(region.height), 1),
    });

  if (resizeWidth && resizeHeight) {
    pipeline = pipeline.resize(resizeWidth, resizeHeight);
  }

  return pipeline.normalize().raw().toBuffer({ resolveWithObject: true });
}

module.exports = {
  convertToRGB,
  normalizePixels,
  convertNHWCtoNCHW,
  flipHorizontal,
  getImageMetadata,
  extractRegion,
};
