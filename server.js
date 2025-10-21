// src/server.js
const express = require('express');
const multer = require('multer');
const { loadModels, detectFaces, preprocessFaceCrop, runArcFace } = require('./modelloader');
const { cosineSimilarity } = require('./utils');
const sharp = require('sharp');

// Configuration
const PORT = process.env.PORT || 3000;
const SIMILARITY_THRESHOLD = parseFloat(process.env.SIM_THRESHOLD || '0.45');
const FACE_EXPAND_RATIO = parseFloat(process.env.FACE_EXPAND_RATIO || '0.10');
const MAX_FILE_SIZE = 5 * 1024 * 1024; // 5MB

// Setup Express
const app = express();
const upload = multer({ 
  storage: multer.memoryStorage(), 
  limits: { fileSize: MAX_FILE_SIZE } 
});
app.use(express.json());

// ==================== Helper Functions ====================

/**
 * Choose the best face from detection results
 * Filters by size, position, and aspect ratio, then scores by detection confidence and centrality
 */
function chooseBestFace(detectOut, imgWidth, imgHeight) {
  if (!detectOut?.boxes?.length) return null;

  const minSize = Math.max(20, Math.min(imgWidth, imgHeight) * 0.08);
  const maxSize = Math.min(imgWidth, imgHeight) * 0.65;
  const edgeMargin = { x: imgWidth * 0.05, y: imgHeight * 0.05 };

  const candidates = detectOut.boxes
    .map(box => {
      // Convert normalized coordinates to pixels
      const x = Math.max(0, Math.round(box.x * imgWidth));
      const y = Math.max(0, Math.round(box.y * imgHeight));
      let w = Math.max(1, Math.round(box.width * imgWidth));
      let h = Math.max(1, Math.round(box.height * imgHeight));
      
      // Clamp to image bounds
      if (x + w > imgWidth) w = imgWidth - x;
      if (y + h > imgHeight) h = imgHeight - y;

      return { x, y, width: w, height: h, score: box.score ?? 0.5 };
    })
    .filter(box => {
      // Size filters
      if (box.width < minSize || box.height < minSize) return false;
      if (box.width > maxSize || box.height > maxSize) return false;
      
      // Edge filter - avoid faces touching edges
      if (box.x < edgeMargin.x || box.y < edgeMargin.y) return false;
      if (box.x + box.width > imgWidth - edgeMargin.x) return false;
      if (box.y + box.height > imgHeight - edgeMargin.y) return false;
      
      // Aspect ratio filter (0.6 to 1.8)
      const aspectRatio = box.width / box.height;
      return aspectRatio >= 0.6 && aspectRatio <= 1.8;
    })
    .map(box => {
      // Calculate composite score
      const centerX = box.x + box.width / 2;
      const centerY = box.y + box.height / 2;
      const dx = Math.abs(centerX - imgWidth / 2) / (imgWidth / 2);
      const dy = Math.abs(centerY - imgHeight / 2) / (imgHeight / 2);
      const centrality = 1 - Math.min(1, Math.sqrt(dx * dx + dy * dy));
      const sizeRatio = Math.min(1, (box.width * box.height) / (imgWidth * imgHeight));
      
      const compositeScore = 0.65 * box.score + 0.25 * centrality + 0.1 * sizeRatio;
      return { ...box, compositeScore };
    });

  if (candidates.length === 0) return null;
  
  candidates.sort((a, b) => b.compositeScore - a.compositeScore);
  return candidates[0];
}

/**
 * Expand face bounding box by a ratio
 */
function expandFaceBox(box, imgWidth, imgHeight, expandRatio = FACE_EXPAND_RATIO) {
  const expandX = Math.round(box.width * expandRatio / 2);
  const expandY = Math.round(box.height * expandRatio / 2);
  
  return {
    x: Math.max(0, box.x - expandX),
    y: Math.max(0, box.y - expandY),
    width: Math.min(imgWidth - (box.x - expandX), box.width + 2 * expandX),
    height: Math.min(imgHeight - (box.y - expandY), box.height + 2 * expandY)
  };
}

/**
 * Make a box square (center-preserving) and clamp within image bounds
 */
function ensureSquareBox(box, imgWidth, imgHeight) {
  const cx = box.x + box.width / 2;
  const cy = box.y + box.height / 2;
  const side = Math.round(Math.max(box.width, box.height));
  let x = Math.round(cx - side / 2);
  let y = Math.round(cy - side / 2);
  if (x < 0) x = 0;
  if (y < 0) y = 0;
  if (x + side > imgWidth) x = imgWidth - side;
  if (y + side > imgHeight) y = imgHeight - side;
  return { x: Math.max(0, x), y: Math.max(0, y), width: Math.min(side, imgWidth), height: Math.min(side, imgHeight) };
}

/**
 * Extract and process face crop from image buffer
 */
async function extractFaceCrop(buffer, box) {
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
 */
async function detectAndValidateFace(buffer) {
  const metadata = await sharp(buffer).metadata();
  const detectOut = await detectFaces(buffer);
  
  if (!detectOut.boxes?.length) {
    throw new Error('No face detected. Please provide a clear image with a visible face.');
  }
  
  const bestFace = chooseBestFace(detectOut, metadata.width, metadata.height);
  if (!bestFace) {
    throw new Error('No face detected. Please provide a clear image with a visible face.');
  }
  
  // Reduce background a bit for tighter crops
  const defaultExpand = isNaN(FACE_EXPAND_RATIO) ? 0.25 : FACE_EXPAND_RATIO;
  const tunedExpand = Math.min(0.20, defaultExpand); // cap at 0.20 by default
  let expandedBox = expandFaceBox(bestFace, metadata.width, metadata.height, tunedExpand);

  // Optionally enforce square crops (often improves recognition consistency)
  const useSquare = String(process.env.SQUARE_CROP || 'true').toLowerCase() !== 'false';
  if (useSquare) {
    expandedBox = ensureSquareBox(expandedBox, metadata.width, metadata.height);
  }
  
  if (expandedBox.width < 20 || expandedBox.height < 20) {
    throw new Error('Detected face is too small. Please provide a clearer image with a larger face.');
  }
  
  console.log(`Face detected (score ${bestFace.score.toFixed(3)}) at (${expandedBox.x}, ${expandedBox.y}) size ${expandedBox.width}x${expandedBox.height}`);
  
  return expandedBox;
}

// ==================== API Endpoints ====================

/**
 * POST /cropface - Return cropped face image for visual inspection
 */
app.post('/cropface', upload.single('image'), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ success: false, error: 'No image provided' });
    }
    
    const faceBox = await detectAndValidateFace(req.file.buffer);
    const crop = await extractFaceCrop(req.file.buffer, faceBox);
    
    res.set('Content-Type', 'image/png');
    res.send(crop);
  } catch (err) {
    console.error('Error in /cropface:', err);
    const status = err.message.includes('No face') || err.message.includes('too small') ? 400 : 500;
    res.status(status).json({ success: false, error: err.message || 'Failed to crop face.' });
  }
});

/**
 * POST /encode - Generate face embedding from image
 */
app.post('/encode', upload.single('image'), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ success: false, error: 'No image provided' });
    }
    
    const faceBox = await detectAndValidateFace(req.file.buffer);
    const preprocessed = await preprocessFaceCrop(req.file.buffer, faceBox);
    const embedding = await runArcFace(preprocessed.tensorDataNCHW, preprocessed.tensorDataNHWC);
    
    if (!embedding || embedding.length !== 512) {
      throw new Error('Failed to generate valid face embedding. Please try with a different image.');
    }
    
    res.json({ success: true, embedding });
  } catch (err) {
    console.error('Error in /encode:', err);
    const status = err.message.includes('No face') || err.message.includes('too small') || err.message.includes('Invalid') ? 400 : 500;
    res.status(status).json({ success: false, error: err.message || 'Processing error. Please try with a different image.' });
  }
});

/**
 * POST /compare - Compare face against stored embedding
 */
app.post('/compare', upload.single('image'), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ success: false, error: 'No image provided' });
    }
    
    if (!req.body.storedEmbedding) {
      return res.status(400).json({ success: false, error: 'Missing storedEmbedding parameter' });
    }
    
    // Parse and validate stored embedding
    let storedEmbedding;
    try {
      storedEmbedding = JSON.parse(req.body.storedEmbedding);
      
      if (!Array.isArray(storedEmbedding) || storedEmbedding.length !== 512) {
        throw new Error('storedEmbedding must be an array of 512 numbers');
      }
      
      if (!storedEmbedding.every(val => typeof val === 'number' && !isNaN(val))) {
        throw new Error('storedEmbedding must contain only valid numbers');
      }
    } catch (e) {
      return res.status(400).json({ success: false, error: `Invalid embedding format: ${e.message}` });
    }
    
    // Detect face and generate embedding
    const faceBox = await detectAndValidateFace(req.file.buffer);
    const preprocessed = await preprocessFaceCrop(req.file.buffer, faceBox);
    const newEmbedding = await runArcFace(preprocessed.tensorDataNCHW, preprocessed.tensorDataNHWC);
    
    if (!newEmbedding || newEmbedding.length !== 512) {
      throw new Error('Failed to generate valid face embedding. Please try with a different image.');
    }
    
    // Calculate similarity
    const similarity = cosineSimilarity(newEmbedding, storedEmbedding);
    if (similarity === null) {
      throw new Error('Failed to calculate similarity. Embeddings may be invalid.');
    }
    
    const isMatch = similarity >= SIMILARITY_THRESHOLD;
    
    // Enhanced debugging output
    console.log(`\n=== COMPARISON RESULT ===`);
    console.log(`Similarity: ${similarity.toFixed(4)}`);
    console.log(`Threshold: ${SIMILARITY_THRESHOLD}`);
    console.log(`Is Match: ${isMatch}`);
    console.log(`New embedding range: [${Math.min(...newEmbedding).toFixed(4)}, ${Math.max(...newEmbedding).toFixed(4)}]`);
    console.log(`Stored embedding range: [${Math.min(...storedEmbedding).toFixed(4)}, ${Math.max(...storedEmbedding).toFixed(4)}]`);
    console.log(`New embedding first 5: [${newEmbedding.slice(0, 5).map(v => v.toFixed(4)).join(', ')}]`);
    console.log(`Stored embedding first 5: [${storedEmbedding.slice(0, 5).map(v => v.toFixed(4)).join(', ')}]`);
    console.log(`========================\n`);
    
    res.json({ 
      success: true, 
      isMatch, 
      similarity: parseFloat(similarity.toFixed(4))
    });
  } catch (err) {
    console.error('Error in /compare:', err);
    const status = err.message.includes('No face') || err.message.includes('too small') || err.message.includes('Invalid') ? 400 : 500;
    res.status(status).json({ success: false, error: err.message || 'Processing error. Please try with a different image.' });
  }
});

// ==================== Server Startup ====================

(async () => {
  try {
    await loadModels();
    app.listen(PORT, () => {
      console.log(`✅ Face verification server running on port ${PORT}`);
      console.log(`📊 Similarity threshold: ${SIMILARITY_THRESHOLD}`);
      console.log(`📏 Face expand ratio: ${FACE_EXPAND_RATIO}`);
    });
  } catch (e) {
    console.error('❌ Failed to load models:', e);
    process.exit(1);
  }
})();
