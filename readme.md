# Face Verification API

A simple Node.js microservice for face verification using AI models (BlazeFace + ArcFace).

## Features

- 🔍 **Face Detection** - Automatically detect faces in images
- 🧬 **Face Encoding** - Generate 512-dimensional embeddings
- ✅ **Face Verification** - Compare faces with cosine similarity
- 🚀 **Fast** - ~150-300ms per request
- 📦 **Simple** - Only 3 REST endpoints

## Quick Start

### 1. Install

```bash
npm install
```

### 2. Run

```bash
npm start
```

Server runs on `http://localhost:3000`

## API Endpoints

### 1. `/cropface` - Preview Face Crop

Returns the detected face crop as PNG image.

```bash
curl -X POST http://localhost:3000/cropface \
  -F "image=@photo.jpg" \
  --output face.png
```

**Response**: PNG image (112×112)

---

### 2. `/encode` - Register Face

Generate embedding for a face image.

```bash
curl -X POST http://localhost:3000/encode \
  -F "image=@photo.jpg"
```

**Response:**
```json
{
  "success": true,
  "embedding": [0.0123, -0.0456, ... 512 numbers]
}
```

**Errors:**
- `No image provided` - Missing image file
- `No face detected` - No valid face found
- `Detected face is too small` - Face too small to process

---

### 3. `/compare` - Verify Face

Compare a new image against a stored embedding.

```bash
curl -X POST http://localhost:3000/compare \
  -F "image=@verify.jpg" \
  -F 'storedEmbedding=[0.0123,-0.0456,...]'
```

**Response:**
```json
{
  "success": true,
  "isMatch": true,
  "similarity": 0.8234
}
```

**Fields:**
- `isMatch` - `true` if similarity ≥ threshold (default 0.65)
- `similarity` - Score from 0.0 to 1.0

**Errors:**
- `Missing storedEmbedding parameter` - No embedding provided
- `Invalid embedding format` - Wrong format or size
- `No face detected` - No face in verification image

## Configuration

Set environment variables before starting:

```bash
# Windows PowerShell
$env:PORT = "3000"
$env:SIM_THRESHOLD = "0.65"
$env:FACE_EXPAND_RATIO = "0.25"
npm start

# Linux/Mac
PORT=3000 SIM_THRESHOLD=0.65 npm start
```

### Options

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | `3000` | Server port |
| `SIM_THRESHOLD` | `0.65` | Match threshold (0.0 - 1.0) |
| `FACE_EXPAND_RATIO` | `0.25` | Face box expansion (0.0 - 0.5) |

### Threshold Guide

- `0.5-0.6` - More permissive (fewer false negatives)
- `0.6-0.7` - Balanced (recommended)
- `0.7-0.8` - Stricter (fewer false positives)

## Usage Example

### PowerShell

```powershell
# 1. Register a face
$enc = Invoke-RestMethod -Uri http://localhost:3000/encode -Method Post -Form @{
  image = Get-Item "C:\photos\person1.jpg"
}

# Save embedding
$enc.embedding | ConvertTo-Json -Compress | Set-Content "emb.json"

# 2. Verify same person
$embJson = Get-Content "emb.json" -Raw
$cmp = Invoke-RestMethod -Uri http://localhost:3000/compare -Method Post -Form @{
  image = Get-Item "C:\photos\person1_2.jpg"
  storedEmbedding = $embJson
}

Write-Host "Match: $($cmp.isMatch), Similarity: $($cmp.similarity)"
```

### cURL

```bash
# 1. Register
response=$(curl -s -X POST http://localhost:3000/encode -F "image=@person1.jpg")
embedding=$(echo $response | jq -c '.embedding')

# 2. Verify
curl -X POST http://localhost:3000/compare \
  -F "image=@person1_verify.jpg" \
  -F "storedEmbedding=$embedding"
```

## Similarity Scores

| Range | Meaning |
|-------|---------|
| 0.0 - 0.3 | Different people |
| 0.3 - 0.6 | Uncertain |
| 0.6 - 1.0 | Same person ✅ |

## Project Structure

```
SOFINDEX/
├── src/
│   ├── server.js         # API endpoints
│   ├── modelLoader.js    # AI model inference
│   └── utils.js          # Helper functions
├── models/
│   ├── blazeface.onnx    # Face detection
│   └── arcface.onnx      # Face recognition
└── package.json
```

## Requirements

- Node.js 16+
- 2GB RAM minimum
- ONNX models in `models/` folder

## Tips

### Best Results

- Use clear, well-lit photos
- Face should be clearly visible
- Avoid heavy shadows or occlusion
- Frontal face works best

### Troubleshooting

**"No face detected"**
- Ensure face is clearly visible
- Try better lighting
- Face shouldn't be too small or at edges

**Low similarity for same person**
- Try adjusting `SIM_THRESHOLD` to 0.6
- Reduce `FACE_EXPAND_RATIO` to 0.2
- Use more similar photos (angle, lighting)

**Models not loading**
- Check `models/` folder contains `.onnx` files
- Ensure sufficient RAM available

## Performance

- **Startup**: ~1-2 seconds (model loading)
- **Face Detection**: ~50-100ms
- **Encoding**: ~100-200ms
- **Comparison**: ~150-300ms total
- **Memory**: ~500MB-1GB

## License

ISC

## Author

Ahmed Haggag

**Scenario**: Register a face with a clear, well-lit image

```bash
curl -X POST http://localhost:3000/encode \
  -F "image=@test_images/person1_clear.jpg"
```

**Expected Result**: Returns `success: true` with a 512-element embedding array

---

### Test Case 2: Successful Face Verification (Same Person)

**Scenario**: Verify the same person with a different photo

```bash
# Step 1: Encode the registration image
response1=$(curl -X POST http://localhost:3000/encode -F "image=@test_images/person1_photo1.jpg")
embedding=$(echo $response1 | jq -c '.embedding')

# Step 2: Compare with another image of the same person
curl -X POST http://localhost:3000/compare \
  -F "image=@test_images/person1_photo2.jpg" \
  -F "storedEmbedding=$embedding"
```

**Expected Result**: Returns `isMatch: true` with similarity ≥ 0.6

---

### Test Case 3: Failed Verification (Different Person)

**Scenario**: Verify a different person

```bash
# Step 1: Encode person 1
response1=$(curl -X POST http://localhost:3000/encode -F "image=@test_images/person1.jpg")
embedding=$(echo $response1 | jq -c '.embedding')

# Step 2: Compare with person 2's image
curl -X POST http://localhost:3000/compare \
  -F "image=@test_images/person2.jpg" \
  -F "storedEmbedding=$embedding"
```

**Expected Result**: Returns `isMatch: false` with similarity < 0.6

---

### Test Case 4: Poor Lighting Conditions

**Scenario**: Test with a poorly lit image

```bash
curl -X POST http://localhost:3000/encode \
  -F "image=@test_images/poor_lighting.jpg"
```

**Expected Result**: Should still work using center-crop fallback or return appropriate error if quality is too poor

---

### Test Case 5: Partial Face / Occluded Face

**Scenario**: Test with partially visible face (wearing sunglasses, mask, etc.)

```bash
curl -X POST http://localhost:3000/encode \
  -F "image=@test_images/partial_face.jpg"
```

**Expected Result**: May use center-crop fallback. Embedding will be generated but may have lower accuracy for verification.

---

### Test Case 6: Invalid Image Format

**Scenario**: Upload a non-image file

```bash
curl -X POST http://localhost:3000/encode \
  -F "image=@test_files/document.pdf"
```

**Expected Result**: Returns error: `"Invalid image format. Please provide a valid JPEG or PNG image."`

---

### Test Case 7: No Image Provided

**Scenario**: Make request without uploading an image

```bash
curl -X POST http://localhost:3000/encode
```

**Expected Result**: Returns error: `"No image provided"`

---

### Test Case 8: Invalid storedEmbedding Format

**Scenario**: Provide invalid embedding in /compare

```bash
curl -X POST http://localhost:3000/compare \
  -F "image=@test_images/person1.jpg" \
  -F "storedEmbedding=invalid_data"
```

**Expected Result**: Returns error: `"Invalid embedding format: ..."`

---

### Test Case 9: Very Small Image

**Scenario**: Upload an image smaller than minimum resolution

```bash
curl -X POST http://localhost:3000/encode \
  -F "image=@test_images/tiny_50x50.jpg"
```

**Expected Result**: Returns error: `"Image resolution too low. Please provide an image of at least 112x112 pixels."`

---

### Test Case 10: Multiple Faces in Image

**Scenario**: Upload an image with multiple faces

```bash
curl -X POST http://localhost:3000/encode \
  -F "image=@test_images/group_photo.jpg"
```

**Expected Result**: Uses the first detected face (highest confidence) or center-crop if detection fails

---

## 📊 Understanding Results

### Similarity Scores

The `/compare` endpoint returns a similarity score between 0.0 and 1.0:

- **0.0 - 0.3**: Different people (very low similarity)
- **0.3 - 0.6**: Uncertain (may be same person with different conditions)
- **0.6 - 1.0**: Same person (high similarity) ✅
- **≥ 0.6**: Considered a match (default threshold)

### Adjusting the Threshold

You can adjust the similarity threshold by setting the `SIM_THRESHOLD` environment variable:

```env
SIM_THRESHOLD=0.7  # More strict (fewer false positives)
SIM_THRESHOLD=0.5  # More lenient (fewer false negatives)
```

## 🛠️ Project Structure

```
SOFINDEX/
├── models/                    # ONNX model files
│   ├── arcface.onnx          # Face recognition model
│   └── blazeface.onnx        # Face detection model
├── src/
│   ├── server.js             # Express server and API endpoints
│   ├── modelLoader.js        # Model loading and inference logic
│   ├── db.js                 # PostgreSQL database functions
│   └── utils.js              # Utility functions (cosine similarity)
├── migrations/
│   └── create_users_table.sql # Database schema
├── package.json
└── README.md
```

## 🔧 Error Handling

The service includes comprehensive error handling for:

- ✅ Missing or invalid images
- ✅ Corrupted image files
- ✅ Images with no detectable faces
- ✅ Images with very small faces
- ✅ Low-resolution images
- ✅ Invalid embedding formats
- ✅ Model inference failures
- ✅ Poor lighting conditions (handled via fallback)
- ✅ Partial faces (handled via fallback)

## 🚀 Performance Considerations

- **Model Loading**: Models are loaded once at startup (takes 1-2 seconds)
- **Inference Time**: 
  - Face detection: ~50-100ms
  - Face encoding: ~100-200ms
  - Total per request: ~150-300ms
- **Memory Usage**: ~500MB-1GB (models loaded in memory)
- **Concurrent Requests**: Supports multiple concurrent requests

## 📝 Development Notes

### Face Detection Strategy

1. **Primary**: Use BlazeFace to detect faces and extract bounding boxes
2. **Fallback**: If no face detected, use center-crop (60% of image)
3. **Validation**: Ensure detected faces meet minimum size requirements

### Preprocessing Pipeline

1. Detect face region (or use center-crop)
2. Crop to face bounding box
3. Resize to 112x112 pixels
4. Normalize pixel values: `(pixel - 127.5) / 128.0`
5. Convert to NHWC format `[1, 112, 112, 3]` for ArcFace model

### Embedding Normalization

All embeddings are L2-normalized (unit vectors), which ensures cosine similarity works correctly.

## 🐛 Troubleshooting

### Issue: "Cannot find module 'onnxruntime-node'"

**Solution**: Reinstall dependencies
```bash
npm install
```

### Issue: Models not loading

**Solution**: Ensure model files are in the correct location
```bash
ls -lh models/
# Should show arcface.onnx and blazeface.onnx
```

### Issue: "Failed to generate valid face embedding"

**Possible causes**:
- Image quality too poor
- Face too small or not visible
- Image heavily occluded

**Solution**: Try with a clearer, well-lit image with a visible face

## 📄 License

ISC

## 👤 Author

Ahmed Haggag

## 🔗 Links

- **Hugging Face Models**: [Link to models used]
- **ONNX Runtime**: https://onnxruntime.ai/
- **Repository**: [Your Git Repository URL]

---

**Note**: This microservice is designed for demonstration and educational purposes. For production use, consider adding authentication, rate limiting, and additional security measures.
