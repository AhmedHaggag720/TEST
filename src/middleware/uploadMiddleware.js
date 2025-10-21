// src/middleware/uploadMiddleware.js
/**
 * File upload middleware configuration
 */

const multer = require('multer');
const config = require('../config');

/**
 * Configure multer for image uploads
 */
const upload = multer({
  storage: multer.memoryStorage(),
  limits: {
    fileSize: config.server.maxFileSize,
  },
  fileFilter: (req, file, cb) => {
    // Accept only image files
    if (file.mimetype.startsWith('image/')) {
      cb(null, true);
    } else {
      cb(new Error('Only image files are allowed'), false);
    }
  },
});

module.exports = upload;
