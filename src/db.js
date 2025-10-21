// src/db.js
const { Pool } = require('pg');

const pool = new Pool({
  connectionString: process.env.DATABASE_URL || 'postgresql://localhost/face_verif_db'
});

async function saveUserEmbedding({ id, name, email, embedding }) {
  const client = await pool.connect();
  try {
    await client.query(
      'INSERT INTO users (id, name, email, embedding) VALUES ($1, $2, $3, $4)',
      [id, name || null, email || null, embedding]
    );
  } finally {
    client.release();
  }
}

module.exports = { pool, saveUserEmbedding };
