const express = require('express');
const path = require('path');
const axios = require('axios');
const multer = require('multer');
const FormData = require('form-data');
const fs = require('fs');

const app = express();
const PORT = 3000;
const FLASK_URL = 'http://localhost:5000/predict';

const upload = multer({ storage: multer.memoryStorage() });

app.use(express.static(path.join(__dirname, 'public')));

app.post('/predict', upload.single('oct_scan'), async (req, res) => {
    if (!req.file) {
        return res.status(400).json({ error: 'No OCT scan file uploaded.' });
    }

    const file = req.file;
    const formData = new FormData();
    formData.append('oct_scan', file.buffer, {
        filename: file.originalname,
        contentType: file.mimetype,
    });

    try {
        const flaskResponse = await axios.post(FLASK_URL, formData, {
            headers: { ...formData.getHeaders() },
            timeout: 30000
        });
        res.json(flaskResponse.data);
    } catch (error) {
        if (error.code === 'ECONNREFUSED' || error.code === 'ENOTFOUND') {
            return res.status(503).json({
                error: "Failed to connect to the prediction service (Python/Flask). Please ensure it is running on http://localhost:5000."
            });
        }
        const errorMessage = error.response?.data?.error || 'Unknown error during prediction process.';
        res.status(500).json({ error: errorMessage });
    }
});

app.listen(PORT, () => {
    console.log(`Node.js Server Running - Access at: http://localhost:${PORT}`);
    console.log("Ensure Python Flask service is running on Port 5000.");
});
