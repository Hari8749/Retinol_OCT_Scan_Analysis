// This line MUST be at the very top to read the .env file for local testing
require('dotenv').config(); 

const express = require('express');
const path = require('path');
const axios = require('axios');
const multer = require('multer');
const FormData = require('form-data');
const fs = require('fs');

const app = express();

// 1. GET PORT FROM ENVIRONMENT
// On Render, process.env.PORT is set automatically.
// Locally, it will default to 3000.
const PORT = process.env.PORT || 3000; 

// 2. GET FLASK URL FROM ENVIRONMENT
// On Render, this is the variable you set in the dashboard (e.g., https://retinal-api.onrender.com)
// Locally, this is read from your .env file (http://localhost:5000)
const FLASK_URL = process.env.FLASK_SERVICE_URL + '/predict';

// Configure Multer to store the file in memory
const upload = multer({ storage: multer.memoryStorage() }); 

// Serve static files (like index.html) from the 'public' directory
app.use(express.static(path.join(__dirname, 'public')));

// --- Prediction Proxy Route ---
app.post('/predict', upload.single('oct_scan'), async (req, res) => {
    
    if (!req.file) {
        return res.status(400).json({ error: 'No OCT scan file uploaded.' });
    }

    const file = req.file;
    const formData = new FormData();
    // The name 'oct_scan' MUST match the name expected by Flask
    formData.append('oct_scan', file.buffer, {
        filename: file.originalname,
        contentType: file.mimetype,
    });

    console.log(`Forwarding request to Flask service at: ${FLASK_URL}`);

    try {
        // Send the request to the Python Flask service URL
        const flaskResponse = await axios.post(FLASK_URL, formData, {
            headers: { 
                ...formData.getHeaders(),
            },
            timeout: 90000 // 90 second timeout for model inference
        });

        res.json(flaskResponse.data);

    } catch (error) {
        console.error("Error communicating with Python Flask server:", error.message);
        
        let userMessage = 'Unknown error during prediction process.';
        if (error.code === 'ECONNREFUSED' || error.code === 'ENOTFOUND') {
            // 3. UPDATED ERROR MESSAGE
            userMessage = "Failed to connect to the Python prediction service. Please ensure the service is running and the URL is correct.";
        } else if (error.response?.data?.error) {
            userMessage = error.response.data.error;
        } else if (error.code === 'ETIMEDOUT' || error.message.includes('timeout')) {
            userMessage = "Prediction failed: The model server (Python) took too long to respond. This is common on free tiers, please try again.";
        }

        res.status(500).json({ error: userMessage });
    }
});

// --- Server Startup ---
app.listen(PORT, () => {
    console.log(`Node.js proxy server running on port ${PORT}`);
    console.log(`Forwarding /predict requests to ${FLASK_URL}`);
});