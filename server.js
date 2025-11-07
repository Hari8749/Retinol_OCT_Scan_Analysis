// This line MUST be at the very top
require('dotenv').config(); 

const express = require('express');
const path = require('path');
const axios = require('axios');
const multer = require('multer');
const FormData = require('form-data');
const fs = require('fs');

const app = express();

// This code now works for BOTH local and Render
// On Render, 'PORT' is set by Render. Locally, it defaults to 3000.
const PORT = process.env.PORT || 3000; 

// --- THIS IS THE CRITICAL FIX ---
// It reads from your .env file (http://localhost:5000) 
// AND correctly adds the '/predict' route
const FLASK_URL = process.env.FLASK_SERVICE_URL + '/predict';

// Configure Multer to store the file in memory
const upload = multer({ storage: multer.memoryStorage() }); 

// Serve static files (like index.html) from the 'public' directory
app.use(express.static(path.join(__dirname, 'public')));

// --- Prediction Proxy Route ---
// This route listens for uploads from your index.html
app.post('/predict', upload.single('oct_scan'), async (req, res) => {
    
    // Check if a file was uploaded
    if (!req.file) {
        return res.status(400).json({ error: 'No OCT scan file uploaded.' });
    }

    const file = req.file;

    // Create FormData to send the binary file buffer to the Python service
    const formData = new FormData();
    // The name 'oct_scan' MUST match the name expected by Flask
    formData.append('oct_scan', file.buffer, {
        filename: file.originalname,
        contentType: file.mimetype,
    });

    // This log is for your terminal, so you can see the correct URL
    console.log(`Forwarding request to Flask service at: ${FLASK_URL}`);

    try {
        // Send the POST request to the Python Flask service URL
        const flaskResponse = await axios.post(FLASK_URL, formData, {
            headers: { 
                ...formData.getHeaders(),
            },
            timeout: 30000 // 30 second timeout for model inference
        });

        // Relay Flask's JSON response back to the client
        res.json(flaskResponse.data);

    } catch (error) {
        console.error("Error communicating with Python Flask server:", error.message);
        
        let userMessage = 'Unknown error during prediction process.';
        if (error.code === 'ECONNREFUSED' || error.code === 'ENOTFOUND') {
            userMessage = "Failed to connect to the Python prediction service. Please ensure the service is running and the URL is correct.";
        } else if (error.response?.status === 405) {
            // This specifically catches the "Method Not Allowed" error
            userMessage = "Method Not Allowed. The Node.js server sent a request to the wrong Python URL. Check the FLASK_URL variable in server.js";
        } else if (error.response?.data?.error) {
            // Pass the error from the Python server to the user
            userMessage = error.response.data.error;
        }

        res.status(500).json({ error: userMessage });
    }
});

// --- Server Startup ---
app.listen(PORT, () => {
    console.log(`Node.js proxy server running on port ${PORT}`);
    // This log message will confirm the full, correct URL
    console.log(`Forwarding /predict requests to ${FLASK_URL}`);
});