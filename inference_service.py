import os
import io
import torch
import torch.nn as nn
from torchvision import models, transforms
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image

# --- 1. App Setup ---
app = Flask(__name__)
# Enable CORS to allow requests from your Node.js frontend
CORS(app)

# --- 2. Class Names (from your notebook) ---
class_names = ['CNV', 'DME', 'DRUSEN', 'NORMAL']

# --- 3. Educational Data (as requested) ---
DISEASE_DATA = [
    {
        "class": "CNV",
        "fullName": "Choroidal Neovascularization (Wet AMD)",
        "description": "An advanced stage of 'wet' Age-related Macular Degeneration (AMD) where abnormal, fragile blood vessels grow beneath the retina. These vessels leak fluid or blood, causing rapid and severe vision loss.",
        "causes": ["Age-related Macular Degeneration (AMD)", "Severe myopia (nearsightedness)", "Inflammatory eye conditions (e.g., punctate inner choroidopathy)", "Genetic predisposition"],
        "prevention_management": ["Regular eye exams (especially if high-risk)", "Adoption of the AREDS2 vitamin formula", "Regular self-monitoring with an Amsler grid"],
        "medicines_treatments": ["Anti-VEGF injections (Aflibercept, Ranibizumab, Bevacizumab) – Primary treatment", "Photodynamic Therapy (PDT) – Used in select cases", "Steroid injections (Triamcinolone)"]
    },
    {
        "class": "DME",
        "fullName": "Diabetic Macular Edema",
        "description": "Swelling in the macula (the central part of the retina) caused by damaged blood vessels leaking fluid due to high blood sugar levels. It is the most common cause of vision loss in people with diabetes.",
        "causes": ["Poorly controlled Type 1 or Type 2 Diabetes (high HbA1c levels)", "Hypertension (high blood pressure)", "High cholesterol levels", "Diabetic Nephropathy (kidney disease)"],
        "prevention_management": ["Strict control of blood sugar (HbA1c levels)", "Manage blood pressure and cholesterol", "Regular comprehensive dilated eye exams"],
        "medicines_treatments": ["Anti-VEGF injections (Ranibizumab, Aflibercept) – First line therapy", "Steroid injections (Dexamethasone implant, Fluocinolone implant)", "Laser photocoagulation (Focal or grid laser)"]
    },
    {
        "class": "DRUSEN",
        "fullName": "Drusen (Early/Intermediate AMD)",
        "description": "Yellow deposits of fatty protein material that accumulate under the retina. While small, hard drusen are common and usually harmless, large, soft drusen are a significant risk factor for progression to advanced AMD (CNV).",
        "causes": ["Aging (primary risk factor)", "Smoking (increases risk significantly)", "Family history/Genetics", "Caucasian ethnicity"],
        "prevention_management": ["Regular self-monitoring with an Amsler grid", "Quit smoking immediately", "Eat a diet rich in green leafy vegetables (lutein, zeaxanthin)"],
        "medicines_treatments": ["AREDS2 high-dose antioxidant and mineral supplements (only for intermediate/advanced stages)", "N/A (No medical intervention typically required for early stages)"]
    },
    {
        "class": "NORMAL",
        "fullName": "Healthy Retina",
        "description": "The retina is clear with a normal foveal contour. No signs of pathological fluid, choroidal neovascularization, or significant drusen deposits are present, indicating good health.",
        "causes": ["N/A (This represents the absence of the specific pathologies being classified)"],
        "prevention_management": ["Maintain a healthy lifestyle (diet, exercise)", "Protect eyes from UV light", "Schedule routine annual eye examinations (especially after age 40)", "Avoid smoking"],
        "medicines_treatments": ["N/A (No treatment necessary for a healthy retina)"]
    }
]

# --- 4. Load Model ---
# Define the exact model architecture from your notebook [cite: classificaion_model.ipynb]
model = models.resnet50(weights=None)
num_ftrs = model.fc.in_features
model.fc = nn.Sequential(
    nn.Linear(num_ftrs, 256),
    nn.ReLU(),
    nn.Dropout(0.5),  # Using 0.5 as seen in your notebook
    nn.Linear(256, len(class_names))
)

# Load the trained model weights (make sure 'best_model.pth' is in your GitHub repo)
# Use 'cpu' as Render's free tier does not have a GPU
try:
    model.load_state_dict(torch.load('best_model.pth', map_location='cpu'))
except FileNotFoundError:
    print("WARNING: 'best_model.pth' not found. Using random weights.")
except Exception as e:
    print(f"Error loading model: {e}")

model.eval()  # Set model to evaluation mode

# --- 5. Preprocessing (Must match training from notebook) ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# --- 6. Prediction Function ---
def predict_oct(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension

    with torch.no_grad():
        outputs = model(image_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        confidence, pred_idx = torch.max(probs, 1)
    
    predicted_class = class_names[pred_idx.item()]
    
    # Find the matching disease info
    info = next((d for d in DISEASE_DATA if d['class'] == predicted_class), None)

    return {
        "predicted_class": predicted_class,
        "confidence": f"{confidence.item() * 100:.2f}%",
        "disease_info": info
    }

# --- 7. Flask API Route ---
@app.route('/predict', methods=['POST'])
def predict_route():
    if 'oct_scan' not in request.files:
        return jsonify({'error': 'No file part. Expected field name: "oct_scan"'}), 400
    
    file = request.files['oct_scan']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    try:
        image_bytes = file.read()
        result = predict_oct(image_bytes)
        return jsonify(result)
        
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return jsonify({'error': f'Prediction processing failed: {str(e)}'}), 500

# --- 8. Health Check Route ---
@app.route('/')
def health_check():
    return "Python/Flask ML service is running."

# --- 9. Run Server ---
if __name__ == '__main__':
    # Get port from environment variable for Render, default to 5000 for local dev
    port = int(os.environ.get("PORT", 5000))
    # Run on 0.0.0.0 to be accessible in the Render network
    app.run(host='0.0.0.0', port=port, debug=True)