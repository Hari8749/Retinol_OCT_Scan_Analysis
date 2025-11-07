from flask import Flask, request, jsonify
from PIL import Image
import torch
import torchvision.transforms as transforms
import io
from torchvision import models
import torch.nn as nn

app = Flask(__name__)

# --- Class names ---
class_names = ['CNV', 'DME', 'DRUSEN', 'NORMAL']

# --- Model definition (ResNet50 with your exact FC layer) ---
model = models.resnet50(weights=None)
num_features = model.fc.in_features
model.fc = nn.Sequential(
    nn.Linear(num_features, 256),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(256, len(class_names))
)

# --- Load trained weights ---
state_dict = torch.load('oct_model.pth', map_location='cpu')
model.load_state_dict(state_dict)
model.eval()

# --- Preprocessing (must match training) ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# --- Prediction ---
def predict_oct(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(image)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        confidence, pred = torch.max(probs, 1)
    predicted_class = class_names[pred.item()]
    return {
        "predicted_class": predicted_class,
        "confidence": f"{confidence.item() * 100:.2f}%",
        "disease_info": {
            "fullName": predicted_class,
            "description": f"The model detected the OCT scan as {predicted_class}."
        }
    }

# --- Flask route ---
@app.route('/predict', methods=['POST'])
def predict():
    if 'oct_scan' not in request.files:
        return jsonify({'error': 'No file part'}), 400
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
        return jsonify({'error': f'Prediction processing failed: {e}'}), 500

if __name__ == '__main__':
    app.run(port=5000, debug=True)
