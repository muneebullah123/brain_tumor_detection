import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from PIL import Image
from flask import Flask, request, jsonify, render_template

# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load pretrained ResNet18 and modify last layer for your number of classes
num_classes = 4  # Change this if your model has a different number of classes
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)

# Load your trained weights
model.load_state_dict(torch.load('best_brain_tumor_model.pt', map_location=device))
model.eval()

# Define preprocessing pipeline (must match training preprocessing)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  # Imagenet means
        std=[0.229, 0.224, 0.225]    # Imagenet stds
    ),
])

# Mapping label indices to class names
label_map = {
    0: 'No Tumor',
    1: 'Glioma Tumor',
    2: 'Meningioma Tumor',
    3: 'Pituitary Tumor'
}

# Initialize Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')  # make sure this file is in templates/

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    try:
        img = Image.open(file).convert('RGB')
    except Exception:
        return jsonify({'error': 'Invalid image file'}), 400

    # Preprocess and prepare image tensor
    input_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        pred_class = probs.argmax(dim=1).item()
        confidence = probs.max().item()

    result = {
        'prediction': label_map.get(pred_class, 'Unknown'),
        'confidence': round(confidence * 100, 2)
    }
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
