from flask import Flask, request, jsonify
from PIL import Image
from torchvision import transforms
import torch
from torchvision import models
from torch import nn
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load your pre-trained PyTorch model
vgg16 = models.vgg16(pretrained=False)  # Set pretrained to False since you're loading your own weights
num_classes = 10  # Adjust this to the actual number of classes in your dataset
vgg16.classifier[6] = nn.Linear(vgg16.classifier[6].in_features, num_classes)

# Load the pre-trained weights from the 20th epoch
model_path = "C:\\Users\\Sujal Joshi\\OneDrive\\Desktop\\CNN_TOMATO_PLANT\\tomato_vgg16_model.pth"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
vgg16.load_state_dict(torch.load(model_path, map_location=device))
vgg16.eval()
vgg16.to(device)

# Define the transformation for input images
image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


def process_image(image):
    img = Image.open(image)
    img = image_transform(img)
    img = img.unsqueeze(0)  # Add a batch dimension
    img = img.to(device)  # Move to GPU if available
    return img

def predict_disease(image):
    with torch.no_grad():
        output = vgg16(image)
    # You need to define how to interpret the model output based on your specific model architecture
    # For example, if your model outputs class probabilities, you might return the class with the highest probability
    _, predicted_class = torch.max(output, 1)
    return int(predicted_class)

@app.route('/detect_disease', methods=['GET', 'POST'])
def detect_disease():
    try:
        if request.method == 'POST' or request.method == 'GET':
            # Handle POST request
            if 'Authorization' not in request.headers:
                return jsonify({'error': 'Missing Authorization header'})

            auth_token = request.headers['Authorization']
            # Verify the validity of the authentication token
            # ...

            image = request.files['image']
            processed_image = process_image(image)
            result = predict_disease(processed_image)
            return jsonify({'result': result})

    except Exception as e:
        return jsonify({'error': str(e)})


if __name__ == '__main__':
    app.run(debug=True)
