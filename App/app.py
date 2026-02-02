import os
from flask import Flask, render_template, request
from PIL import Image
import torch
import pandas as pd
from torchvision import transforms, models
import torch.nn as nn

# Load CSV files
disease_info = pd.read_csv('disease_info.csv', encoding='cp1252')
supplement_info = pd.read_csv('supplement_info.csv', encoding='cp1252')

NUM_CLASSES = 39

# Load ResNet50 model
model = models.resnet50()

# Replace final layer for 39 classes
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)

# Load trained weights
model.load_state_dict(
    torch.load("trained_model.pth", map_location=torch.device('cpu'))
)

model.eval()

# Image transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Prediction function
def predict(file_path):
    img = Image.open(file_path).convert('RGB')
    img = transform(img)
    img = img.unsqueeze(0)

    with torch.no_grad():
        output = model(img)
        _, predicted = torch.max(output, 1)

    return predicted.item()


# Flask app
app = Flask(__name__)

@app.route('/')
def home_page():
    return render_template('home.html')

@app.route('/contact')
def contact():
    return render_template('contact-us.html')

@app.route('/index')
def ai_engine_page():
    return render_template('index.html')

@app.route('/mobile-device')
def mobile_device_detected_page():
    return render_template('mobile-device.html')


@app.route('/submit', methods=['GET', 'POST'])
def submit():
    if request.method == 'POST':
        image = request.files['image']
        filename = image.filename

        upload_folder = 'static/uploads'
        os.makedirs(upload_folder, exist_ok=True)

        file_path = os.path.join(upload_folder, filename)
        image.save(file_path)

        pred = predict(file_path)

        title = disease_info['disease_name'][pred]
        description = disease_info['description'][pred]
        prevent = disease_info['Possible Steps'][pred]
        image_url = disease_info['image_url'][pred]

        supplement_name = supplement_info['supplement name'][pred]
        supplement_image_url = supplement_info['supplement image'][pred]
        supplement_buy_link = supplement_info['buy link'][pred]

        return render_template(
            'submit.html',
            title=title,
            desc=description,
            prevent=prevent,
            image_url=image_url,
            pred=pred,
            sname=supplement_name,
            simage=supplement_image_url,
            buy_link=supplement_buy_link
        )


@app.route('/market', methods=['GET', 'POST'])
def market():
    return render_template(
        'market.html',
        supplement_image=list(supplement_info['supplement image']),
        supplement_name=list(supplement_info['supplement name']),
        disease=list(disease_info['disease_name']),
        buy=list(supplement_info['buy link'])
    )


if __name__ == '__main__':
    app.run(debug=True)
