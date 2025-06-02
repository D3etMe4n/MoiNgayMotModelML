# main.py
from fastapi import FastAPI, HTTPException
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.datasets import CIFAR10
from PIL import Image
import numpy as np
import base64
from io import BytesIO

app = FastAPI()

# Define Logistic Regression Model
class LogisticRegression(nn.Module):
    def __init__(self, input_dim):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))

# Constants
input_dim = 32 * 32 * 3
batch_size = 64
learning_rate = 0.001
epochs = 5

# Transform pipeline
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

# Load CIFAR-10 dataset (binary classification: class 0 vs class 1)
train_dataset = CIFAR10(root='./data', train=True, transform=transform, download=True)
test_dataset = CIFAR10(root='./data', train=False, transform=transform, download=True)

def filter_binary(dataset):
    images, labels = [], []
    for img, label in dataset:
        if label in [0, 1]:  # Binary classification
            images.append(img.view(-1))
            labels.append(torch.tensor(label).float().unsqueeze(0))
    return torch.stack(images), torch.tensor(labels)

X_train, y_train = filter_binary(train_dataset)
X_test, y_test = filter_binary(test_dataset)

# Initialize model
model = LogisticRegression(input_dim)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train model
def train_model():
    print("Training logistic regression model on CIFAR-10 binary subset...")
    for epoch in range(epochs):
        outputs = model(X_train)
        loss = criterion(outputs, y_train.unsqueeze(1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Accuracy
        predicted = (outputs > 0.5).float()
        correct = (predicted == y_train.unsqueeze(1)).sum().item()
        accuracy = correct / y_train.shape[0]

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}, Accuracy: {accuracy:.4f}")

@app.on_event("startup")
async def startup_event():
    train_model()
    print("Model trained.")

# Map class indices to names
class_names = ['airplane', 'automobile']

def tensor_to_base64(tensor_image):
    """Converts normalized tensor image back to displayable RGB image in base64."""
    # Denormalize
    denorm = tensor_image * 0.5 + 0.5  # Undo Normalize((0.5, 0.5, 0.5), ...)
    image_np = denorm.numpy().transpose(1, 2, 0)  # CxHxW -> HxWxC
    image_np = (image_np * 255).astype(np.uint8)

    # Convert to PIL image
    pil_img = Image.fromarray(image_np)
    buffered = BytesIO()
    pil_img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

@app.get("/prediction/logistic/{image_id}")
async def predict_logistic(image_id: int):
    if image_id < 0 or image_id >= len(X_test):
        raise HTTPException(status_code=404, detail=f"Image ID must be between 0 and {len(X_test)-1}")

    with torch.no_grad():
        image_tensor = X_test[image_id].unsqueeze(0)
        output = model(image_tensor)
        prediction = (output > 0.5).float().item()
        true_label = y_test[image_id].item()

        # Get original image from test_dataset for visualization
        original_img_tensor, label = test_dataset[image_id]
        while label not in [0, 1]:
            image_id += 1
            original_img_tensor, label = test_dataset[image_id]

        base64_image = tensor_to_base64(original_img_tensor)

    return {
        "model": "logistic",
        "image_id": image_id,
        "predicted_class": int(prediction),
        "predicted_class_name": class_names[int(prediction)],
        "true_class": int(true_label),
        "true_class_name": class_names[int(true_label)],
        "image_b64": base64_image
    }
