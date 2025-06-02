import os
import logging
from fastapi import FastAPI, HTTPException
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from PIL import Image
import numpy as np
import base64
from io import BytesIO

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = FastAPI()

# Define Softmax Regression Model
class SoftmaxRegression(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(SoftmaxRegression, self).__init__()
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.linear(x)

# Constants
input_dim = 32 * 32 * 3
num_classes = 10
batch_size = 256               # Increased batch size for faster training
learning_rate = 0.0005         # Lowered learning rate for better convergence
weight_decay = 5e-3            # Stronger L2 regularization
epochs = 25                    # More epochs to allow full convergence

# Transform pipeline
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

# Load CIFAR-10 dataset
train_dataset = datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.CIFAR10(root='./data', train=False, transform=transform, download=True)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# Initialize model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SoftmaxRegression(input_dim, num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

# Training metrics
training_losses = []
training_accuracies = []
validation_accuracies = []

def evaluate_model():
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.view(-1, input_dim).to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    return accuracy

def train_model():
    logger.info("Starting training of multinomial logistic regression model on CIFAR-10...")
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images = images.view(-1, input_dim).to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        avg_loss = total_loss / len(train_loader)
        train_acc = correct / total
        val_acc = evaluate_model()

        training_losses.append(avg_loss)
        training_accuracies.append(train_acc)
        validation_accuracies.append(val_acc)

        logger.info(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}, "
                    f"Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")

    logger.info("Model training completed.")

@app.on_event("startup")
async def startup_event():
    train_model()

# Map class indices to names
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

def tensor_to_base64(tensor_image):
    """Converts normalized tensor image back to displayable RGB image in base64."""
    denorm = tensor_image * 0.5 + 0.5
    image_np = denorm.numpy().transpose(1, 2, 0)
    image_np = (image_np * 255).astype(np.uint8)
    pil_img = Image.fromarray(image_np)
    buffered = BytesIO()
    pil_img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

@app.get("/prediction/softmax/{image_id}")
async def predict_softmax(image_id: int):
    if image_id < 0 or image_id >= len(test_dataset):
        raise HTTPException(status_code=404, detail=f"Image ID must be between 0 and {len(test_dataset)-1}")

    image_tensor, true_label = test_dataset[image_id]
    input_tensor = image_tensor.view(-1, input_dim).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        _, predicted_class = torch.max(output.data, 1)

    predicted_class_int = predicted_class.item() if isinstance(predicted_class, torch.Tensor) else predicted_class
    true_label_int = true_label.item() if isinstance(true_label, torch.Tensor) else true_label

    predicted_name = class_names[predicted_class_int]
    true_name = class_names[true_label_int]

    base64_image = tensor_to_base64(image_tensor)

    return {
        "model": "softmax",
        "image_id": image_id,
        "predicted_class": predicted_class_int,
        "predicted_class_name": predicted_name,
        "true_class": true_label_int,
        "true_class_name": true_name,
        "image_b64": base64_image
    }

@app.get("/training/results")
async def get_training_results():
    return {
        "total_epochs": epochs,
        "final_train_accuracy": training_accuracies[-1],
        "final_validation_accuracy": validation_accuracies[-1],
        "final_loss": training_losses[-1],
        "history": {
            "losses": training_losses,
            "train_accuracies": training_accuracies,
            "validation_accuracies": validation_accuracies
        }
    }

@app.get("/training/logs")
async def get_logs():
    try:
        with open("training.log", "r") as f:
            log_content = f.read()
        return {"logs": log_content}
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Log file not found.")