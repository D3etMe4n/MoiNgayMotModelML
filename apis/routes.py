# apis/routes.py
from fastapi import APIRouter, HTTPException
from app_state import app_state
from fastapi.responses import JSONResponse
from fastapi import FastAPI

import torch

from data_loader import get_data_loaders
from models.softmax import SoftmaxRegression
from models.svm import SVMModelWrapper
from utils.helpers import tensor_to_base64, class_names
from trainers.softmax_trainer import ModelTrainer as SoftmaxTrainer
from trainers.svm_trainer import SVMTrainer
from model_loader import load_models_and_data

router = APIRouter()

def setup_routes(app: FastAPI):
    app.include_router(router)

@router.on_event("startup")
async def initialize_models_and_data():
    load_models_and_data()
    return {"status": "Models and data loaded"}

@router.post("/train/{model_name}")
async def train_model(model_name: str):
    if model_name not in ["softmax", "svm"]:
        raise HTTPException(status_code=400, detail="Model name must be 'softmax' or 'svm'")

    if model_name == "softmax":
        if app_state.is_softmax_trained:
            return {"message": "Softmax model already trained"}

        app_state.softmax_trainer.train(app_state.EPOCHS)
        app_state.is_softmax_trained = True
        return {"message": "Softmax model trained successfully"}

    elif model_name == "svm":
        if app_state.is_svm_trained:
            return {"message": "SVM model already trained"}

        app_state.svm_trainer.train()
        app_state.is_svm_trained = True
        return {"message": "SVM model trained successfully"}

@router.get("/prediction/softmax/{image_id}")
async def predict_softmax(image_id: int):
    if not app_state.is_softmax_trained:
        await train_model("softmax")

    if image_id < 0 or image_id >= len(app_state.test_dataset):
        raise HTTPException(
            status_code=404,
            detail=f"Image ID must be between 0 and {len(app_state.test_dataset)-1}"
        )

    image_tensor, true_label = app_state.test_dataset[image_id]
    input_tensor = image_tensor.view(1, -1).to(app_state.device)

    with torch.no_grad():
        output = app_state.softmax_model(input_tensor)
        _, predicted_class = torch.max(output.data, 1)

    # chuyển true_label về int nếu cần
    true_idx = int(true_label) if isinstance(true_label, int) else true_label.item()
    pred_idx = predicted_class.item()

    return {
        "model": "softmax",
        "image_id": image_id,
        "predicted_class": pred_idx,
        "predicted_class_name": class_names[pred_idx],
        "true_class": true_idx,
        "true_class_name": class_names[true_idx],
        "image_b64": tensor_to_base64(image_tensor)
    }

@router.get("/prediction/svm/{image_id}")
async def predict_svm(image_id: int):
    # nếu chưa train thì train luôn
    if not app_state.is_svm_trained:
        app_state.svm_trainer.train()
        app_state.is_svm_trained = True

    # kiểm tra range của image_id
    if image_id < 0 or image_id >= len(app_state.test_dataset):
        raise HTTPException(
            status_code=404,
            detail=f"Image ID must be between 0 and {len(app_state.test_dataset)-1}"
        )

    image_tensor, true_label = app_state.test_dataset[image_id]
    input_tensor = image_tensor.view(1, -1).to(app_state.device)
    pred_tensor = app_state.svm_model.predict_tensor(input_tensor)
    pred_idx = pred_tensor.item()
    # true_label có thể là int hoặc Tensor
    true_idx = int(true_label) if isinstance(true_label, int) else true_label.item()

    return {
        "model": "svm",
        "image_id": image_id,
        "predicted_class": pred_idx,
        "predicted_class_name": class_names[pred_idx],
        "true_class": true_idx,
        "true_class_name": class_names[true_idx],
        "image_b64": tensor_to_base64(image_tensor)
    }

@router.get("/training/results")
async def get_training_results():
    svm_acc = app_state.svm_trainer.evaluate() if app_state.is_svm_trained else None
    softmax_loss = app_state.softmax_trainer.training_losses[-1] if app_state.is_softmax_trained else None
    softmax_train_acc = app_state.softmax_trainer.training_accuracies[-1] if app_state.is_softmax_trained else None
    softmax_val_acc = app_state.softmax_trainer.validation_accuracies[-1] if app_state.is_softmax_trained else None

    return {
        "total_epochs": app_state.EPOCHS,
        "final_train_accuracy_softmax": softmax_train_acc,
        "final_validation_accuracy_softmax": softmax_val_acc,
        "final_loss_softmax": softmax_loss,
        "validation_accuracy_svm": svm_acc,
        "history": {
            "losses": app_state.softmax_trainer.training_losses if app_state.is_softmax_trained else [],
            "train_accuracies_softmax": app_state.softmax_trainer.training_accuracies if app_state.is_softmax_trained else [],
            "validation_accuracies_softmax": app_state.softmax_trainer.validation_accuracies if app_state.is_softmax_trained else []
        }
    }