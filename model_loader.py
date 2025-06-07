# model_loader.py
import torch

from config.settings import INPUT_DIM, NUM_CLASSES, EPOCHS, BATCH_SIZE
from data_loader import get_data_loaders
from models.softmax import SoftmaxRegression
from models.svm import SVMModelWrapper
from models.xgboost import XGBoostWrapper
from trainers.softmax_trainer import ModelTrainer as SoftmaxTrainer
from trainers.svm_trainer import SVMTrainer
from trainers.xgboost_trainer import XGBoostTrainer
from app_state import app_state

def load_models_and_data():
    # Thiết lập device và các thông số toàn cục
    app_state.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    app_state.INPUT_DIM = INPUT_DIM
    app_state.NUM_CLASSES = NUM_CLASSES
    app_state.EPOCHS = EPOCHS

    # Load dữ liệu
    train_loader, test_loader, test_dataset = get_data_loaders(BATCH_SIZE)

    # Gán vào app_state
    app_state.train_loader = train_loader
    app_state.test_loader = test_loader
    app_state.test_dataset = test_dataset

    # Khởi tạo mô hình Softmax
    app_state.softmax_model = SoftmaxRegression(INPUT_DIM, NUM_CLASSES)
    app_state.softmax_trainer = SoftmaxTrainer(
        app_state.softmax_model,
        train_loader,
        test_loader,
        INPUT_DIM,
        lr=0.0005,
        weight_decay=5e-4
    )

    # Khởi tạo mô hình SVM
    app_state.svm_model = SVMModelWrapper(
    INPUT_DIM,
    NUM_CLASSES,
    kernel="rbf",
    C=1e-2,
    gamma=0.01
    )
    app_state.svm_trainer = SVMTrainer(
        app_state.svm_model,
        train_loader,
        test_loader,
        INPUT_DIM
    )

    # Khởi tạo mô hình XGBoost
    app_state.xgboost_model = XGBoostWrapper(INPUT_DIM, NUM_CLASSES)
    app_state.xgboost_trainer = XGBoostTrainer(
        app_state.xgboost_model, train_loader, test_loader, INPUT_DIM
    )