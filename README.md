# MoiNgayMotModelML - CIFAR-10 Classification with FastAPI, Docker, and Machine Learning Models

Welcome to **MoiNgayMotModelML**, a project dedicated to exploring machine learning models for image classification using the **CIFAR-10 dataset**. This project leverages tools like **FastAPI** for building RESTful APIs and **Docker** for containerized deployment.

The goal is to classify images from the CIFAR-10 dataset using both basic machine learning models and deep learning techniques such as LSTM and CNN, while exposing functionality through well-defined API endpoints: `/preprocess` and `/prediction/{model}`.

---

## 1. Dataset and Tools

### CIFAR-10
The **CIFAR-10** dataset consists of **60,000 32x32 color images** in **10 classes**, with **6,000 images per class**. The dataset includes objects like airplanes, cars, birds, cats, deer, dogs, frogs, horses, ships, and trucks . It is commonly used as a benchmark for evaluating machine learning and deep learning models .

### FastAPI
[FastAPI](https://fastapi.tiangolo.com/) is a modern, fast (high-performance), web framework for building APIs with Python based on standard Python type hints. It provides automatic documentation via Swagger UI and ReDoc.

- **API Endpoints**:
  - `POST /preprocess`: Accepts raw image data or paths for preprocessing.
  - `GET /prediction/{model}`: Returns predictions using the specified model (`logistic`, `svm`, `random_forest`, `lstm`, or `cnn`).

- **Port**: The application will run on port **8000** by default.

Example `curl` usage:
```bash
# Preprocess an image
curl -X POST http://localhost:8000/preprocess -H "Content-Type: application/json" -d '{"image_path": "path/to/image.png"}'

# Predict using a specific model
curl http://localhost:8000/prediction/random_forest
```

### Docker
[Docker](https://www.docker.com/) is used to containerize the application, ensuring consistency across development, testing, and production environments. A multi-stage build is employed to reduce final image size and enhance security .

---

## 2. Feature Engineering

Since we are not using CNN-based methods for some models, feature engineering plays a critical role in extracting meaningful representations from raw pixel data:

### Methods Used:
- **Pixel Intensity Features**: Flatten the 32x32 RGB image into a 1D vector of 3072 features (3 channels × 32 × 32) .
- **Color Histograms**: Extract histograms for each RGB channel to capture color distribution within the image.
- **PCA (Principal Component Analysis)**: Reduce dimensionality while retaining significant variance (e.g., 95% explained variance).
- **Local Binary Patterns (LBP)**: Capture texture information, useful for distinguishing patterns like fur, wood, etc.
- **Histogram of Oriented Gradients (HOG)**: Useful for capturing edge and shape information in images.

These engineered features are then fed into traditional ML models like Logistic Regression, SVM, and Random Forest for classification.

---

## 3. Prediction Models

We implement a variety of models for comparison and educational purposes:

### Basic Machine Learning Models
- **Logistic Regression**
- **Support Vector Machine (SVM)**
- **Random Forest**

These models use preprocessed and engineered features from the images for classification.

### Deep Learning Models
- **Long Short-Term Memory (LSTM)**: Although typically used for sequential data, LSTMs can be applied to flattened image vectors treated as sequences.
- **Convolutional Neural Networks (CNN)**: Standard CNN architectures like LeNet or simple custom CNNs are trained directly on raw pixel data for better accuracy .

Each model is accessible via its own endpoint under `/prediction/{model_name}`.

---

## Project Structure

```
moingaymotmodelml/
│
├── main.py                  # Entry point - chạy FastAPI app
├── requirements.txt         # Danh sách thư viện cần thiết
│
├── config/                  # Cấu hình chung (hằng số)
│   └── settings.py          # INPUT_DIM, NUM_CLASSES, BATCH_SIZE,...
│
├── models/                  # Mô hình học máy
│   ├── softmax.py           # SoftmaxRegression class
│   ├── svm.py               # SVMModelWrapper class
│   └── xgboost.py           # XGBoostWrapper class
│
├── data/                    # Xử lý dữ liệu
│   └── data_loader.py       # Hàm tải CIFAR-10 dataset
│
├── trainers/                # Huấn luyện từng mô hình
│   ├── softmax_trainer.py   # ModelTrainer cho Softmax
│   ├── svm_trainer.py       # SVMTrainer cho SVM
│   └── xgboost_trainer.py   # XGBoostTrainer cho XGBoost
│
├── apis/                    # API routes
│   └── routes.py            # Các route chính
│
├── utils/                   # Hàm tiện ích
│   └── helpers.py           # tensor_to_base64, class_names,...
│
├── app_state.py             # Quản lý trạng thái ứng dụng
├── logging_setup.py         # Thiết lập logging toàn ứng dụng
└── Dockerfile               # Docker build file (tùy chọn)
```

---

## Getting Started

### Prerequisites
- Python 3.8+
- Docker installed

### Build and Run with Docker

```bash
# Build the Docker image
docker build -t moingaymotmodelml .

# Run the container
docker run -p 8000:8000 -v $(pwd)/data:/app/data moingaymotmodelml
```

Now you can access the API at `http://localhost:8000`.

Swagger UI is available at `http://localhost:8000/docs`.

---

## Acknowledgements

This project uses the **CIFAR-10 dataset**, collected by Alex Krizhevsky, Vinod Nair, and Geoffrey Hinton.
