# config/settings.py

INPUT_DIM = 32 * 32 * 3  # CIFAR-10 image size: 32x32 RGB
NUM_CLASSES = 10         # CIFAR-10 có 10 lớp
BATCH_SIZE = 256
LEARNING_RATE = 0.0005
WEIGHT_DECAY = 5e-4      # L2 regularization
EPOCHS = 25              # Số epoch huấn luyện mô hình softmax