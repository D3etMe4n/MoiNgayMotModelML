# app_state.py

class AppState:
    def __init__(self):
        self.is_softmax_trained = False
        self.is_svm_trained = False
        self.softmax_model = None
        self.svm_model = None
        self.softmax_trainer = None
        self.svm_trainer = None
        self.is_xgboost_trained = False
        self.xgboost_model = None
        self.xgboost_trainer = None
        self.test_dataset = None
        self.device = None
        self.INPUT_DIM = None
        self.NUM_CLASSES = None
        self.EPOCHS = None

# Khởi tạo một instance duy nhất để dùng toàn cục
app_state = AppState()