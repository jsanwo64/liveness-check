import os

class TrainingConfig:
    # Common configurations
    BATCH_SIZE = 32
    EPOCHS = 50
    VALIDATION_SPLIT = 0.2
    IMAGE_SIZE = (224, 224)
    
    # Face Liveness specific
    LIVENESS_CLASSES = ['real', 'fake']
    LIVENESS_LEARNING_RATE = 0.001
    
    # Anti-spoofing specific
    SPOOF_CLASSES = ['real', 'print', 'replay', 'mask']
    SPOOF_LEARNING_RATE = 0.0005
    
    # Emotion specific
    EMOTION_CLASSES = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
    EMOTION_LEARNING_RATE = 0.001
    
    # Data augmentation
    AUGMENTATION_PARAMS = {
        'rotation_range': 20,
        'width_shift_range': 0.2,
        'height_shift_range': 0.2,
        'horizontal_flip': True,
        'fill_mode': 'nearest'
    }
    
    # Paths
    BASE_PATH = os.path.dirname(os.path.abspath(__file__))
    DATA_PATH = os.path.join(BASE_PATH, 'data')
    MODELS_PATH = os.path.join(BASE_PATH, '..', 'models')
    
    # Ensure directories exist
    os.makedirs(MODELS_PATH, exist_ok=True)
    
    # Model paths
    LIVENESS_MODEL_PATH = os.path.join(MODELS_PATH, 'face_liveness_model.h5')
    SPOOF_MODEL_PATH = os.path.join(MODELS_PATH, 'anti_spoof_model.h5')
    EMOTION_MODEL_PATH = os.path.join(MODELS_PATH, 'emotion_model.h5') 