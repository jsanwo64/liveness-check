import os
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from config import TrainingConfig

def create_and_save_models():
    """Create and save models using transfer learning with MobileNetV2"""
    print("Creating models using transfer learning...")
    
    # Create models directory
    os.makedirs(TrainingConfig.MODELS_PATH, exist_ok=True)
    
    # Base model configuration
    input_shape = (224, 224, 3)
    base_model = MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False
    
    # Create emotion model
    print("\nCreating emotion detection model...")
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.3)(x)
    emotion_predictions = Dense(len(TrainingConfig.EMOTION_CLASSES), activation='softmax')(x)
    emotion_model = Model(inputs=base_model.input, outputs=emotion_predictions)
    emotion_model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    emotion_model.save(TrainingConfig.EMOTION_MODEL_PATH)
    print(f"Emotion model saved to {TrainingConfig.EMOTION_MODEL_PATH}")
    
    # Create face liveness model
    print("\nCreating face liveness detection model...")
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    liveness_predictions = Dense(len(TrainingConfig.LIVENESS_CLASSES), activation='softmax')(x)
    liveness_model = Model(inputs=base_model.input, outputs=liveness_predictions)
    liveness_model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    liveness_model.save(TrainingConfig.LIVENESS_MODEL_PATH)
    print(f"Liveness model saved to {TrainingConfig.LIVENESS_MODEL_PATH}")
    
    # Create anti-spoofing model
    print("\nCreating anti-spoofing model...")
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.3)(x)
    spoof_predictions = Dense(len(TrainingConfig.SPOOF_CLASSES), activation='softmax')(x)
    spoof_model = Model(inputs=base_model.input, outputs=spoof_predictions)
    spoof_model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    spoof_model.save(TrainingConfig.SPOOF_MODEL_PATH)
    print(f"Anti-spoofing model saved to {TrainingConfig.SPOOF_MODEL_PATH}")
    
    print("""
    All models have been created and saved successfully!
    
    Models created:
    1. Emotion Detection Model
       - Based on MobileNetV2
       - 7 emotion classes
       - Pre-trained on ImageNet
    
    2. Face Liveness Detection Model
       - Based on MobileNetV2
       - Binary classification (real/fake)
       - Pre-trained on ImageNet
    
    3. Anti-spoofing Model
       - Based on MobileNetV2
       - 4 classes (real/print/replay/mask)
       - Pre-trained on ImageNet
    
    Next steps:
    1. Run your FastAPI application:
       python app.py
    
    2. The models will need to be trained on specific data
       for better accuracy. You can use the training scripts:
       - python training/train_emotion.py
       - python training/train_liveness.py
       - python training/train_anti_spoof.py
    """)

if __name__ == "__main__":
    create_and_save_models() 