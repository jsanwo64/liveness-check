import os
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from config import TrainingConfig
from utils import prepare_data_generators, get_callbacks, create_base_model

def create_emotion_model():
    """Create and compile the emotion detection model"""
    base_model, x = create_base_model()
    
    # Add more layers for complex emotion detection
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.4)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.3)(x)
    predictions = Dense(len(TrainingConfig.EMOTION_CLASSES), activation='softmax')(x)
    
    # Create model
    model = Model(inputs=base_model.input, outputs=predictions)
    
    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=TrainingConfig.EMOTION_LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train_emotion_model():
    """Train the emotion detection model"""
    print("Preparing data generators...")
    data_dir = os.path.join(TrainingConfig.DATA_PATH, 'emotion')
    
    if not os.path.exists(data_dir):
        raise ValueError(f"Data directory not found: {data_dir}")
    
    train_generator, val_generator = prepare_data_generators(
        data_dir,
        TrainingConfig.EMOTION_CLASSES
    )
    
    print("Creating model...")
    model = create_emotion_model()
    
    print("Training model...")
    model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=TrainingConfig.EPOCHS,
        callbacks=get_callbacks(TrainingConfig.EMOTION_MODEL_PATH),
        workers=4,
        use_multiprocessing=True
    )
    
    print(f"Model saved to {TrainingConfig.EMOTION_MODEL_PATH}")

if __name__ == "__main__":
    # Enable mixed precision training for faster training
    tf.keras.mixed_precision.set_global_policy('mixed_float16')
    
    try:
        train_emotion_model()
    except Exception as e:
        print(f"Training failed: {e}") 