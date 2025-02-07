import os
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from config import TrainingConfig
from utils import prepare_data_generators, get_callbacks, create_base_model

def create_liveness_model():
    """Create and compile the face liveness detection model"""
    base_model, x = create_base_model()
    
    # Add output layer
    predictions = Dense(len(TrainingConfig.LIVENESS_CLASSES), activation='softmax')(x)
    
    # Create model
    model = Model(inputs=base_model.input, outputs=predictions)
    
    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=TrainingConfig.LIVENESS_LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train_liveness_model():
    """Train the face liveness detection model"""
    print("Preparing data generators...")
    data_dir = os.path.join(TrainingConfig.DATA_PATH, 'face_liveness')
    
    if not os.path.exists(data_dir):
        raise ValueError(f"Data directory not found: {data_dir}")
    
    train_generator, val_generator = prepare_data_generators(
        data_dir,
        TrainingConfig.LIVENESS_CLASSES
    )
    
    print("Creating model...")
    model = create_liveness_model()
    
    print("Training model...")
    model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=TrainingConfig.EPOCHS,
        callbacks=get_callbacks(TrainingConfig.LIVENESS_MODEL_PATH),
        workers=4,
        use_multiprocessing=True
    )
    
    print(f"Model saved to {TrainingConfig.LIVENESS_MODEL_PATH}")

if __name__ == "__main__":
    # Enable mixed precision training for faster training
    tf.keras.mixed_precision.set_global_policy('mixed_float16')
    
    try:
        train_liveness_model()
    except Exception as e:
        print(f"Training failed: {e}") 