import os
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from config import TrainingConfig
from utils import prepare_data_generators, get_callbacks, create_base_model

def create_anti_spoof_model():
    """Create and compile the anti-spoofing model"""
    base_model, x = create_base_model()
    
    # Add output layer with more neurons for multiple spoof types
    x = Dense(256, activation='relu')(x)
    predictions = Dense(len(TrainingConfig.SPOOF_CLASSES), activation='softmax')(x)
    
    # Create model
    model = Model(inputs=base_model.input, outputs=predictions)
    
    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=TrainingConfig.SPOOF_LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train_anti_spoof_model():
    """Train the anti-spoofing model"""
    print("Preparing data generators...")
    data_dir = os.path.join(TrainingConfig.DATA_PATH, 'anti_spoof')
    
    if not os.path.exists(data_dir):
        raise ValueError(f"Data directory not found: {data_dir}")
    
    train_generator, val_generator = prepare_data_generators(
        data_dir,
        TrainingConfig.SPOOF_CLASSES
    )
    
    print("Creating model...")
    model = create_anti_spoof_model()
    
    print("Training model...")
    model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=TrainingConfig.EPOCHS,
        callbacks=get_callbacks(TrainingConfig.SPOOF_MODEL_PATH),
        workers=4,
        use_multiprocessing=True
    )
    
    print(f"Model saved to {TrainingConfig.SPOOF_MODEL_PATH}")

if __name__ == "__main__":
    # Enable mixed precision training for faster training
    tf.keras.mixed_precision.set_global_policy('mixed_float16')
    
    try:
        train_anti_spoof_model()
    except Exception as e:
        print(f"Training failed: {e}") 