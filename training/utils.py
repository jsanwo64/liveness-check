import os
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from config import TrainingConfig

def create_data_generator(augmentation=True):
    """Create data generator with optional augmentation"""
    if augmentation:
        return ImageDataGenerator(
            preprocessing_function=preprocess_input,
            validation_split=TrainingConfig.VALIDATION_SPLIT,
            **TrainingConfig.AUGMENTATION_PARAMS
        )
    return ImageDataGenerator(
        preprocessing_function=preprocess_input,
        validation_split=TrainingConfig.VALIDATION_SPLIT
    )

def load_and_preprocess_image(image_path):
    """Load and preprocess a single image"""
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, TrainingConfig.IMAGE_SIZE)
    return preprocess_input(img)

def get_callbacks(model_path, patience=10):
    """Get common training callbacks"""
    return [
        ModelCheckpoint(
            model_path,
            monitor='val_loss',
            save_best_only=True,
            mode='min',
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=5,
            min_lr=1e-6,
            verbose=1
        )
    ]

def prepare_data_generators(data_dir, classes, batch_size=TrainingConfig.BATCH_SIZE):
    """Prepare train and validation generators"""
    datagen = create_data_generator(augmentation=True)
    
    train_generator = datagen.flow_from_directory(
        data_dir,
        target_size=TrainingConfig.IMAGE_SIZE,
        batch_size=batch_size,
        classes=classes,
        subset='training',
        shuffle=True
    )
    
    val_generator = datagen.flow_from_directory(
        data_dir,
        target_size=TrainingConfig.IMAGE_SIZE,
        batch_size=batch_size,
        classes=classes,
        subset='validation',
        shuffle=False
    )
    
    return train_generator, val_generator

def create_base_model(input_shape=(224, 224, 3)):
    """Create base model using MobileNetV2"""
    from tensorflow.keras.applications import MobileNetV2
    from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
    from tensorflow.keras.models import Model
    
    base_model = MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )
    
    # Freeze the base model layers
    base_model.trainable = False
    
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.3)(x)
    
    return base_model, x 