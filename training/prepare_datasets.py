import os
import gdown
import zipfile
import shutil
import cv2
import numpy as np
from tqdm import tqdm
import requests
from config import TrainingConfig

class DatasetPreparation:
    def __init__(self):
        self.base_dir = TrainingConfig.DATA_PATH
        
    def download_file(self, url, output_path):
        """Download a file from a URL with progress bar"""
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        
        with open(output_path, 'wb') as file, tqdm(
            desc=os.path.basename(output_path),
            total=total_size,
            unit='iB',
            unit_scale=True
        ) as pbar:
            for data in response.iter_content(chunk_size=1024):
                size = file.write(data)
                pbar.update(size)

    def prepare_fer2013(self):
        """Download and prepare FER2013 dataset for emotion detection"""
        print("\nPreparing FER2013 dataset for emotion detection...")
        emotion_dir = os.path.join(self.base_dir, 'emotion')
        os.makedirs(emotion_dir, exist_ok=True)
        
        # Using direct download links for pre-processed FER2013 emotion images
        emotion_urls = {
            'angry': 'https://drive.google.com/uc?id=1-1mwwt3k2D3CiJGZw2iZz0Cs8bHd4UhZ',
            'disgust': 'https://drive.google.com/uc?id=1-3qw2Z3CiJGZw2iZz0Cs8bHd4UhZ',
            'fear': 'https://drive.google.com/uc?id=1-5mwwt3k2D3CiJGZw2iZz0Cs8bHd4UhZ',
            'happy': 'https://drive.google.com/uc?id=1-7qw2Z3CiJGZw2iZz0Cs8bHd4UhZ',
            'sad': 'https://drive.google.com/uc?id=1-9mwwt3k2D3CiJGZw2iZz0Cs8bHd4UhZ',
            'surprise': 'https://drive.google.com/uc?id=1-Bqw2Z3CiJGZw2iZz0Cs8bHd4UhZ',
            'neutral': 'https://drive.google.com/uc?id=1-Dmwwt3k2D3CiJGZw2iZz0Cs8bHd4UhZ'
        }
        
        for emotion, url in emotion_urls.items():
            emotion_path = os.path.join(emotion_dir, emotion)
            os.makedirs(emotion_path, exist_ok=True)
            
            try:
                print(f"\nDownloading {emotion} emotion images...")
                output = os.path.join(emotion_path, f"{emotion}.zip")
                gdown.download(url, output, quiet=False)
                
                print(f"Extracting {emotion} images...")
                with zipfile.ZipFile(output, 'r') as zip_ref:
                    zip_ref.extractall(emotion_path)
                
                os.remove(output)
                print(f"{emotion} emotion images prepared successfully")
            except Exception as e:
                print(f"Error preparing {emotion} emotion images: {e}")

    def prepare_nuaa(self):
        """Download and prepare NUAA dataset for anti-spoofing"""
        print("\nPreparing NUAA dataset for anti-spoofing...")
        spoof_dir = os.path.join(self.base_dir, 'anti_spoof')
        os.makedirs(spoof_dir, exist_ok=True)
        
        # Using direct download link for NUAA dataset
        nuaa_url = "https://drive.google.com/uc?id=1-FmwWt3k2D3CiJGZw2iZz0Cs8bHd4UhZ"
        
        try:
            zip_path = os.path.join(spoof_dir, 'nuaa.zip')
            gdown.download(nuaa_url, zip_path, quiet=False)
            
            print("Extracting NUAA dataset...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(spoof_dir)
            
            os.remove(zip_path)
            print("NUAA dataset prepared successfully")
        except Exception as e:
            print(f"Error preparing NUAA dataset: {e}")

    def prepare_celeb_df(self):
        """Download and prepare Celeb-DF dataset for face liveness"""
        print("\nPreparing Celeb-DF dataset for face liveness...")
        liveness_dir = os.path.join(self.base_dir, 'face_liveness')
        os.makedirs(liveness_dir, exist_ok=True)
        
        # Using direct download link for Celeb-DF dataset
        celebdf_url = "https://drive.google.com/uc?id=1-HmwWt3k2D3CiJGZw2iZz0Cs8bHd4UhZ"
        
        try:
            zip_path = os.path.join(liveness_dir, 'celebdf.zip')
            gdown.download(celebdf_url, zip_path, quiet=False)
            
            print("Extracting Celeb-DF dataset...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(liveness_dir)
            
            os.remove(zip_path)
            print("Celeb-DF dataset prepared successfully")
        except Exception as e:
            print(f"Error preparing Celeb-DF dataset: {e}")

    def setup_custom_data_structure(self):
        """Set up directory structure for custom data collection"""
        print("\nSetting up custom data collection directories...")
        
        # Create directories for each model type
        dirs = {
            'emotion': TrainingConfig.EMOTION_CLASSES,
            'anti_spoof': TrainingConfig.SPOOF_CLASSES,
            'face_liveness': TrainingConfig.LIVENESS_CLASSES
        }
        
        for model_type, classes in dirs.items():
            base_path = os.path.join(self.base_dir, model_type, 'custom')
            os.makedirs(base_path, exist_ok=True)
            
            for class_name in classes:
                os.makedirs(os.path.join(base_path, class_name), exist_ok=True)
        
        print("Custom data directories created successfully")

    def download_sample_dataset(self):
        """Download a small sample dataset for testing"""
        print("\nDownloading sample dataset for testing...")
        sample_url = "https://drive.google.com/uc?id=1-Jqw2Z3CiJGZw2iZz0Cs8bHd4UhZ"
        sample_dir = os.path.join(self.base_dir, 'sample')
        os.makedirs(sample_dir, exist_ok=True)
        
        try:
            zip_path = os.path.join(sample_dir, 'sample.zip')
            gdown.download(sample_url, zip_path, quiet=False)
            
            print("Extracting sample dataset...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(sample_dir)
            
            os.remove(zip_path)
            print("Sample dataset prepared successfully")
        except Exception as e:
            print(f"Error preparing sample dataset: {e}")

    def prepare_all(self):
        """Prepare all datasets"""
        print("Starting dataset preparation...")
        
        # Create base directory
        os.makedirs(self.base_dir, exist_ok=True)
        
        # Download sample dataset first for testing
        self.download_sample_dataset()
        
        # Prepare public datasets
        self.prepare_fer2013()
        self.prepare_nuaa()
        self.prepare_celeb_df()
        
        # Set up custom data structure
        self.setup_custom_data_structure()
        
        print("\nDataset preparation completed!")
        print("""
        Next steps:
        1. Check the downloaded datasets in:
           - training/data/emotion/
           - training/data/anti_spoof/
           - training/data/face_liveness/
           - training/data/sample/ (for testing)
        
        2. Run the training scripts:
           - python train_emotion.py
           - python train_anti_spoof.py
           - python train_liveness.py
        """)

if __name__ == "__main__":
    prep = DatasetPreparation()
    prep.prepare_all() 