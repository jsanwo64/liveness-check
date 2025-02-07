import os
import gdown
import zipfile
from tqdm import tqdm
from config import TrainingConfig

def download_emotion_dataset():
    """Download pre-processed emotion dataset"""
    print("Downloading pre-processed emotion dataset...")
    
    # Create emotion directory
    emotion_dir = os.path.join(TrainingConfig.DATA_PATH, 'emotion')
    os.makedirs(emotion_dir, exist_ok=True)
    
    # Using a pre-processed version of the FER2013 dataset from Google Drive
    dataset_urls = {
        'train': '1YlXqBhKzJ2-P5jQRyqH7LHqV9QmrhiNh',  # Training set
        'test': '1Yp3L97Mpbz-_7b9d_yxrxEiZLEYu2Qs8'   # Test set
    }
    
    try:
        for split, file_id in dataset_urls.items():
            print(f"\nDownloading {split} dataset...")
            output = os.path.join(emotion_dir, f'fer2013_{split}.zip')
            
            # Download using file ID
            gdown.download(f'https://drive.google.com/uc?id={file_id}', output, quiet=False)
            
            print(f"Extracting {split} dataset...")
            with zipfile.ZipFile(output, 'r') as zip_ref:
                zip_ref.extractall(os.path.join(emotion_dir, split))
            
            # Clean up
            os.remove(output)
        
        print("\nDataset downloaded and extracted successfully!")
        print("""
        Dataset structure:
        - training/data/emotion/
            - train/
                - angry/
                - disgust/
                - fear/
                - happy/
                - sad/
                - surprise/
                - neutral/
            - test/
                [same structure as train]
        """)
        
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        print("Please try again or check your internet connection.")
        
    print("""
    Next steps:
    1. Run the emotion model training script:
       python training/train_emotion.py
    
    2. The model will be saved as:
       models/emotion_model.h5
    """)

if __name__ == "__main__":
    download_emotion_dataset() 