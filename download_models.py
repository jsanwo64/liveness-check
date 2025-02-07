import os
import requests
import bz2
from tqdm import tqdm

def download_file(url, filename):
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(filename, 'wb') as file, tqdm(
        desc=filename,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as progress_bar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            progress_bar.update(size)

def main():
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Download facial landmarks model
    print("Downloading facial landmarks model...")
    landmark_url = "https://github.com/davisking/dlib-models/raw/master/shape_predictor_68_face_landmarks.dat.bz2"
    compressed_file = "models/shape_predictor_68_face_landmarks.dat.bz2"
    
    download_file(landmark_url, compressed_file)
    
    print("Extracting facial landmarks model...")
    with bz2.open(compressed_file) as f_in, open('models/shape_predictor_68_face_landmarks.dat', 'wb') as f_out:
        f_out.write(f_in.read())
    
    # Clean up compressed file
    os.remove(compressed_file)
    
    # Download other models from their respective sources
    # Note: You'll need to replace these URLs with actual URLs to your pre-trained models
    models = {
        'face_liveness_model.h5': 'URL_TO_FACE_LIVENESS_MODEL',
        'anti_spoof_model.h5': 'URL_TO_ANTI_SPOOF_MODEL',
        'emotion_model.h5': 'URL_TO_EMOTION_MODEL'
    }
    
    for model_name, url in models.items():
        print(f"\nDownloading {model_name}...")
        try:
            download_file(url, f"models/{model_name}")
        except Exception as e:
            print(f"Failed to download {model_name}: {e}")
            print("You'll need to manually provide this model.")

if __name__ == "__main__":
    main() 