import os
import cv2
import numpy as np
from datetime import datetime
import argparse
from config import TrainingConfig

class DataCollector:
    def __init__(self, model_type, class_name):
        self.model_type = model_type
        self.class_name = class_name
        self.output_dir = os.path.join(
            TrainingConfig.DATA_PATH,
            model_type,
            'custom',
            class_name
        )
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
    
    def capture_images(self, num_images=100):
        """Capture images from webcam"""
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Could not open webcam")
            return
        
        count = 0
        while count < num_images:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame")
                break
            
            # Convert to RGB for display
            display_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Detect faces
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )
            
            # Draw rectangles around faces
            for (x, y, w, h) in faces:
                cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Display frame
            cv2.imshow('Capture', cv2.cvtColor(display_frame, cv2.COLOR_RGB2BGR))
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s') and len(faces) > 0:
                # Save the image
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                filename = f"{self.class_name}_{timestamp}.jpg"
                filepath = os.path.join(self.output_dir, filename)
                
                # Save the largest face
                if len(faces) > 0:
                    # Get the largest face
                    largest_face = max(faces, key=lambda rect: rect[2] * rect[3])
                    x, y, w, h = largest_face
                    
                    # Add margin
                    margin = int(0.2 * max(w, h))
                    x = max(0, x - margin)
                    y = max(0, y - margin)
                    w = min(frame.shape[1] - x, w + 2 * margin)
                    h = min(frame.shape[0] - y, h + 2 * margin)
                    
                    # Crop and save face
                    face_img = frame[y:y+h, x:x+w]
                    face_img = cv2.resize(face_img, TrainingConfig.IMAGE_SIZE)
                    cv2.imwrite(filepath, face_img)
                    
                    count += 1
                    print(f"Saved image {count}/{num_images}: {filename}")
        
        cap.release()
        cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description='Collect custom training data')
    parser.add_argument('model_type', choices=['emotion', 'anti_spoof', 'face_liveness'],
                      help='Type of model to collect data for')
    parser.add_argument('class_name', help='Class name to collect data for')
    parser.add_argument('--num_images', type=int, default=100,
                      help='Number of images to collect')
    
    args = parser.parse_args()
    
    # Verify valid class name
    valid_classes = {
        'emotion': TrainingConfig.EMOTION_CLASSES,
        'anti_spoof': TrainingConfig.SPOOF_CLASSES,
        'face_liveness': TrainingConfig.LIVENESS_CLASSES
    }
    
    if args.class_name not in valid_classes[args.model_type]:
        print(f"Error: Invalid class name for {args.model_type}")
        print(f"Valid classes are: {valid_classes[args.model_type]}")
        return
    
    collector = DataCollector(args.model_type, args.class_name)
    
    print(f"""
    Starting data collection for {args.model_type} - {args.class_name}
    
    Instructions:
    - Position your face in the frame
    - Press 's' to save an image when face is detected
    - Press 'q' to quit
    
    Target: {args.num_images} images
    """)
    
    collector.capture_images(args.num_images)

if __name__ == "__main__":
    main() 