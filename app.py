import os
import cv2
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Request
from fastapi.responses import JSONResponse
from datetime import datetime, timedelta
from typing import Optional
import secrets
from redis import asyncio as aioredis
import logging
from PIL import Image
import io
import dlib
import mediapipe as mp
from scipy.spatial import distance
import json
from ratelimit import limits, RateLimitException
import uvicorn
from contextlib import asynccontextmanager
from dotenv import load_dotenv
import psutil

# Load environment variables
load_dotenv()

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SecurityConfig:
    RATE_LIMIT_CALLS = int(os.getenv("RATE_LIMIT_CALLS", 100))
    RATE_LIMIT_PERIOD = int(os.getenv("RATE_LIMIT_PERIOD", 3600))
    MAX_IMAGE_SIZE = int(os.getenv("MAX_IMAGE_SIZE", 10 * 1024 * 1024))
    ALLOWED_IMAGE_TYPES = {'image/jpeg', 'image/png'}
    CHALLENGE_EXPIRY = int(os.getenv("CHALLENGE_EXPIRY", 30))
    SESSION_EXPIRY = int(os.getenv("SESSION_EXPIRY", 3600))
    MIN_FACE_SIZE = int(os.getenv("MIN_FACE_SIZE", 100))
    CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", 0.95))
    MAX_ATTEMPTS = int(os.getenv("MAX_ATTEMPTS", 3))
    
    REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")

class ModelLoader:
    @staticmethod
    async def load_model(model_path: str) -> tf.keras.Model:
        try:
            # Check if model exists in current directory or models directory
            model_locations = [
                model_path,
                os.path.join('models', model_path),
                os.path.join(os.path.dirname(__file__), model_path),
                os.path.join(os.path.dirname(__file__), 'models', model_path)
            ]
            
            for location in model_locations:
                if os.path.exists(location):
                    return tf.keras.models.load_model(location)
            
            logger.error(f"Model not found in any of these locations: {model_locations}")
            raise FileNotFoundError(f"Model not found: {model_path}")
        except Exception as e:
            logger.error(f"Failed to load model {model_path}: {e}")
            raise RuntimeError(f"Model loading failed: {model_path}")

class LivenessDetector:
    def __init__(self):
        self.models = {}
        self.face_detector = None
        self.landmark_predictor = None
        self.face_mesh = None
        
    async def initialize(self):
        try:
            # Initialize face detector first as it's required
            self.face_detector = dlib.get_frontal_face_detector()
            
            # Try to load landmark predictor
            landmark_predictor_paths = [
                'shape_predictor_68_face_landmarks.dat',
                os.path.join('models', 'shape_predictor_68_face_landmarks.dat'),
                os.path.join(os.path.dirname(__file__), 'shape_predictor_68_face_landmarks.dat'),
                os.path.join(os.path.dirname(__file__), 'models', 'shape_predictor_68_face_landmarks.dat')
            ]
            
            predictor_loaded = False
            for path in landmark_predictor_paths:
                if os.path.exists(path):
                    self.landmark_predictor = dlib.shape_predictor(path)
                    predictor_loaded = True
                    break
            
            if not predictor_loaded:
                logger.warning("Landmark predictor model not found. Some functionality may be limited.")
            
            # Try to load the models, but continue if some are missing
            model_files = {
                'face': 'face_liveness_model.h5',
                'spoof': 'anti_spoof_model.h5',
                'emotion': 'emotion_model.h5'
            }
            
            for model_name, model_file in model_files.items():
                try:
                    self.models[model_name] = await ModelLoader.load_model(model_file)
                except Exception as e:
                    logger.warning(f"Failed to load {model_name} model: {e}")
            
            self.mp_face_mesh = mp.solutions.face_mesh
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                max_num_faces=1,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            
            if not self.models:
                raise RuntimeError("No models could be loaded successfully")
                
        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            raise

class LivenessAPI:
    def __init__(self):
        self.detector = None
        self.redis_client = None
        
    async def initialize(self):
        try:
            self.redis_client = await aioredis.from_url(
                SecurityConfig.REDIS_URL,
                encoding="utf-8",
                decode_responses=True
            )
            
            self.detector = LivenessDetector()
            await self.detector.initialize()
            
        except Exception as e:
            logger.error(f"API initialization failed: {e}")
            raise

    @staticmethod
    def generate_session_id() -> str:
        return secrets.token_urlsafe(32)

    @staticmethod
    def generate_challenge() -> str:
        return secrets.token_urlsafe(16)

    async def store_challenge(self, session_id: str, challenge: str):
        try:
            await self.redis_client.setex(
                f"challenge:{session_id}",
                SecurityConfig.CHALLENGE_EXPIRY,
                challenge
            )
        except Exception as e:
            logger.error(f"Failed to store challenge: {e}")
            raise HTTPException(status_code=500, detail="Failed to generate challenge")

@asynccontextmanager
async def lifespan(app: FastAPI):
    api = LivenessAPI()
    try:
        await api.initialize()
        app.state.api = api
        yield
    except Exception as e:
        logger.error(f"Application startup failed: {e}")
        raise
    finally:
        if hasattr(app.state, 'api') and app.state.api.redis_client:
            await app.state.api.redis_client.close()

app = FastAPI(lifespan=lifespan)

@app.get("/health")
async def health_check():
    """
    Comprehensive health check endpoint that verifies:
    1. Redis connection
    2. Model availability and loading
    3. System resources
    """
    health_status = {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "components": {
            "redis": {"status": "unknown"},
            "models": {
                "emotion": {"status": "unknown"},
                "liveness": {"status": "unknown"},
                "anti_spoof": {"status": "unknown"}
            },
            "system": {
                "status": "unknown",
                "memory_usage": None
            }
        }
    }
    
    try:
        # Check Redis connection
        try:
            await app.state.api.redis_client.ping()
            health_status["components"]["redis"] = {
                "status": "healthy",
                "message": "Connection successful"
            }
        except Exception as e:
            health_status["components"]["redis"] = {
                "status": "unhealthy",
                "message": str(e)
            }
            health_status["status"] = "degraded"
        
        # Check models
        models = app.state.api.detector.models
        for model_name in ["face", "spoof", "emotion"]:
            if model_name in models and models[model_name] is not None:
                health_status["components"]["models"][model_name] = {
                    "status": "healthy",
                    "message": "Model loaded successfully"
                }
            else:
                health_status["components"]["models"][model_name] = {
                    "status": "unhealthy",
                    "message": "Model not loaded"
                }
                health_status["status"] = "degraded"
        
        # Check system resources
        memory = psutil.virtual_memory()
        health_status["components"]["system"] = {
            "status": "healthy",
            "memory_usage": {
                "total": memory.total,
                "available": memory.available,
                "percent": memory.percent
            }
        }
        
        if memory.percent > 90:  # Warning if memory usage is above 90%
            health_status["components"]["system"]["status"] = "warning"
            health_status["status"] = "degraded"
        
        # Overall status determination
        if any(comp["status"] == "unhealthy" for comp in [
            health_status["components"]["redis"],
            *health_status["components"]["models"].values()
        ]):
            health_status["status"] = "unhealthy"
        elif any(comp["status"] == "warning" for comp in [
            health_status["components"]["redis"],
            *health_status["components"]["models"].values(),
            health_status["components"]["system"]
        ]):
            health_status["status"] = "degraded"
        
        return health_status
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "timestamp": datetime.now().isoformat(),
                "error": str(e)
            }
        )

@app.post("/initialize-session/")
async def initialize_session():
    try:
        session_id = app.state.api.generate_session_id()
        challenge = app.state.api.generate_challenge()
        await app.state.api.store_challenge(session_id, challenge)
        return JSONResponse(content={
            "session_id": session_id,
            "initial_challenge": challenge
        })
    except Exception as e:
        logger.error(f"Session initialization failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to initialize session")

@app.post("/liveness/check")
async def check_liveness(
    request: Request,
    file: UploadFile = File(...),
    session_id: Optional[str] = None
):
    """
    Check liveness from uploaded image
    Returns liveness score, emotion, and anti-spoofing results
    """
    try:
        # Validate session if provided
        if session_id:
            challenge = await app.state.api.redis_client.get(f"challenge:{session_id}")
            if not challenge:
                raise HTTPException(
                    status_code=400,
                    detail="Invalid or expired session"
                )

        # Validate file size
        contents = await file.read()
        if len(contents) > SecurityConfig.MAX_IMAGE_SIZE:
            raise HTTPException(
                status_code=400,
                detail="File size too large"
            )

        # Validate file type
        if file.content_type not in SecurityConfig.ALLOWED_IMAGE_TYPES:
            raise HTTPException(
                status_code=400,
                detail="Invalid file type"
            )

        # Convert image to format needed by models
        image = Image.open(io.BytesIO(contents))
        image = image.convert('RGB')
        image_np = np.array(image)
        
        # Detect faces
        faces = app.state.api.detector.face_detector(image_np)
        if not faces:
            raise HTTPException(
                status_code=400,
                detail="No face detected in image"
            )
        
        # Get the largest face
        face = max(faces, key=lambda rect: rect.width() * rect.height())
        
        # Check minimum face size
        if face.width() < SecurityConfig.MIN_FACE_SIZE or face.height() < SecurityConfig.MIN_FACE_SIZE:
            raise HTTPException(
                status_code=400,
                detail="Face too small in image"
            )

        # Extract face region with margin
        margin = int(0.2 * max(face.width(), face.height()))
        x = max(0, face.left() - margin)
        y = max(0, face.top() - margin)
        w = min(image_np.shape[1] - x, face.width() + 2 * margin)
        h = min(image_np.shape[0] - y, face.height() + 2 * margin)
        
        face_img = image_np[y:y+h, x:x+w]
        face_img = cv2.resize(face_img, (224, 224))  # Size expected by models
        face_img = face_img.astype('float32') / 255.0
        face_img = np.expand_dims(face_img, axis=0)

        # Get predictions from all models
        results = {
            "liveness_score": None,
            "anti_spoof_result": None,
            "emotion": None,
            "confidence_scores": {}
        }

        # Face liveness detection
        if 'face' in app.state.api.detector.models:
            liveness_pred = app.state.api.detector.models['face'].predict(face_img, verbose=0)
            results["liveness_score"] = float(liveness_pred[0][1])  # Probability of being real
            results["confidence_scores"]["liveness"] = {
                "real": float(liveness_pred[0][1]),
                "fake": float(liveness_pred[0][0])
            }

        # Anti-spoofing detection
        if 'spoof' in app.state.api.detector.models:
            spoof_pred = app.state.api.detector.models['spoof'].predict(face_img, verbose=0)
            spoof_class_idx = np.argmax(spoof_pred[0])
            results["anti_spoof_result"] = TrainingConfig.SPOOF_CLASSES[spoof_class_idx]
            results["confidence_scores"]["anti_spoof"] = {
                class_name: float(score)
                for class_name, score in zip(TrainingConfig.SPOOF_CLASSES, spoof_pred[0])
            }

        # Emotion detection
        if 'emotion' in app.state.api.detector.models:
            emotion_pred = app.state.api.detector.models['emotion'].predict(face_img, verbose=0)
            emotion_class_idx = np.argmax(emotion_pred[0])
            results["emotion"] = TrainingConfig.EMOTION_CLASSES[emotion_class_idx]
            results["confidence_scores"]["emotion"] = {
                class_name: float(score)
                for class_name, score in zip(TrainingConfig.EMOTION_CLASSES, emotion_pred[0])
            }

        # Add face location in image
        results["face_location"] = {
            "x": int(x),
            "y": int(y),
            "width": int(w),
            "height": int(h)
        }

        # Determine overall liveness status
        is_live = results["liveness_score"] is not None and results["liveness_score"] > SecurityConfig.CONFIDENCE_THRESHOLD
        is_real = results["anti_spoof_result"] == "real" if results["anti_spoof_result"] else None

        results["overall_status"] = {
            "is_live": is_live,
            "is_real": is_real,
            "passed": is_live and (is_real or is_real is None)
        }

        return JSONResponse(content=results)

    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error processing liveness check: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing image: {str(e)}"
        )
    finally:
        # Reset file pointer and close
        await file.seek(0)
        await file.close()

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
