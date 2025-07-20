import cv2
import numpy as np
from deepface import DeepFace
import os
from typing import Tuple, Dict, Any
import time
import warnings
warnings.filterwarnings('ignore')

def detect_face(img1_path: str, img2_path: str) -> Dict[str, Any]:
    """
    Detect and compare faces in two images using DeepFace's FaceNet512 model.
    
    Args:
        img1_path (str): Path to the first image
        img2_path (str): Path to the second image
    
    Returns:
        Dict[str, Any]: Dictionary containing all analysis results and statistics
    """
    
    # Verify that both image files exist
    if not os.path.exists(img1_path):
        raise FileNotFoundError(f"Image 1 not found: {img1_path}")
    if not os.path.exists(img2_path):
        raise FileNotFoundError(f"Image 2 not found: {img2_path}")
    
    # Start timing
    start_time = time.time()
    
    # Initialize results dictionary
    results = {
        'verified': None,
        'distance': None,
        'threshold': 0.40,  # Custom threshold
        'model_name': 'FaceNet512',
        'distance_metric': 'cosine',
        'faces_detected_img1': 0,
        'faces_detected_img2': 0,
        'processing_time': 0,
        'face1_cropped': None,
        'face2_cropped': None
    }
    
    try:
        # Step 1: Face Verification using FaceNet512 with custom threshold
        verification_result = DeepFace.verify(
            img1_path=img1_path,
            img2_path=img2_path,
            model_name="Facenet512",
            enforce_detection=True,
            detector_backend="opencv",
            distance_metric="cosine"
        )
        
        # Apply custom threshold
        distance = verification_result['distance']
        results['distance'] = distance
        results['verified'] = distance < results['threshold']
        
        # Step 2: Extract face embeddings and count faces
        # Load images
        img1 = cv2.imread(img1_path)
        img2 = cv2.imread(img2_path)
        
        # Convert BGR to RGB for display
        img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
        
        # Extract face embeddings using FaceNet512
        embedding1 = DeepFace.represent(
            img_path=img1_path,
            model_name="Facenet512",
            enforce_detection=True,
            detector_backend="opencv"
        )
        
        embedding2 = DeepFace.represent(
            img_path=img2_path,
            model_name="Facenet512",
            enforce_detection=True,
            detector_backend="opencv"
        )
        
        # Count faces detected
        results['faces_detected_img1'] = len(embedding1)
        results['faces_detected_img2'] = len(embedding2)
        
        # Step 3: Extract cropped face regions
        # Load the face cascade classifier for face detection
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Detect faces
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        
        faces1 = face_cascade.detectMultiScale(gray1, 1.1, 4)
        faces2 = face_cascade.detectMultiScale(gray2, 1.1, 4)
        
        # Select the largest face from each image
        def get_largest_face(faces):
            if len(faces) == 0:
                return None
            # Sort by area (width * height) and return the largest
            largest_face = max(faces, key=lambda x: x[2] * x[3])
            return largest_face
        
        face1_bbox = get_largest_face(faces1)
        face2_bbox = get_largest_face(faces2)
        
        # Extract cropped face regions
        if face1_bbox is not None:
            x, y, w, h = face1_bbox
            face1_region = img1_rgb[y:y+h, x:x+w]
            face1_region = cv2.resize(face1_region, (160, 160))
            results['face1_cropped'] = face1_region
        else:
            results['face1_cropped'] = None
            
        if face2_bbox is not None:
            x, y, w, h = face2_bbox
            face2_region = img2_rgb[y:y+h, x:x+w]
            face2_region = cv2.resize(face2_region, (160, 160))
            results['face2_cropped'] = face2_region
        else:
            results['face2_cropped'] = None
        
        # Calculate processing time
        results['processing_time'] = time.time() - start_time
        
        return results
        
    except Exception as e:
        # Calculate processing time even if error occurs
        results['processing_time'] = time.time() - start_time
        raise e