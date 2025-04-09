import cv2
import numpy as np

def preprocess_face_for_siamese(face_img, target_size=(100, 100)):
    """
    Preprocess face image for Siamese Network
    
    Args:
        face_img: Face image from face detector
        target_size: Target size for the model
        
    Returns:
        processed_face: Face image ready for Siamese Network input
    """
    # Resize to target size
    face_resized = cv2.resize(face_img, target_size)
    
    # Convert to RGB if needed
    if len(face_resized.shape) == 2:  # Grayscale
        face_resized = cv2.cvtColor(face_resized, cv2.COLOR_GRAY2RGB)
    elif face_resized.shape[2] == 4:  # RGBA
        face_resized = cv2.cvtColor(face_resized, cv2.COLOR_RGBA2RGB)
    
    # Normalize pixel values to [0, 1]
    face_normalized = face_resized / 255.0
    
    # Add batch dimension
    face_batch = np.expand_dims(face_normalized, axis=0)
    
    return face_batch

def preprocess_face_for_ethnicity(face_img, target_size=(224, 224)):
    """
    Preprocess face image for Ethnicity Classifier
    
    Args:
        face_img: Face image from face detector
        target_size: Target size for the model
        
    Returns:
        processed_face: Face image ready for Ethnicity Classifier input
    """
    # Resize to target size
    face_resized = cv2.resize(face_img, target_size)
    
    # Convert to RGB if needed
    if len(face_resized.shape) == 2:  # Grayscale
        face_resized = cv2.cvtColor(face_resized, cv2.COLOR_GRAY2RGB)
    elif face_resized.shape[2] == 4:  # RGBA
        face_resized = cv2.cvtColor(face_resized, cv2.COLOR_RGBA2RGB)
    
    # Normalize pixel values to [0, 1]
    face_normalized = face_resized / 255.0
    
    # Add batch dimension
    face_batch = np.expand_dims(face_normalized, axis=0)
    
    return face_batch

def normalize_face(face_img):
    """
    Apply additional normalization techniques to enhance face image
    
    Args:
        face_img: Face image
        
    Returns:
        normalized_face: Enhanced face image
    """
    # Convert to BGR if grayscale
    if len(face_img.shape) == 2:
        face_img = cv2.cvtColor(face_img, cv2.COLOR_GRAY2BGR)
        
    # Convert to LAB color space for CLAHE
    lab = cv2.cvtColor(face_img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE to L channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    
    # Merge channels back
    normalized_lab = cv2.merge((cl, a, b))
    
    # Convert back to BGR
    normalized_face = cv2.cvtColor(normalized_lab, cv2.COLOR_LAB2BGR)
    
    return normalized_face