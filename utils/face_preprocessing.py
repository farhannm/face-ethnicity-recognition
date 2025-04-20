import cv2
import numpy as np
import os

def preprocess_face_for_facenet(face_img, target_size=(160, 160)):
    """
    Preprocess face image for FaceNet model input
    
    Args:
        face_img: Aligned face image
        target_size: Target size for FaceNet (default: 160x160)
        
    Returns:
        processed_face: Face image ready for FaceNet input
    """
    # Resize to FaceNet input size
    face_resized = cv2.resize(face_img, target_size)
    
    # Convert to RGB if needed (FaceNet expects RGB)
    if len(face_resized.shape) == 2:  # Grayscale
        face_resized = cv2.cvtColor(face_resized, cv2.COLOR_GRAY2RGB)
    elif face_resized.shape[2] == 4:  # RGBA
        face_resized = cv2.cvtColor(face_resized, cv2.COLOR_RGBA2RGB)
    elif face_resized.shape[2] == 3:  # BGR (OpenCV default)
        face_resized = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
    
    # Convert to float32
    face_normalized = face_resized.astype(np.float32)
    
    # Scale pixel values to [0, 1]
    face_normalized /= 255.0
    
    # Standardize image (mean=0, std=1)
    mean = np.mean(face_normalized, axis=(0, 1, 2))
    std = np.std(face_normalized, axis=(0, 1, 2))
    std_adj = np.maximum(std, 1.0/np.sqrt(face_normalized.size))
    face_normalized = (face_normalized - mean) / std_adj
    
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

def align_face_with_landmarks(face_img, landmarks, target_size=(224, 224)):
    """
    Align face based on eye landmarks
    
    Args:
        face_img: Input face image
        landmarks: Dictionary containing facial landmarks
        target_size: Target size for aligned face
        
    Returns:
        aligned_face: Aligned face image
    """
    # Get eye landmarks
    left_eye = landmarks['left_eye']
    right_eye = landmarks['right_eye']
    
    # Calculate angle for alignment
    dx = right_eye[0] - left_eye[0]
    dy = right_eye[1] - left_eye[1]
    
    if dx == 0:
        angle = 0
    else:
        angle = np.degrees(np.arctan2(dy, dx))
    
    # Get center point between eyes
    center_x = (left_eye[0] + right_eye[0]) // 2
    center_y = (left_eye[1] + right_eye[1]) // 2
    center = (center_x, center_y)
    
    # Get rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # Determine output image size
    h, w = face_img.shape[:2]
    
    # Apply rotation
    aligned_face = cv2.warpAffine(
        face_img, 
        rotation_matrix, 
        (w, h),
        flags=cv2.INTER_CUBIC, 
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0)
    )
    
    # Resize to target size
    aligned_face = cv2.resize(aligned_face, target_size)
    
    return aligned_face

def parse_image_metadata(filename):
    """
    Parse metadata from image filename
    
    Expected format: Nama_Suku_Ekspresi_Sudut_Pencahayaan_Jarak.ext
    Or simpler format: Nama_Suku_Ekspresi_Sudut.ext
    
    Args:
        filename: Image filename
        
    Returns:
        metadata: Dictionary with extracted metadata
    """
    # Remove extension
    base_name = os.path.splitext(filename)[0]
    parts = base_name.split('_')
    
    metadata = {
        'nama': parts[0] if len(parts) > 0 else 'unknown',
        'suku': parts[1] if len(parts) > 1 else 'unknown',
        'ekspresi': parts[2] if len(parts) > 2 else 'unknown',
        'sudut': parts[3] if len(parts) > 3 else 'unknown',
    }
    
    # Optional metadata
    if len(parts) > 4:
        metadata['pencahayaan'] = parts[4]
    else:
        metadata['pencahayaan'] = 'unknown'
        
    if len(parts) > 5:
        metadata['jarak'] = parts[5]
    else:
        # Try to infer distance from angle field (e.g., "Frontal_Dekat")
        if '_' in metadata['sudut']:
            angle_parts = metadata['sudut'].split('_')
            if len(angle_parts) > 1:
                metadata['sudut'] = angle_parts[0]
                metadata['jarak'] = angle_parts[1]
            else:
                metadata['jarak'] = 'unknown'
        else:
            metadata['jarak'] = 'unknown'
    
    return metadata

def compare_face_features(embedding1, embedding2, metric='cosine'):
    """
    Compare two face embeddings to determine similarity
    
    Args:
        embedding1: First face embedding
        embedding2: Second face embedding
        metric: Similarity metric ('cosine' or 'euclidean')
        
    Returns:
        similarity: Similarity score (0-1, higher means more similar)
    """
    # Flatten embeddings
    emb1 = embedding1.flatten()
    emb2 = embedding2.flatten()
    
    # Calculate similarity
    if metric == 'cosine':
        # Compute cosine similarity
        dot_product = np.dot(emb1, emb2)
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        
        # Check for zero division
        if norm1 == 0 or norm2 == 0:
            return 0.0
            
        cos_sim = dot_product / (norm1 * norm2)
        
        # Convert to similarity (0-1 range)
        similarity = (cos_sim + 1) / 2
    else:  # euclidean
        # Compute Euclidean distance
        euclidean_dist = np.linalg.norm(emb1 - emb2)
        
        # Convert to similarity (0-1 range) using exponential decay
        similarity = np.exp(-euclidean_dist)
    
    return similarity

def enhance_face_for_comparison(face_img, target_size=(224, 224)):
    """
    Enhanced preprocessing for face comparison
    
    Args:
        face_img: Input face image
        target_size: Target size for processed face
        
    Returns:
        enhanced_face: Enhanced face image optimized for comparison
    """
    # Resize to target size
    face_resized = cv2.resize(face_img, target_size)
    
    # Convert to grayscale for better invariance to color/lighting
    face_gray = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)
    
    # Apply histogram equalization for lighting normalization
    face_equalized = cv2.equalizeHist(face_gray)
    
    # Apply Gaussian blur to reduce noise
    face_filtered = cv2.GaussianBlur(face_equalized, (5, 5), 0)
    
    # Apply CLAHE for better local contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    face_clahe = clahe.apply(face_filtered)
    
    # Convert back to BGR for compatibility with other functions
    enhanced_face = cv2.cvtColor(face_clahe, cv2.COLOR_GRAY2BGR)
    
    return enhanced_face

def remove_background(face_img):
    """
    Attempt to remove background from face image
    
    Args:
        face_img: Input face image
        
    Returns:
        face_without_bg: Face image with background removed or minimized
    """
    # Convert to grayscale
    gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    
    # Apply GaussianBlur to reduce noise
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    
    # Create a binary mask using adaptive thresholding
    _, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Find the largest contour (assumed to be the face)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return face_img  # Return original if no contours found
    
    # Find largest contour by area
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Create a new mask with only the largest contour
    face_mask = np.zeros_like(mask)
    cv2.drawContours(face_mask, [largest_contour], 0, 255, -1)
    
    # Apply morphological operations to improve mask
    kernel = np.ones((9, 9), np.uint8)
    face_mask = cv2.dilate(face_mask, kernel, iterations=3)
    face_mask = cv2.GaussianBlur(face_mask, (11, 11), 0)
    
    # Create 3-channel mask
    face_mask_3channel = cv2.cvtColor(face_mask, cv2.COLOR_GRAY2BGR)
    
    # Normalize mask to range 0-1
    face_mask_3channel = face_mask_3channel.astype(np.float32) / 255.0
    
    # Apply mask to original image
    face_without_bg = face_img.astype(np.float32) * face_mask_3channel
    
    # Convert back to uint8
    face_without_bg = face_without_bg.astype(np.uint8)
    
    return face_without_bg