import cv2
import numpy as np

class FaceEmbedder:
    """
    A face embedder that extracts face features using OpenCV and reduces dimensions to match model
    """
    def __init__(self, target_dimensions=512):
        """Initialize the face embedder"""
        # Use OpenCV's HOG descriptor for face feature extraction
        self.hog = cv2.HOGDescriptor()
        self.input_shape = (128, 128)  # Standard size for processing
        self.target_dimensions = target_dimensions  # Match KNN model expectations
        print(f"Face embedder initialized with target dimensions: {target_dimensions}")
        
    def get_embedding(self, face_img):
        """
        Extract embedding from face image
        
        Args:
            face_img: Preprocessed face image batch (normalized, shape: batch_size x H x W x 3)
            
        Returns:
            embedding: Feature vector representation of the face
        """
        # Convert to single face if input is a batch
        if len(face_img.shape) == 4:
            # Take the first image from the batch
            img = face_img[0]
        else:
            img = face_img
            
        # Convert to BGR if necessary
        if img.shape[2] == 3 and np.mean(img[:, :, 0]) > np.mean(img[:, :, 2]):  # RGB to BGR check
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            
        # Convert to grayscale
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
            
        # Ensure image is uint8 and in the right range for HOG
        if gray.dtype != np.uint8:
            # Handle conversion to uint8 carefully to avoid warning
            # First, handle any NaN values
            gray = np.nan_to_num(gray)
            
            # Scale to 0-255 range if normalized
            if np.max(gray) <= 1.0:
                gray = np.clip(gray * 255.0, 0, 255).astype(np.uint8)
            else:
                # Clip values to valid range for uint8
                gray = np.clip(gray, 0, 255).astype(np.uint8)
                
        # Resize to standard size
        resized = cv2.resize(gray, self.input_shape)
        
        # Extract features
        features = self._extract_features(resized)
        
        # Reduce dimensions to match model expectations using simple approach
        reduced_features = self._simple_reduce_dimensions(features)
        
        # Reshape to match the expected output format
        embedding = reduced_features.reshape(1, -1)
        
        return embedding
    
    def _extract_features(self, img):
        """Extract multiple features and combine them"""
        # 1. HOG features (simpler version with fewer features)
        winSize = (img.shape[1], img.shape[0])  # Use full image size
        blockSize = (16, 16)
        blockStride = (8, 8)
        cellSize = (8, 8)
        nbins = 9
        try:
            hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins)
            hog_features = hog.compute(img)
        except Exception as e:
            print(f"HOG computation error: {e}")
            # Create empty array with approximately right dimensions
            hog_features = np.zeros((256,), dtype=np.float32)
        
        # 2. Simple histogram features
        hist_features = self._extract_histogram_features(img, bins=64)
        
        # 3. Downsampled image as features (robust)
        try:
            small_img = cv2.resize(img, (32, 32)).flatten() / 255.0
        except Exception as e:
            print(f"Resizing error: {e}")
            small_img = np.zeros((32*32,), dtype=np.float32)
        
        # 4. Edge features using Canny
        try:
            edges = cv2.Canny(img, 100, 200)
            edge_features = cv2.resize(edges, (16, 16)).flatten() / 255.0
        except Exception as e:
            print(f"Edge detection error: {e}")
            edge_features = np.zeros((16*16,), dtype=np.float32)
        
        # Combine all features
        combined = np.concatenate([
            hog_features.flatten(),
            hist_features.flatten(),
            small_img.flatten(),
            edge_features.flatten()
        ])
        
        return combined
    
    def _simple_reduce_dimensions(self, features):
        """Reduce feature dimensions using a simple method"""
        feature_length = len(features)
        
        # If features already match target dimensions, return as is
        if feature_length == self.target_dimensions:
            return features
            
        # If features are smaller than target, pad with zeros
        if feature_length < self.target_dimensions:
            padding = np.zeros(self.target_dimensions - feature_length)
            return np.concatenate([features, padding])
            
        # If features are larger, use simple selection (evenly spaced)
        indices = np.linspace(0, feature_length-1, self.target_dimensions, dtype=int)
        return features[indices]
        
    def _extract_histogram_features(self, gray_img, bins=64):
        """Extract histogram features from image"""
        try:
            hist = cv2.calcHist([gray_img], [0], None, [bins], [0, 256])
            hist = cv2.normalize(hist, hist).flatten()
        except Exception as e:
            print(f"Histogram error: {e}")
            hist = np.zeros((bins,), dtype=np.float32)
        return hist