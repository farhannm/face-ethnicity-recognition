import numpy as np
import cv2
import tensorflow as tf
import os

class FaceNetEmbedder:
    """
    A class for extracting face embeddings using FaceNet-inspired approach
    
    This is a lightweight implementation that mimics FaceNet's embedding
    process using available tools without requiring the actual FaceNet model.
    """
    def __init__(self, embedding_size=512):
        """
        Initialize the FaceNet embedder
        
        Args:
            embedding_size: Size of embedding vector (default: 512)
        """
        self.embedding_size = embedding_size
        self.input_shape = (160, 160)  # Standard FaceNet input size
        
        # Initialize feature extractors
        self.initialize_extractors()
        
        print(f"FaceNet embedder initialized with embedding size: {embedding_size}")
        
    def initialize_extractors(self):
        """Initialize the feature extractors"""
        # Use OpenCV's DNN face recognizer if available
        try:
            model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                     '../models/opencv_face_recognition.caffemodel')
            config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                     '../models/opencv_face_recognition.prototxt')
            
            if os.path.exists(model_path) and os.path.exists(config_path):
                self.face_recognizer = cv2.dnn.readNetFromCaffe(config_path, model_path)
                self.use_dnn = True
                print("Using OpenCV DNN face recognizer")
            else:
                self.use_dnn = False
                print("OpenCV DNN model not found, using fallback feature extraction")
        except Exception as e:
            print(f"Error initializing DNN: {e}")
            self.use_dnn = False
    
    def get_embedding(self, face_img):
        """
        Extract embedding from face image
        
        Args:
            face_img: Preprocessed face image (normalized, shape: batch_size x H x W x 3)
            
        Returns:
            embedding: Face embedding vector
        """
        # Handle batch input (expected shape: batch_size x H x W x 3)
        if len(face_img.shape) == 4:
            img = face_img[0]  # Take first image from batch
        else:
            img = face_img
        
        # Ensure image is in the right format for OpenCV processing
        # Convert to uint8 if in float format
        if img.dtype == np.float32 or img.dtype == np.float64:
            # Check if values are in range [0,1]
            if np.max(img) <= 1.0:
                img = (img * 255).astype(np.uint8)
            else:
                # Clip values to valid range for uint8
                img = np.clip(img, 0, 255).astype(np.uint8)
        
        # Ensure we have BGR format for OpenCV
        if img.shape[2] == 3:
            # Check if RGB (approximation: red channel has higher mean than blue)
            if np.mean(img[:,:,0]) > np.mean(img[:,:,2]):
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        # Resize to input shape if needed
        if img.shape[:2] != self.input_shape:
            img = cv2.resize(img, self.input_shape)
        
        # Extract features
        if self.use_dnn:
            embedding = self._extract_dnn_features(img)
        else:
            embedding = self._extract_hand_crafted_features(img)
        
        # Ensure embedding is the right size
        embedding = self._resize_embedding(embedding)
        
        return embedding
    
    def _extract_dnn_features(self, img):
        """Extract features using DNN model"""
        try:
            # Prepare input blob
            blob = cv2.dnn.blobFromImage(img, 1.0, self.input_shape, 
                                         (104, 177, 123), swapRB=False, crop=False)
            
            # Set input and forward pass
            self.face_recognizer.setInput(blob)
            embedding = self.face_recognizer.forward()
            
            # Normalize embedding
            embedding = embedding / np.linalg.norm(embedding)
            
            return embedding
        except Exception as e:
            print(f"Error in DNN feature extraction: {e}")
            # Fall back to hand-crafted features if DNN fails
            return self._extract_hand_crafted_features(img)
    
    def _extract_hand_crafted_features(self, img):
        """Extract hand-crafted features as fallback"""
        features = []
        
        # Ensure img is uint8 
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)
        
        # Convert to grayscale if needed
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()  # Already grayscale
            
        # Ensure it's uint8 again
        if gray.dtype != np.uint8:
            gray = np.clip(gray, 0, 255).astype(np.uint8)
        
        # 1. HOG features
        try:
            win_size = (160, 160)
            block_size = (16, 16)
            block_stride = (8, 8)
            cell_size = (8, 8)
            nbins = 9
            
            hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins)
            hog_features = hog.compute(gray)
            features.append(hog_features.flatten())
        except Exception as e:
            print(f"Error computing HOG features: {e}")
            # Create placeholder for HOG features
            hog_features = np.zeros((324,), dtype=np.float32)
            features.append(hog_features)
        
        # 2. LBP-like features
        # Simple LBP-inspired texture features
        try:
            texture_features = []
            for i in range(1, gray.shape[0]-1, 4):  # Skip pixels for efficiency
                for j in range(1, gray.shape[1]-1, 4):
                    # Center pixel
                    center = gray[i, j]
                    # Compare with neighbors (simplified LBP)
                    code = 0
                    if gray[i-1, j] > center: code += 1
                    if gray[i+1, j] > center: code += 2
                    if gray[i, j-1] > center: code += 4
                    if gray[i, j+1] > center: code += 8
                    texture_features.append(code)
            
            # Downsample to reduce dimension
            texture_features = np.array(texture_features, dtype=np.float32)
            texture_features = np.histogram(texture_features, bins=16, range=(0, 16))[0]
            features.append(texture_features / (np.sum(texture_features) + 1e-10))
        except Exception as e:
            print(f"Error computing LBP features: {e}")
            # Create placeholder for LBP features
            texture_features = np.zeros((16,), dtype=np.float32)
            features.append(texture_features)
        
        # 3. Histogram features
        try:
            hist_features = cv2.calcHist([gray], [0], None, [64], [0, 256])
            hist_features = hist_features.flatten() / (np.sum(hist_features) + 1e-10)
            features.append(hist_features)
        except Exception as e:
            print(f"Error computing histogram features: {e}")
            # Create placeholder for histogram features
            hist_features = np.zeros((64,), dtype=np.float32)
            features.append(hist_features)
        
        # 4. Gabor-like edge features
        try:
            # Simple edge detection in different directions
            edges_h = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            edges_v = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            # Magnitude and direction
            magnitude = np.sqrt(edges_h**2 + edges_v**2)
            direction = np.arctan2(edges_v, edges_h)
            # Histogram of directions
            direction_hist = np.histogram(direction.flatten(), bins=16, range=(-np.pi, np.pi))[0]
            direction_hist = direction_hist / (np.sum(direction_hist) + 1e-10)
            features.append(direction_hist)
        except Exception as e:
            print(f"Error computing edge features: {e}")
            # Create placeholder for edge features
            direction_hist = np.zeros((16,), dtype=np.float32)
            features.append(direction_hist)
        
        # 5. Mean and std per region
        try:
            # Divide image into 4x4 grid
            h, w = gray.shape
            region_h, region_w = h // 4, w // 4
            region_features = []
            for i in range(4):
                for j in range(4):
                    region = gray[i*region_h:(i+1)*region_h, j*region_w:(j+1)*region_w]
                    region_features.append(np.mean(region))
                    region_features.append(np.std(region))
            features.append(np.array(region_features, dtype=np.float32))
        except Exception as e:
            print(f"Error computing region features: {e}")
            # Create placeholder for region features
            region_features = np.zeros((32,), dtype=np.float32)
            features.append(region_features)
        
        # Combine all features
        try:
            combined_features = np.concatenate(features)
            
            # Handle NaN or Inf values
            combined_features = np.nan_to_num(combined_features)
            
            # Normalize (with safeguard against zero division)
            norm = np.linalg.norm(combined_features)
            if norm > 1e-10:
                combined_features = combined_features / norm
            
            return combined_features
        except Exception as e:
            print(f"Error combining features: {e}")
            # Return a fallback vector with correct dimensions
            return np.random.rand(self.embedding_size)
    
    def _resize_embedding(self, embedding):
        """Resize embedding to target size"""
        try:
            embedding = embedding.flatten()
            
            # Already the right size
            if embedding.shape[0] == self.embedding_size:
                return embedding.reshape(1, -1)
                
            # Need to reduce
            if embedding.shape[0] > self.embedding_size:
                # Use PCA-inspired approach (take evenly spaced samples)
                indices = np.linspace(0, embedding.shape[0]-1, self.embedding_size, dtype=int)
                resized = embedding[indices]
                
            # Need to increase
            else:
                # Pad with zeros
                resized = np.zeros(self.embedding_size)
                resized[:embedding.shape[0]] = embedding
                
            # Add batch dimension
            return resized.reshape(1, -1)
        except Exception as e:
            print(f"Error resizing embedding: {e}")
            # Return a fallback vector with correct dimensions
            return np.random.rand(1, self.embedding_size)
    
    def compute_similarity(self, embedding1, embedding2, metric='cosine'):
        """
        Compute similarity between two face embeddings
        
        Args:
            embedding1: First face embedding
            embedding2: Second face embedding
            metric: Similarity metric ('cosine' or 'euclidean')
            
        Returns:
            similarity: Similarity score (0-1, higher means more similar)
            is_same_person: Boolean indicating whether embeddings likely represent same person
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
                return 0.0, False
                
            cos_sim = dot_product / (norm1 * norm2)
            
            # Convert to similarity (0-1 range)
            similarity = (cos_sim + 1) / 2
        else:  # euclidean
            # Compute Euclidean distance
            euclidean_dist = np.linalg.norm(emb1 - emb2)
            
            # Convert to similarity (0-1 range) using exponential decay
            similarity = np.exp(-euclidean_dist)
        
        # Determine if same person (default threshold: 0.6)
        is_same_person = similarity >= 0.6
        
        return similarity, is_same_person