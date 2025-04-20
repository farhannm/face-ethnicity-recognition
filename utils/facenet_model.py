import os
import numpy as np
import tensorflow as tf
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

class FaceNetModel:
    """
    Class to handle FaceNet pre-trained model integration for face embeddings
    """
    def __init__(self, model_path=None):
        """
        Initialize FaceNet model
        
        Args:
            model_path: Path to pre-trained FaceNet model (Keras .h5 format)
                        If None, will try to use a default path
        """
        # Set default model paths to check
        default_paths = [
            # Path if model is in the models directory
            os.path.join(os.path.dirname(os.path.abspath(__file__)), '../models/facenet_keras.h5'),
            # Path if model is in the current directory
            os.path.join(os.path.dirname(os.path.abspath(__file__)), 'facenet_keras.h5'),
        ]
        
        # Try to load the model
        self.model = None
        self.model_loaded = False
        
        # First try with user-provided path
        if model_path is not None and os.path.exists(model_path):
            try:
                self.model = load_model(model_path)
                self.model_loaded = True
                print(f"Loaded FaceNet model from: {model_path}")
            except Exception as e:
                print(f"Error loading FaceNet model from {model_path}: {e}")
                
        # If that didn't work, try default paths
        if not self.model_loaded:
            for path in default_paths:
                if os.path.exists(path):
                    try:
                        self.model = load_model(path)
                        self.model_loaded = True
                        print(f"Loaded FaceNet model from: {path}")
                        break
                    except Exception as e:
                        print(f"Error loading FaceNet model from {path}: {e}")
        
        # If model couldn't be loaded, let user know
        if not self.model_loaded:
            print("Warning: Failed to load FaceNet model. Please download the model or specify correct path.")
            print("You can download the pre-trained Keras FaceNet model from: https://github.com/nyoki-mtl/keras-facenet")
            
        # Model input/output configuration
        self.input_shape = (160, 160)  # Standard FaceNet input size
        self.embedding_size = 128  # FaceNet embedding size (varies by model)
        
        # If model loaded successfully, get the actual embedding size
        if self.model_loaded:
            self.embedding_size = self.model.output_shape[-1]
            print(f"FaceNet embedding size: {self.embedding_size}")
            
    def is_model_loaded(self):
        """Check if model was loaded successfully"""
        return self.model_loaded
    
    def get_embedding(self, face_img):
        """
        Generate face embedding using FaceNet
        
        Args:
            face_img: Face image (can be single image or batch)
                     Should be preprocessed for FaceNet
        
        Returns:
            embedding: Face embedding vector
        """
        if not self.model_loaded:
            # Return random embedding if model not loaded (for testing)
            print("Model not loaded, returning random embedding")
            embedding = np.random.rand(1, self.embedding_size)
            return embedding
        
        try:
            # Handle batch input
            if len(face_img.shape) == 4:
                # Already in batch format (batch_size, height, width, channels)
                processed_img = face_img
            else:
                # Add batch dimension if single image
                processed_img = np.expand_dims(face_img, axis=0)
                
            # Ensure correct input shape
            if processed_img.shape[1:3] != self.input_shape:
                # Resize images to expected input shape
                resized_img = np.zeros((processed_img.shape[0], self.input_shape[0], 
                                        self.input_shape[1], processed_img.shape[3]))
                
                for i in range(processed_img.shape[0]):
                    resized_img[i] = cv2.resize(processed_img[i], self.input_shape)
                
                processed_img = resized_img
            
            # Ensure pixel values are in [0, 1]
            if np.max(processed_img) > 1.0:
                processed_img = processed_img / 255.0
                
            # Generate embedding
            embedding = self.model.predict(processed_img)
            
            # Normalize embedding to unit length
            embedding_norm = np.linalg.norm(embedding, axis=1, keepdims=True)
            normalized_embedding = embedding / embedding_norm
            
            return normalized_embedding
            
        except Exception as e:
            print(f"Error generating embedding: {e}")
            # Return random embedding in case of error
            return np.random.rand(1, self.embedding_size)
    
    def preprocess_face(self, face_img):
        """
        Preprocess face for FaceNet input
        
        Args:
            face_img: Face image (BGR format from OpenCV)
            
        Returns:
            preprocessed_face: Image ready for FaceNet
        """
        try:
            # Convert to RGB if needed (FaceNet expects RGB)
            if len(face_img.shape) == 3 and face_img.shape[2] == 3:
                rgb_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            else:
                # If grayscale, convert to RGB
                if len(face_img.shape) == 2:
                    rgb_img = cv2.cvtColor(face_img, cv2.COLOR_GRAY2RGB)
                else:
                    rgb_img = face_img
            
            # Resize to expected input shape
            resized_img = cv2.resize(rgb_img, self.input_shape)
            
            # Convert to float32 and scale to [0, 1]
            preprocessed_img = resized_img.astype(np.float32) / 255.0
            
            return preprocessed_img
            
        except Exception as e:
            print(f"Error preprocessing face: {e}")
            # Return blank image in case of error
            return np.zeros((self.input_shape[0], self.input_shape[1], 3), dtype=np.float32)
    
    def compute_similarity(self, embedding1, embedding2, metric='cosine'):
        """
        Compute similarity between two embeddings
        
        Args:
            embedding1: First face embedding
            embedding2: Second face embedding
            metric: Similarity metric ('cosine' or 'euclidean')
            
        Returns:
            similarity: Similarity score (0-1, higher means more similar)
        """
        # Flatten embeddings to 1D
        if len(embedding1.shape) > 1:
            embedding1 = embedding1.flatten()
        if len(embedding2.shape) > 1:
            embedding2 = embedding2.flatten()
            
        # Calculate similarity based on metric
        if metric == 'cosine':
            # Compute dot product (cosine similarity for normalized embeddings)
            similarity = np.dot(embedding1, embedding2)
            # Ensure in [0, 1] range
            similarity = max(0, min(1, similarity))
        else:  # euclidean
            # Compute Euclidean distance
            distance = np.linalg.norm(embedding1 - embedding2)
            # Convert to similarity score [0, 1]
            similarity = 1.0 / (1.0 + distance)
            
        return similarity