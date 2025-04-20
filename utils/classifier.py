import numpy as np
import tensorflow as tf
import os
import joblib
from sklearn.neighbors import KNeighborsClassifier
from tensorflow.keras.models import load_model

class FaceNetEmbedder:
    """
    Class for extracting face embeddings using FaceNet model
    """
    def __init__(self, model_path=None):
        """
        Initialize FaceNet model
        
        Args:
            model_path: Path to pre-trained FaceNet model (If None, attempts to use a default path)
        """
        # Hide TensorFlow warnings
        tf.get_logger().setLevel('ERROR')
        
        if model_path and os.path.exists(model_path):
            self.model = load_model(model_path)
        else:
            # If no model path provided, try common paths
            possible_paths = [
                'models/facenet/facenet_keras.h5',
                os.path.join(os.path.dirname(__file__), '../models/facenet/facenet_keras.h5')
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    try:
                        print(f"Attempting to load FaceNet model from: {path}")
                        # Check file size to make sure it's not too small to be a valid model
                        file_size_mb = os.path.getsize(path) / (1024 * 1024)
                        print(f"File size: {file_size_mb:.2f} MB")
                        
                        if file_size_mb < 1.0:
                            print(f"Warning: File size is suspiciously small ({file_size_mb:.2f} MB)")
                        
                        self.model = load_model(path)
                        self.input_shape = self.model.inputs[0].shape[1:3]  # Should be (160, 160)
                        print(f"FaceNet model loaded successfully with input shape: {self.input_shape}")
                        self.mock_model = False
                        break
                    except Exception as e:
                        import traceback
                        print(f"Error loading FaceNet model from {path}: {e}")
                        print(traceback.format_exc())
                        continue
            else:
                print("FaceNet model not found or could not be loaded. Running in demo mode with a mock embedder.")
                self.input_shape = (160, 160)  # Default FaceNet input shape
                self.mock_model = True
                print(f"Mock FaceNet embedder initialized with input shape: {self.input_shape}")
    
    def get_embedding(self, face_img):
        """
        Extract embedding from face image
        
        Args:
            face_img: Preprocessed face image batch
            
        Returns:
            embedding: 128-dimensional embedding vector
        """
        # Ensure face image is the right shape
        if face_img.shape[1:3] != self.input_shape:
            raise ValueError(f"Input face must be resized to {self.input_shape}, got {face_img.shape[1:3]}")
            
        # If we're in mock mode, generate a random embedding
        if hasattr(self, 'mock_model') and self.mock_model:
            # Generate random 128-dimensional embedding
            np.random.seed(int(np.sum(face_img) * 100) % 10000)  # Seed based on image content
            embedding = np.random.randn(1, 128)
            
            # Normalize embedding (L2 norm)
            embedding = embedding / np.linalg.norm(embedding, axis=1, keepdims=True)
            
            return embedding
        
        # Get embedding from the real model
        embedding = self.model.predict(face_img, verbose=0)
        
        # Normalize embedding (L2 norm)
        embedding = embedding / np.linalg.norm(embedding, axis=1, keepdims=True)
        
        return embedding

class EthnicityClassifier:
    """
    Ethnicity classifier using FaceNet embeddings and KNN
    """
    def __init__(self, facenet_model_path=None, knn_model_path=None):
        """
        Initialize the ethnicity classifier
        
        Args:
            facenet_model_path: Path to pre-trained FaceNet model
            knn_model_path: Path to trained KNN model for ethnicity classification
        """
        # Initialize FaceNet model for embeddings
        self.embedder = FaceNetEmbedder(facenet_model_path)
        
        # Load KNN model
        self.knn_model = None
        if knn_model_path and os.path.exists(knn_model_path):
            self.load_model(knn_model_path)
        else:
            # Try common paths
            possible_paths = [
                'models/facenet_knn_model.pkl',
                os.path.join(os.path.dirname(__file__), '../models/facenet_knn_model.pkl')
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    self.load_model(path)
                    break
            else:
                print("KNN model not found. Running in demo mode with a mock classifier.")
                self._initialize_mock_classifier()
    
    def load_model(self, model_path):
        """
        Load KNN model from file
        
        Args:
            model_path: Path to saved KNN model
        """
        try:
            print(f"Attempting to load KNN model from: {model_path}")
            # Check file size to make sure it's not too small to be a valid model
            file_size_kb = os.path.getsize(model_path) / 1024
            print(f"File size: {file_size_kb:.2f} KB")
            
            if file_size_kb < 1.0:
                print(f"Warning: File size is suspiciously small ({file_size_kb:.2f} KB)")
            
            self.knn_model = joblib.load(model_path)
            self.class_names = self.knn_model.classes_
            print(f"KNN model loaded successfully with {len(self.class_names)} classes: {self.class_names}")
        except Exception as e:
            import traceback
            print(f"Error loading KNN model: {e}")
            print(traceback.format_exc())
            self.knn_model = None
    
    def _initialize_mock_classifier(self):
        """
        Initialize a mock classifier for demo mode when the real model is not available
        """
        self.knn_model = "mock_model"  # Just a marker to indicate we have a mock model
        self.class_names = ["Jawa", "Sunda", "Melayu"]  # Default ethnicity classes
        print(f"Mock classifier initialized with classes: {self.class_names}")
    
    def predict_ethnicity(self, face_img):
        """
        Predict ethnicity for a face image
        
        Args:
            face_img: Preprocessed face image
            
        Returns:
            predicted_class: Predicted ethnicity class name
            confidence: Confidence score for the prediction
            all_confidences: Dictionary of confidences for all classes
        """
        # Check if we're in demo mode with mock model
        if self.knn_model == "mock_model":
            # Generate random confidences for demo
            import random
            all_confidences = {}
            
            # Create random confidences that sum to 1
            random_values = [random.random() for _ in range(len(self.class_names))]
            total = sum(random_values)
            normalized_values = [val/total for val in random_values]
            
            for i, class_name in enumerate(self.class_names):
                all_confidences[class_name] = float(normalized_values[i])
            
            # Determine the class with highest confidence
            predicted_class = max(all_confidences, key=all_confidences.get)
            confidence = all_confidences[predicted_class]
            
            return predicted_class, confidence, all_confidences
        
        # Real model prediction
        elif self.knn_model is None:
            raise ValueError("KNN model not loaded. Cannot predict ethnicity.")
        
        # Get face embedding
        embedding = self.embedder.get_embedding(face_img)[0]  # Get first embedding (batch size 1)
        
        # Get nearest neighbors
        neighbors = self.knn_model.kneighbors([embedding], return_distance=True)
        distances = neighbors[0][0]
        indices = neighbors[1][0]
        
        # Get predicted labels
        predicted_labels = [self.knn_model.classes_[i] for i in indices]
        
        # Count occurrences of each label in the k nearest neighbors
        unique_labels, counts = np.unique(predicted_labels, return_counts=True)
        
        # Calculate confidence score for each class based on its frequency among k-nearest neighbors
        all_confidences = {}
        for label in self.class_names:
            if label in unique_labels:
                idx = np.where(unique_labels == label)[0][0]
                conf = counts[idx] / len(indices)
                all_confidences[label] = float(conf)  # Convert to native Python float
            else:
                all_confidences[label] = 0.0
        
        # Get the predicted class (most frequent among k-nearest neighbors)
        predicted_class = self.knn_model.predict([embedding])[0]
        confidence = all_confidences[predicted_class]
        
        return predicted_class, confidence, all_confidences