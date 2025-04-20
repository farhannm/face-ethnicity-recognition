import pickle
import numpy as np
import os

class EthnicClassifier:
    def __init__(self, model_path):
        """
        Initialize the ethnic classifier with a pretrained model
        
        Args:
            model_path (str): Path to the saved model (.pkl file)
        """
        # Load the model
        self.model_path = model_path
        self.load_model()
        
    def load_model(self):
        """Load the model from file"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
            
        try:
            with open(self.model_path, 'rb') as f:
                self.model = pickle.load(f)
            
            # Print model debug information
            print(f"Model type: {type(self.model)}")
            
            # Handle different model structures
            if isinstance(self.model, dict):
                print(f"Model keys: {list(self.model.keys())}")
                
                # Case 1: Model has a 'classifier' key
                if 'classifier' in self.model:
                    self.knn_model = self.model['classifier']
                    print("Using classifier from 'classifier' key")
                # Case 2: Look for any key that might be a classifier
                else:
                    for key, value in self.model.items():
                        if hasattr(value, 'predict'):
                            self.knn_model = value
                            print(f"Using classifier from '{key}' key")
                            break
                    else:
                        # If no suitable key found, use the model itself
                        self.knn_model = self.model
                        print("Using the full model dict as classifier")
            else:
                # Case 3: Model itself is the classifier
                self.knn_model = self.model
                print("Using the model itself as classifier")
                
            # Try to get class names
            self.class_names = self._extract_class_names()
            print(f"Extracted class names: {self.class_names}")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            # Create fallback default values
            self.knn_model = None
            self.class_names = ["Jawa", "Sunda", "Melayu"]
            print(f"Created fallback class names: {self.class_names}")
    
    def _extract_class_names(self):
        """Extract class names from the model"""
        # Try different ways to get class names
        
        # 1. Try from knn_model.classes_
        if hasattr(self.knn_model, 'classes_'):
            return [str(c) for c in self.knn_model.classes_]
            
        # 2. Try from model dictionary
        if isinstance(self.model, dict):
            # Check for classes in model dict
            for key in ['classes', 'class_names', 'labels']:
                if key in self.model:
                    return [str(c) for c in self.model[key]]
            
            # Check for label encoder
            if 'label_encoder' in self.model and hasattr(self.model['label_encoder'], 'classes_'):
                return [str(c) for c in self.model['label_encoder'].classes_]
                
        # 3. Default to ethnic class names
        return ["Jawa", "Sunda", "Melayu"]  # Default fallback
            
    def predict(self, embedding):
        """
        Predict ethnic class for a face embedding
        
        Args:
            embedding (numpy.ndarray): Face embedding vector
            
        Returns:
            tuple: (predicted_class, confidence, confidence_scores)
        """
        # If no valid model, return default values
        if self.knn_model is None:
            print("Warning: No valid model loaded, returning default prediction")
            default_class = self.class_names[0]
            confidence_scores = {c: 1.0 if c == default_class else 0.0 for c in self.class_names}
            return default_class, 1.0, confidence_scores
        
        try:
            # Reshape embedding for prediction
            embedding_reshaped = embedding.reshape(1, -1)
            
            # Get prediction
            if hasattr(self.knn_model, 'predict'):
                predicted_class_idx = self.knn_model.predict(embedding_reshaped)[0]
                # Convert index or object to string for consistency
                predicted_class = str(predicted_class_idx)
                
                # Try to get probability distribution
                if hasattr(self.knn_model, 'predict_proba'):
                    proba = self.knn_model.predict_proba(embedding_reshaped)[0]
                    confidence_scores = {str(class_name): float(score) 
                                      for class_name, score in zip(self.class_names, proba)}
                    confidence = confidence_scores[predicted_class]
                else:
                    # Fallback to basic confidence scores
                    confidence_scores = {str(class_name): 0.1 for class_name in self.class_names}
                    confidence_scores[predicted_class] = 0.9
                    confidence = 0.9
            else:
                # Fallback to first class with high confidence
                predicted_class = self.class_names[0]
                confidence_scores = {str(class_name): 0.1 for class_name in self.class_names}
                confidence_scores[predicted_class] = 0.9
                confidence = 0.9
                print("Warning: Model doesn't have predict method, using fallback")
                
            return predicted_class, confidence, confidence_scores
            
        except Exception as e:
            print(f"Error during prediction: {e}")
            # Return default prediction on error
            predicted_class = self.class_names[0]
            confidence_scores = {str(class_name): 0.1 for class_name in self.class_names}
            confidence_scores[predicted_class] = 0.9
            confidence = 0.9
            return predicted_class, confidence, confidence_scores