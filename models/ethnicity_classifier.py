import numpy as np
import tensorflow as tf

# Explicitly import from tensorflow.keras
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout

class EthnicityClassifier:
    """`
    CNN with Transfer Learning for ethnicity classification
    """
    def __init__(self, num_classes=3, input_shape=(224, 224, 3)):
        """
        Initialize the ethnicity classifier
        
        Args:
            num_classes: Number of ethnicity classes
            input_shape: Input shape of face images
        """
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.model = self._build_model()
        self.class_names = ['Jawa', 'Sunda', 'Melayu']  # Default class names
        
    def _build_model(self):
        """
        Build the CNN model with transfer learning
        
        Returns:
            model: Keras Model instance
        """
        # Load pre-trained MobileNetV2 without top layer
        try:
            base_model = MobileNetV2(
                weights='imagenet',
                include_top=False,
                input_shape=self.input_shape
            )
        except Exception as e:
            print(f"Error loading MobileNetV2: {e}")
            print("Falling back to MobileNetV2 without pretrained weights")
            base_model = MobileNetV2(
                weights=None,
                include_top=False,
                input_shape=self.input_shape
            )
        
        # Add custom top layers
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(1024, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.3)(x)
        predictions = Dense(self.num_classes, activation='softmax')(x)
        
        # Create the full model
        model = Model(inputs=base_model.input, outputs=predictions)
        
        # Freeze the base model layers
        for layer in base_model.layers:
            layer.trainable = False
            
        # Compile the model
        try:
            model.compile(
                optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
        except Exception as e:
            print(f"Error compiling model: {e}")
            print("Trying with explicit optimizer")
            model.compile(
                optimizer=Adam(learning_rate=0.001),
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
        
        return model
    
    def predict_ethnicity(self, face_image):
        """
        Predict ethnicity for a face image
        
        Args:
            face_image: Preprocessed face image (normalized, resized)
            
        Returns:
            predicted_class: Predicted ethnicity class name
            confidence: Confidence score for the prediction
            all_confidences: Dictionary of confidences for all classes
        """
        # Make prediction
        predictions = self.model.predict(face_image)
        
        # Get the predicted class index
        predicted_class_idx = np.argmax(predictions[0])
        
        # Get the confidence score
        confidence = predictions[0][predicted_class_idx]
        
        # Get the predicted class name
        predicted_class = self.class_names[predicted_class_idx]
        
        # Create dictionary with all confidence scores
        all_confidences = {self.class_names[i]: float(predictions[0][i]) for i in range(len(self.class_names))}
        
        return predicted_class, confidence, all_confidences
    
    def set_class_names(self, class_names):
        """
        Set the ethnicity class names
        
        Args:
            class_names: List of class names
        """
        if len(class_names) != self.num_classes:
            raise ValueError(f"Expected {self.num_classes} class names, got {len(class_names)}")
        self.class_names = class_names
    
    def load_weights(self, weights_path):
        """
        Load weights from a saved model file
        
        Args:
            weights_path: Path to the weights file
        """
        try:
            self.model.load_weights(weights_path)
        except Exception as e:
            print(f"Error loading weights: {e}")
            print("Continuing with uninitialized weights")
        
    def save_weights(self, weights_path):
        """
        Save model weights to a file
        
        Args:
            weights_path: Path where to save the weights
        """
        self.model.save_weights(weights_path)
        
    def fine_tune(self, unfreeze_layers=23):
        """
        Prepare the model for fine-tuning by unfreezing some layers
        
        Args:
            unfreeze_layers: Number of layers to unfreeze from the end of the base model
        """
        # Get the base model
        base_model = self.model.layers[0]
        
        # Unfreeze the specified number of layers from the end
        for layer in base_model.layers[-unfreeze_layers:]:
            layer.trainable = True
            
        # Recompile with a lower learning rate for fine-tuning
        try:
            self.model.compile(
                optimizer=Adam(learning_rate=1e-5),
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
        except Exception as e:
            print(f"Error recompiling model: {e}")
            # Alternative compile method for older TensorFlow versions
            self.model.compile(
                optimizer=Adam(1e-5),
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )