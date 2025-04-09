import numpy as np
from scipy.spatial.distance import cosine
import tensorflow as tf

# Handle different TensorFlow/Keras import scenarios
try:
    # Try the new standalone keras import first
    import keras
    from keras.models import Model, Sequential
    from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Lambda, BatchNormalization, Dropout, GlobalAveragePooling2D
    from keras.regularizers import l2
except ImportError:
    try:
        # Fall back to tensorflow.keras if standalone keras fails
        from tensorflow import keras
        from tensorflow.keras.models import Model, Sequential
        from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Lambda, BatchNormalization, Dropout, GlobalAveragePooling2D
        from tensorflow.keras.regularizers import l2
    except ImportError:
        # Provide clear error message if import fails
        raise ImportError(
            "Could not import Keras. Please install TensorFlow 2.x with: pip install tensorflow==2.13.0"
        )

class SiameseNetwork:
    """
    Siamese Network for face similarity detection
    """
    def __init__(self, input_shape=(100, 100, 3), embedding_dim=128):
        """
        Initialize the Siamese Network
        
        Args:
            input_shape: Input shape of face images
            embedding_dim: Dimension of the face embedding vector
        """
        self.input_shape = input_shape
        self.embedding_dim = embedding_dim
        self.model = self._build_model()
        self.threshold = 0.5  # Default threshold for similarity decision
        
    def _build_model(self):
        """
        Build the base network for feature extraction
        
        Returns:
            model: Keras Model instance
        """
        # Define the base network (shared weights CNN)
        base_network = Sequential([
            # First convolutional block
            Conv2D(32, (3, 3), activation='relu', input_shape=self.input_shape, kernel_regularizer=l2(1e-4)),
            BatchNormalization(),
            Conv2D(32, (3, 3), activation='relu', kernel_regularizer=l2(1e-4)),
            BatchNormalization(),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.25),
            
            # Second convolutional block
            Conv2D(64, (3, 3), activation='relu', kernel_regularizer=l2(1e-4)),
            BatchNormalization(),
            Conv2D(64, (3, 3), activation='relu', kernel_regularizer=l2(1e-4)),
            BatchNormalization(),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.25),
            
            # Third convolutional block
            Conv2D(128, (3, 3), activation='relu', kernel_regularizer=l2(1e-4)),
            BatchNormalization(),
            Conv2D(128, (3, 3), activation='relu', kernel_regularizer=l2(1e-4)),
            BatchNormalization(),
            GlobalAveragePooling2D(),
            Dropout(0.4),
            
            # Dense layers
            Dense(512, activation='relu', kernel_regularizer=l2(1e-4)),
            BatchNormalization(),
            Dropout(0.5),
            Dense(self.embedding_dim, activation=None, kernel_regularizer=l2(1e-4))
        ])
        
        # Create the Siamese network structure for training (not used for inference)
        input_a = Input(shape=self.input_shape)
        input_b = Input(shape=self.input_shape)
        
        embedding_a = base_network(input_a)
        embedding_b = base_network(input_b)
        
        # Define L2 normalization Lambda layer
        l2_norm = Lambda(lambda x: tf.math.l2_normalize(x, axis=1))
        
        # L2 normalize embeddings
        norm_a = l2_norm(embedding_a)
        norm_b = l2_norm(embedding_b)
        
        # Compute the Euclidean distance between the embeddings
        distance = Lambda(lambda x: tf.reduce_sum(tf.square(x[0] - x[1]), axis=1, keepdims=True))(
            [norm_a, norm_b])
        
        # Create the siamese model
        siamese_model = Model(inputs=[input_a, input_b], outputs=distance)
        
        # For inference, we only need the base network to extract embeddings
        return base_network
    
    def get_embedding(self, face_image):
        """
        Extract face embedding from an image
        
        Args:
            face_image: Preprocessed face image (normalized, resized)
            
        Returns:
            embedding: Face embedding vector (normalized)
        """
        # Extract the embedding
        embedding = self.model.predict(face_image)
        
        # Normalize the embedding (L2 norm)
        embedding = embedding / np.linalg.norm(embedding)
        
        return embedding
    
    def compare_faces(self, face1, face2, distance_metric='euclidean'):
        """
        Compare two face images and calculate similarity score
        
        Args:
            face1: First preprocessed face image
            face2: Second preprocessed face image
            distance_metric: 'euclidean' or 'cosine'
            
        Returns:
            similarity: Similarity score (0-1, higher = more similar)
            match: Boolean indicating if faces match based on threshold
        """
        # Get embeddings
        embedding1 = self.get_embedding(face1)
        embedding2 = self.get_embedding(face2)
        
        # Calculate distance based on selected metric
        if distance_metric == 'cosine':
            distance = cosine(embedding1.flatten(), embedding2.flatten())
            # Convert cosine distance to similarity score (0-1, higher = more similar)
            similarity = 1 - distance
        else:  # euclidean
            distance = np.linalg.norm(embedding1 - embedding2)
            # Convert euclidean distance to similarity score (0-1, higher = more similar)
            # Clip maximum distance to 2.0 for normalization
            similarity = 1 - min(distance / 2.0, 1.0)
        
        # Determine match based on threshold
        match = similarity >= self.threshold
        
        return similarity, match
    
    def set_threshold(self, threshold):
        """
        Set the threshold for face matching
        
        Args:
            threshold: New threshold value (0-1)
        """
        self.threshold = threshold
    
    def load_weights(self, weights_path):
        """
        Load weights from a saved model file
        
        Args:
            weights_path: Path to the weights file
        """
        self.model.load_weights(weights_path)
        
    def save_weights(self, weights_path):
        """
        Save model weights to a file
        
        Args:
            weights_path: Path where to save the weights
        """
        self.model.save_weights(weights_path)