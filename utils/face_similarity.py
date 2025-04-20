import numpy as np
from scipy.spatial.distance import cosine, euclidean
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import cv2

class FaceSimilarity:
    """
    Class to compute face similarity between face embeddings
    """
    def __init__(self, threshold=0.6, distance_metric='cosine'):
        """
        Initialize face similarity
        
        Args:
            threshold: Similarity threshold for match decision (default: 0.6)
            distance_metric: Type of distance metric ('cosine' or 'euclidean')
        """
        self.threshold = threshold
        self.distance_metric = distance_metric
        
    def compute_similarity(self, embedding1, embedding2):
        """
        Compute similarity between two face embeddings
        
        Args:
            embedding1: First face embedding
            embedding2: Second face embedding
            
        Returns:
            similarity_score: Similarity score (0-1, higher means more similar)
            is_match: Boolean indicating whether faces match
        """
        try:
            # Ensure embeddings are flattened
            embedding1 = embedding1.flatten()
            embedding2 = embedding2.flatten()
            
            # Handle potential NaN or inf values
            embedding1 = np.nan_to_num(embedding1)
            embedding2 = np.nan_to_num(embedding2)
            
            # Calculate distance
            if self.distance_metric == 'cosine':
                # Handle potential zero vectors
                if np.linalg.norm(embedding1) < 1e-10 or np.linalg.norm(embedding2) < 1e-10:
                    similarity_score = 0.0
                else:
                    distance = cosine(embedding1, embedding2)
                    # Handle potential nan or inf values
                    if np.isnan(distance) or np.isinf(distance):
                        similarity_score = 0.0
                    else:
                        # Convert to similarity (1 - distance)
                        similarity_score = 1.0 - distance
            else:  # euclidean
                distance = euclidean(embedding1, embedding2)
                # Handle potential nan or inf values
                if np.isnan(distance) or np.isinf(distance):
                    similarity_score = 0.0
                else:
                    # Convert to similarity (exponential decay)
                    similarity_score = np.exp(-distance)
            
            # Determine if match based on threshold
            is_match = similarity_score >= self.threshold
            
            return similarity_score, is_match
            
        except Exception as e:
            print(f"Error during similarity computation: {e}")
            # Return default values in case of error
            return 0.0, False
    
    def safe_compute_similarity(self, embedding1, embedding2):
        """
        A safer version of compute_similarity with better error handling
        
        Args:
            embedding1: First face embedding
            embedding2: Second face embedding
            
        Returns:
            similarity_score: Similarity score (0-1, higher means more similar)
            is_match: Boolean indicating whether faces match
        """
        # Input validation
        if embedding1 is None or embedding2 is None:
            print("Error: None embeddings provided")
            return 0.0, False
            
        try:
            # Ensure embeddings are numpy arrays
            if not isinstance(embedding1, np.ndarray):
                embedding1 = np.array(embedding1)
            if not isinstance(embedding2, np.ndarray):
                embedding2 = np.array(embedding2)
                
            # Check embedding shapes
            if embedding1.size == 0 or embedding2.size == 0:
                print("Error: Empty embeddings provided")
                return 0.0, False
                
            # Ensure embeddings are flattened
            embedding1 = embedding1.flatten()
            embedding2 = embedding2.flatten()
            
            # Check if embeddings have the same size
            if embedding1.size != embedding2.size:
                print(f"Warning: Embeddings have different sizes: {embedding1.size} vs {embedding2.size}")
                # Resize to match the smaller size
                min_size = min(embedding1.size, embedding2.size)
                embedding1 = embedding1[:min_size]
                embedding2 = embedding2[:min_size]
            
            # Normalize embeddings to unit length 
            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)
            
            if norm1 > 1e-10:
                embedding1 = embedding1 / norm1
            if norm2 > 1e-10:
                embedding2 = embedding2 / norm2
            
            # Replace any NaN or infinite values
            embedding1 = np.nan_to_num(embedding1)
            embedding2 = np.nan_to_num(embedding2)
            
            # Calculate similarity
            if self.distance_metric == 'cosine':
                # Compute cosine similarity directly with dot product
                # Since vectors are normalized, dot product equals cosine similarity
                similarity_score = np.dot(embedding1, embedding2)
                
                # Adjust range from [-1,1] to [0,1]
                similarity_score = (similarity_score + 1) / 2
            else:  # euclidean
                distance = euclidean(embedding1, embedding2)
                # Convert to similarity (exponential decay)
                similarity_score = np.exp(-distance)
            
            # Ensure similarity is in [0,1] range
            similarity_score = max(0, min(1, similarity_score))
            
            # Determine if match based on threshold
            is_match = similarity_score >= self.threshold
            
            return similarity_score, is_match
            
        except Exception as e:
            print(f"Error during similarity computation: {e}")
            # Return default values in case of error
            return 0.0, False
    
    def evaluate_threshold(self, same_pairs, diff_pairs):
        """
        Evaluate threshold using pairs of same and different identities
        
        Args:
            same_pairs: List of (embedding1, embedding2) pairs from same identity
            diff_pairs: List of (embedding1, embedding2) pairs from different identities
            
        Returns:
            optimal_threshold: Optimal threshold value
            accuracy: Accuracy at optimal threshold
            auc_score: Area under ROC curve
            fpr: False positive rates
            tpr: True positive rates
            thresholds: Thresholds used for ROC curve
        """
        # Calculate similarity scores for same identity pairs
        same_scores = []
        for emb1, emb2 in same_pairs:
            score, _ = self.safe_compute_similarity(emb1, emb2)
            same_scores.append(score)
        
        # Calculate similarity scores for different identity pairs
        diff_scores = []
        for emb1, emb2 in diff_pairs:
            score, _ = self.safe_compute_similarity(emb1, emb2)
            diff_scores.append(score)
        
        # Create labels (1 for same, 0 for different)
        y_true = [1] * len(same_scores) + [0] * len(diff_scores)
        y_scores = same_scores + diff_scores
        
        # Calculate ROC curve
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        
        # Calculate area under curve
        auc_score = auc(fpr, tpr)
        
        # Find optimal threshold (maximum accuracy)
        accuracies = []
        for threshold in thresholds:
            y_pred = [1 if score >= threshold else 0 for score in y_scores]
            accuracy = np.mean(np.array(y_pred) == np.array(y_true))
            accuracies.append(accuracy)
        
        optimal_idx = np.argmax(accuracies)
        optimal_threshold = thresholds[optimal_idx]
        accuracy = accuracies[optimal_idx]
        
        return optimal_threshold, accuracy, auc_score, fpr, tpr, thresholds
    
    def plot_similarity_distribution(self, same_scores, diff_scores, title="Face Similarity Distribution"):
        """
        Plot distribution of similarity scores
        
        Args:
            same_scores: List of similarity scores for same identity pairs
            diff_scores: List of similarity scores for different identity pairs
            title: Plot title
            
        Returns:
            fig: Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot histograms
        bins = np.linspace(0, 1, 50)
        ax.hist(same_scores, bins=bins, alpha=0.5, label='Same Identity', color='green')
        ax.hist(diff_scores, bins=bins, alpha=0.5, label='Different Identity', color='red')
        
        # Plot threshold line
        ax.axvline(x=self.threshold, color='blue', linestyle='--', label=f'Threshold ({self.threshold:.2f})')
        
        # Add labels and legend
        ax.set_xlabel('Similarity Score')
        ax.set_ylabel('Frequency')
        ax.set_title(title)
        ax.legend()
        
        return fig
    
    def plot_roc_curve(self, fpr, tpr, thresholds, auc_score, optimal_threshold):
        """
        Plot ROC curve with optimal threshold
        
        Args:
            fpr: False positive rates
            tpr: True positive rates
            thresholds: Thresholds for ROC curve
            auc_score: Area under ROC curve
            optimal_threshold: Optimal threshold value
            
        Returns:
            fig: Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot ROC curve
        ax.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {auc_score:.3f})')
        
        # Plot diagonal line
        ax.plot([0, 1], [0, 1], color='gray', linestyle='--')
        
        # Mark optimal threshold
        # Find index for optimal threshold
        idx = (np.abs(thresholds - optimal_threshold)).argmin()
        ax.plot(fpr[idx], tpr[idx], 'ro', markersize=10, 
                label=f'Optimal threshold ({optimal_threshold:.3f})')
        
        # Add labels and legend
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('Receiver Operating Characteristic (ROC) Curve')
        ax.legend(loc='lower right')
        
        # Set axis limits
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.05])
        
        return fig
    
    def visualize_similarity_examples(self, face_pairs, scores, matches, expected_matches, size=(112, 112)):
        """
        Visualize examples of TP, FP, TN, FN predictions
        
        Args:
            face_pairs: List of (face1, face2) image pairs
            scores: Similarity scores for each pair
            matches: Binary predictions for each pair
            expected_matches: Ground truth for each pair
            size: Size to resize faces
            
        Returns:
            fig: Matplotlib figure
        """
        # Create a figure
        fig, axes = plt.subplots(4, 4, figsize=(16, 16))
        
        # Categories
        categories = [
            ('True Positive', True, True),  # match, expected_match
            ('False Positive', True, False),
            ('True Negative', False, False),
            ('False Negative', False, True)
        ]
        
        # Find examples for each category
        for row, (category, pred_match, exp_match) in enumerate(categories):
            examples = []
            
            # Find indexes for this category
            for i, (match, expected) in enumerate(zip(matches, expected_matches)):
                if match == pred_match and expected == exp_match:
                    examples.append(i)
            
            # Display up to 4 examples (or fewer if not available)
            for col in range(4):
                ax = axes[row, col]
                
                if col < len(examples):
                    idx = examples[col]
                    face1, face2 = face_pairs[idx]
                    
                    # Resize if needed
                    face1_resized = cv2.resize(face1, size) if face1.shape[:2] != size else face1
                    face2_resized = cv2.resize(face2, size) if face2.shape[:2] != size else face2
                    
                    # Convert BGR to RGB if needed
                    if len(face1_resized.shape) == 3 and face1_resized.shape[2] == 3:
                        face1_resized = cv2.cvtColor(face1_resized, cv2.COLOR_BGR2RGB)
                        face2_resized = cv2.cvtColor(face2_resized, cv2.COLOR_BGR2RGB)
                    
                    # Create composite image (side by side)
                    composite = np.hstack((face1_resized, face2_resized))
                    
                    # Display image
                    ax.imshow(composite)
                    score = scores[idx]
                    title = f"Score: {score:.3f}"
                    ax.set_title(title)
                
                # Remove axis
                ax.axis('off')
            
            # Add category label
            fig.text(0.01, 0.75 - row * 0.25, category, fontsize=14)
        
        plt.tight_layout()
        plt.subplots_adjust(left=0.1)
        
        return fig