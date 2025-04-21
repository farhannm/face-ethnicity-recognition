import numpy as np
from deepface import DeepFace
import cv2
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import traceback

class DeepFaceSimilarity:
    """
    Class to compute face similarity between face images using DeepFace
    """
    def __init__(self, threshold=0.6, model_name='VGG-Face', distance_metric='cosine'):
        """
        Initialize face similarity with DeepFace
        
        Args:
            threshold: Similarity threshold for match decision (default: 0.6)
            model_name: DeepFace model to use ('VGG-Face', 'Facenet', 'Facenet512', 'OpenFace', 'DeepFace', 'DeepID', 'ArcFace', 'Dlib')
            distance_metric: Type of distance metric ('cosine', 'euclidean', 'euclidean_l2')
        """
        self.threshold = threshold
        self.model_name = model_name
        self.distance_metric = distance_metric
        self.scores_same = []  # Store similarity scores for same identity pairs
        self.scores_diff = []  # Store similarity scores for different identity pairs
    
    def compute_similarity(self, img1, img2):
        """
        Compute similarity between two face images using DeepFace
        
        Args:
            img1: First face image (numpy array, BGR format)
            img2: Second face image (numpy array, BGR format)
            
        Returns:
            similarity_score: Similarity score (0-1, higher means more similar)
            is_match: Boolean indicating whether faces match
        """
        try:
            # Ensure images are in RGB format (DeepFace expects RGB)
            if img1.shape[2] == 3:  # Check if image has 3 channels
                img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
            else:
                img1_rgb = img1
                
            if img2.shape[2] == 3:
                img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
            else:
                img2_rgb = img2
            
            # Verify faces using DeepFace
            result = DeepFace.verify(
                img1_rgb, 
                img2_rgb,
                model_name=self.model_name,
                distance_metric=self.distance_metric,
                enforce_detection=False  # Skip face detection as images are already aligned faces
            )
            
            # Extract similarity score and match decision
            distance = result['distance']
            
            # Convert distance to similarity score (0-1, higher means more similar)
            if self.distance_metric == 'cosine':
                # Cosine distance is already between 0-1
                # But it represents dissimilarity, so we invert it
                similarity_score = 1 - distance
            else:
                # For euclidean distances, use exponential decay
                similarity_score = np.exp(-distance)
            
            # Determine if match based on threshold
            is_match = similarity_score >= self.threshold
            
            return similarity_score, is_match
            
        except Exception as e:
            print(f"Error during DeepFace similarity computation: {e}")
            traceback.print_exc()
            # Return default values in case of error
            return 0.0, False
    
    def get_embeddings(self, img):
        """
        Get face embeddings using DeepFace
        
        Args:
            img: Face image (numpy array, BGR format)
            
        Returns:
            embedding: Face embedding vector
        """
        try:
            # Ensure image is in RGB format (DeepFace expects RGB)
            if img.shape[2] == 3:  # Check if image has 3 channels
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            else:
                img_rgb = img
            
            # Extract embeddings using DeepFace
            embedding_obj = DeepFace.represent(
                img_rgb,
                model_name=self.model_name,
                enforce_detection=False  # Skip face detection as image is already an aligned face
            )
            
            if isinstance(embedding_obj, list):
                embedding = np.array(embedding_obj[0]['embedding'])
            else:
                embedding = np.array(embedding_obj['embedding'])
                
            return embedding
            
        except Exception as e:
            print(f"Error during DeepFace embedding extraction: {e}")
            traceback.print_exc()
            return None
            
    def compute_similarity_from_embeddings(self, embedding1, embedding2):
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
            # Ensure embeddings are numpy arrays
            embedding1 = np.array(embedding1)
            embedding2 = np.array(embedding2)
            
            # Calculate distance based on metric
            if self.distance_metric == 'cosine':
                from scipy.spatial.distance import cosine
                distance = cosine(embedding1, embedding2)
                # Convert to similarity (1 - distance)
                similarity_score = 1.0 - distance
            elif self.distance_metric == 'euclidean':
                from scipy.spatial.distance import euclidean
                distance = euclidean(embedding1, embedding2)
                # Convert to similarity (exponential decay)
                similarity_score = np.exp(-distance)
            else:  # euclidean_l2
                # Normalize vectors to unit length
                embedding1 = embedding1 / np.linalg.norm(embedding1)
                embedding2 = embedding2 / np.linalg.norm(embedding2)
                # Calculate euclidean distance
                distance = np.linalg.norm(embedding1 - embedding2)
                # Convert to similarity
                similarity_score = 1.0 - distance / 2.0  # Normalize to 0-1 range
            
            # Ensure similarity is in [0,1] range
            similarity_score = max(0, min(1, similarity_score))
            
            # Determine if match based on threshold
            is_match = similarity_score >= self.threshold
            
            return similarity_score, is_match
            
        except Exception as e:
            print(f"Error during similarity computation from embeddings: {e}")
            traceback.print_exc()
            # Return default values in case of error
            return 0.0, False
    
    def add_pair_score(self, score, is_same_person):
        """
        Add a similarity score to the appropriate collection for performance evaluation
        
        Args:
            score: Similarity score from compute_similarity
            is_same_person: Boolean indicating if the pair is actually the same person
        """
        if is_same_person:
            self.scores_same.append(score)
        else:
            self.scores_diff.append(score)
    
    def evaluate_threshold(self):
        """
        Evaluate threshold using stored pairs
        
        Returns:
            optimal_threshold: Optimal threshold value
            metrics: Dictionary of performance metrics
        """
        if len(self.scores_same) == 0 or len(self.scores_diff) == 0:
            print("No scores available for evaluation. Add pairs with add_pair_score() first.")
            return self.threshold, {}
        
        # Convert lists to numpy arrays
        scores_same = np.array(self.scores_same)
        scores_diff = np.array(self.scores_diff)
        
        # Create labels (1 for same, 0 for different)
        y_true = np.hstack([np.ones(len(scores_same)), np.zeros(len(scores_diff))])
        y_scores = np.hstack([scores_same, scores_diff])
        
        # Calculate ROC curve
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        
        # Calculate area under curve
        auc_score = auc(fpr, tpr)
        
        # Calculate Equal Error Rate (EER)
        fnr = 1 - tpr
        abs_diffs = np.abs(fpr - fnr)
        eer_idx = np.argmin(abs_diffs)
        eer = (fpr[eer_idx] + fnr[eer_idx]) / 2
        eer_threshold = thresholds[eer_idx]
        
        # Find threshold for best accuracy
        accuracies = []
        for threshold in thresholds:
            y_pred = [1 if score >= threshold else 0 for score in y_scores]
            accuracy = np.mean(np.array(y_pred) == np.array(y_true))
            accuracies.append(accuracy)
        
        optimal_idx = np.argmax(accuracies)
        optimal_threshold = thresholds[optimal_idx]
        best_accuracy = accuracies[optimal_idx]
        
        # Calculate precision and recall curve
        precision, recall, pr_thresholds = precision_recall_curve(y_true, y_scores)
        
        # Calculate F1 scores
        f1_scores = 2 * precision * recall / (precision + recall + 1e-10)  # Avoid division by zero
        best_f1_idx = np.argmax(f1_scores)
        best_f1 = f1_scores[best_f1_idx]
        f1_threshold = pr_thresholds[min(best_f1_idx, len(pr_thresholds)-1)]
        
        # Compile metrics
        metrics = {
            'auc': auc_score,
            'eer': eer,
            'eer_threshold': eer_threshold,
            'optimal_threshold': optimal_threshold,
            'best_accuracy': best_accuracy,
            'best_f1': best_f1,
            'f1_threshold': f1_threshold,
            'fpr': fpr,
            'tpr': tpr,
            'thresholds': thresholds,
            'precision': precision,
            'recall': recall,
            'f1_scores': f1_scores
        }
        
        return optimal_threshold, metrics
    
    def plot_similarity_distribution(self):
        """
        Plot distribution of similarity scores
        
        Returns:
            fig: Matplotlib figure
        """
        if len(self.scores_same) == 0 or len(self.scores_diff) == 0:
            print("No scores available for plotting. Add pairs with add_pair_score() first.")
            return None
        
        # Convert lists to numpy arrays
        scores_same = np.array(self.scores_same)
        scores_diff = np.array(self.scores_diff)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot histograms
        bins = np.linspace(0, 1, 50)
        ax.hist(scores_same, bins=bins, alpha=0.5, label='Same Identity', color='green')
        ax.hist(scores_diff, bins=bins, alpha=0.5, label='Different Identity', color='red')
        
        # Plot threshold line
        ax.axvline(x=self.threshold, color='blue', linestyle='--', label=f'Threshold ({self.threshold:.2f})')
        
        # Add labels and legend
        ax.set_xlabel('Similarity Score')
        ax.set_ylabel('Frequency')
        ax.set_title(f'Face Similarity Distribution ({self.model_name}, {self.distance_metric})')
        ax.legend()
        
        return fig