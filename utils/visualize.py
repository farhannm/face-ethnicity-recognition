import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from utils.metrics import FaceRecognitionMetrics

class FaceSimilarityVisualizer:
    @staticmethod
    def visualize_embeddings(embeddings, labels, title="Face Embedding Visualization"):
        """
        Visualize face embeddings using dimensionality reduction
        
        Args:
            embeddings: Array of face embeddings
            labels: Corresponding labels
            title: Plot title
        
        Returns:
            matplotlib.figure.Figure
        """
        from sklearn.manifold import TSNE
        
        # Reduce dimensionality
        tsne = TSNE(n_components=2, random_state=42)
        reduced_embeddings = tsne.fit_transform(embeddings)
        
        # Create plot
        plt.figure(figsize=(10, 8))
        
        # Get unique labels
        unique_labels = np.unique(labels)
        
        # Use different colors for different labels
        colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
        
        for label, color in zip(unique_labels, colors):
            mask = labels == label
            plt.scatter(
                reduced_embeddings[mask, 0], 
                reduced_embeddings[mask, 1], 
                c=[color], 
                label=f'Label {label}', 
                alpha=0.7
            )
        
        plt.title(title)
        plt.xlabel('t-SNE Dimension 1')
        plt.ylabel('t-SNE Dimension 2')
        plt.legend()
        
        return plt.gcf()
    
    @staticmethod
    def create_comparison_grid(face_pairs, scores, matches, expected_matches, output_dir):
        """
        Create a grid of face comparisons showing different match scenarios
        
        Args:
            face_pairs: List of face image pairs
            scores: Similarity scores
            matches: Predicted matches
            expected_matches: Ground truth matches
            output_dir: Directory to save the visualization
        
        Returns:
            matplotlib.figure.Figure
        """
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Create figure
        fig, axes = plt.subplots(4, 4, figsize=(16, 16))
        
        # Categories of comparisons
        categories = [
            ('True Positive', True, True),
            ('False Positive', True, False),
            ('True Negative', False, False),
            ('False Negative', False, True)
        ]
        
        # Iterate through categories
        for row, (category, pred_match, exp_match) in enumerate(categories):
            # Find matching examples
            matching_indices = [
                i for i, (pred, exp) in enumerate(zip(matches, expected_matches))
                if pred == pred_match and exp == exp_match
            ]
            
            # Display up to 4 examples
            for col in range(4):
                ax = axes[row, col]
                
                if col < len(matching_indices):
                    idx = matching_indices[col]
                    face1, face2 = face_pairs[idx]
                    score = scores[idx]
                    
                    # Convert to RGB if needed
                    if len(face1.shape) == 3 and face1.shape[2] == 3:
                        face1 = cv2.cvtColor(face1, cv2.COLOR_BGR2RGB)
                        face2 = cv2.cvtColor(face2, cv2.COLOR_BGR2RGB)
                    
                    # Combine images side by side
                    combined = np.hstack((face1, face2))
                    
                    # Display
                    ax.imshow(combined)
                    ax.set_title(f"Score: {score:.3f}")
                
                ax.axis('off')
            
            # Add category label
            fig.text(0.01, 0.75 - row * 0.25, category, fontsize=14)
        
        plt.tight_layout()
        
        # Save the figure
        output_path = os.path.join(output_dir, 'face_comparison_grid.png')
        plt.savefig(output_path)
        
        return fig