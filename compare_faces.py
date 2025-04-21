import os
import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt
from deepface import DeepFace

from utils.metrics import FaceRecognitionMetrics
from utils.visualize import FaceSimilarityVisualizer

class FaceComparer:
    def __init__(self, test_dir, model_name='VGG-Face'):
        """
        Initialize face comparer with test directory and DeepFace model
        
        Args:
            test_dir: Directory containing test images
            model_name: DeepFace recognition model 
        """
        self.test_dir = test_dir
        self.model_name = model_name
        self.supported_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    
    def load_face_images(self):
        """
        Load face images from test directory
        
        Returns:
            List of face images
            List of corresponding labels/filenames
        """
        face_images = []
        labels = []
        
        # Walk through directory
        for root, _, files in os.walk(self.test_dir):
            # Skip empty directories
            if not files:
                continue
            
            # Use directory name as label
            label = os.path.basename(root)
            
            for file in files:
                # Check file extension
                if any(file.lower().endswith(ext) for ext in self.supported_extensions):
                    # Full path to image
                    full_path = os.path.join(root, file)
                    
                    # Read image
                    try:
                        # Use DeepFace to verify if it's a valid face
                        result = DeepFace.extract_faces(full_path, enforce_detection=False)
                        
                        # Read image with OpenCV
                        image = cv2.imread(full_path)
                        
                        if image is not None:
                            face_images.append(image)
                            labels.append(label)
                    except Exception as e:
                        print(f"Error processing {full_path}: {e}")
        
        return face_images, labels
    
    def compute_face_embedding(self, image):
        """
        Compute face embedding using DeepFace
        
        Args:
            image: Input image
        
        Returns:
            Face embedding vector
        """
        try:
            # Temporarily save image to use with DeepFace
            temp_path = 'temp_face_image.jpg'
            cv2.imwrite(temp_path, image)
            
            # Extract embedding
            embedding = DeepFace.represent(
                img_path=temp_path, 
                model_name=self.model_name,
                enforce_detection=False
            )[0]['embedding']
            
            # Remove temporary file
            os.remove(temp_path)
            
            return np.array(embedding)
        except Exception as e:
            print(f"Error computing embedding: {e}")
            return None
    
    def compute_embeddings(self, face_images):
        """
        Compute face embeddings for all images
        
        Args:
            face_images: List of face images
        
        Returns:
            Numpy array of embeddings
        """
        embeddings = []
        for image in face_images:
            embedding = self.compute_face_embedding(image)
            if embedding is not None:
                embeddings.append(embedding)
        
        return np.array(embeddings)
    
    def compare_faces(self, output_dir='evaluation'):
        """
        Compare faces and generate performance metrics
        
        Args:
            output_dir: Directory to save evaluation results
        
        Returns:
            Dict of performance metrics
        """
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Load face images
        face_images, labels = self.load_face_images()
        
        # Compute embeddings
        embeddings = self.compute_embeddings(face_images)
        
        # Create face pairs and compute similarities
        face_pairs = []
        scores = []
        matches = []
        expected_matches = []
        
        # Generate all possible pairs
        for i in range(len(embeddings)):
            for j in range(i+1, len(embeddings)):
                face_pairs.append((face_images[i], face_images[j]))
                
                # Compute similarity using cosine similarity
                from scipy.spatial.distance import cosine
                similarity = 1 - cosine(embeddings[i], embeddings[j])
                scores.append(similarity)
                
                # Expected match is based on label
                expected_match = labels[i] == labels[j]
                expected_matches.append(expected_match)
                
                # Predict match (use adaptive threshold based on model)
                model_thresholds = {
                    'VGG-Face': 0.5,
                    'Facenet': 0.4,
                    'OpenFace': 0.35,
                    'DeepFace': 0.45,
                    'DeepID': 0.4,
                    'ArcFace': 0.55,
                    'Dlib': 0.45
                }
                threshold = model_thresholds.get(self.model_name, 0.5)
                matches.append(similarity > threshold)
        
        # Convert to numpy arrays
        scores = np.array(scores)
        matches = np.array(matches)
        expected_matches = np.array(expected_matches)
        
        # Compute metrics
        metrics = FaceRecognitionMetrics.compute_rates(
            expected_matches, matches, scores
        )
        
        # Log model details
        metrics['Model'] = self.model_name
        
        # Visualize embeddings
        FaceSimilarityVisualizer.visualize_embeddings(
            embeddings, 
            np.array(labels), 
            os.path.join(output_dir, 'embedding_visualization.png')
        )
        
        # Create comparison grid
        FaceSimilarityVisualizer.create_comparison_grid(
            face_pairs, scores, matches, expected_matches, 
            os.path.join(output_dir, 'face_comparison_grid.png')
        )
        
        # Visualize ROC curve
        FaceSimilarityVisualizer.plot_roc_curve(
            metrics['ROC Curve']['False Positive Rates'],
            metrics['ROC Curve']['True Positive Rates'],
            metrics['Area Under Curve (AUC)'],
            os.path.join(output_dir, 'roc_curve.png')
        )
        
        # Save metrics report
        FaceRecognitionMetrics.save_metrics_report(
            metrics, 
            os.path.join(output_dir, 'performance_metrics.txt')
        )
        
        return metrics
    
def compare_two_faces(image1_path, image2_path, model_name='VGG-Face'):
    """
    Compare two faces manually with enhanced visualizations
    
    Args:
        image1_path: Path to first image
        image2_path: Path to second image
        model_name: Face recognition model to use
    
    Returns:
        Comparison result and visualization
    """
    import os
    import numpy as np
    import cv2
    import matplotlib.pyplot as plt
    from scipy.spatial.distance import cosine
    from deepface import DeepFace
    from utils.metrics import FaceRecognitionMetrics
    from utils.visualize import FaceSimilarityVisualizer
    
    # Output directory
    output_dir = 'evaluation'
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Validate image paths
    if not os.path.exists(image1_path):
        raise FileNotFoundError(f"Image 1 not found: {image1_path}")
    if not os.path.exists(image2_path):
        raise FileNotFoundError(f"Image 2 not found: {image2_path}")
    
    # Read images
    img1 = cv2.imread(image1_path)
    img2 = cv2.imread(image2_path)
    
    # Ensure images are read correctly
    if img1 is None or img2 is None:
        raise ValueError("Could not read one or both images")
    
    try:
        # Use DeepFace to compare faces
        verification_result = DeepFace.verify(
            img1_path=image1_path, 
            img2_path=image2_path, 
            model_name=model_name,
            enforce_detection=False
        )
        
        # Extract embeddings for both images
        embedding1 = DeepFace.represent(
            img_path=image1_path, 
            model_name=model_name,
            enforce_detection=False
        )[0]['embedding']
        
        embedding2 = DeepFace.represent(
            img_path=image2_path, 
            model_name=model_name,
            enforce_detection=False
        )[0]['embedding']
        
        # Convert embeddings to numpy arrays
        embedding1 = np.array(embedding1)
        embedding2 = np.array(embedding2)
        
        # Compute similarity using cosine similarity
        similarity = 1 - cosine(embedding1, embedding2)
        
        # Model thresholds (same as in FaceComparer class)
        model_thresholds = {
            'VGG-Face': 0.5,
            'Facenet': 0.4,
            'OpenFace': 0.35,
            'DeepFace': 0.45,
            'DeepID': 0.4,
            'ArcFace': 0.55,
            'Dlib': 0.45
        }
        threshold = model_thresholds.get(model_name, 0.5)
        
        # Determine if images match based on threshold
        is_match = similarity > threshold
        
        # Prepare data for visualizations
        embeddings = np.array([embedding1, embedding2])
        labels = np.array([os.path.basename(os.path.dirname(image1_path)), 
                           os.path.basename(os.path.dirname(image2_path))])
        face_pairs = [(img1, img2)]
        scores = np.array([similarity])
        matches = np.array([is_match])
        expected_matches = np.array([labels[0] == labels[1]])  # Assuming same folder = same person
        
        # 1. Create basic comparison visualization
        def convert_to_rgb(img):
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize images to have same height
        height = min(img1.shape[0], img2.shape[0])
        
        # Resize first image
        aspect_ratio1 = img1.shape[1] / img1.shape[0]
        new_width1 = int(height * aspect_ratio1)
        img1_resized = cv2.resize(img1, (new_width1, height))
        
        # Resize second image
        aspect_ratio2 = img2.shape[1] / img2.shape[0]
        new_width2 = int(height * aspect_ratio2)
        img2_resized = cv2.resize(img2, (new_width2, height))
        
        # Create white separator
        separator = np.ones((height, 5, 3), dtype=np.uint8) * 255
        
        # Combine images
        combined = np.hstack([
            convert_to_rgb(img1_resized), 
            separator, 
            convert_to_rgb(img2_resized)
        ])
        
        # Save comparison image
        comparison_path = os.path.join(output_dir, 'face_comparison.png')
        cv2.imwrite(comparison_path, cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))
        
        # 2. Create simple embedding visualization instead of using t-SNE
        # (which requires more than 2 samples for perplexity)
        embedding_viz_path = os.path.join(output_dir, 'embedding_visualization.png')
        
        # Create a custom embedding visualization instead of using FaceSimilarityVisualizer
        plt.figure(figsize=(10, 8))
        
        # Create a basic 2D representation using PCA-like projection (first 2 dimensions)
        # This is a simplification just to show the relationship between two embeddings
        embedding_diff = embedding2 - embedding1
        magnitude = np.linalg.norm(embedding_diff)
        
        # Plot as vectors in 2D space
        plt.scatter(0, 0, color='red', s=100, label=labels[0])
        plt.scatter(1, 0, color='blue', s=100, label=labels[1])
        
        # Draw connection line with width based on similarity
        plt.plot([0, 1], [0, 0], linewidth=2, color='gray', alpha=0.7)
        
        # Add similarity score text
        plt.text(0.5, 0.1, f"Similarity: {similarity:.4f}", 
                 horizontalalignment='center', fontsize=12)
        
        # Add match/no match text
        match_text = "MATCH" if is_match else "NO MATCH"
        match_color = "green" if is_match else "red"
        plt.text(0.5, -0.1, match_text, 
                 horizontalalignment='center', fontsize=14, 
                 color=match_color, weight='bold')
        
        plt.title(f"Face Embedding Relationship ({model_name})")
        plt.xlim(-0.5, 1.5)
        plt.ylim(-0.5, 0.5)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.savefig(embedding_viz_path)
        plt.close()
        
        # 3. Create comparison grid
        grid_path = os.path.join(output_dir, 'face_comparison_grid.png')
        
        # Create a custom grid visualization instead of using FaceSimilarityVisualizer
        plt.figure(figsize=(12, 6))
        
        # Face 1
        plt.subplot(1, 2, 1)
        plt.imshow(convert_to_rgb(img1))
        plt.title(f"Image 1: {os.path.basename(image1_path)}")
        plt.axis('off')
        
        # Face 2
        plt.subplot(1, 2, 2)
        plt.imshow(convert_to_rgb(img2))
        plt.title(f"Image 2: {os.path.basename(image2_path)}")
        plt.axis('off')
        
        # Add similarity information in the figure
        plt.suptitle(f"Similarity: {similarity:.4f} | {'MATCH' if is_match else 'NO MATCH'}", 
                    fontsize=16, color='black' if is_match else 'red')
        
        plt.tight_layout()
        plt.savefig(grid_path)
        plt.close()
        
        # 4. Compute metrics for the comparison
        # For one pair, we need to create some additional metrics beyond basic match/mismatch
        # Simulate a ROC curve with just a single point plus endpoints
        fpr = np.array([0, 1-is_match, 1])  # 0, 0 or 1, 1
        tpr = np.array([0, expected_matches[0], 1])  # 0, 0 or 1, 1
        thresholds = np.array([1, threshold, 0])
        
        # Calculate AUC (simple approximation for this special case)
        roc_auc = 0.5  # Default for random classifier
        if expected_matches[0] == is_match:
            roc_auc = 1.0  # Perfect classification for this sample
            
        # Calculate Precision, Recall, F1-Score berdasarkan hasil klasifikasi
        # Hitung Precision, Recall, F1-Score dengan benar untuk satu pasang wajah
        if is_match:  # Jika prediksi = match
            if expected_matches[0]:  # dan harapan = match (True Positive)
                precision = 1.0
                recall = 1.0
                f1_score = 1.0
                tar = 1.0
                far = 0.0
                frr = 0.0
            else:  # tapi harapan = non-match (False Positive)
                precision = 0.0
                recall = 0.0  # Tidak relevan karena tidak ada positif sebenarnya
                f1_score = 0.0
                tar = 0.0  # Tidak relevan karena tidak ada positif sebenarnya
                far = 1.0
                frr = 0.0  # Tidak relevan karena tidak ada positif sebenarnya
        else:  # Jika prediksi = non-match
            if expected_matches[0]:  # tapi harapan = match (False Negative)
                precision = 0.0  # Tidak relevan karena tidak ada positif prediksi
                recall = 0.0
                f1_score = 0.0
                tar = 0.0
                far = 0.0
                frr = 1.0
            else:  # dan harapan = non-match (True Negative)
                precision = 1.0  # Semua prediksi negatif benar
                recall = 1.0  # Semua negatif sebenarnya terdeteksi
                f1_score = 1.0  # Perfect untuk kasus ini
                tar = 1.0  # Tidak relevan untuk kasus negatif
                far = 0.0
                frr = 0.0
        
        metrics = {
            'Model': model_name,
            'Similarity Score': similarity,
            'Threshold': threshold,
            'Predicted Match': is_match,
            'Expected Match': expected_matches[0],
            'True Positive': expected_matches[0] and is_match,
            'False Positive': not expected_matches[0] and is_match,
            'True Negative': not expected_matches[0] and not is_match,
            'False Negative': expected_matches[0] and not is_match,
            'Area Under Curve (AUC)': roc_auc,
            'Precision': precision,
            'Recall': recall, 
            'F1-Score': f1_score,
            'True Acceptance Rate (TAR)': tar,
            'False Acceptance Rate (FAR)': far,
            'False Rejection Rate (FRR)': frr,
            'Equal Error Rate (EER)': 0.5 * (not (expected_matches[0] == is_match)),
            'ROC Curve': {
                'False Positive Rates': fpr,
                'True Positive Rates': tpr,
                'Thresholds': thresholds
            }
        }
        
        # Save metrics report
        metrics_path = os.path.join(output_dir, 'comparison_metrics.txt')
        
        # Create a simple metrics report manually
        with open(metrics_path, 'w') as f:
            f.write("=== Face Comparison Metrics ===\n\n")
            for key, value in metrics.items():
                if key != 'ROC Curve':  # Skip the ROC curve data
                    f.write(f"{key}: {value}\n")
        
        # 5. Add visualizations for requested metrics
        
        # 5.1. Create a bar chart for TAR, FAR, FRR, EER
        plt.figure(figsize=(10, 6))
        metrics_names = ['True Acceptance Rate (TAR)', 'False Acceptance Rate (FAR)', 
                       'False Rejection Rate (FRR)', 'Equal Error Rate (EER)']
        metrics_values = [metrics[name] for name in metrics_names]
        bars = plt.bar(metrics_names, metrics_values, color=['green', 'red', 'orange', 'purple'], alpha=0.7)
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.2f}', ha='center', va='bottom')
        
        plt.title('Face Recognition Rates')
        plt.ylabel('Rate')
        plt.ylim(0, 1.1)
        plt.xticks(rotation=15, ha='right')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'rates_chart.png'))
        plt.close()
        
        # 5.2. Create a bar chart for Precision, Recall, F1
        plt.figure(figsize=(10, 6))
        metrics_names = ['Precision', 'Recall', 'F1-Score']
        metrics_values = [metrics[name] for name in metrics_names]
        
        # Debug output untuk melihat nilai yang akan divisualisasikan
        print("\nDebug - Metrik untuk visualisasi Precision, Recall, F1:")
        for name, value in zip(metrics_names, metrics_values):
            print(f"{name}: {value}")
            
        bars = plt.bar(metrics_names, metrics_values, color=['blue', 'green', 'purple'], alpha=0.7)
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.2f}', ha='center', va='bottom')
        
        plt.title('Classification Performance Metrics')
        plt.ylabel('Score')
        plt.ylim(0, 1.1)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        # Simpan visualisasi ke file tertentu
        precision_recall_path = os.path.join(output_dir, 'precision_recall_f1.png')
        plt.savefig(precision_recall_path)
        print(f"Menyimpan visualisasi Precision, Recall, F1 ke: {precision_recall_path}")
        plt.close()
        
        # 5.3. Visualize AUC
        plt.figure(figsize=(8, 6))
        
        # Create a filled area to represent AUC
        plt.fill_between([0, 1], [0, 1], color='lightgray', alpha=0.5, label='Random Classifier (AUC=0.5)')
        
        # Draw a curve to represent our classifier
        x = np.linspace(0, 1, 100)
        # This is a simplified curve that approximates an ROC curve with the given AUC
        beta = 4  # Controls the shape of the curve
        y = x ** (1/beta) * roc_auc
        y = np.minimum(y, 1)  # Ensure y doesn't exceed 1
        
        plt.plot(x, y, 'r-', linewidth=3, label=f'Classifier (AUC={roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
        
        # Plot our single data point
        plt.scatter(fpr[1], tpr[1], c='red', s=100, zorder=5)
        plt.annotate(f'Threshold={threshold:.2f}', (fpr[1], tpr[1]), 
                   xytext=(20, -20), textcoords='offset points', 
                   arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2'))
        
        plt.xlim([0, 1])
        plt.ylim([0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - Area Under Curve (AUC) = {roc_auc:.2f}')
        plt.legend(loc='lower right')
        plt.grid(linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'auc_visualization.png'))
        plt.close()
        
        # 5.4. Create a threshold impact visualization
        plt.figure(figsize=(10, 6))
        
        # Create sample thresholds
        sample_thresholds = np.linspace(0, 1, 11)
        
        # Calculate rates at each threshold
        sample_far = np.zeros_like(sample_thresholds)
        sample_tar = np.zeros_like(sample_thresholds)
        sample_frr = np.zeros_like(sample_thresholds)
        
        for i, t in enumerate(sample_thresholds):
            # For a single comparison, this is simplified
            would_match = similarity > t
            sample_far[i] = 1.0 if not expected_matches[0] and would_match else 0.0
            sample_tar[i] = 1.0 if expected_matches[0] and would_match else 0.0
            sample_frr[i] = 1.0 if expected_matches[0] and not would_match else 0.0
        
        plt.plot(sample_thresholds, sample_far, 'ro-', label='False Acceptance Rate (FAR)')
        plt.plot(sample_thresholds, sample_tar, 'go-', label='True Acceptance Rate (TAR)')
        plt.plot(sample_thresholds, sample_frr, 'bo-', label='False Rejection Rate (FRR)')
        
        # Mark the current threshold
        plt.axvline(x=threshold, color='black', linestyle='--', alpha=0.7)
        plt.text(threshold, 0.5, f'Current Threshold: {threshold:.2f}', 
               rotation=90, verticalalignment='center')
        
        plt.xlabel('Threshold')
        plt.ylabel('Rate')
        plt.title('Impact of Threshold on Classification Rates')
        plt.legend()
        plt.grid(linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'threshold_impact.png'))
        plt.close()
        
        # Print detailed results
        print("\nFace Comparison Results:")
        print(f"Model Used: {model_name}")
        print(f"Verified: {verification_result['verified']}")
        print(f"Distance: {verification_result['distance']}")
        print(f"Similarity Score: {similarity:.4f}")
        print(f"Threshold: {threshold:.2f}")
        
        print(f"\nVisualization files saved to {output_dir}:")
        print(f"- Basic comparison: {os.path.basename(comparison_path)}")
        print(f"- Embedding visualization: {os.path.basename(embedding_viz_path)}")
        print(f"- Comparison grid: {os.path.basename(grid_path)}")
        print(f"- Metrics report: {os.path.basename(metrics_path)}")
        print(f"- Performance visualizations:")
        print(f"  * rates_chart.png: TAR, FAR, FRR, EER")
        print(f"  * precision_recall_f1.png: Precision, Recall, F1-Score")
        print(f"  * auc_visualization.png: Area Under Curve (AUC)")
        print(f"  * threshold_impact.png: Impact of threshold on rates")
        
        return verification_result
    
    except Exception as e:
        print(f"Error comparing faces: {e}")
        import traceback
        traceback.print_exc()  # Print full error traceback for debugging
        return None

def main():
    """
    Main function to manually compare two face images
    """
    import argparse
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Compare Two Face Images')
    parser.add_argument('image1', type=str, help='Path to first image')
    parser.add_argument('image2', type=str, help='Path to second image')
    parser.add_argument(
        '--model', 
        type=str, 
        default='VGG-Face', 
        choices=['VGG-Face', 'Facenet', 'OpenFace', 'DeepFace', 'DeepID', 'ArcFace', 'Dlib'],
        help='Face recognition model to use'
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Compare faces
    compare_two_faces(args.image1, args.image2, args.model)

if __name__ == "__main__":
    main()