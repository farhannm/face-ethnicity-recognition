import matplotlib.pyplot as plt
import numpy as np
import cv2
import io
from PIL import Image

def plot_similarity_result(face1, face2, similarity_score, is_match):
    """
    Create a visualization of face similarity result
    
    Args:
        face1: First face image
        face2: Second face image
        similarity_score: Similarity score (0-1)
        is_match: Boolean indicating if faces match
        
    Returns:
        fig: Matplotlib figure with the visualization
    """
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    # Convert BGR to RGB for display
    face1_rgb = cv2.cvtColor(face1, cv2.COLOR_BGR2RGB)
    face2_rgb = cv2.cvtColor(face2, cv2.COLOR_BGR2RGB)
    
    # Display face images
    axes[0].imshow(face1_rgb)
    axes[0].set_title("Face 1")
    axes[0].axis('off')
    
    axes[2].imshow(face2_rgb)
    axes[2].set_title("Face 2")
    axes[2].axis('off')
    
    # Display similarity visualization
    axes[1].set_xlim(0, 1)
    axes[1].set_ylim(0, 1)
    axes[1].axvline(x=0.5, color='gray', linestyle='--', alpha=0.5)  # Threshold line
    
    # Draw similarity gauge
    color = 'green' if is_match else 'red'
    axes[1].barh(0.5, similarity_score, height=0.3, color=color)
    
    # Add similarity score text
    axes[1].text(0.5, 0.8, f"Similarity: {similarity_score:.2f}", 
                ha='center', va='center', fontsize=12)
    
    # Add match/no match text
    match_text = "MATCH" if is_match else "NO MATCH"
    axes[1].text(0.5, 0.2, match_text, 
                ha='center', va='center', fontsize=14, 
                fontweight='bold', color=color)
    
    axes[1].set_title("Similarity Result")
    axes[1].axis('off')
    
    plt.tight_layout()
    
    return fig

def plot_ethnicity_prediction(face_img, ethnicity, confidences):
    """
    Create a visualization of ethnicity prediction
    
    Args:
        face_img: Face image
        ethnicity: Predicted ethnicity
        confidences: Dictionary of confidence scores for each ethnicity
        
    Returns:
        fig: Matplotlib figure with the visualization
    """
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), gridspec_kw={'width_ratios': [1, 1.5]})
    
    # Convert BGR to RGB for display
    face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    
    # Display face image
    axes[0].imshow(face_rgb)
    axes[0].set_title("Input Face")
    axes[0].axis('off')
    
    # Sort confidences for bar chart
    ethnicities = list(confidences.keys())
    scores = [confidences[e] for e in ethnicities]
    
    # Sort by confidence score
    sorted_indices = np.argsort(scores)
    ethnicities = [ethnicities[i] for i in sorted_indices]
    scores = [scores[i] for i in sorted_indices]
    
    # Set colors (highlight the predicted ethnicity)
    colors = ['lightgray'] * len(ethnicities)
    try:
        colors[ethnicities.index(ethnicity)] = 'green'
    except ValueError:
        # Handle case where predicted ethnicity is not in the confidences
        pass
    
    # Plot horizontal bar chart
    y_pos = np.arange(len(ethnicities))
    axes[1].barh(y_pos, scores, color=colors)
    axes[1].set_yticks(y_pos)
    axes[1].set_yticklabels(ethnicities)
    axes[1].set_xlim(0, 1)
    axes[1].set_xlabel('Confidence Score')
    axes[1].set_title('Ethnicity Prediction')
    
    # Add confidence values as text
    for i, v in enumerate(scores):
        axes[1].text(v + 0.01, i, f'{v:.2f}', va='center')
    
    plt.tight_layout()
    
    return fig

def draw_bounding_box(image, face_rect, label=None, color=(0, 255, 0), thickness=2):
    """
    Draw a bounding box around a face with optional label
    
    Args:
        image: Input image
        face_rect: Face rectangle coordinates (x, y, w, h)
        label: Optional text label
        color: Box color in BGR
        thickness: Line thickness
        
    Returns:
        image_with_box: Image with drawn bounding box
    """
    if face_rect is None:
        return image
        
    image_with_box = image.copy()
    x, y, w, h = face_rect
    
    # Draw rectangle
    cv2.rectangle(image_with_box, (x, y), (x+w, y+h), color, thickness)
    
    # Add label if provided
    if label:
        # Calculate text position and size
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_thickness = 1
        text_size = cv2.getTextSize(label, font, font_scale, font_thickness)[0]
        
        # Draw text background
        cv2.rectangle(image_with_box, (x, y - text_size[1] - 5), (x + text_size[0], y), color, -1)
        
        # Draw text
        cv2.putText(image_with_box, label, (x, y - 5), font, font_scale, (0, 0, 0), font_thickness)
    
    return image_with_box

def fig_to_image(fig):
    """
    Convert a matplotlib figure to an image
    
    Args:
        fig: Matplotlib figure
        
    Returns:
        image: PIL Image
    """
    # Save figure to a byte buffer
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100)
    buf.seek(0)
    
    # Create PIL image
    img = Image.open(buf)
    
    return img