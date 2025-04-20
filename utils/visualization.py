import matplotlib.pyplot as plt
import numpy as np
import cv2
import io
from PIL import Image

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
    if len(face_img.shape) == 3 and face_img.shape[2] == 3:
        face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    else:
        face_rgb = face_img  # Already RGB or grayscale
    
    # Display face image
    axes[0].imshow(face_rgb)
    axes[0].set_title("Input Face")
    axes[0].axis('off')
    
    # Sort confidences for bar chart
    ethnicities = list(confidences.keys())
    scores = [confidences[e] for e in ethnicities]
    
    # Sort by confidence score (descending)
    sorted_indices = np.argsort(scores)[::-1]
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

def draw_landmarks(image, landmarks, radius=2, color=(0, 0, 255), thickness=-1):
    """
    Draw facial landmarks on an image
    
    Args:
        image: Input image
        landmarks: Dictionary containing facial landmarks
        radius: Circle radius for landmarks
        color: Color for landmarks in BGR
        thickness: Circle thickness
        
    Returns:
        image_with_landmarks: Image with drawn landmarks
    """
    image_with_landmarks = image.copy()
    
    # Draw each landmark
    for point_name, point in landmarks.items():
        cv2.circle(image_with_landmarks, point, radius, color, thickness)
        
    return image_with_landmarks

def create_processing_steps_visualization(original_img, face_detected_img, aligned_img, normalized_img):
    """
    Create a visualization of all preprocessing steps
    
    Args:
        original_img: Original input image
        face_detected_img: Image with detected face
        aligned_img: Aligned face image
        normalized_img: Normalized face image
        
    Returns:
        fig: Matplotlib figure with the visualization
    """
    # Create figure
    fig, axes = plt.subplots(1, 4, figsize=(15, 4))
    
    # Convert BGR to RGB for display
    original_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    face_detected_rgb = cv2.cvtColor(face_detected_img, cv2.COLOR_BGR2RGB)
    aligned_rgb = cv2.cvtColor(aligned_img, cv2.COLOR_BGR2RGB)
    normalized_rgb = cv2.cvtColor(normalized_img, cv2.COLOR_BGR2RGB)
    
    # Display images
    axes[0].imshow(original_rgb)
    axes[0].set_title("Original Image")
    axes[0].axis('off')
    
    axes[1].imshow(face_detected_rgb)
    axes[1].set_title("Face Detection")
    axes[1].axis('off')
    
    axes[2].imshow(aligned_rgb)
    axes[2].set_title("Face Alignment")
    axes[2].axis('off')
    
    axes[3].imshow(normalized_rgb)
    axes[3].set_title("Normalization")
    axes[3].axis('off')
    
    plt.tight_layout()
    
    return fig

def fig_to_image(fig):
    """
    Convert a matplotlib figure to a PIL Image
    
    Args:
        fig: Matplotlib figure
        
    Returns:
        img: PIL Image
    """
    # Save figure to a byte buffer
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100)
    buf.seek(0)
    
    # Create PIL image
    img = Image.open(buf)
    
    return img