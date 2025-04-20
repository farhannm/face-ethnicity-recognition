import matplotlib.pyplot as plt
import numpy as np
import cv2
import io
from PIL import Image

def plot_face_comparison(face1, face2, similarity_score, is_match, threshold):
    """
    Create visualization of face comparison results
    
    Args:
        face1: First face image
        face2: Second face image
        similarity_score: Similarity score (0-1)
        is_match: Boolean indicating if faces match
        threshold: Threshold used for decision
        
    Returns:
        fig: Matplotlib figure with the visualization
    """
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), 
                             gridspec_kw={'width_ratios': [1, 1, 1.5]})
    
    try:
        # Ensure images are uint8
        if face1.dtype != np.uint8:
            face1 = np.clip(face1, 0, 255).astype(np.uint8)
        if face2.dtype != np.uint8:
            face2 = np.clip(face2, 0, 255).astype(np.uint8)
        
        # Convert BGR to RGB for display
        if len(face1.shape) == 3 and face1.shape[2] == 3:
            face1_rgb = cv2.cvtColor(face1, cv2.COLOR_BGR2RGB)
        else:
            face1_rgb = face1  # Already RGB or grayscale
            
        if len(face2.shape) == 3 and face2.shape[2] == 3:
            face2_rgb = cv2.cvtColor(face2, cv2.COLOR_BGR2RGB)
        else:
            face2_rgb = face2  # Already RGB or grayscale
        
        # Display first face
        axes[0].imshow(face1_rgb)
        axes[0].set_title("Face 1")
        axes[0].axis('off')
        
        # Display second face
        axes[1].imshow(face2_rgb)
        axes[1].set_title("Face 2")
        axes[1].axis('off')
        
    except Exception as e:
        print(f"Error displaying face images: {e}")
        # Create placeholder images in case of error
        placeholder = np.ones((224, 224, 3), dtype=np.uint8) * 200
        cv2.putText(placeholder, "Error loading image", (20, 112), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
        axes[0].imshow(placeholder)
        axes[0].set_title("Face 1 (Error)")
        axes[0].axis('off')
        
        axes[1].imshow(placeholder)
        axes[1].set_title("Face 2 (Error)")
        axes[1].axis('off')
    
    # Create similarity gauge
    ax = axes[2]
    
    # Draw similarity gauge (semicircle)
    gauge_plot(ax, similarity_score, threshold, is_match)
    
    plt.tight_layout()
    return fig

def gauge_plot(ax, value, threshold, is_match):
    """
    Create a gauge plot for similarity score
    
    Args:
        ax: Matplotlib axis
        value: Similarity value (0-1)
        threshold: Threshold for decision
        is_match: Whether faces match
    """
    # Ensure value is in valid range
    value = max(0, min(1, value))
    threshold = max(0, min(1, threshold))
    
    # Clear axis
    ax.clear()
    
    # Set labels
    ax.set_title("Similarity Score", fontsize=14)
    
    # Add match/no match text
    match_text = "MATCH" if is_match else "NO MATCH"
    match_color = "green" if is_match else "red"
    ax.text(0.5, 0.15, match_text, 
            horizontalalignment='center',
            verticalalignment='center',
            fontsize=20, 
            fontweight='bold', 
            color=match_color)
    
    # Add score text
    ax.text(0.5, 0.3, f"Score: {value:.4f}", 
            horizontalalignment='center',
            verticalalignment='center',
            fontsize=14)
    
    # Add threshold text
    ax.text(0.5, 0.4, f"Threshold: {threshold:.2f}", 
            horizontalalignment='center',
            verticalalignment='center',
            fontsize=12,
            color='blue')
    
    # Create gauge
    theta = np.linspace(0, np.pi, 100)
    r = 0.8
    
    # Circle border
    x0, y0 = r * np.cos(theta), r * np.sin(theta)
    ax.plot(x0, y0, 'k')
    
    # Add ticks
    for i in range(0, 11):
        val = i / 10
        angle = np.pi * val
        xpos = r * np.cos(angle)
        ypos = r * np.sin(angle)
        xtext = 0.92 * r * np.cos(angle)
        ytext = 0.92 * r * np.sin(angle)
        ax.plot([xpos, 0.95*xpos], [ypos, 0.95*ypos], 'k')
        ax.text(xtext, ytext, f"{val:.1f}", 
                horizontalalignment='center',
                verticalalignment='center',
                fontsize=8)
    
    # Add colored regions (red-yellow-green gradient)
    for i in range(0, 100):
        val = i / 100
        val_next = (i+1) / 100
        
        angle = np.pi * val
        angle_next = np.pi * val_next
        
        # Color gradient from red to green
        if val < 0.4:
            color = (1, val/0.4, 0)  # Red to yellow
        else:
            color = (1 - (val-0.4)/0.6, 1, 0)  # Yellow to green
            
        # Draw sector
        angles = np.linspace(angle, angle_next, 10)
        xs = r * np.cos(angles)
        ys = r * np.sin(angles)
        ax.fill_between(xs, 0, ys, color=color, alpha=0.7)
    
    # Add threshold line
    threshold_angle = np.pi * threshold
    xthresh = [0, r * np.cos(threshold_angle)]
    ythresh = [0, r * np.sin(threshold_angle)]
    ax.plot(xthresh, ythresh, 'b--', linewidth=2)
    
    # Add needle
    value_angle = np.pi * value
    xneedle = [0, r * np.cos(value_angle)]
    yneedle = [0, r * np.sin(value_angle)]
    ax.plot(xneedle, yneedle, 'k-', linewidth=3)
    
    # Add needle center
    ax.plot(0, 0, 'ko', markersize=8)
    
    # Set equal aspect ratio
    ax.set_aspect('equal')
    
    # Hide axis
    ax.axis('off')
    
    # Set limits slightly larger than the gauge
    ax.set_xlim(-1, 1)
    ax.set_ylim(-0.1, 1.1)

def create_face_pair_grid(face_pairs, similarity_scores, is_match_list, num_rows=3, num_cols=3):
    """
    Create a grid visualization of face pairs with similarity scores
    
    Args:
        face_pairs: List of face image pairs (face1, face2)
        similarity_scores: List of similarity scores
        is_match_list: List of boolean match decisions
        num_rows: Number of rows in grid
        num_cols: Number of columns in grid
        
    Returns:
        fig: Matplotlib figure
    """
    # Create figure
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 15))
    
    # Flatten axes if needed
    if num_rows > 1 and num_cols > 1:
        axes = axes.flatten()
    
    # Limit to available pairs
    num_pairs = min(len(face_pairs), num_rows * num_cols)
    
    # Display each pair
    for i in range(num_pairs):
        try:
            face1, face2 = face_pairs[i]
            score = similarity_scores[i]
            is_match = is_match_list[i]
            
            # Ensure images are uint8
            if face1.dtype != np.uint8:
                face1 = np.clip(face1, 0, 255).astype(np.uint8)
            if face2.dtype != np.uint8:
                face2 = np.clip(face2, 0, 255).astype(np.uint8)
            
            # Convert BGR to RGB for display
            if len(face1.shape) == 3 and face1.shape[2] == 3:
                face1_rgb = cv2.cvtColor(face1, cv2.COLOR_BGR2RGB)
                face2_rgb = cv2.cvtColor(face2, cv2.COLOR_BGR2RGB)
            else:
                face1_rgb = face1  # Already RGB or grayscale
                face2_rgb = face2  # Already RGB or grayscale
            
            # Resize to same height
            height = min(face1_rgb.shape[0], face2_rgb.shape[0])
            
            ratio1 = height / face1_rgb.shape[0]
            new_width1 = int(face1_rgb.shape[1] * ratio1)
            face1_resized = cv2.resize(face1_rgb, (new_width1, height))
            
            ratio2 = height / face2_rgb.shape[0]
            new_width2 = int(face2_rgb.shape[1] * ratio2)
            face2_resized = cv2.resize(face2_rgb, (new_width2, height))
            
            # Add a separator (3 pixels wide white line)
            separator = np.ones((height, 3, 3), dtype=np.uint8) * 255
            
            # Combine images horizontally
            combined = np.hstack((face1_resized, separator, face2_resized))
            
            # Display in axis
            ax = axes[i] if isinstance(axes, np.ndarray) else axes
            ax.imshow(combined)
            
            # Display score with color
            match_label = "MATCH" if is_match else "NO MATCH"
            label_color = "green" if is_match else "red"
            
            # Add custom title
            ax.set_title(f"{match_label} (Score: {score:.3f})", 
                        color=label_color,
                        fontweight='bold')
            ax.axis('off')
            
        except Exception as e:
            print(f"Error displaying face pair {i}: {e}")
            # Create placeholder image in case of error
            placeholder = np.ones((224, 224, 3), dtype=np.uint8) * 200
            cv2.putText(placeholder, "Error displaying face pair", (20, 112), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            
            ax = axes[i] if isinstance(axes, np.ndarray) else axes
            ax.imshow(placeholder)
            ax.set_title(f"Error in pair {i}")
            ax.axis('off')
    
    # Hide unused subplots
    for i in range(num_pairs, num_rows * num_cols):
        if isinstance(axes, np.ndarray) and i < len(axes):
            axes[i].axis('off')
    
    plt.tight_layout()
    return fig