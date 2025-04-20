import os
import sys
import cv2
import numpy as np
import pandas as pd
import tempfile
import streamlit as st
import time
import matplotlib.pyplot as plt
from PIL import Image
from scipy.spatial.distance import cosine, euclidean
from utils.facenet_model import FaceNetModel
from deepface import DeepFace
# Set page config must be the first Streamlit command
st.set_page_config(
    page_title="Face Recognition System",
    layout="wide"
)

# Add utils directory to Python path if needed
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import custom modules
from utils.face_detector import MTCNNFaceDetector
from utils.face_preprocessing import preprocess_face_for_facenet, normalize_face, compare_face_features
from utils.face_embedder import FaceEmbedder
from utils.ethnic_classifier import EthnicClassifier
from utils.face_similarity import FaceSimilarity
from utils.similarity_visualization import plot_face_comparison, create_face_pair_grid
from utils.facenet_embedder import FaceNetEmbedder
from utils.visualization import (
    plot_ethnicity_prediction,
    draw_bounding_box,
    draw_landmarks,
    create_processing_steps_visualization
)

# Define model paths
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'facenet_svm_model.pkl')

@st.cache_resource
def load_face_detector():
    """Load face detector with caching"""
    return MTCNNFaceDetector()

@st.cache_resource
def load_face_embedder():
    """Load face embedder with caching"""
    # Initialize with target dimensions matching KNN model (512 features)
    return FaceEmbedder(target_dimensions=512)

@st.cache_resource
def load_facenet_embedder():
    """Load FaceNet-inspired embedder with caching"""
    return FaceNetEmbedder(embedding_size=512)

@st.cache_resource
def load_ethnicity_classifier():
    """Load ethnicity classifier with caching"""
    return EthnicClassifier(model_path=MODEL_PATH)

@st.cache_resource
def load_face_similarity():
    """Load face similarity with caching"""
    return FaceSimilarity(threshold=0.6, distance_metric='cosine')

@st.cache_resource
def load_facenet_model():
    """Load FaceNet model with caching"""
    return FaceNetModel()

# Function to convert uploaded file to OpenCV image
def load_image_from_upload(uploaded_file):
    """Convert uploaded file to OpenCV image"""
    # Read file as bytes
    bytes_data = uploaded_file.getvalue()
    
    # Convert to numpy array
    nparr = np.frombuffer(bytes_data, np.uint8)
    
    # Decode image
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    return img

def perform_deepface_analysis(image):
    """
    Perform age, gender, and emotion analysis using DeepFace
    
    Args:
        image: Input image in BGR format
        
    Returns:
        dict: Containing age, gender, and emotion predictions
    """
    try:
        # Convert BGR to RGB for DeepFace
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Perform DeepFace analysis
        result = DeepFace.analyze(
            img_path=rgb_image, 
            actions=['age', 'gender', 'emotion'],
            enforce_detection=False
        )
        
        # If multiple faces are detected, take the first one
        if isinstance(result, list):
            result = result[0]
        
        return result
    
    except Exception as e:
        st.warning(f"DeepFace analysis error: {e}")
        return None

def process_image(image, face_detector, face_embedder, ethnicity_classifier, detection_threshold=0.5):
    """Process image for ethnicity, age, and gender detection"""
    try:
        # Create columns for steps visualization
        col1, col2 = st.columns(2)
        
        with col1:
            # Step 1: Detect and align face
            with st.spinner("Detecting and aligning face..."):
                start_time = time.time()
                
                # Detect face using MTCNN
                normalized_face, face_found, face_rect, confidence = face_detector.detect_and_align_face(
                    image, target_size=(224, 224))
                
                detection_time = time.time() - start_time
                
            # Check if face was detected
            if not face_found or (confidence and confidence < detection_threshold):
                st.warning(f"No face detected clearly (confidence: {confidence if confidence else 'unknown'}). Results may not be accurate.")
            
            # Draw face bounding box if detected
            if face_rect is not None:
                # Add confidence to label if available
                label = f"Face: {confidence:.2f}" if confidence else "Face"
                image_with_box = draw_bounding_box(image, face_rect, label)
                st.image(cv2.cvtColor(image_with_box, cv2.COLOR_BGR2RGB), caption="Detected Face")
            
            # Display aligned and normalized face
            st.markdown("### Preprocessed Face")
            st.image(cv2.cvtColor(normalized_face, cv2.COLOR_BGR2RGB), caption="Aligned and normalized face")
        
        with col2:
            # Step 2: Extract face embeddings
            with st.spinner("Extracting face features..."):
                start_time = time.time()
                
                # Process face image for feature extraction
                face_processed = preprocess_face_for_facenet(normalized_face)
                
                # Get embedding
                embedding = face_embedder.get_embedding(face_processed)
                
                embedding_time = time.time() - start_time
                
                # Log embedding shape for debugging
                st.write(f"Embedding shape: {embedding.shape}")
            
            # Step 3: Classify ethnicity, perform DeepFace analysis
            with st.spinner("Analyzing face..."):
                start_time = time.time()
                
                try:
                    # Classify ethnicity
                    ethnicity, confidence, all_confidences = ethnicity_classifier.predict(embedding)
                    
                    # Perform DeepFace analysis for age, gender, emotion
                    deepface_result = perform_deepface_analysis(normalized_face)
                    
                    classification_time = time.time() - start_time
                    
                    # Display results
                    st.markdown("### Results")
                    
                    # Ethnicity Prediction
                    st.markdown("#### Ethnicity Prediction")
                    st.markdown(f"**Predicted Ethnicity:** {ethnicity}")
                    st.markdown(f"**Confidence:** {confidence:.4f}")
                    
                    # DeepFace Analysis Results
                    if deepface_result:
                        st.markdown("#### Additional Face Analysis")
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Estimated Age", str(deepface_result['age']))
                        
                        with col2:
                            # Handle gender result more robustly
                            gender_result = deepface_result['gender']
                            if isinstance(gender_result, dict):
                                # If it's a probability dictionary, choose the most likely gender
                                gender = max(gender_result, key=gender_result.get)
                            else:
                                # If it's already a string, use it directly
                                gender = str(gender_result)
                            st.metric("Gender", gender)
                        
                        with col3:
                            # Find dominant emotion
                            emotions = deepface_result['emotion']
                            dominant_emotion = max(emotions, key=emotions.get)
                            st.metric("Dominant Emotion", dominant_emotion)
                    
                    # Map numeric class names to ethnic names if needed
                    class_name_mapping = {
                        "0": "Jawa",
                        "1": "Sunda", 
                        "2": "Melayu"
                    }
                    
                    # Create readable confidence display with mapping
                    readable_confidences = {}
                    for class_name, conf in all_confidences.items():
                        readable_name = class_name_mapping.get(class_name, class_name)
                        readable_confidences[readable_name] = conf
                    
                    # Check if using numeric classes and convert for visualization
                    if set(all_confidences.keys()).issubset({"0", "1", "2"}):
                        st.info("Using mapped ethnic names for visualization: 0=Jawa, 1=Sunda, 2=Melayu")
                        all_confidences = readable_confidences
                        ethnicity = class_name_mapping.get(ethnicity, ethnicity)
                    
                    # Timing information
                    # st.markdown("### Processing Times")
                    # st.markdown(f"- Face Detection: {detection_time:.3f} seconds")
                    # st.markdown(f"- Feature Extraction: {embedding_time:.3f} seconds")
                    # st.markdown(f"- Classification: {classification_time:.3f} seconds")
                    # st.markdown(f"- Total Time: {detection_time + embedding_time + classification_time:.3f} seconds")
                    
                    # Visualization of confidence scores
                    st.markdown("### Confidence Scores")
                    
                    # Create columns for confidence bars
                    conf_cols = st.columns(len(all_confidences))
                    
                    # Sort confidences by value (descending)
                    sorted_confidences = sorted(all_confidences.items(), key=lambda x: x[1], reverse=True)
                    
                    # Display confidence bars
                    for i, (eth, conf) in enumerate(sorted_confidences):
                        with conf_cols[i]:
                            st.metric(eth, f"{conf:.4f}")
                            st.progress(float(conf))
                    
                    # Visualization using matplotlib
                    fig = plot_ethnicity_prediction(normalized_face, ethnicity, all_confidences)
                    st.pyplot(fig)
                    
                except Exception as e:
                    st.error(f"Error during analysis: {e}")
                    st.exception(e)  # This will print the full traceback
        
    except Exception as e:
        st.error(f"Error during face processing: {e}")
        st.exception(e) 

def compare_faces(face1_img, face2_img, face_detector, facenet_embedder, face_similarity, detection_threshold=0.5):
    """Compare two face images for similarity"""
    try:
        # Step 1: Detect and align faces
        with st.spinner("Detecting and aligning faces..."):
            # Process first face
            face1_normalized, face1_found, face1_rect, face1_conf = face_detector.detect_and_align_face(
                face1_img, target_size=(224, 224))
            
            # Process second face
            face2_normalized, face2_found, face2_rect, face2_conf = face_detector.detect_and_align_face(
                face2_img, target_size=(224, 224))
            
        # Check if faces were detected
        if not face1_found or (face1_conf and face1_conf < detection_threshold):
            st.warning(f"No clear face detected in the first image (confidence: {face1_conf if face1_conf else 'unknown'}).")
            return None, None, None, None
            
        if not face2_found or (face2_conf and face2_conf < detection_threshold):
            st.warning(f"No clear face detected in the second image (confidence: {face2_conf if face2_conf else 'unknown'}).")
            return None, None, None, None
        
        # Step 2: Extract face embeddings
        with st.spinner("Extracting face features..."):
            # Make sure faces are uint8 before processing
            face1_normalized = np.clip(face1_normalized, 0, 255).astype(np.uint8)
            face2_normalized = np.clip(face2_normalized, 0, 255).astype(np.uint8)
            
            # Get embeddings directly from the normalized faces
            embedding1 = facenet_embedder.get_embedding(face1_normalized)
            embedding2 = facenet_embedder.get_embedding(face2_normalized)
        
        # Step 3: Calculate similarity
        with st.spinner("Calculating face similarity..."):
            # Use the safer computation method
            similarity_score, is_match = face_similarity.safe_compute_similarity(embedding1, embedding2)
        
        # Return results
        return face1_normalized, face2_normalized, similarity_score, is_match
        
    except Exception as e:
        st.error(f"Error during face comparison: {e}")
        st.exception(e)  # This will print the full traceback
        return None, None, None, None

def plot_roc_curve_simple(threshold=0.5):
    """
    Create a simple ROC curve visualization for face similarity
    
    Args:
        threshold: Current similarity threshold
        
    Returns:
        fig: Matplotlib figure
    """
    # Create a simple informative ROC curve plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Sample ROC curve data (this would be real data in a full implementation)
    fpr = [0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    tpr = [0, 0.7, 0.8, 0.87, 0.9, 0.92, 0.94, 0.96, 0.97, 0.98, 0.99, 1.0]
    
    # Calculate approximate AUC
    auc_value = sum([(fpr[i+1] - fpr[i]) * (tpr[i+1] + tpr[i])/2 for i in range(len(fpr)-1)])
    
    # Plot ROC curve
    ax.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC curve (AUC ≈ {auc_value:.2f})')
    
    # Plot diagonal reference line
    ax.plot([0, 1], [0, 1], 'k--', label='Random classifier')
    
    # Find the closest point on the ROC curve to the current threshold
    # This is a simplification - in a real implementation we'd have the actual mapping
    threshold_idx = min(int(threshold * 10), len(fpr) - 2)
    
    # Mark current threshold
    ax.plot(fpr[threshold_idx], tpr[threshold_idx], 'ro', markersize=10, 
            label=f'Current threshold ({threshold:.2f})')
    
    # Axis labels and title
    ax.set_xlabel('False Positive Rate (Different people incorrectly matched)')
    ax.set_ylabel('True Positive Rate (Same person correctly matched)')
    ax.set_title('Receiver Operating Characteristic (ROC) Curve')
    ax.legend(loc='lower right')
    
    # Set axis limits
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Add explanatory text
    ax.text(0.5, 0.1, "Note: This is an illustrative ROC curve.\nIn a production system, this would be based on a \nvalidation dataset of known face pairs.",
            bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    return fig

def plot_similarity_distribution(threshold=0.5):
    """
    Create a simple similarity distribution visualization
    
    Args:
        threshold: Current similarity threshold
        
    Returns:
        fig: Matplotlib figure
    """
    # Create a simple distribution plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Generate sample data
    # These would be real measurements in a full implementation
    np.random.seed(42)  # For reproducibility
    same_person = np.random.beta(8, 2, size=100)  # Right-skewed distribution centered around 0.8
    different_people = np.random.beta(2, 8, size=100)  # Left-skewed distribution centered around 0.2
    
    # Plot histograms
    bins = np.linspace(0, 1, 30)
    ax.hist(same_person, bins=bins, alpha=0.5, label='Same Person', color='green')
    ax.hist(different_people, bins=bins, alpha=0.5, label='Different People', color='red')
    
    # Add threshold line
    ax.axvline(x=threshold, color='blue', linestyle='--', label=f'Threshold ({threshold:.2f})')
    
    # Add labels and legend
    ax.set_xlabel('Similarity Score')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Face Similarity Scores')
    ax.legend()
    
    # Add explanatory text
    ax.text(0.5, 0.7 * ax.get_ylim()[1], 
            "Note: This is an illustrative distribution.\nIn a production system, this would be based on a \nvalidation dataset of known face pairs.",
            bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    return fig

def plot_example_matches():
    """
    Create visualization of example face match results
    
    Returns:
        fig: Matplotlib figure
    """
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Generate placeholders for the match examples
    # In a real implementation, these would be actual face images
    
    # Create a placeholder function to generate example faces with text
    def make_example_face(text, match_type, score):
        img = np.ones((224, 224, 3), dtype=np.uint8) * 240  # Light gray background
        
        # Add text
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, text, (50, 100), font, 1, (0, 0, 0), 2)
        cv2.putText(img, match_type, (50, 150), font, 0.7, (0, 0, 0), 1)
        cv2.putText(img, f"Score: {score:.2f}", (50, 180), font, 0.7, (0, 0, 0), 1)
        
        return img
    
    # Types of match examples
    examples = [
        ("True Positive", "Same person correctly matched", 0.82),
        ("False Positive", "Different people incorrectly matched", 0.65),
        ("True Negative", "Different people correctly not matched", 0.35),
        ("False Negative", "Same person incorrectly not matched", 0.45)
    ]
    
    # Create and display examples
    for i, (match_type, description, score) in enumerate(examples):
        row, col = i // 2, i % 2
        
        # Create two example faces
        face1 = make_example_face("Example Face 1", match_type, score)
        face2 = make_example_face("Example Face 2", description, score)
        
        # Convert to RGB
        face1_rgb = cv2.cvtColor(face1, cv2.COLOR_BGR2RGB)
        face2_rgb = cv2.cvtColor(face2, cv2.COLOR_BGR2RGB)
        
        # Display side by side
        composite = np.hstack((face1_rgb, face2_rgb))
        axes[row, col].imshow(composite)
        axes[row, col].set_title(match_type)
        axes[row, col].axis('off')
    
    plt.tight_layout()
    return fig

def main():
    try:
        # Set app title
        st.title("Face Recognition System")
        
        # Load models with error handling
        try:
            with st.spinner("Loading face detector (MTCNN)..."):
                face_detector = load_face_detector()
                
            with st.spinner("Loading face embedder..."):
                face_embedder = load_face_embedder()
                
            with st.spinner("Loading FaceNet embedder..."):
                facenet_embedder = load_facenet_embedder()
                
            with st.spinner("Loading ethnicity classifier..."):
                ethnicity_classifier = load_ethnicity_classifier()
                
            with st.spinner("Loading face similarity module..."):
                face_similarity = load_face_similarity()
                
            # Check if we're running in demo mode
            if not os.path.exists(MODEL_PATH):
                st.warning("""
                ⚠️ Running in DEMO MODE - KNN Model not found
                
                The KNN model was not found at the expected location:
                - KNN model should be at: `models/facenet_knn_model.pkl`
                
                The app is running with a fallback model that may generate random predictions.
                For accurate results, please place the correct model in the expected location.
                """)
        except Exception as e:
            st.error(f"Error loading models: {e}")
            st.exception(e)  # Show the full exception for debugging
            st.stop()
            
        # Sidebar
        st.sidebar.title("Menu")
        
        # Choose app function
        app_mode = st.sidebar.selectbox("Choose the app mode", ["About", "Face Analysis", "Face Similarity"])
        
        # About section
        if app_mode == "About":
            st.markdown("""
            ## Fitur dan Algoritma

            ### Main (Utama)

            #### 1. Face Similarity
            Memungkinkan sistem untuk mengidentifikasi dan membandingkan wajah untuk menentukan apakah dua gambar wajah yang berbeda berasal dari orang yang sama.

            **Teknologi:**
            - FaceNet Pre-train Models

            #### 2. Deteksi Suku/Etnis
            Mengklasifikasikan wajah seseorang ke dalam kategori suku/etnis berdasarkan fitur wajah menggunakan teknik computer vision.

            **Teknologi:**
            - MTCNN (deteksi wajah)
            - FaceNet Pre-train Models
            - Classifiers (KNN dan SVM)

            ### Fitur Pendamping

            #### 1. Gender Detection
            Memungkinkan sistem untuk mengidentifikasi jenis kelamin dari seseorang berdasarkan citra wajahnya. Sistem menganalisis struktur wajah untuk mengklasifikasikan gender secara real-time, lengkap dengan tingkat kepercayaan (confidence score).

            **Teknologi:**
            - Convolutional Neural Network (CNN) dengan pre-train model DeepFace

            #### 2. Age Estimation
            Memperkirakan usia seseorang berdasarkan karakteristik visual dari wajahnya. Sistem tidak hanya melihat ciri-ciri yang mencolok seperti kerutan atau tekstur kulit, tetapi juga memanfaatkan fitur-fitur halus yang diperoleh dari lapisan-lapisan model deep learning untuk menghasilkan estimasi usia yang lebih akurat.

            **Teknologi:**
            - Convolutional Neural Network (CNN) dengan pre-train model DeepFace

            ## Penggunaan

            - Unggah gambar atau gunakan webcam
            - Bandingkan wajah untuk kemiripan atau klasifikasi etnis
            - Lihat visualisasi detail hasil

            ## Pemetaan Kelas Etnis

            - **0**: Jawa
            - **1**: Sunda
            - **2**: Melayu

            PCD 2025 - PraTubes
            """)
        
        # Ethnicity Detection section
        elif app_mode == "Face Analysis":
            st.header("Comprehensive Face Analysis")
            
            # Display MTCNN parameters in sidebar
            st.sidebar.markdown("## Detection Parameters")
            
            min_face_size = st.sidebar.slider(
                "Minimum Face Size", 
                min_value=20, 
                max_value=100, 
                value=40,
                help="Minimum size of faces to detect (in pixels)"
            )
            
            detection_threshold = st.sidebar.slider(
                "Detection Confidence Threshold", 
                min_value=0.0, 
                max_value=1.0, 
                value=0.5,
                help="Minimum confidence required for face detection"
            )
            
            # Choose input method
            input_method = st.radio("Choose input method:", ["Upload Image", "Use Webcam"])
            
            # Process based on input method
            if input_method == "Upload Image":
                # Upload image
                st.markdown("### Upload Face Image")
                uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
                
                # Process image if uploaded
                if uploaded_file is not None:
                    # Load image
                    image = load_image_from_upload(uploaded_file)
                    
                    # Display original image
                    st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="Original Image")
                    
                    # Process button
                    if st.button("Analyze Face"):
                        process_image(image, face_detector, face_embedder, ethnicity_classifier, detection_threshold)
                        
            else:  # Webcam option
                st.markdown("### Capture from Webcam")
                
                # Capture button
                picture = st.camera_input("Take a picture")
                
                if picture:
                    # Convert to OpenCV format
                    image = load_image_from_upload(picture)
                    
                    # Process the captured image
                    process_image(image, face_detector, face_embedder, ethnicity_classifier, detection_threshold)
        
        # Face Similarity section
        elif app_mode == "Face Similarity":
            st.header("Face Similarity Comparison")
            
            # Display parameters in sidebar
            st.sidebar.markdown("## Similarity Parameters")
            
            detection_threshold = st.sidebar.slider(
                "Face Detection Threshold", 
                min_value=0.0, 
                max_value=1.0, 
                value=0.5,
                help="Minimum confidence required for face detection"
            )
            
            similarity_threshold = st.sidebar.slider(
                "Similarity Threshold", 
                min_value=0.0, 
                max_value=1.0, 
                value=face_similarity.threshold,
                help="Threshold for determining face match"
            )
            
            distance_metric = st.sidebar.selectbox(
                "Distance Metric",
                ["cosine", "euclidean"],
                help="Method to calculate similarity between face embeddings"
            )
            
            # Update similarity settings
            face_similarity.threshold = similarity_threshold
            face_similarity.distance_metric = distance_metric
            
            # Choose operation mode
            operation_mode = st.radio(
                "Choose operation mode:", 
                ["Compare Two Faces", "View Similarity Information"]
            )
            
            if operation_mode == "Compare Two Faces":
                st.markdown("### Upload Face Images for Comparison")
                
                # Create two columns for face uploads
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### Face 1")
                    uploaded_file1 = st.file_uploader("Choose first face image", type=["jpg", "jpeg", "png"])
                    
                    # Show placeholder or uploaded image
                    if uploaded_file1 is not None:
                        # Load image
                        face1_img = load_image_from_upload(uploaded_file1)
                        
                        # Display image
                        st.image(cv2.cvtColor(face1_img, cv2.COLOR_BGR2RGB), caption="Face 1")
                    
                with col2:
                    st.markdown("#### Face 2")
                    uploaded_file2 = st.file_uploader("Choose second face image", type=["jpg", "jpeg", "png"])
                    
                    # Show placeholder or uploaded image
                    if uploaded_file2 is not None:
                        # Load image
                        face2_img = load_image_from_upload(uploaded_file2)
                        
                        # Display image
                        st.image(cv2.cvtColor(face2_img, cv2.COLOR_BGR2RGB), caption="Face 2")
                
                # Compare button
                if uploaded_file1 is not None and uploaded_file2 is not None:
                    if st.button("Compare Faces"):
                        # Load images
                        face1_img = load_image_from_upload(uploaded_file1)
                        face2_img = load_image_from_upload(uploaded_file2)
                        
                        # Compare faces
                        face1_norm, face2_norm, similarity_score, is_match = compare_faces(
                            face1_img, face2_img, face_detector, facenet_embedder, face_similarity, detection_threshold)
                        
                        if face1_norm is not None and face2_norm is not None:
                            # Display comparison results
                            st.markdown("### Face Comparison Results")
                            
                            # Create visualization
                            fig = plot_face_comparison(
                                face1_norm, face2_norm, similarity_score, is_match, face_similarity.threshold)
                            st.pyplot(fig)
                            
                            # Detailed metrics
                            st.markdown("### Detailed Metrics")
                            st.markdown(f"**Similarity Score:** {similarity_score:.4f}")
                            st.markdown(f"**Threshold:** {face_similarity.threshold:.4f}")
                            st.markdown(f"**Match Decision:** {'MATCH' if is_match else 'NO MATCH'}")
                            st.markdown(f"**Distance Metric:** {face_similarity.distance_metric}")
                        else:
                            st.error("Face comparison failed. Please try with different images.")
                            
            elif operation_mode == "View Similarity Information":
                st.markdown("### Face Similarity Visualizations")
                
                # Add explanatory text about face similarity
                st.markdown("""
                Face similarity comparison works by measuring the distance between face embeddings.
                A lower distance (higher similarity) indicates a higher likelihood that the faces belong to the same person.
                
                The key components of face similarity are:
                
                1. **Face Detection and Alignment** - Properly locating and aligning faces for consistent comparison
                2. **Feature Extraction** - Converting face images to numerical embeddings (feature vectors)
                3. **Similarity Calculation** - Measuring distance between embeddings using metrics like cosine similarity
                4. **Threshold Decision** - Determining if the similarity score is high enough to consider a match
                
                The visualizations below show how similarity scores are distributed and how the threshold affects 
                matching accuracy.
                """)
                
                # Show distribution of similarity scores
                st.subheader("Distribution of Similarity Scores")
                fig_dist = plot_similarity_distribution(similarity_threshold)
                st.pyplot(fig_dist)
                
                # Show ROC curve
                st.subheader("ROC Curve for Face Matching")
                fig_roc = plot_roc_curve_simple(similarity_threshold)
                st.pyplot(fig_roc)
                
                # Show example match results
                st.subheader("Example Match Results")
                st.markdown("""
                These visualizations show examples of how face matching works in practice:
                
                - **True Positive**: Same person correctly identified as a match
                - **False Positive**: Different people incorrectly identified as a match
                - **True Negative**: Different people correctly identified as not a match
                - **False Negative**: Same person incorrectly identified as not a match
                
                The choice of similarity threshold affects the balance between these outcomes.
                """)
                
                fig_examples = plot_example_matches()
                st.pyplot(fig_examples)
                
                # Show impact of threshold
                st.subheader("Impact of Similarity Threshold")
                st.markdown(f"""
                Current threshold: **{similarity_threshold:.2f}**
                
                - **Lower threshold** (e.g., 0.3): More matches will be found, but more false positives
                - **Higher threshold** (e.g., 0.8): Fewer matches, but higher confidence in each match
                
                Adjust the threshold in the sidebar to balance between:
                - **Security** (fewer false positives)
                - **Convenience** (fewer false negatives)
                
                The optimal threshold depends on your specific use case and security requirements.
                """)
    
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        st.exception(e)  # This will print the full traceback
        st.info("Please check your installation and try again. You may need to install the required dependencies.")

# Run the app
if __name__ == "__main__":
    main()