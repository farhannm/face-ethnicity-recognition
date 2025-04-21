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
from utils.deepface_similarity import DeepFaceSimilarity
from utils.similarity_visualization import plot_face_comparison, create_face_pair_grid
from utils.facenet_embedder import FaceNetEmbedder
from utils.performance_visualization import (
    plot_tar_far_frr,
    plot_roc_with_metrics,
    plot_precision_recall_f1,
    create_performance_dashboard,
    plot_performance_summary
)
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
def load_deepface_similarity():
    """Load DeepFace similarity with caching"""
    return DeepFaceSimilarity(
        threshold=0.6, 
        model_name='VGG-Face',  # Options: 'VGG-Face', 'Facenet', 'Facenet512', 'OpenFace', 'DeepFace', 'DeepID', 'ArcFace'
        distance_metric='cosine'  # Options: 'cosine', 'euclidean', 'euclidean_l2'
    )

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

def compare_faces_deepface(face1_img, face2_img, face_detector, deepface_similarity, detection_threshold=0.5):
    """Compare two face images for similarity using DeepFace"""
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
        
        # Step 2: Calculate similarity using DeepFace
        with st.spinner("Calculating face similarity..."):
            # Make sure faces are uint8 before processing
            face1_normalized = np.clip(face1_normalized, 0, 255).astype(np.uint8)
            face2_normalized = np.clip(face2_normalized, 0, 255).astype(np.uint8)
            
            # Compute similarity directly from the normalized face images
            similarity_score, is_match = deepface_similarity.compute_similarity(face1_normalized, face2_normalized)
        
        # Return results
        return face1_normalized, face2_normalized, similarity_score, is_match
        
    except Exception as e:
        st.error(f"Error during face comparison: {e}")
        st.exception(e)  # This will print the full traceback
        return None, None, None, None

def generate_demo_similarity_data(num_pairs=100):
    """
    Generate demo data for performance metric visualizations
    
    Args:
        num_pairs: Number of pairs to generate for each category
        
    Returns:
        scores_same: Numpy array of similarity scores for same identity pairs
        scores_diff: Numpy array of similarity scores for different identity pairs
    """
    # Create synthetic data that approximates realistic face similarity distributions
    np.random.seed(42)  # For reproducibility
    
    # Same identity pairs typically have high similarity (0.7-1.0)
    scores_same = np.random.beta(8, 2, size=num_pairs)
    
    # Different identity pairs typically have low similarity (0.0-0.5)
    scores_diff = np.random.beta(2, 8, size=num_pairs)
    
    return scores_same, scores_diff

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
                
            with st.spinner("Loading DeepFace similarity module..."):
                deepface_similarity = load_deepface_similarity()
                
            # Check if we're running in demo mode
            if not os.path.exists(MODEL_PATH):
                st.warning("""
                ⚠️ Running in DEMO MODE - SVM Model not found
                
                The SVM model was not found at the expected location:
                - SVM model should be at: `models/facenet_svm_model_v1.pkl`
                
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
            - DeepFace Pre-train Models (VGG-Face, FaceNet, ArcFace)

            #### 2. Deteksi Suku/Etnis
            Mengklasifikasikan wajah seseorang ke dalam kategori suku/etnis berdasarkan fitur wajah menggunakan teknik computer vision.

            **Teknologi:**
            - MTCNN (deteksi wajah)
            - FaceNet Pre-train Models
            - Classifiers (SVM)

            ### Fitur Pendamping

            #### 1. Gender Detection
            Memungkinkan sistem untuk mengidentifikasi jenis kelamin dari seseorang berdasarkan citra wajahnya. Sistem menganalisis struktur wajah untuk mengklasifikasikan gender secara real-time, lengkap dengan tingkat kepercayaan (confidence score).

            **Teknologi:**
            - Convolutional Neural Network (CNN) dengan pre-train model DeepFace

            #### 2. Age Estimation
            Memperkirakan usia seseorang berdasarkan karakteristik visual dari wajahnya. Sistem tidak hanya melihat ciri-ciri yang mencolok seperti kerutan atau tekstur kulit, tetapi juga memanfaatkan fitur-fitur halus yang diperoleh dari lapisan-lapisan model deep learning untuk menghasilkan estimasi usia yang lebih akurat.

            **Teknologi:**
            - Convolutional Neural Network (CNN) dengan pre-train model DeepFace

            ## Performance Metrics
            
            Sistem ini sekarang menampilkan metrik performa berikut untuk evaluasi face similarity:
            
            - **True Acceptance Rate (TAR)**: Persentase wajah yang sama yang teridentifikasi benar sebagai sama
            - **False Acceptance Rate (FAR)**: Persentase wajah berbeda yang teridentifikasi salah sebagai sama
            - **False Rejection Rate (FRR)**: Persentase wajah sama yang teridentifikasi salah sebagai berbeda
            - **Equal Error Rate (EER)**: Titik dimana FAR dan FRR memiliki nilai yang sama
            - **Area Under Curve (AUC)**: Ukuran keseluruhan performa dari sistem
            - **Precision, Recall, F1-Score**: Metrik evaluasi model klasifikasi

            ## Penggunaan

            - Unggah gambar atau gunakan webcam
            - Bandingkan wajah untuk kemiripan atau klasifikasi etnis
            - Lihat visualisasi detail hasil dan metrik performa

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
                value=deepface_similarity.threshold,
                help="Threshold for determining face match"
            )
            
            model_name = st.sidebar.selectbox(
                "DeepFace Model",
                ["VGG-Face", "Facenet", "Facenet512", "ArcFace", "OpenFace", "DeepFace", "DeepID"],
                index=0,
                help="Deep learning model to use for face embedding extraction"
            )
            
            distance_metric = st.sidebar.selectbox(
                "Distance Metric",
                ["cosine", "euclidean", "euclidean_l2"],
                index=0,
                help="Method to calculate similarity between face embeddings"
            )
            
            # Update similarity settings
            deepface_similarity.threshold = similarity_threshold
            deepface_similarity.model_name = model_name
            deepface_similarity.distance_metric = distance_metric
            
            # Choose operation mode
            operation_mode = st.radio(
                "Choose operation mode:", 
                ["Compare Two Faces", "View Performance Metrics"]
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
                        
                        # Compare faces using DeepFace
                        face1_norm, face2_norm, similarity_score, is_match = compare_faces_deepface(
                            face1_img, face2_img, face_detector, deepface_similarity, detection_threshold)
                        
                        if face1_norm is not None and face2_norm is not None:
                            # Display comparison results
                            st.markdown("### Face Comparison Results")
                            
                            # Create visualization
                            fig = plot_face_comparison(
                                face1_norm, face2_norm, similarity_score, is_match, deepface_similarity.threshold)
                            st.pyplot(fig)
                            
                            # Detailed metrics
                            st.markdown("### Detailed Metrics")
                            st.markdown(f"**Similarity Score:** {similarity_score:.4f}")
                            st.markdown(f"**Threshold:** {deepface_similarity.threshold:.4f}")
                            st.markdown(f"**Match Decision:** {'MATCH' if is_match else 'NO MATCH'}")
                            st.markdown(f"**Model Used:** {deepface_similarity.model_name}")
                            st.markdown(f"**Distance Metric:** {deepface_similarity.distance_metric}")
                        else:
                            st.error("Face comparison failed. Please try with different images.")
                            
            elif operation_mode == "View Performance Metrics":
                st.markdown("### Face Recognition Performance Metrics")
                
                # Add explanatory text about performance metrics
                st.markdown("""
                Face recognition systems need to balance between two types of errors:
                
                1. **False Accepts (Type I Error)**: When the system incorrectly matches two different individuals.
                2. **False Rejects (Type II Error)**: When the system fails to match two images of the same person.
                
                The following visualizations show key performance metrics that help evaluate the system and 
                choose the optimal threshold.
                """)
                
                # Get the demo similarity scores
                scores_same, scores_diff = generate_demo_similarity_data(num_pairs=200)
                
                # Create dashboard using performance_visualization module
                performance_figs, metrics = create_performance_dashboard(
                    scores_same, scores_diff, current_threshold=similarity_threshold)
                
                # Display key metrics at current threshold
                st.subheader("Current Performance at Threshold = {:.2f}".format(similarity_threshold))
                
                # Create metrics display
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("True Acceptance Rate (TAR)", "{:.2f}%".format(metrics['tar_current'] * 100))
                    st.metric("False Acceptance Rate (FAR)", "{:.2f}%".format(metrics['far_current'] * 100))
                with col2:
                    st.metric("False Rejection Rate (FRR)", "{:.2f}%".format(metrics['frr_current'] * 100))
                    st.metric("Equal Error Rate (EER)", "{:.2f}%".format(metrics['eer'] * 100))
                with col3:
                    st.metric("Precision", "{:.2f}%".format(metrics['precision_current'] * 100))
                    st.metric("F1-Score", "{:.2f}".format(metrics['f1_current']))
                
                # ROC curve with AUC
                st.subheader("ROC Curve")
                st.markdown("""
                The Receiver Operating Characteristic (ROC) curve shows the trade-off between True Positive Rate (TPR) 
                and False Positive Rate (FPR) at different threshold values. The Area Under Curve (AUC) is a measure of 
                the model's ability to discriminate between positive and negative classes.
                
                **Higher AUC value (closer to 1.0) indicates better performance.**
                """)
                st.pyplot(performance_figs[1])
                
                # TAR, FAR, FRR vs Threshold
                st.subheader("TAR, FAR, and FRR vs. Threshold")
                st.markdown("""
                This graph shows how the True Acceptance Rate (TAR), False Acceptance Rate (FAR), and 
                False Rejection Rate (FRR) change with different threshold values.
                
                **The Equal Error Rate (EER) point is where FAR equals FRR**. Lower EER indicates better performance.
                """)
                st.pyplot(performance_figs[0])
                
                # Precision, Recall, F1-Score
                st.subheader("Precision, Recall, and F1-Score vs. Threshold")
                st.markdown("""
                - **Precision**: Proportion of positive identifications that were actually correct.
                - **Recall (TAR)**: Proportion of actual positives that were correctly identified.
                - **F1-Score**: Harmonic mean of precision and recall, providing a balance between the two.
                
                The optimal threshold often maximizes the F1-Score, balancing precision and recall.
                """)
                st.pyplot(performance_figs[2])
                
                # Summary of optimal thresholds
                st.subheader("Optimal Thresholds")
                st.markdown(f"""
                Based on the performance analysis, the following thresholds are recommended:
                
                - **Current Threshold**: {metrics['current_threshold']:.3f}
                - **Equal Error Rate (EER) Threshold**: {metrics['eer_threshold']:.3f} (EER = {metrics['eer']:.3f})
                - **Optimal F1-Score Threshold**: {metrics['best_f1_threshold']:.3f} (F1 = {metrics['best_f1']:.3f})
                
                The choice of threshold depends on your specific security requirements:
                - For higher security (fewer false accepts), use a higher threshold
                - For better user experience (fewer false rejects), use a lower threshold
                """)
                
                # Summary of model configuration
                st.subheader("Current Model Configuration")
                st.markdown(f"""
                - **Deep Learning Model**: {deepface_similarity.model_name}
                - **Distance Metric**: {deepface_similarity.distance_metric}
                - **Similarity Threshold**: {deepface_similarity.threshold:.3f}
                """)
                
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        st.exception(e)  # This will print the full traceback
        st.info("Please check your installation and try again. You may need to install the required dependencies.")

# Run the app
if __name__ == "__main__":
    main()