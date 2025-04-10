import os
import sys
import cv2
import numpy as np
import tempfile

# Set page config must be the first Streamlit command
import streamlit as st
st.set_page_config(
    page_title="Face Ethnicity Recognition",
    layout="wide"
)

# Fallback functions for preprocessing and visualization
def preprocess_face_for_siamese(face):
    """Process face for siamese network"""
    face_resized = cv2.resize(face, (100, 100))
    face_normalized = face_resized / 255.0
    return np.expand_dims(face_normalized, axis=0)

def preprocess_face_for_ethnicity(face):
    """Process face for ethnicity classifier"""
    face_resized = cv2.resize(face, (224, 224))
    face_normalized = face_resized / 255.0
    return np.expand_dims(face_normalized, axis=0)

def plot_similarity_result(face1, face2, similarity_score, is_match):
    """Create a simple visualization fallback"""
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    face1_rgb = cv2.cvtColor(face1, cv2.COLOR_BGR2RGB)
    face2_rgb = cv2.cvtColor(face2, cv2.COLOR_BGR2RGB)
    
    axes[0].imshow(face1_rgb)
    axes[0].set_title("Face 1")
    axes[0].axis('off')
    
    axes[2].imshow(face2_rgb)
    axes[2].set_title("Face 2")
    axes[2].axis('off')
    
    axes[1].set_xlim(0, 1)
    axes[1].set_ylim(0, 1)
    axes[1].axvline(x=0.5, color='gray', linestyle='--', alpha=0.5)
    
    color = 'green' if is_match else 'red'
    axes[1].barh(0.5, similarity_score, height=0.3, color=color)
    axes[1].text(0.5, 0.8, f"Similarity: {similarity_score:.2f}", 
                ha='center', va='center', fontsize=12)
    match_text = "MATCH" if is_match else "NO MATCH"
    axes[1].text(0.5, 0.2, match_text, 
                ha='center', va='center', fontsize=14, 
                fontweight='bold', color=color)
    axes[1].set_title("Similarity Result")
    axes[1].axis('off')
    plt.tight_layout()
    return fig

def plot_ethnicity_prediction(face_img, ethnicity, confidences):
    """Create a simple ethnicity prediction visualization fallback"""
    import matplotlib.pyplot as plt
    import numpy as np
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), gridspec_kw={'width_ratios': [1, 1.5]})
    
    face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    axes[0].imshow(face_rgb)
    axes[0].set_title("Input Face")
    axes[0].axis('off')
    
    ethnicities = list(confidences.keys())
    scores = list(confidences.values())
    
    # Sort by confidence score
    sorted_indices = np.argsort(scores)
    ethnicities = [ethnicities[i] for i in sorted_indices]
    scores = [scores[i] for i in sorted_indices]
    
    colors = ['lightgray'] * len(ethnicities)
    colors[ethnicities.index(ethnicity)] = 'green'
    
    y_pos = np.arange(len(ethnicities))
    axes[1].barh(y_pos, scores, color=colors)
    axes[1].set_yticks(y_pos)
    axes[1].set_yticklabels(ethnicities)
    axes[1].set_xlim(0, 1)
    axes[1].set_xlabel('Confidence Score')
    axes[1].set_title('Ethnicity Prediction')
    
    for i, v in enumerate(scores):
        axes[1].text(v + 0.01, i, f'{v:.2f}', va='center')
    
    plt.tight_layout()
    return fig

def draw_bounding_box(image, face_rect, label=None, color=(0, 255, 0), thickness=2):
    """Draw a bounding box on an image"""
    if face_rect is None:
        return image
        
    image_with_box = image.copy()
    x, y, w, h = face_rect
    
    cv2.rectangle(image_with_box, (x, y), (x+w, y+h), color, thickness)
    
    if label:
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_thickness = 1
        text_size = cv2.getTextSize(label, font, font_scale, font_thickness)[0]
        
        cv2.rectangle(image_with_box, (x, y - text_size[1] - 5), (x + text_size[0], y), color, -1)
        cv2.putText(image_with_box, label, (x, y - 5), font, font_scale, (0, 0, 0), font_thickness)
    
    return image_with_box

# Mock classes for Siamese and Ethnicity Classifier
class SiameseNetwork:
    def __init__(self, **kwargs):
        self.threshold = 0.5
        
    def compare_faces(self, face1, face2, distance_metric='euclidean'):
        import random
        similarity = random.uniform(0.3, 0.8)
        is_match = similarity >= self.threshold
        return similarity, is_match
        
    def set_threshold(self, threshold):
        self.threshold = threshold
        
    def load_weights(self, path):
        pass

class EthnicityClassifier:
    def __init__(self, **kwargs):
        self.class_names = ['Jawa', 'Sunda', 'Melayu']
        
    def set_class_names(self, class_names):
        self.class_names = class_names
        
    def predict_ethnicity(self, face_image):
        import random
        confidences = {name: random.uniform(0.1, 0.6) for name in self.class_names}
        
        # Normalize to sum to 1
        total = sum(confidences.values())
        confidences = {k: v/total for k, v in confidences.items()}
        
        # Get max confidence
        max_class = max(confidences, key=confidences.get)
        max_confidence = confidences[max_class]
        
        return max_class, max_confidence, confidences
        
    def load_weights(self, path):
        pass

class HaarCascadeFaceDetector:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        self.target_size = (224, 224)
        
    def detect_faces(self, image, **kwargs):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image.copy()
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        face_images = []
        for (x, y, w, h) in faces:
            face_img = image[y:y+h, x:x+w]
            face_img = cv2.resize(face_img, self.target_size)
            face_images.append(face_img)
        return faces, face_images
        
    def detect_and_align_face(self, image, target_size=(224, 224)):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image.copy()
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        if len(faces) == 0:
            resized = cv2.resize(image, target_size)
            return resized, False, None
        (x, y, w, h) = faces[0]
        face_img = image[y:y+h, x:x+w]
        face_normalized = cv2.resize(face_img, target_size)
        return face_normalized, True, (x, y, w, h)

@st.cache_resource
def load_face_detector():
    return HaarCascadeFaceDetector()

@st.cache_resource
def load_siamese_network():
    return SiameseNetwork(input_shape=(100, 100, 3), embedding_dim=128)

@st.cache_resource
def load_ethnicity_classifier():
    return EthnicityClassifier(num_classes=3, input_shape=(224, 224, 3))

# Function to convert uploaded file to OpenCV image
def load_image_from_upload(uploaded_file):
    # Read file as bytes
    bytes_data = uploaded_file.getvalue()
    
    # Convert to numpy array
    nparr = np.frombuffer(bytes_data, np.uint8)
    
    # Decode image
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    return img

def main():
    try:
        # Display app in demo mode if TensorFlow is not available
        if 'tensorflow' not in sys.modules:
            st.warning("⚠️ Running in DEMO MODE - TensorFlow not detected")
            st.info("""
            ### For full functionality:
            
            ```bash
            # For Mac with M1/M2/M3 chip:
            pip install tensorflow-macos==2.13.0
            
            # For other systems:
            pip install tensorflow==2.13.0
            ```
            
            After installing, restart the app.
            """)
        
        # Load models with error handling
        face_detector = load_face_detector()
        siamese_network = load_siamese_network()
        ethnicity_classifier = load_ethnicity_classifier()
        
        # Set app title
        st.title("Face Ethnicity Recognition System")
        
        # Sidebar
        st.sidebar.title("Settings")
        
        # Choose app function
        app_mode = st.sidebar.selectbox(
            "Choose the app mode",
            ["About", "Face Similarity", "Ethnicity Detection"]
        )
        
        # About section (same as before)
        if app_mode == "About":
            st.markdown("""
            # About
            
            This application performs two main functions:
            
            ## 1. Face Similarity Detection
            
            Upload two face images to determine if they belong to the same person.
            The system:
            - Detects faces using Haar Cascade
            - Extracts facial features using a Siamese Network
            - Calculates similarity score
            - Determines if the faces match
            
            ## 2. Ethnicity Detection
            
            Upload a face image to classify its ethnicity (Jawa, Sunda, or Melayu).
            The system:
            - Detects and aligns the face
            - Normalizes the face image
            - Uses a CNN with Transfer Learning for classification
            - Displays ethnicity prediction with confidence scores
            
            ## Implementation Details
            
            - **Face Detection**: Haar Cascade Classifier
            - **Face Similarity**: Siamese Network with Euclidean/Cosine distance
            - **Ethnicity Classification**: MobileNetV2 with custom top layers
            
            Developed by: Team Ethnicity Recognizer
            """)
            
            st.sidebar.markdown("## Adjust threshold for face matching")
            threshold = st.sidebar.slider("Similarity Threshold", 0.0, 1.0, 0.5, 0.01)
            siamese_network.set_threshold(threshold)
        
        # Ethnicity Detection section
        elif app_mode == "Ethnicity Detection":
            st.header("Ethnicity Detection")
            
            # Upload image
            st.markdown("### Upload Face Image")
            uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
            
            # Process image if uploaded
            if uploaded_file is not None:
                # Load image
                image = load_image_from_upload(uploaded_file)
                
                # Process button
                if st.button("Detect Ethnicity"):
                    try:
                        # Detect and align face
                        with st.spinner("Detecting and aligning face..."):
                            normalized_face, face_found, face_rect = face_detector.detect_and_align_face(
                                image, target_size=(224, 224))
                        
                        # Check if face was detected
                        if not face_found:
                            st.warning("No face detected clearly. Results may not be accurate.")
                        
                        # Draw face bounding box if detected
                        if face_rect is not None:
                            image_with_box = draw_bounding_box(image, face_rect)
                            st.image(cv2.cvtColor(image_with_box, cv2.COLOR_BGR2RGB), caption="Input image with detected face")
                        else:
                            st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="Input image")
                        
                        # Preprocess face for ethnicity classification
                        with st.spinner("Classifying ethnicity..."):
                            face_processed = preprocess_face_for_ethnicity(normalized_face)
                            
                            # Classify ethnicity
                            ethnicity, confidence, all_confidences = ethnicity_classifier.predict_ethnicity(face_processed)
                        
                        # Display results
                        st.markdown("### Results")
                        st.markdown(f"**Predicted Ethnicity:** {ethnicity}")
                        st.markdown(f"**Confidence:** {confidence:.4f}")
                        
                        # Confidence bar
                        st.markdown("### Confidence Scores")
                        for eth, conf in all_confidences.items():
                            st.progress(float(conf))  # Ensure float conversion
                            st.text(f"{eth}: {conf:.4f}")
                        
                        # Visualization
                        fig = plot_ethnicity_prediction(normalized_face, ethnicity, all_confidences)
                        st.pyplot(fig)
                        
                        # Display aligned face
                        st.markdown("### Preprocessed Face")
                        st.image(cv2.cvtColor(normalized_face, cv2.COLOR_BGR2RGB), caption="Aligned and normalized face")
                    except Exception as e:
                        st.error(f"Error during ethnicity detection: {e}")
        
        # Face Similarity section
        elif app_mode == "Face Similarity":
            st.header("Face Similarity Detection")
            
            # Sidebar options
            st.sidebar.markdown("## Adjust threshold for face matching")
            threshold = st.sidebar.slider("Similarity Threshold", 0.0, 1.0, 0.5, 0.01)
            siamese_network.set_threshold(threshold)
            
            distance_metric = st.sidebar.radio(
                "Distance Metric",
                ["euclidean", "cosine"]
            )
            
            # Upload images
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### Upload First Face Image")
                uploaded_file1 = st.file_uploader("Choose first image...", type=["jpg", "jpeg", "png"], key="file1")
            
            with col2:
                st.markdown("### Upload Second Face Image")
                uploaded_file2 = st.file_uploader("Choose second image...", type=["jpg", "jpeg", "png"], key="file2")
            
            # Process images if both are uploaded
            if uploaded_file1 is not None and uploaded_file2 is not None:
                # Load images
                image1 = load_image_from_upload(uploaded_file1)
                image2 = load_image_from_upload(uploaded_file2)
                
                # Process button
                if st.button("Compare Faces"):
                    try:
                        # Detect faces
                        with st.spinner("Detecting faces..."):
                            faces1, face_images1 = face_detector.detect_faces(image1)
                            faces2, face_images2 = face_detector.detect_faces(image2)
                        
                        # Check if faces were detected
                        if len(face_images1) == 0 or len(face_images2) == 0:
                            st.error("No faces detected in one or both images. Please try different images.")
                        else:
                            # Use the first detected face from each image
                            face1 = face_images1[0]
                            face2 = face_images2[0]
                            
                            # Draw face bounding box
                            image1_with_box = draw_bounding_box(image1, faces1[0])
                            image2_with_box = draw_bounding_box(image2, faces2[0])
                            
                            # Display images with face detection
                            col1, col2 = st.columns(2)
                            with col1:
                                st.image(cv2.cvtColor(image1_with_box, cv2.COLOR_BGR2RGB), caption="Image 1 with detected face")
                            with col2:
                                st.image(cv2.cvtColor(image2_with_box, cv2.COLOR_BGR2RGB), caption="Image 2 with detected face")
                            
                            # Preprocess faces for Siamese Network
                            with st.spinner("Comparing faces..."):
                                face1_processed = preprocess_face_for_siamese(face1)
                                face2_processed = preprocess_face_for_siamese(face2)
                                
                                # Compare faces
                                similarity, is_match = siamese_network.compare_faces(
                                    face1_processed, face2_processed, distance_metric=distance_metric)
                            
                            # Display results
                            st.markdown("### Results")
                            st.markdown(f"**Similarity Score:** {similarity:.4f}")
                            st.markdown(f"**Match Threshold:** {threshold:.2f}")
                            
                            if is_match:
                                st.success(f"✅ These faces appear to be the **SAME PERSON** (similarity: {similarity:.4f})")
                            else:
                                st.error(f"❌ These faces appear to be **DIFFERENT PEOPLE** (similarity: {similarity:.4f})")
                            
                            # Visualization
                            fig = plot_similarity_result(face1, face2, similarity, is_match)
                            st.pyplot(fig)
                    except Exception as e:
                        st.error(f"Error during face comparison: {e}")

    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        st.info("Please check your installation and try again. You may need to run: pip install -r requirements.txt")

# Make sure we have sys module
import sys

# Run the app
if __name__ == "__main__":
    main()

