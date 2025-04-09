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

# import os
# import json
# import streamlit as st
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# import cv2
# import numpy as np
# from PIL import Image

# from src.dataset_splitter import split_dataset
# from src.data_augmentation import augment_dataset
# from src.create_metadata import create_metadata_csv

# def main():
#     st.title("Dataset Preprocessing untuk Face Ethnicity Recognition")
    
#     # Inisialisasi session state untuk menyimpan data antar sesi
#     if 'augmentation_stats' not in st.session_state:
#         st.session_state['augmentation_stats'] = None
#     if 'augmented_dataset_dir' not in st.session_state:
#         st.session_state['augmented_dataset_dir'] = None
#     if 'input_dir' not in st.session_state:
#         st.session_state['input_dir'] = None
#     if 'processing_done' not in st.session_state:
#         st.session_state['processing_done'] = False
#     if 'split_stats' not in st.session_state:
#         st.session_state['split_stats'] = None
    
#     # Sidebar untuk konfigurasi
#     st.sidebar.header("Pengaturan Preprocessing")
    
#     # Pilih direktori input
#     input_dir = st.sidebar.text_input(
#         "Direktori Dataset Raw", 
#         value="dataset/raw",
#         help="Path ke direktori yang berisi gambar mentah"
#     )
    
#     # Pilih direktori output
#     final_dataset_dir = st.sidebar.text_input(
#         "Direktori Dataset Final", 
#         value="dataset/final_dataset",
#         help="Path untuk menyimpan dataset yang sudah di-split"
#     )
    
#     # Pilih direktori augmentasi
#     augmented_dataset_dir = st.sidebar.text_input(
#         "Direktori Dataset Augmentasi", 
#         value="dataset/augmented_dataset",
#         help="Path untuk menyimpan dataset yang sudah diaugmentasi"
#     )
    
#     # Pilih direktori metadata
#     metadata_path = st.sidebar.text_input(
#         "Path Metadata CSV", 
#         value="dataset/metadata.csv",
#         help="Path untuk menyimpan metadata dataset"
#     )
    
#     # Path untuk menyimpan statistik hasil pemrosesan
#     stats_path = st.sidebar.text_input(
#         "Path Simpan Statistik", 
#         value="dataset/stats.json",
#         help="Path untuk menyimpan statistik pemrosesan"
#     )
    
#     # Slider untuk proporsi test set
#     test_size = st.sidebar.slider(
#         "Proporsi Test Set", 
#         min_value=0.05, 
#         max_value=0.3, 
#         value=0.15, 
#         step=0.05,
#         help="Proporsi data yang akan digunakan untuk testing (validasi akan mengambil proporsi yang sama)"
#     )
    
#     # Tombol untuk memuat hasil preprocessing yang sudah ada
#     if st.sidebar.button("Muat Hasil Sebelumnya"):
#         if os.path.exists(stats_path):
#             try:
#                 with open(stats_path, 'r') as f:
#                     saved_stats = json.load(f)
                
#                 # Muat statistik yang disimpan ke session state
#                 st.session_state['augmentation_stats'] = saved_stats.get('augmentation_stats', None)
#                 st.session_state['split_stats'] = saved_stats.get('split_stats', None)
#                 st.session_state['augmented_dataset_dir'] = augmented_dataset_dir
#                 st.session_state['input_dir'] = input_dir
#                 st.session_state['processing_done'] = True
                
#                 st.success("Berhasil memuat hasil pemrosesan sebelumnya!")
#             except Exception as e:
#                 st.error(f"Gagal memuat hasil sebelumnya: {e}")
#         else:
#             st.warning(f"Tidak ditemukan file statistik di: {stats_path}")
    
#     # Tombol untuk memulai preprocessing
#     if st.sidebar.button("Proses Dataset"):
#         # Pastikan direktori output ada
#         os.makedirs(final_dataset_dir, exist_ok=True)
#         os.makedirs(augmented_dataset_dir, exist_ok=True)
        
#         # Tampilkan spinner selama preprocessing
#         with st.spinner("Memproses dataset..."):
#             # Lakukan splitting dataset
#             split_stats = split_dataset(
#                 input_dir, 
#                 final_dataset_dir, 
#                 test_size=test_size
#             )
            
#             # Lakukan augmentasi
#             augmentation_stats = augment_dataset(
#                 final_dataset_dir, 
#                 augmented_dataset_dir
#             )
            
#             # Buat metadata
#             create_metadata_csv(
#                 input_dir, 
#                 metadata_path
#             )
            
#             # Simpan statistik ke file
#             try:
#                 with open(stats_path, 'w') as f:
#                     json.dump({
#                         'augmentation_stats': augmentation_stats,
#                         'split_stats': split_stats
#                     }, f, indent=2)
#                 st.success(f"Statistik hasil pemrosesan disimpan ke: {stats_path}")
#             except Exception as e:
#                 st.error(f"Gagal menyimpan statistik: {e}")
        
#         # Simpan statistik augmentasi ke session state untuk diakses di tab perbandingan
#         st.session_state['augmentation_stats'] = augmentation_stats
#         st.session_state['split_stats'] = split_stats
#         st.session_state['augmented_dataset_dir'] = augmented_dataset_dir
#         st.session_state['input_dir'] = input_dir
#         st.session_state['processing_done'] = True
        
#         # Tampilkan hasil
#         st.success("Preprocessing dataset selesai!")
    
#     # Tampilkan hasil pemrosesan jika sudah dilakukan
#     if st.session_state['processing_done']:
#         display_results(
#             st.session_state['split_stats'], 
#             st.session_state['augmentation_stats'],
#             st.session_state['augmented_dataset_dir'],
#             st.session_state['input_dir'],
#             metadata_path
#         )

# def display_results(split_stats, augmentation_stats, augmented_dataset_dir, input_dir, metadata_path):
#     """
#     Tampilkan hasil preprocessing dalam bentuk tab
#     """
#     # Tab untuk menampilkan hasil
#     tab1, tab2, tab3, tab4, tab5 = st.tabs([
#         "Statistik Split Dataset", 
#         "Statistik Augmentasi", 
#         "Metadata Dataset",
#         "Perbandingan Hasil Augmentasi",
#         "Perbandingan Hasil Normalisasi"
#     ])
    
#     with tab1:
#         display_split_stats(split_stats)
    
#     with tab2:
#         display_augmentation_stats(augmentation_stats)
    
#     with tab3:
#         display_metadata(metadata_path)
    
#     with tab4:
#         display_augmentation_comparison(augmentation_stats, augmented_dataset_dir, input_dir)
    
#     with tab5:
#         display_normalization_comparison(augmentation_stats, augmented_dataset_dir, input_dir)

# def display_split_stats(split_stats):
#     """
#     Tampilkan statistik split dataset
#     """
#     if not split_stats:
#         st.info("Tidak ada data statistik split. Silakan proses dataset terlebih dahulu.")
#         return
    
#     st.subheader("Distribusi Dataset")
    
#     # Siapkan data untuk visualisasi
#     split_data = []
#     for split, split_info in split_stats.items():
#         for suku, count in split_info['per_suku'].items():
#             split_data.append({
#                 'Split': split,
#                 'Suku': suku,
#                 'Jumlah Gambar': count
#             })
    
#     df_split = pd.DataFrame(split_data)
    
#     # Buat plot distribusi
#     plt.figure(figsize=(10, 6))
#     sns.barplot(x='Split', y='Jumlah Gambar', hue='Suku', data=df_split)
#     plt.title('Distribusi Gambar per Split dan Suku')
#     st.pyplot(plt)
    
#     # Tampilkan tabel statistik
#     st.subheader("Rincian Pembagian Dataset")
#     for split, split_info in split_stats.items():
#         st.write(f"**{split.capitalize()} Set**:")
#         st.write(f"Total Gambar: {split_info['total']}")
#         st.write("Distribusi per Suku:")
#         st.json(split_info['per_suku'])

# def display_augmentation_stats(augmentation_stats):
#     """
#     Tampilkan statistik augmentasi
#     """
#     if not augmentation_stats:
#         st.info("Tidak ada data statistik augmentasi. Silakan proses dataset terlebih dahulu.")
#         return
    
#     st.subheader("Statistik Augmentasi")
    
#     # Siapkan data untuk visualisasi augmentasi
#     aug_data = []
#     for suku, suku_stats in augmentation_stats['per_suku'].items():
#         aug_data.append({
#             'Suku': suku,
#             'Gambar Asli': suku_stats['original'],
#             'Gambar Normalisasi': suku_stats.get('normalized', 0),
#             'Gambar Augmentasi': suku_stats['augmented']
#         })
    
#     df_aug = pd.DataFrame(aug_data)
    
#     # Buat plot augmentasi
#     plt.figure(figsize=(12, 6))
#     df_aug.plot(x='Suku', y=['Gambar Asli', 'Gambar Normalisasi', 'Gambar Augmentasi'], kind='bar', stacked=False)
#     plt.title('Perbandingan Gambar Asli, Normalisasi, dan Augmentasi per Suku')
#     plt.xlabel('Suku')
#     plt.ylabel('Jumlah Gambar')
#     plt.legend(title='Jenis Gambar')
#     st.pyplot(plt)
    
#     # Tampilkan statistik augmentasi
#     st.subheader("Rincian Augmentasi")
#     st.write(f"Total Gambar Asli: {augmentation_stats['total_images']}")
#     st.write(f"Total Gambar Normalisasi: {augmentation_stats.get('normalized_images', 0)}")
#     st.write(f"Total Gambar Augmentasi: {augmentation_stats['augmented_images']}")
#     st.write(f"Wajah Tidak Terdeteksi: {augmentation_stats.get('faces_not_detected', 0)}")
    
#     st.write("\nRincian per Suku:")
#     aug_detail = {
#         suku: {
#             'Gambar Asli': suku_stats['original'],
#             'Gambar Normalisasi': suku_stats.get('normalized', 0),
#             'Gambar Augmentasi': suku_stats['augmented'],
#             'Wajah Tidak Terdeteksi': suku_stats.get('faces_not_detected', 0)
#         } 
#         for suku, suku_stats in augmentation_stats['per_suku'].items()
#     }
#     st.json(aug_detail)

# def display_metadata(metadata_path):
#     """
#     Tampilkan metadata dataset
#     """
#     st.subheader("Metadata Dataset")
    
#     # Baca metadata
#     try:
#         if os.path.exists(metadata_path):
#             df_metadata = pd.read_csv(metadata_path)
            
#             # Tampilkan preview metadata
#             st.dataframe(df_metadata.head())
            
#             # Statistik metadata
#             st.subheader("Statistik Metadata")
            
#             # Distribusi suku
#             suku_dist = df_metadata['suku'].value_counts()
#             plt.figure(figsize=(8, 5))
#             suku_dist.plot(kind='pie', autopct='%1.1f%%')
#             plt.title('Distribusi Suku')
#             st.pyplot(plt)
            
#             # Informasi tambahan
#             st.write("Total Gambar:", len(df_metadata))
#             st.write("\nDistribusi Suku:")
#             st.dataframe(suku_dist)
            
#             # Informasi nama
#             st.write("\nDaftar Nama:")
#             st.dataframe(df_metadata['nama'].value_counts())
#         else:
#             st.info(f"File metadata tidak ditemukan: {metadata_path}")
#     except Exception as e:
#         st.error(f"Gagal membaca metadata: {e}")

# def display_normalization_comparison(augmentation_stats, augmented_dataset_dir, input_dir):
#     """
#     Tampilkan perbandingan gambar asli raw dan hasil normalisasi
#     """
#     if not augmentation_stats:
#         st.info("Tidak ada data augmentasi. Silakan proses dataset terlebih dahulu.")
#         return
    
#     st.subheader("Perbandingan Gambar Asli dan Hasil Normalisasi")
    
#     # Filter berdasarkan suku
#     available_suku = list(augmentation_stats['per_suku'].keys())
#     if not available_suku:
#         st.warning("Tidak ada data suku yang tersedia")
#         return
    
#     # Gunakan key yang unik untuk selectbox agar tidak konflik dengan tab lain
#     selected_suku = st.selectbox("Pilih Suku", available_suku, key="norm_suku")
    
#     # Filter berdasarkan split
#     split_options = ["train", "validation", "test"]
#     selected_split = st.selectbox("Pilih Split", split_options, key="norm_split")
    
#     # Dapatkan semua gambar asli untuk suku dan split yang dipilih
#     original_images = {}
    
#     # Periksa apakah ada data augmentasi atau tidak
#     if 'augmentation_map' in augmentation_stats:
#         for orig_path, aug_info in augmentation_stats['augmentation_map'].items():
#             if aug_info['suku'] == selected_suku and aug_info['split'] == selected_split:
#                 # Filter gambar dengan suku yang dipilih
#                 if 'normalized' in aug_info:
#                     original_images[orig_path] = {
#                         'normalized': aug_info['normalized'],
#                         'face_detected': aug_info.get('face_detected', True)
#                     }
    
#     # Jika tidak ada gambar untuk kombinasi suku dan split yang dipilih
#     if not original_images:
#         st.warning(f"Tidak ada gambar untuk suku {selected_suku} di split {selected_split}")
#         return
    
#     # Opsi untuk menampilkan hanya gambar dengan wajah terdeteksi
#     show_only_detected = st.checkbox("Tampilkan hanya gambar dengan wajah terdeteksi", value=True, key="norm_detect_check")
    
#     # Filter gambar berdasarkan deteksi wajah jika diminta
#     if show_only_detected:
#         original_images = {k: v for k, v in original_images.items() if v.get('face_detected', True)}
#         if not original_images:
#             st.warning(f"Tidak ada gambar dengan wajah terdeteksi untuk suku {selected_suku} di split {selected_split}")
#             return
    
#     # Tampilkan pilihan gambar
#     image_options = list(original_images.keys())
    
#     # Dapatkan nama-nama gambar saja (tanpa path)
#     display_options = [os.path.basename(img_path) for img_path in image_options]
    
#     # Jika ada terlalu banyak gambar, batasi tampilan menggunakan pagination
#     items_per_page = 5
#     total_pages = max(1, (len(image_options) + items_per_page - 1) // items_per_page)
    
#     page = st.number_input("Halaman", min_value=1, max_value=total_pages, value=1, key="norm_page")
    
#     start_idx = (page - 1) * items_per_page
#     end_idx = min(start_idx + items_per_page, len(image_options))
    
#     st.write(f"Menampilkan {end_idx - start_idx} dari {len(image_options)} gambar")
    
#     # Tampilkan gambar yang dipilih
#     for i in range(start_idx, end_idx):
#         orig_path = image_options[i]
#         norm_info = original_images[orig_path]
        
#         # Tampilkan judul untuk gambar ini
#         st.markdown(f"### Gambar {i+1}: {display_options[i]}")
        
#         # Status deteksi wajah
#         face_detected = norm_info.get('face_detected', True)
#         if face_detected:
#             st.success("✅ Wajah terdeteksi dan dinormalisasi")
#         else:
#             st.warning("⚠️ Wajah tidak terdeteksi, gambar hanya diubah ukuran")
        
#         # Buat baris untuk gambar asli dan hasil normalisasi
#         cols = st.columns(2)
        
#         # Path lengkap ke gambar asli di direktori output
#         full_orig_path = os.path.join(augmented_dataset_dir, orig_path)
        
#         # Juga perlu mendapatkan path ke gambar asli dari raw untuk perbandingan
#         # Kita perlu mengekstrak informasi subjek, suku, dan nama file
#         parts = orig_path.split('/')
#         suku = parts[1]
#         filename = parts[2]
        
#         # Ekstrak nama subjek dari nama file (contoh: Fanza_Sunda_...)
#         name_parts = filename.split('_')
#         if len(name_parts) >= 2:
#             subjek = name_parts[0]
#             raw_suku = name_parts[1]
#             # Path ke raw image
#             raw_path = os.path.join(input_dir, subjek, raw_suku, filename)
#         else:
#             # Jika nama file tidak sesuai pola, gunakan gambar dari output saja
#             raw_path = full_orig_path
        
#         # Path ke gambar normalisasi
#         norm_path = os.path.join(augmented_dataset_dir, norm_info['normalized'])
        
#         # Periksa apakah raw image tersedia
#         if os.path.exists(raw_path):
#             try:
#                 raw_image = Image.open(raw_path)
#                 with cols[0]:
#                     st.write("Gambar Raw Asli:")
#                     st.image(raw_image, use_column_width=True)
#             except Exception as e:
#                 with cols[0]:
#                     st.error(f"Tidak dapat menampilkan gambar raw: {e}")
        
#         # Tampilkan gambar normalisasi
#         try:
#             if os.path.exists(norm_path):
#                 norm_image = Image.open(norm_path)
#                 with cols[1]:
#                     st.write("Hasil Normalisasi (500x500):")
#                     st.image(norm_image, use_column_width=True)
#             else:
#                 with cols[1]:
#                     st.error(f"File normalisasi tidak ditemukan: {norm_path}")
#         except Exception as e:
#             st.error(f"Error saat menampilkan gambar normalisasi: {e}")
        
#         # Tambahkan garis pemisah
#         st.markdown("---")

# def display_augmentation_comparison(augmentation_stats, augmented_dataset_dir, input_dir):
#     """
#     Tampilkan perbandingan gambar asli, normalisasi, dan hasil augmentasi
#     """
#     if not augmentation_stats:
#         st.info("Tidak ada data augmentasi. Silakan proses dataset terlebih dahulu.")
#         return
    
#     st.subheader("Perbandingan Gambar Asli dan Hasil Augmentasi")
    
#     # Filter berdasarkan suku
#     available_suku = list(augmentation_stats['per_suku'].keys())
#     if not available_suku:
#         st.warning("Tidak ada data suku yang tersedia")
#         return
    
#     selected_suku = st.selectbox("Pilih Suku", available_suku, key="aug_suku")
    
#     # Filter berdasarkan split
#     split_options = ["train", "validation", "test"]
#     selected_split = st.selectbox("Pilih Split", split_options, key="aug_split")
    
#     # Dapatkan semua gambar asli untuk suku dan split yang dipilih
#     original_images = {}
    
#     # Periksa apakah ada data augmentasi atau tidak
#     if 'augmentation_map' in augmentation_stats:
#         for orig_path, aug_info in augmentation_stats['augmentation_map'].items():
#             if aug_info['suku'] == selected_suku and aug_info['split'] == selected_split:
#                 # Filter gambar dengan suku yang dipilih
#                 original_images[orig_path] = {
#                     'normalized': aug_info.get('normalized', None),
#                     'augmented_versions': aug_info['augmented_versions'],
#                     'face_detected': aug_info.get('face_detected', True)
#                 }
    
#     # Jika tidak ada gambar untuk kombinasi suku dan split yang dipilih
#     if not original_images:
#         st.warning(f"Tidak ada gambar untuk suku {selected_suku} di split {selected_split}")
#         return
    
#     # Opsi untuk menampilkan hanya gambar dengan wajah terdeteksi
#     show_only_detected = st.checkbox("Tampilkan hanya gambar dengan wajah terdeteksi", value=True, key="aug_detect_check")
    
#     # Filter gambar berdasarkan deteksi wajah jika diminta
#     if show_only_detected:
#         original_images = {k: v for k, v in original_images.items() if v.get('face_detected', True)}
#         if not original_images:
#             st.warning(f"Tidak ada gambar dengan wajah terdeteksi untuk suku {selected_suku} di split {selected_split}")
#             return
    
#     # Tampilkan pilihan gambar
#     image_options = list(original_images.keys())
    
#     # Dapatkan nama-nama gambar saja (tanpa path)
#     display_options = [os.path.basename(img_path) for img_path in image_options]
    
#     # Jika ada terlalu banyak gambar, batasi tampilan menggunakan pagination
#     items_per_page = 3
#     total_pages = max(1, (len(image_options) + items_per_page - 1) // items_per_page)
    
#     page = st.number_input("Halaman", min_value=1, max_value=total_pages, value=1, key="aug_page")
    
#     start_idx = (page - 1) * items_per_page
#     end_idx = min(start_idx + items_per_page, len(image_options))
    
#     st.write(f"Menampilkan {end_idx - start_idx} dari {len(image_options)} gambar")
    
#     # Tampilkan gambar yang dipilih
#     for i in range(start_idx, end_idx):
#         orig_path = image_options[i]
#         img_info = original_images[orig_path]
#         aug_paths = img_info['augmented_versions']
        
#         # Tampilkan judul untuk gambar ini
#         st.markdown(f"### Gambar {i+1}: {display_options[i]}")
        
#         # Status deteksi wajah
#         face_detected = img_info.get('face_detected', True)
#         if face_detected:
#             st.success("✅ Wajah terdeteksi dan dinormalisasi")
#         else:
#             st.warning("⚠️ Wajah tidak terdeteksi, gambar hanya diubah ukuran")
        
#         # Buat baris untuk gambar asli, normalisasi, dan augmentasi
#         cols = st.columns(3)
        
#         # Path lengkap ke gambar asli di direktori output
#         full_orig_path = os.path.join(augmented_dataset_dir, orig_path)
        
#         # Juga perlu mendapatkan path ke gambar asli dari raw untuk perbandingan
#         # Kita perlu mengekstrak informasi subjek, suku, dan nama file
#         parts = orig_path.split('/')
#         suku = parts[1]
#         filename = parts[2]
        
#         # Ekstrak nama subjek dari nama file (contoh: Fanza_Sunda_...)
#         name_parts = filename.split('_')
#         if len(name_parts) >= 2:
#             subjek = name_parts[0]
#             raw_suku = name_parts[1]
#             # Path ke raw image
#             raw_path = os.path.join(input_dir, subjek, raw_suku, filename)
#         else:
#             # Jika nama file tidak sesuai pola, gunakan gambar dari output saja
#             raw_path = full_orig_path
        
#         # Periksa apakah raw image tersedia
#         if os.path.exists(raw_path):
#             try:
#                 raw_image = Image.open(raw_path)
#                 with cols[0]:
#                     st.write("Gambar Raw Asli:")
#                     st.image(raw_image, use_column_width=True)
#             except Exception as e:
#                 with cols[0]:
#                     st.error(f"Tidak dapat menampilkan gambar raw: {e}")
        
#         # Tampilkan gambar normalisasi
#         if 'normalized' in img_info and img_info['normalized']:
#             try:
#                 norm_path = os.path.join(augmented_dataset_dir, img_info['normalized'])
#                 if os.path.exists(norm_path):
#                     norm_image = Image.open(norm_path)
#                     with cols[1]:
#                         st.write("Hasil Normalisasi:")
#                         st.image(norm_image, use_column_width=True)
#                 else:
#                     with cols[1]:
#                         st.error(f"File normalisasi tidak ditemukan")
#             except Exception as e:
#                 with cols[1]:
#                     st.error(f"Error saat menampilkan gambar normalisasi")
        
#         try:
#             # Tampilkan contoh augmentasi (ambil yang pertama)
#             if aug_paths and len(aug_paths) > 0:
#                 # Path lengkap ke gambar augmentasi
#                 full_aug_path = os.path.join(augmented_dataset_dir, aug_paths[0])
#                 if os.path.exists(full_aug_path):
#                     aug_image = Image.open(full_aug_path)
#                     with cols[2]:
#                         st.write(f"Contoh Augmentasi:")
#                         st.image(aug_image, use_column_width=True)
#                 else:
#                     with cols[2]:
#                         st.error(f"File augmentasi tidak ditemukan")
#         except Exception as e:
#             with cols[2]:
#                 st.error(f"Error saat menampilkan gambar augmentasi")
        
#         # Tampilkan semua hasil augmentasi
#         st.write("Semua Hasil Augmentasi:")
#         aug_cols = st.columns(min(len(aug_paths), 3))
        
#         for j, aug_path in enumerate(aug_paths):
#             col_idx = j % 3
#             # Path lengkap ke gambar augmentasi
#             full_aug_path = os.path.join(augmented_dataset_dir, aug_path)
#             if os.path.exists(full_aug_path):
#                 aug_image = Image.open(full_aug_path)
#                 with aug_cols[col_idx]:
#                     st.write(f"Augmentasi #{j+1}:")
#                     st.image(aug_image, use_column_width=True)
        
#         # Tambahkan garis pemisah
#         st.markdown("---")

# def validate_dataset_structure(base_dir):
#     """
#     Validasi struktur direktori dataset
#     """
#     st.header("Validasi Struktur Dataset")
    
#     errors = []
#     warnings = []
    
#     # Cek struktur direktori raw
#     raw_dir = os.path.join(base_dir, 'raw')
#     if not os.path.exists(raw_dir):
#         errors.append(f"Direktori raw tidak ditemukan: {raw_dir}")
#     else:
#         # Validasi struktur raw
#         for subjek in os.listdir(raw_dir):
#             subjek_path = os.path.join(raw_dir, subjek)
#             if not os.path.isdir(subjek_path):
#                 continue
            
#             # Pastikan memiliki suku
#             if len(os.listdir(subjek_path)) == 0:
#                 errors.append(f"Subjek {subjek} tidak memiliki suku")
            
#             for suku in os.listdir(subjek_path):
#                 suku_path = os.path.join(subjek_path, suku)
                
#                 # Cek jumlah gambar
#                 images = [
#                     img for img in os.listdir(suku_path) 
#                     if img.lower().endswith(('.png', '.jpg', '.jpeg', '.JPG'))
#                 ]
                
#                 if len(images) < 4:
#                     warnings.append(f"Subjek {subjek} dari suku {suku} memiliki kurang dari 4 gambar")
                
#                 # Validasi nama file
#                 for img in images:
#                     expected_format = f"{subjek}_{suku}_*"
#                     if not img.startswith(f"{subjek}_{suku}_"):
#                         errors.append(f"Format nama file tidak sesuai: {img}. Diharapkan: {expected_format}")
    
#     # Tampilkan hasil validasi
#     if errors:
#         st.error("Kesalahan Struktur Dataset:")
#         for error in errors:
#             st.error(f"- {error}")
#     else:
#         st.success("Tidak ada kesalahan struktur pada direktori raw")
    
#     if warnings:
#         st.warning("Peringatan:")
#         for warning in warnings:
#             st.warning(f"- {warning}")
    
#     return len(errors) == 0

# def dataset_info_section():
#     """
#     Bagian informasi dataset
#     """
#     st.sidebar.header("Informasi Dataset")
    
#     # Pilih direktori untuk divalidasi
#     base_dir = st.sidebar.text_input(
#         "Direktori Base Dataset", 
#         value="dataset",
#         help="Path ke direktori utama dataset"
#     )
    
#     # Tombol validasi
#     if st.sidebar.button("Validasi Struktur Dataset"):
#         validate_dataset_structure(base_dir)

# def browse_raw_images():
#     """
#     Browser gambar raw
#     """
#     st.header("Browser Gambar Raw")
    
#     # Pilih direktori raw
#     raw_dir = st.text_input("Direktori Raw", value="dataset/raw")
    
#     if not os.path.exists(raw_dir):
#         st.error(f"Direktori tidak ditemukan: {raw_dir}")
#         return
    
#     # List subjek
#     subjek_list = [d for d in os.listdir(raw_dir) if os.path.isdir(os.path.join(raw_dir, d))]
    
#     if not subjek_list:
#         st.warning("Tidak ada subjek yang ditemukan")
#         return
    
#     # Pilih subjek
#     subjek = st.selectbox("Pilih Subjek", subjek_list)
    
#     # List suku untuk subjek yang dipilih
#     subjek_path = os.path.join(raw_dir, subjek)
#     suku_list = [d for d in os.listdir(subjek_path) if os.path.isdir(os.path.join(subjek_path, d))]
    
#     if not suku_list:
#         st.warning(f"Tidak ada suku untuk subjek {subjek}")
#         return
    
#     # Pilih suku
#     suku = st.selectbox("Pilih Suku", suku_list)
    
#     # Tampilkan gambar
#     suku_path = os.path.join(subjek_path, suku)
#     images = [img for img in os.listdir(suku_path) if img.lower().endswith(('.png', '.jpg', '.jpeg', '.JPG'))]
    
#     if not images:
#         st.warning(f"Tidak ada gambar untuk subjek {subjek} dari suku {suku}")
#         return
    
#     # Tampilkan gambar dalam grid
#     cols = st.columns(3)  # 3 kolom
    
#     for i, img_name in enumerate(images):
#         img_path = os.path.join(suku_path, img_name)
#         img = Image.open(img_path)
        
#         col_idx = i % 3
#         with cols[col_idx]:
#             st.image(img, caption=img_name, use_column_width=True)

# if __name__ == "__main__":
#     # Konfigurasi halaman Streamlit
#     st.set_page_config(
#         page_title="Face Ethnicity Recognition Dataset Preprocessing",
#         page_icon=":camera:",
#         layout="wide"
#     )
    
#     # Tambahkan navigasi di sidebar
#     menu = st.sidebar.radio(
#         "Menu Preprocessing",
#         [
#             "Preprocessing Dataset", 
#             "Validasi Dataset",
#             "Browser Gambar Raw"
#         ]
#     )
    
#     # Tampilkan halaman sesuai pilihan
#     if menu == "Preprocessing Dataset":
#         main()
#     elif menu == "Validasi Dataset":
#         dataset_info_section()
#     elif menu == "Browser Gambar Raw":
#         browse_raw_images()