import cv2
import numpy as np
import os
from scipy.spatial import distance

def detect_and_align_face(image, face_cascade_path=None, eye_cascade_path=None, target_size=(500, 500)):
    """
    Deteksi wajah, melakukan alignment, dan normalisasi ukuran
    
    Args:
        image: Gambar input (format BGR dari OpenCV)
        face_cascade_path: Path ke file XML Haar Cascade untuk deteksi wajah
        eye_cascade_path: Path ke file XML Haar Cascade untuk deteksi mata
        target_size: Ukuran target output (width, height)
        
    Returns:
        normalized_face: Wajah yang sudah dinormalisasi ukuran dan alignment
        face_found: Boolean yang menunjukkan apakah wajah terdeteksi
        face_rect: Koordinat wajah yang terdeteksi (x, y, w, h)
    """
    # Konversi ke grayscale untuk deteksi
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Inisialisasi face detector dengan Haar Cascade
    try:
        # Gunakan cascade yang disediakan atau default OpenCV
        if face_cascade_path and os.path.exists(face_cascade_path):
            face_cascade = cv2.CascadeClassifier(face_cascade_path)
        else:
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            
        if eye_cascade_path and os.path.exists(eye_cascade_path):
            eye_cascade = cv2.CascadeClassifier(eye_cascade_path)
        else:
            eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    except Exception as e:
        print(f"Error loading cascade classifiers: {e}")
        # Jika gagal memuat detector, kembalikan gambar asli
        resized = cv2.resize(image, target_size)
        return resized, False, None
    
    # Deteksi wajah - coba beberapa parameter untuk meningkatkan deteksi
    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
    
    # Jika tidak ada wajah yang terdeteksi, coba parameter yang lebih toleran
    if len(faces) == 0:
        faces = face_cascade.detectMultiScale(gray, 1.1, 3, minSize=(30, 30))
        
    # Jika masih tidak ada wajah, coba scaling yang berbeda
    if len(faces) == 0:
        faces = face_cascade.detectMultiScale(gray, 1.05, 4, minSize=(20, 20))
    
    # Jika tidak ada wajah yang terdeteksi, kembalikan gambar yang diubah ukuran saja
    if len(faces) == 0:
        print("No face detected, resizing original image")
        resized = cv2.resize(image, target_size)
        return resized, False, None
    
    # Ambil wajah terbesar jika ada beberapa wajah
    face_areas = [w*h for (x, y, w, h) in faces]
    largest_face_idx = np.argmax(face_areas)
    x, y, w, h = faces[largest_face_idx]
    face_rect = (x, y, w, h)
    
    # Ekstrak region wajah dengan margin
    margin_x = int(w * 0.2)  # 20% margin
    margin_y = int(h * 0.2)
    
    # Pastikan koordinat tidak melebihi batas gambar
    x1 = max(0, x - margin_x)
    y1 = max(0, y - margin_y)
    x2 = min(image.shape[1], x + w + margin_x)
    y2 = min(image.shape[0], y + h + margin_y)
    
    face_img = image[y1:y2, x1:x2]
    
    # Deteksi mata di region wajah untuk alignment
    if len(face_img.shape) == 3:
        face_gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    else:
        face_gray = face_img.copy()
    
    eyes = eye_cascade.detectMultiScale(face_gray, 1.1, 3, minSize=(20, 20))
    
    # Jika kurang dari 2 mata terdeteksi, langsung resize saja
    if len(eyes) < 2:
        aligned_face = face_img
    else:
        # Ambil 2 titik mata dengan jarak terjauh untuk alignment
        eye_centers = [(ex + ew//2, ey + eh//2) for ex, ey, ew, eh in eyes]
        
        # Hitung semua kemungkinan pasangan mata
        eye_pairs = [(i, j) for i in range(len(eye_centers)) for j in range(i+1, len(eye_centers))]
        
        # Pilih pasangan mata dengan jarak horizontal terbesar
        if eye_pairs:
            best_pair = max(eye_pairs, key=lambda pair: abs(eye_centers[pair[0]][0] - eye_centers[pair[1]][0]))
            left_eye, right_eye = eye_centers[best_pair[0]], eye_centers[best_pair[1]]
            
            # Pastikan mata kiri ada di sebelah kiri
            if left_eye[0] > right_eye[0]:
                left_eye, right_eye = right_eye, left_eye
            
            # Hitung sudut untuk alignment
            dx = right_eye[0] - left_eye[0]
            dy = right_eye[1] - left_eye[1]
            
            if dx > 0:
                angle = np.degrees(np.arctan2(dy, dx))
                
                # Pusat rotasi adalah tengah-tengah kedua mata
                center_x = int((left_eye[0] + right_eye[0]) / 2)
                center_y = int((left_eye[1] + right_eye[1]) / 2)
                center = (center_x, center_y)
                
                # Ambil matrix rotasi dan lakukan warp affine
                rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
                aligned_face = cv2.warpAffine(face_img, rotation_matrix, (face_img.shape[1], face_img.shape[0]),
                                            flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT)
            else:
                aligned_face = face_img
        else:
            aligned_face = face_img
    
    # Resize ke ukuran target
    normalized_face = cv2.resize(aligned_face, target_size)
    
    # Normalisasi pencahayaan dengan CLAHE
    if len(normalized_face.shape) == 3:
        # Untuk gambar berwarna, normalisasi di channel L dari Lab
        lab = cv2.cvtColor(normalized_face, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Terapkan CLAHE pada channel L
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        
        # Gabungkan kembali
        normalized_lab = cv2.merge((cl, a, b))
        normalized_face = cv2.cvtColor(normalized_lab, cv2.COLOR_LAB2BGR)
    else:
        # Untuk gambar grayscale
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        normalized_face = clahe.apply(normalized_face)
    
    return normalized_face, True, face_rect

def preprocess_face_for_ethnicity(face_img, target_size=(224, 224)):
    """
    Preprocess face image for Ethnicity Classifier
    
    Args:
        face_img: Face image from face detector
        target_size: Target size for the model
        
    Returns:
        processed_face: Face image ready for Ethnicity Classifier input
    """
    # Resize to target size
    face_resized = cv2.resize(face_img, target_size)
    
    # Convert to RGB if needed
    if len(face_resized.shape) == 2:  # Grayscale
        face_resized = cv2.cvtColor(face_resized, cv2.COLOR_GRAY2RGB)
    elif face_resized.shape[2] == 4:  # RGBA
        face_resized = cv2.cvtColor(face_resized, cv2.COLOR_RGBA2RGB)
    
    # Normalize pixel values to [0, 1]
    face_normalized = face_resized / 255.0
    
    # Add batch dimension
    face_batch = np.expand_dims(face_normalized, axis=0)
    
    return face_batch

def normalize_face(face_img):
    """
    Apply additional normalization techniques to enhance face image
    
    Args:
        face_img: Face image
        
    Returns:
        normalized_face: Enhanced face image
    """
    # Convert to BGR if grayscale
    if len(face_img.shape) == 2:
        face_img = cv2.cvtColor(face_img, cv2.COLOR_GRAY2BGR)
        
    # Convert to LAB color space for CLAHE
    lab = cv2.cvtColor(face_img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE to L channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    
    # Merge channels back
    normalized_lab = cv2.merge((cl, a, b))
    
    # Convert back to BGR
    normalized_face = cv2.cvtColor(normalized_lab, cv2.COLOR_LAB2BGR)
    
    return normalized_face

def parse_image_metadata(filename):
    """
    Parse metadata from image filename
    
    Expected format: Nama_Suku_Ekspresi_Sudut_Pencahayaan_Jarak.ext
    Or simpler format: Nama_Suku_Ekspresi_Sudut.ext
    
    Args:
        filename: Image filename
        
    Returns:
        metadata: Dictionary with extracted metadata
    """
    # Remove extension
    base_name = os.path.splitext(filename)[0]
    parts = base_name.split('_')
    
    metadata = {
        'nama': parts[0] if len(parts) > 0 else 'unknown',
        'suku': parts[1] if len(parts) > 1 else 'unknown',
        'ekspresi': parts[2] if len(parts) > 2 else 'unknown',
        'sudut': parts[3] if len(parts) > 3 else 'unknown',
    }
    
    # Optional metadata
    if len(parts) > 4:
        metadata['pencahayaan'] = parts[4]
    else:
        metadata['pencahayaan'] = 'unknown'
        
    if len(parts) > 5:
        metadata['jarak'] = parts[5]
    else:
        # Try to infer distance from angle field (e.g., "Frontal_Dekat")
        if '_' in metadata['sudut']:
            angle_parts = metadata['sudut'].split('_')
            if len(angle_parts) > 1:
                metadata['sudut'] = angle_parts[0]
                metadata['jarak'] = angle_parts[1]
            else:
                metadata['jarak'] = 'unknown'
        else:
            metadata['jarak'] = 'unknown'
    
    return metadata