import os
import cv2
import numpy as np
import imgaug.augmenters as iaa
from scipy.spatial import distance

def detect_and_align_face(image, target_size=(500, 500)):
    """
    Deteksi wajah, melakukan alignment, dan normalisasi ukuran
    
    Args:
        image: Gambar input (format BGR dari OpenCV)
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
    # Gunakan model yang sudah ada di OpenCV
    try:
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    except Exception as e:
        print(f"Error loading cascade classifiers: {e}")
        # Jika gagal memuat detector, kembalikan gambar asli
        resized = cv2.resize(image, target_size)
        return resized, False, None
    
    # Deteksi wajah
    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
    
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
                # FIX: Konversi pusat rotasi menjadi tuple dengan nilai integer
                center_x = (left_eye[0] + right_eye[0]) // 2
                center_y = (left_eye[1] + right_eye[1]) // 2
                center = (center_x, center_y)  # Pastikan ini tuple dengan nilai integer
                
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

def augment_image(image):
    """
    Lakukan augmentasi pada gambar
    """
    # Definisikan augmentasi
    augmentation = iaa.Sequential([
        iaa.Affine(rotate=(-15, 15)),  # Rotasi ±15 derajat
        iaa.Fliplr(0.5),  # Flip horizontal
        iaa.MultiplyBrightness((0.8, 1.2)),  # Perubahan brightness
        iaa.LinearContrast((0.8, 1.2)),  # Kontras
        iaa.AdditiveGaussianNoise(scale=(0, 0.05*255))  # Noise Gaussian
    ])
    
    # Lakukan augmentasi
    augmented_images = [
        augmentation.augment_image(image),
        augmentation.augment_image(image)
    ]
    
    return augmented_images

def augment_dataset(input_dir, output_dir):
    """
    Lakukan augmentasi pada seluruh dataset
    """
    # Buat direktori output
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "normalized"), exist_ok=True)
    
    # Statistik augmentasi
    augmentation_stats = {
        'total_images': 0,
        'normalized_images': 0,
        'augmented_images': 0,
        'faces_not_detected': 0,
        'per_suku': {},
        'augmentation_map': {}  # Untuk menyimpan pemetaan gambar asli ke hasil augmentasi dan normalisasi
    }
    
    # Proses setiap split
    for split in ['train', 'validation', 'test']:
        split_input_dir = os.path.join(input_dir, split)
        split_output_dir = os.path.join(output_dir, split)
        split_normalized_dir = os.path.join(output_dir, "normalized", split)
        
        os.makedirs(split_output_dir, exist_ok=True)
        os.makedirs(split_normalized_dir, exist_ok=True)
        
        # Proses setiap suku
        for suku in os.listdir(split_input_dir):
            suku_input_dir = os.path.join(split_input_dir, suku)
            suku_output_dir = os.path.join(split_output_dir, suku)
            suku_normalized_dir = os.path.join(split_normalized_dir, suku)
            
            os.makedirs(suku_output_dir, exist_ok=True)
            os.makedirs(suku_normalized_dir, exist_ok=True)
            
            # Inisialisasi statistik suku
            if suku not in augmentation_stats['per_suku']:
                augmentation_stats['per_suku'][suku] = {
                    'original': 0,
                    'normalized': 0,
                    'augmented': 0,
                    'faces_not_detected': 0
                }
            
            # Proses setiap gambar
            for img_name in os.listdir(suku_input_dir):
                img_path = os.path.join(suku_input_dir, img_name)
                
                try:
                    # Baca gambar
                    image = cv2.imread(img_path)
                    if image is None:
                        print(f"Failed to read image: {img_path}")
                        continue
                    
                    # Normalisasi wajah
                    normalized_face, face_found, face_rect = detect_and_align_face(image, target_size=(500, 500))
                    
                    # Simpan gambar asli
                    dest_path = os.path.join(suku_output_dir, img_name)
                    cv2.imwrite(dest_path, image)
                    
                    # Simpan gambar yang sudah dinormalisasi
                    normalized_name = f"{os.path.splitext(img_name)[0]}_normalized{os.path.splitext(img_name)[1]}"
                    normalized_path = os.path.join(suku_normalized_dir, normalized_name)
                    cv2.imwrite(normalized_path, normalized_face)
                    
                    # Track jalur relatif untuk UI
                    rel_dest_path = os.path.join(split, suku, img_name)
                    rel_normalized_path = os.path.join("normalized", split, suku, normalized_name)
                    
                    augmentation_stats['total_images'] += 1
                    augmentation_stats['per_suku'][suku]['original'] += 1
                    
                    if face_found:
                        augmentation_stats['normalized_images'] += 1
                        augmentation_stats['per_suku'][suku]['normalized'] += 1
                    else:
                        augmentation_stats['faces_not_detected'] += 1
                        augmentation_stats['per_suku'][suku]['faces_not_detected'] += 1
                    
                    # Inisialisasi pemetaan untuk gambar ini
                    augmentation_stats['augmentation_map'][rel_dest_path] = {
                        'suku': suku,
                        'split': split,
                        'normalized': rel_normalized_path,
                        'face_detected': face_found,
                        'augmented_versions': []
                    }
                    
                    # Lakukan augmentasi pada gambar yang sudah dinormalisasi
                    augmented_images = augment_image(normalized_face)
                    
                    # Simpan gambar augmentasi
                    for i, aug_img in enumerate(augmented_images, 1):
                        base_name, ext = os.path.splitext(img_name)
                        aug_img_name = f"{base_name}_aug{i}{ext}"
                        aug_img_path = os.path.join(suku_output_dir, aug_img_name)
                        
                        cv2.imwrite(aug_img_path, aug_img)
                        
                        # Track jalur relatif hasil augmentasi
                        rel_aug_path = os.path.join(split, suku, aug_img_name)
                        augmentation_stats['augmentation_map'][rel_dest_path]['augmented_versions'].append(rel_aug_path)
                        
                        augmentation_stats['augmented_images'] += 1
                        augmentation_stats['per_suku'][suku]['augmented'] += 1
                except Exception as e:
                    print(f"Error processing image {img_path}: {e}")
                    continue
    
    return augmentation_stats