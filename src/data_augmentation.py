import os
import cv2
import numpy as np
import imgaug.augmenters as iaa
from scipy.spatial import distance
from utils.face_preprocessing import detect_and_align_face, parse_image_metadata

def augment_image(image):
    """
    Lakukan augmentasi pada gambar sesuai ketentuan yang diminta:
    - Rotasi (±15°)
    - Horizontal flip
    - Perubahan brightness dan contrast (±20%)
    - Penambahan noise Gaussian ringan
    """
    # Setiap augmentasi akan diterapkan dengan probabilitas tertentu
    # Definisikan augmentasi
    augmentation_pipeline = iaa.Sequential([
        # Rotasi ±15 derajat
        iaa.Affine(rotate=(-15, 15)),
        
        # Horizontal flip dengan probabilitas 50%
        iaa.Fliplr(0.5),
        
        # Perubahan brightness dan contrast (±20%)
        iaa.OneOf([
            iaa.MultiplyBrightness((0.8, 1.2)),
            iaa.LinearContrast((0.8, 1.2)),
        ]),
        
        # Noise Gaussian ringan
        iaa.AdditiveGaussianNoise(scale=(0, 0.05*255))
    ])
    
    # Buat variasi augmentasi - buat 5 versi dari setiap gambar
    augmented_images = []
    for _ in range(5):
        # Pada setiap iterasi, imgaug akan menerapkan augmentasi dengan
        # parameter yang berbeda-beda dari range yang ditentukan
        aug_img = augmentation_pipeline.augment_image(image)
        augmented_images.append(aug_img)
    
    return augmented_images

def augment_dataset(input_dir, output_dir, custom_cascades=None):
    """
    Lakukan augmentasi pada seluruh dataset
    
    Args:
        input_dir: Direktori input (hasil split dataset)
        output_dir: Direktori output untuk menyimpan hasil augmentasi
        custom_cascades: Dictionary berisi path ke cascade files kustom (opsional)
            {'face': path_to_face_cascade, 'eye': path_to_eye_cascade}
    """
    # Buat direktori output
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "normalized"), exist_ok=True)
    
    # Set cascade files
    face_cascade_path = None
    eye_cascade_path = None
    if custom_cascades:
        face_cascade_path = custom_cascades.get('face', None)
        eye_cascade_path = custom_cascades.get('eye', None)
    
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
        
        if not os.path.exists(split_input_dir):
            print(f"Warning: Split directory not found: {split_input_dir}")
            continue
        
        os.makedirs(split_output_dir, exist_ok=True)
        os.makedirs(split_normalized_dir, exist_ok=True)
        
        # Proses setiap suku
        for suku in os.listdir(split_input_dir):
            suku_input_dir = os.path.join(split_input_dir, suku)
            suku_output_dir = os.path.join(split_output_dir, suku)
            suku_normalized_dir = os.path.join(split_normalized_dir, suku)
            
            if not os.path.isdir(suku_input_dir):
                continue
            
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
                # Skip files that are not images
                if not img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                    continue
                    
                img_path = os.path.join(suku_input_dir, img_name)
                
                try:
                    # Baca gambar
                    image = cv2.imread(img_path)
                    if image is None:
                        print(f"Failed to read image: {img_path}")
                        continue
                    
                    # Ekstrak metadata dari nama file
                    metadata = parse_image_metadata(img_name)
                    
                    # Normalisasi wajah - deteksi, align, dan resize
                    normalized_face, face_found, face_rect = detect_and_align_face(
                        image, 
                        face_cascade_path=face_cascade_path,
                        eye_cascade_path=eye_cascade_path,
                        target_size=(500, 500)
                    )
                    
                    # Simpan gambar asli ke direktori output
                    dest_path = os.path.join(suku_output_dir, img_name)
                    cv2.imwrite(dest_path, image)
                    
                    # Simpan gambar yang sudah dinormalisasi
                    base_name, ext = os.path.splitext(img_name)
                    normalized_name = f"{base_name}_normalized{ext}"
                    normalized_path = os.path.join(suku_normalized_dir, normalized_name)
                    cv2.imwrite(normalized_path, normalized_face)
                    
                    # Track jalur relatif untuk UI
                    rel_dest_path = os.path.join(split, suku, img_name)
                    rel_normalized_path = os.path.join("normalized", split, suku, normalized_name)
                    
                    # Update statistik
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
                        'augmented_versions': [],
                        'metadata': metadata
                    }
                    
                    # Hanya lakukan augmentasi pada data training dan jika wajah terdeteksi
                    if split == 'train' and face_found:
                        # Lakukan augmentasi pada gambar yang sudah dinormalisasi
                        augmented_images = augment_image(normalized_face)
                        
                        # Simpan gambar augmentasi
                        for i, aug_img in enumerate(augmented_images, 1):
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