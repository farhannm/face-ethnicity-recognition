import os
import shutil
from sklearn.model_selection import train_test_split
import re

def split_dataset(input_dir, output_dir, test_size=0.15, val_size=0.15, random_state=42):
    """
    Split dataset menjadi train, validation, dan test sesuai dengan ketentuan:
    - 70% training
    - 15% validation
    - 15% testing
    
    Args:
        input_dir: Direktori yang berisi dataset raw
        output_dir: Direktori untuk menyimpan hasil split
        test_size: Proporsi data untuk test set (default: 0.15)
        val_size: Proporsi data untuk validation set (default: 0.15)
        random_state: Random seed untuk reproduksibilitas
    
    Returns:
        dataset_stats: Statistik hasil split dataset
    """
    # Buat direktori output
    os.makedirs(output_dir, exist_ok=True)
    train_dir = os.path.join(output_dir, 'train')
    val_dir = os.path.join(output_dir, 'validation')
    test_dir = os.path.join(output_dir, 'test')
    
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    # Statistik dataset
    dataset_stats = {
        'train': {'total': 0, 'per_suku': {}},
        'validation': {'total': 0, 'per_suku': {}},
        'test': {'total': 0, 'per_suku': {}}
    }
    
    # Dictionary untuk melacak semua gambar per subjek dan suku
    all_images = {}
    
    # Scan direktori input untuk mengumpulkan semua gambar
    for suku in os.listdir(input_dir):
        suku_path = os.path.join(input_dir, suku)
        if not os.path.isdir(suku_path):
            continue

        for subjek in os.listdir(suku_path):
            subjek_path = os.path.join(suku_path, subjek)
            if not os.path.isdir(subjek_path):
                continue

            for img_name in os.listdir(subjek_path):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.JPG', '.bmp', '.tiff')):
                    # Inisialisasi entri suku jika belum ada
                    if suku not in all_images:
                        all_images[suku] = []

                    all_images[suku].append({
                        'subjek': subjek,
                        'suku': suku,
                        'img_name': img_name,
                        'full_path': os.path.join(subjek_path, img_name)
                    })

    
    # Buat split untuk setiap suku
    for suku, images in all_images.items():
        # Buat direktori suku di setiap split
        train_suku_dir = os.path.join(train_dir, suku)
        val_suku_dir = os.path.join(val_dir, suku)
        test_suku_dir = os.path.join(test_dir, suku)
        
        os.makedirs(train_suku_dir, exist_ok=True)
        os.makedirs(val_suku_dir, exist_ok=True)
        os.makedirs(test_suku_dir, exist_ok=True)
        
        # Inisialisasi statistik per suku
        if suku not in dataset_stats['train']['per_suku']:
            dataset_stats['train']['per_suku'][suku] = 0
        if suku not in dataset_stats['validation']['per_suku']:
            dataset_stats['validation']['per_suku'][suku] = 0
        if suku not in dataset_stats['test']['per_suku']:
            dataset_stats['test']['per_suku'][suku] = 0
        
        # Jika jumlah gambar terlalu sedikit, kirim semua ke training
        if len(images) < 5:
            train_images = images
            val_images = []
            test_images = []
        else:
            # Split dataset menggunakan sklearn
            # First, split out test set (15%)
            train_val_images, test_images = train_test_split(
                images, 
                test_size=test_size,
                random_state=random_state
            )
            
            # Then split the remaining into train (70%) and validation (15%)
            # Calculate val_size as a proportion of train_val_images
            effective_val_size = val_size / (1 - test_size)
            
            train_images, val_images = train_test_split(
                train_val_images, 
                test_size=effective_val_size,
                random_state=random_state
            )
        
        # Salin gambar ke masing-masing split
        for img_info in train_images:
            src_path = img_info['full_path']
            dest_path = os.path.join(train_suku_dir, img_info['img_name'])
            shutil.copy2(src_path, dest_path)
            dataset_stats['train']['total'] += 1
            dataset_stats['train']['per_suku'][suku] += 1
        
        for img_info in val_images:
            src_path = img_info['full_path']
            dest_path = os.path.join(val_suku_dir, img_info['img_name'])
            shutil.copy2(src_path, dest_path)
            dataset_stats['validation']['total'] += 1
            dataset_stats['validation']['per_suku'][suku] += 1
        
        for img_info in test_images:
            src_path = img_info['full_path']
            dest_path = os.path.join(test_suku_dir, img_info['img_name'])
            shutil.copy2(src_path, dest_path)
            dataset_stats['test']['total'] += 1
            dataset_stats['test']['per_suku'][suku] += 1
    
    return dataset_stats