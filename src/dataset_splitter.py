import os
import shutil
from sklearn.model_selection import train_test_split

def split_dataset(input_dir, output_dir, test_size=0.3, random_state=42):
    """
    Split dataset menjadi train, validation, dan test
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
    
    # Proses setiap subjek
    for subjek in os.listdir(input_dir):
        subjek_path = os.path.join(input_dir, subjek)
        
        if not os.path.isdir(subjek_path):
            continue
        
        # Proses setiap suku
        for suku in os.listdir(subjek_path):
            suku_path = os.path.join(subjek_path, suku)
            
            if not os.path.isdir(suku_path):
                continue
            
            # Buat direktori suku di setiap split
            train_suku_dir = os.path.join(train_dir, suku)
            val_suku_dir = os.path.join(val_dir, suku)
            test_suku_dir = os.path.join(test_dir, suku)
            
            os.makedirs(train_suku_dir, exist_ok=True)
            os.makedirs(val_suku_dir, exist_ok=True)
            os.makedirs(test_suku_dir, exist_ok=True)
            
            # Ambil semua gambar
            images = [
                img for img in os.listdir(suku_path) 
                if img.lower().endswith(('.png', '.jpg', '.jpeg', '.JPG'))
            ]
            
            # Split dataset
            train_val_images, test_images = train_test_split(
                images, 
                test_size=0.15,  # 15% untuk test
                random_state=random_state
            )

            train_images, val_images = train_test_split(
                train_val_images, 
                test_size=0.176,  # 15% dari total (0.176 * 0.85 = 0.15)
                random_state=random_state
            )
            
            # Salin gambar ke masing-masing split
            def copy_images(image_list, dest_dir):
                for img in image_list:
                    src_path = os.path.join(suku_path, img)
                    dest_path = os.path.join(dest_dir, img)
                    shutil.copy2(src_path, dest_path)
                return len(image_list)
            
            # Salin dan catat statistik
            train_count = copy_images(train_images, train_suku_dir)
            val_count = copy_images(val_images, val_suku_dir)
            test_count = copy_images(test_images, test_suku_dir)
            
            # Update statistik
            dataset_stats['train']['total'] += train_count
            dataset_stats['validation']['total'] += val_count
            dataset_stats['test']['total'] += test_count
            
            dataset_stats['train']['per_suku'][suku] = train_count
            dataset_stats['validation']['per_suku'][suku] = val_count
            dataset_stats['test']['per_suku'][suku] = test_count
    
    return dataset_stats