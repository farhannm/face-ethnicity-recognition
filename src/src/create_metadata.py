import os
import csv
import cv2
from utils.face_preprocessing import parse_image_metadata

def create_metadata_csv(input_dir, output_path):
    """
    Buat metadata CSV dari dataset
    
    Args:
        input_dir: Direktori yang berisi dataset raw
        output_path: Path untuk menyimpan file CSV metadata
    
    Returns:
        metadata: List dari metadata gambar
    """
    # Persiapkan list untuk menyimpan metadata
    metadata = []
    
    # Iterasi setiap suku
    for suku in os.listdir(input_dir):
        suku_path = os.path.join(input_dir, suku)
        if not os.path.isdir(suku_path):
            continue
        
        # Iterasi setiap subjek
        for subjek in os.listdir(suku_path):
            subjek_path = os.path.join(suku_path, subjek)
            if not os.path.isdir(subjek_path):
                continue
            
            # Iterasi setiap gambar
            for img_name in os.listdir(subjek_path):
                if not img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.JPG', '.bmp', '.tiff')):
                    continue
                
                img_path = os.path.join(subjek_path, img_name)
                
                # Baca gambar untuk mendapatkan informasi tambahan
                try:
                    image = cv2.imread(img_path)
                    if image is None:
                        print(f"Error membaca gambar {img_path}: Format tidak didukung")
                        continue
                    
                    height, width = image.shape[:2]
                except Exception as e:
                    print(f"Error membaca gambar {img_path}: {e}")
                    continue
                
                # Parse metadata dari nama file
                file_metadata = parse_image_metadata(img_name)
                
                # Tambahkan informasi dimensi dan path
                file_metadata.update({
                    'path_gambar': img_path,
                    'width': width,
                    'height': height,
                    'suku': suku,
                    'subjek': subjek
                })
                
                # Jika tidak ada nama di metadata, gunakan nama subjek
                if file_metadata['nama'] == 'unknown':
                    file_metadata['nama'] = subjek
                
                # Jika tidak ada suku di metadata, gunakan suku dari direktori
                if file_metadata['suku'] == 'unknown':
                    file_metadata['suku'] = suku
                
                # Tambahkan ke metadata
                metadata.append(file_metadata)
    
    # Tulis ke CSV
    with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
        # Gabungkan semua kemungkinan field dari semua metadata
        all_fields = set()
        for entry in metadata:
            all_fields.update(entry.keys())
        
        fieldnames = sorted(list(all_fields))
        
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for row in metadata:
            writer.writerow(row)
    
    print(f"Metadata CSV berhasil dibuat di {output_path}")
    
    # Tampilkan statistik
    print("\nStatistik Dataset:")
    suku_counts = {}
    for row in metadata:
        suku = row['suku']
        suku_counts[suku] = suku_counts.get(suku, 0) + 1
    
    print("Distribusi per Suku:")
    for suku, count in suku_counts.items():
        print(f"- {suku}: {count} gambar")
    
    return metadata