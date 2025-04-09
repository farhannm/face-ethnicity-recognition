import os
import csv
import cv2

def create_metadata_csv(input_dir, output_path):
    """
    Buat metadata CSV dari dataset
    """
    # Persiapkan list untuk menyimpan metadata
    metadata = []
    
    # Iterasi setiap subjek
    for subjek in os.listdir(input_dir):
        subjek_path = os.path.join(input_dir, subjek)
        
        if not os.path.isdir(subjek_path):
            continue
        
        # Iterasi setiap suku
        for suku in os.listdir(subjek_path):
            suku_path = os.path.join(subjek_path, suku)
            
            if not os.path.isdir(suku_path):
                continue
            
            # Iterasi setiap gambar
            for img_name in os.listdir(suku_path):
                img_path = os.path.join(suku_path, img_name)
                
                # Baca gambar untuk mendapatkan informasi tambahan
                try:
                    image = cv2.imread(img_path)
                    height, width, _ = image.shape
                except Exception as e:
                    print(f"Error membaca gambar {img_path}: {e}")
                    continue
                
                # Ekstrak informasi dari nama file
                file_parts = os.path.splitext(img_name)[0].split('_')
                
                # Default values
                ekspresi = file_parts[-2] if len(file_parts) > 2 else 'tidak_diketahui'
                sudut = file_parts[-1] if len(file_parts) > 3 else 'tidak_diketahui'
                
                # Tambahkan ke metadata
                metadata.append({
                    'path_gambar': img_path,
                    'nama': file_parts[0],
                    'suku': suku,
                    'ekspresi': ekspresi,
                    'sudut': sudut,
                    'pencahayaan': 'tidak_diketahui',
                    'width': width,
                    'height': height
                })
    
    # Tulis ke CSV
    with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = [
            'path_gambar', 'nama', 'suku', 
            'ekspresi', 'sudut', 'pencahayaan', 
            'width', 'height'
        ]
        
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