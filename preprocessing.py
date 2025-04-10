import os
import json
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import numpy as np
from PIL import Image

from src.dataset_splitter import split_dataset
from src.data_augmentation import augment_dataset
from src.create_metadata import create_metadata_csv

def main():
    st.title("Dataset Preprocessing untuk Face Ethnicity Recognition")
    
    # Inisialisasi session state untuk menyimpan data antar sesi
    if 'augmentation_stats' not in st.session_state:
        st.session_state['augmentation_stats'] = None
    if 'augmented_dataset_dir' not in st.session_state:
        st.session_state['augmented_dataset_dir'] = None
    if 'input_dir' not in st.session_state:
        st.session_state['input_dir'] = None
    if 'processing_done' not in st.session_state:
        st.session_state['processing_done'] = False
    if 'split_stats' not in st.session_state:
        st.session_state['split_stats'] = None
    
    # Sidebar untuk konfigurasi
    st.sidebar.header("Pengaturan Preprocessing")
    
    # Pilih direktori input
    input_dir = st.sidebar.text_input(
        "Direktori Dataset Raw", 
        value="dataset/raw",
        help="Path ke direktori yang berisi gambar mentah"
    )
    
    # Pilih direktori output
    final_dataset_dir = st.sidebar.text_input(
        "Direktori Dataset Final", 
        value="dataset/final_dataset",
        help="Path untuk menyimpan dataset yang sudah di-split"
    )
    
    # Pilih direktori augmentasi
    augmented_dataset_dir = st.sidebar.text_input(
        "Direktori Dataset Augmentasi", 
        value="dataset/augmented_dataset",
        help="Path untuk menyimpan dataset yang sudah diaugmentasi"
    )
    
    # Pilih direktori metadata
    metadata_path = st.sidebar.text_input(
        "Path Metadata CSV", 
        value="dataset/metadata.csv",
        help="Path untuk menyimpan metadata dataset"
    )
    
    # Path untuk menyimpan statistik hasil pemrosesan
    stats_path = st.sidebar.text_input(
        "Path Simpan Statistik", 
        value="dataset/stats.json",
        help="Path untuk menyimpan statistik pemrosesan"
    )
    
    # Slider untuk proporsi test set dan validation set
    test_size = st.sidebar.slider(
        "Proporsi Test Set", 
        min_value=0.05, 
        max_value=0.3, 
        value=0.15, 
        step=0.05,
        help="Proporsi data yang akan digunakan untuk testing"
    )
    
    val_size = st.sidebar.slider(
        "Proporsi Validation Set", 
        min_value=0.05, 
        max_value=0.3, 
        value=0.15, 
        step=0.05,
        help="Proporsi data yang akan digunakan untuk validasi"
    )
    
    # Tombol untuk memuat hasil preprocessing yang sudah ada
    if st.sidebar.button("Muat Hasil Sebelumnya"):
        if os.path.exists(stats_path):
            try:
                with open(stats_path, 'r') as f:
                    saved_stats = json.load(f)
                
                # Muat statistik yang disimpan ke session state
                st.session_state['augmentation_stats'] = saved_stats.get('augmentation_stats', None)
                st.session_state['split_stats'] = saved_stats.get('split_stats', None)
                st.session_state['augmented_dataset_dir'] = augmented_dataset_dir
                st.session_state['input_dir'] = input_dir
                st.session_state['processing_done'] = True
                
                st.success("Berhasil memuat hasil pemrosesan sebelumnya!")
            except Exception as e:
                st.error(f"Gagal memuat hasil sebelumnya: {e}")
        else:
            st.warning(f"Tidak ditemukan file statistik di: {stats_path}")
    
    # Tombol untuk memulai preprocessing
    if st.sidebar.button("Proses Dataset"):
        # Pastikan direktori output ada
        os.makedirs(final_dataset_dir, exist_ok=True)
        os.makedirs(augmented_dataset_dir, exist_ok=True)
        
        # Tampilkan spinner selama preprocessing
        with st.spinner("Memproses dataset..."):
            # Step 1: Validasi struktur dataset (opsional)
            st.write("Langkah 1: Validasi struktur dataset...")
            is_valid = validate_dataset_structure(input_dir)
            if not is_valid:
                st.warning("Struktur dataset memiliki beberapa masalah, tetapi akan tetap dilanjutkan.")
            
            # Step 2: Lakukan splitting dataset
            st.write("Langkah 2: Memisahkan dataset menjadi train, validation, dan test...")
            split_stats = split_dataset(
                input_dir, 
                final_dataset_dir, 
                test_size=test_size,
                val_size=val_size
            )
            
            # Step 3: Lakukan augmentasi
            st.write("Langkah 3: Melakukan normalisasi dan augmentasi pada dataset...")
            augmentation_stats = augment_dataset(
                final_dataset_dir, 
                augmented_dataset_dir
            )
            
            # Step 4: Buat metadata
            st.write("Langkah 4: Membuat metadata dataset...")
            create_metadata_csv(
                input_dir, 
                metadata_path
            )
            
            # Simpan statistik ke file
            try:
                with open(stats_path, 'w') as f:
                    json.dump({
                        'augmentation_stats': augmentation_stats,
                        'split_stats': split_stats
                    }, f, indent=2)
                st.success(f"Statistik hasil pemrosesan disimpan ke: {stats_path}")
            except Exception as e:
                st.error(f"Gagal menyimpan statistik: {e}")
        
        # Simpan statistik augmentasi ke session state untuk diakses di tab perbandingan
        st.session_state['augmentation_stats'] = augmentation_stats
        st.session_state['split_stats'] = split_stats
        st.session_state['augmented_dataset_dir'] = augmented_dataset_dir
        st.session_state['input_dir'] = input_dir
        st.session_state['processing_done'] = True
        
        # Tampilkan hasil
        st.success("Preprocessing dataset selesai!")
    
    # Tampilkan hasil pemrosesan jika sudah dilakukan
    if st.session_state['processing_done']:
        display_results(
            st.session_state['split_stats'], 
            st.session_state['augmentation_stats'],
            st.session_state['augmented_dataset_dir'],
            st.session_state['input_dir'],
            metadata_path
        )

def display_results(split_stats, augmentation_stats, augmented_dataset_dir, input_dir, metadata_path):
    """
    Tampilkan hasil preprocessing dalam bentuk tab
    """
    # Tab untuk menampilkan hasil
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Statistik Split Dataset", 
        "Statistik Augmentasi", 
        "Metadata Dataset",
        "Perbandingan Hasil Augmentasi",
        "Perbandingan Hasil Normalisasi"
    ])
    
    with tab1:
        display_split_stats(split_stats)
    
    with tab2:
        display_augmentation_stats(augmentation_stats)
    
    with tab3:
        display_metadata(metadata_path)
    
    with tab4:
        display_augmentation_comparison(augmentation_stats, augmented_dataset_dir, input_dir)
    
    with tab5:
        display_normalization_comparison(augmentation_stats, augmented_dataset_dir, input_dir)

def display_split_stats(split_stats):
    """
    Tampilkan statistik split dataset
    """
    if not split_stats:
        st.info("Tidak ada data statistik split. Silakan proses dataset terlebih dahulu.")
        return
    
    st.subheader("Distribusi Dataset")
    
    # Siapkan data untuk visualisasi
    split_data = []
    for split, split_info in split_stats.items():
        for suku, count in split_info['per_suku'].items():
            split_data.append({
                'Split': split,
                'Suku': suku,
                'Jumlah Gambar': count
            })
    
    df_split = pd.DataFrame(split_data)
    
    # Buat plot distribusi
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Split', y='Jumlah Gambar', hue='Suku', data=df_split)
    plt.title('Distribusi Gambar per Split dan Suku')
    st.pyplot(plt)
    
    # Tampilkan tabel statistik
    st.subheader("Rincian Pembagian Dataset")
    for split, split_info in split_stats.items():
        st.write(f"**{split.capitalize()} Set**:")
        st.write(f"Total Gambar: {split_info['total']}")
        st.write("Distribusi per Suku:")
        st.json(split_info['per_suku'])
    
    # Tampilkan persentase pembagian dataset
    total_images = sum([split_info['total'] for split_info in split_stats.values()])
    if total_images > 0:
        st.subheader("Persentase Pembagian Dataset")
        percentages = {
            split: (split_info['total'] / total_images) * 100 
            for split, split_info in split_stats.items()
        }
        
        # Buat chart pie
        plt.figure(figsize=(8, 8))
        plt.pie(
            percentages.values(), 
            labels=percentages.keys(), 
            autopct='%1.1f%%', 
            startangle=90
        )
        plt.title('Persentase Pembagian Dataset')
        st.pyplot(plt)

def display_augmentation_stats(augmentation_stats):
    """
    Tampilkan statistik augmentasi
    """
    if not augmentation_stats:
        st.info("Tidak ada data statistik augmentasi. Silakan proses dataset terlebih dahulu.")
        return
    
    st.subheader("Statistik Augmentasi")
    
    # Siapkan data untuk visualisasi augmentasi
    aug_data = []
    for suku, suku_stats in augmentation_stats['per_suku'].items():
        aug_data.append({
            'Suku': suku,
            'Gambar Asli': suku_stats['original'],
            'Gambar Normalisasi': suku_stats.get('normalized', 0),
            'Gambar Augmentasi': suku_stats['augmented']
        })
    
    df_aug = pd.DataFrame(aug_data)
    
    # Buat plot augmentasi
    plt.figure(figsize=(12, 6))
    df_aug.plot(x='Suku', y=['Gambar Asli', 'Gambar Normalisasi', 'Gambar Augmentasi'], kind='bar', stacked=False)
    plt.title('Perbandingan Gambar Asli, Normalisasi, dan Augmentasi per Suku')
    plt.xlabel('Suku')
    plt.ylabel('Jumlah Gambar')
    plt.legend(title='Jenis Gambar')
    st.pyplot(plt)
    
    # Tampilkan statistik augmentasi
    st.subheader("Rincian Augmentasi")
    st.write(f"Total Gambar Asli: {augmentation_stats['total_images']}")
    st.write(f"Total Gambar Normalisasi: {augmentation_stats.get('normalized_images', 0)}")
    st.write(f"Total Gambar Augmentasi: {augmentation_stats['augmented_images']}")
    st.write(f"Wajah Tidak Terdeteksi: {augmentation_stats.get('faces_not_detected', 0)}")
    
    # Visualisasi proporsi wajah terdeteksi vs tidak terdeteksi
    if 'faces_not_detected' in augmentation_stats:
        faces_detected = augmentation_stats.get('normalized_images', 0)
        faces_not_detected = augmentation_stats.get('faces_not_detected', 0)
        
        if faces_detected + faces_not_detected > 0:
            plt.figure(figsize=(8, 8))
            plt.pie(
                [faces_detected, faces_not_detected], 
                labels=['Wajah Terdeteksi', 'Wajah Tidak Terdeteksi'], 
                autopct='%1.1f%%', 
                startangle=90,
                colors=['#5cb85c', '#d9534f']
            )
            plt.title('Proporsi Deteksi Wajah')
            st.pyplot(plt)
    
    st.write("\nRincian per Suku:")
    aug_detail = {
        suku: {
            'Gambar Asli': suku_stats['original'],
            'Gambar Normalisasi': suku_stats.get('normalized', 0),
            'Gambar Augmentasi': suku_stats['augmented'],
            'Wajah Tidak Terdeteksi': suku_stats.get('faces_not_detected', 0)
        } 
        for suku, suku_stats in augmentation_stats['per_suku'].items()
    }
    st.json(aug_detail)

def display_metadata(metadata_path):
    """
    Tampilkan metadata dataset
    """
    st.subheader("Metadata Dataset")
    
    # Baca metadata
    try:
        if os.path.exists(metadata_path):
            df_metadata = pd.read_csv(metadata_path)
            
            # Tampilkan preview metadata
            st.dataframe(df_metadata.head())
            
            # Statistik metadata
            st.subheader("Statistik Metadata")
            
            # Distribusi suku
            suku_dist = df_metadata['suku'].value_counts()
            plt.figure(figsize=(8, 5))
            suku_dist.plot(kind='bar')
            plt.title('Distribusi Suku')
            plt.ylabel('Jumlah Gambar')
            plt.xticks(rotation=45)
            st.pyplot(plt)
            
            # Distribusi ekspresi jika ada
            if 'ekspresi' in df_metadata.columns:
                ekspresi_dist = df_metadata['ekspresi'].value_counts()
                plt.figure(figsize=(8, 5))
                ekspresi_dist.plot(kind='bar')
                plt.title('Distribusi Ekspresi Wajah')
                plt.ylabel('Jumlah Gambar')
                plt.xticks(rotation=45)
                st.pyplot(plt)
            
            # Distribusi sudut jika ada
            if 'sudut' in df_metadata.columns:
                sudut_dist = df_metadata['sudut'].value_counts()
                plt.figure(figsize=(8, 5))
                sudut_dist.plot(kind='bar')
                plt.title('Distribusi Sudut Pengambilan')
                plt.ylabel('Jumlah Gambar')
                plt.xticks(rotation=45)
                st.pyplot(plt)
            
            # Informasi tambahan
            st.write("Total Gambar:", len(df_metadata))
            st.write("\nDistribusi Suku:")
            st.dataframe(suku_dist)
            
            # Informasi nama
            if 'nama' in df_metadata.columns:
                st.write("\nDistribusi Nama:")
                st.dataframe(df_metadata['nama'].value_counts())
        else:
            st.info(f"File metadata tidak ditemukan: {metadata_path}")
    except Exception as e:
        st.error(f"Gagal membaca metadata: {e}")

def display_normalization_comparison(augmentation_stats, augmented_dataset_dir, input_dir):
    """
    Tampilkan perbandingan gambar asli raw dan hasil normalisasi
    """
    if not augmentation_stats:
        st.info("Tidak ada data augmentasi. Silakan proses dataset terlebih dahulu.")
        return
    
    st.subheader("Perbandingan Gambar Asli dan Hasil Normalisasi")
    
    # Filter berdasarkan suku
    available_suku = list(augmentation_stats['per_suku'].keys())
    if not available_suku:
        st.warning("Tidak ada data suku yang tersedia")
        return
    
    # Gunakan key yang unik untuk selectbox agar tidak konflik dengan tab lain
    selected_suku = st.selectbox("Pilih Suku", available_suku, key="norm_suku")
    
    # Filter berdasarkan split
    split_options = ["train", "validation", "test"]
    selected_split = st.selectbox("Pilih Split", split_options, key="norm_split")
    
    # Dapatkan semua gambar asli untuk suku dan split yang dipilih
    original_images = {}
    
    # Periksa apakah ada data augmentasi atau tidak
    if 'augmentation_map' in augmentation_stats:
        for orig_path, aug_info in augmentation_stats['augmentation_map'].items():
            if aug_info['suku'] == selected_suku and aug_info['split'] == selected_split:
                # Filter gambar dengan suku yang dipilih
                if 'normalized' in aug_info:
                    original_images[orig_path] = {
                        'normalized': aug_info['normalized'],
                        'face_detected': aug_info.get('face_detected', True),
                        'metadata': aug_info.get('metadata', {})
                    }
    
    # Jika tidak ada gambar untuk kombinasi suku dan split yang dipilih
    if not original_images:
        st.warning(f"Tidak ada gambar untuk suku {selected_suku} di split {selected_split}")
        return
    
    # Opsi untuk menampilkan hanya gambar dengan wajah terdeteksi
    show_only_detected = st.checkbox("Tampilkan hanya gambar dengan wajah terdeteksi", value=True, key="norm_detect_check")
    
    # Filter berdasarkan ekspresi (jika ada dalam metadata)
    ekspresi_options = set()
    for img_info in original_images.values():
        if 'metadata' in img_info and 'ekspresi' in img_info['metadata']:
            ekspresi = img_info['metadata']['ekspresi']
            if ekspresi != 'unknown':
                ekspresi_options.add(ekspresi)
    
    selected_ekspresi = None
    if ekspresi_options:
        ekspresi_options = list(ekspresi_options)
        ekspresi_options.insert(0, "Semua Ekspresi")
        selected_ekspresi = st.selectbox("Filter berdasarkan Ekspresi", ekspresi_options, key="norm_ekspresi")
    
    # Filter berdasarkan sudut (jika ada dalam metadata)
    sudut_options = set()
    for img_info in original_images.values():
        if 'metadata' in img_info and 'sudut' in img_info['metadata']:
            sudut = img_info['metadata']['sudut']
            if sudut != 'unknown':
                sudut_options.add(sudut)
    
    selected_sudut = None
    if sudut_options:
        sudut_options = list(sudut_options)
        sudut_options.insert(0, "Semua Sudut")
        selected_sudut = st.selectbox("Filter berdasarkan Sudut", sudut_options, key="norm_sudut")
    
    # Filter gambar berdasarkan pilihan
    filtered_images = {}
    for path, info in original_images.items():
        # Filter berdasarkan deteksi wajah
        if show_only_detected and not info.get('face_detected', True):
            continue
        
        # Filter berdasarkan ekspresi
        if selected_ekspresi and selected_ekspresi != "Semua Ekspresi":
            if 'metadata' not in info or 'ekspresi' not in info['metadata'] or info['metadata']['ekspresi'] != selected_ekspresi:
                continue
        
        # Filter berdasarkan sudut
        if selected_sudut and selected_sudut != "Semua Sudut":
            if 'metadata' not in info or 'sudut' not in info['metadata'] or info['metadata']['sudut'] != selected_sudut:
                continue
        
        filtered_images[path] = info
    
    # Jika tidak ada gambar yang memenuhi kriteria filter
    if not filtered_images:
        st.warning(f"Tidak ada gambar yang memenuhi kriteria filter")
        return
    
    # Tampilkan pilihan gambar
    image_options = list(filtered_images.keys())
    
    # Dapatkan nama-nama gambar saja (tanpa path)
    display_options = [os.path.basename(img_path) for img_path in image_options]
    
    # Jika ada terlalu banyak gambar, batasi tampilan menggunakan pagination
    items_per_page = 5
    total_pages = max(1, (len(image_options) + items_per_page - 1) // items_per_page)
    
    page = st.number_input("Halaman", min_value=1, max_value=total_pages, value=1, key="norm_page")
    
    start_idx = (page - 1) * items_per_page
    end_idx = min(start_idx + items_per_page, len(image_options))
    
    st.write(f"Menampilkan {end_idx - start_idx} dari {len(image_options)} gambar")
    
    # Tampilkan gambar yang dipilih
    for i in range(start_idx, end_idx):
        orig_path = image_options[i]
        norm_info = filtered_images[orig_path]
        
        # Tampilkan judul untuk gambar ini
        st.markdown(f"### Gambar {i+1}: {display_options[i]}")
        
        # Status deteksi wajah
        face_detected = norm_info.get('face_detected', True)
        if face_detected:
            st.success("✅ Wajah terdeteksi dan dinormalisasi")
        else:
            st.warning("⚠️ Wajah tidak terdeteksi, gambar hanya diubah ukuran")
        
        # Tampilkan metadata gambar jika ada
        if 'metadata' in norm_info:
            metadata = norm_info['metadata']
            if metadata:
                meta_cols = st.columns(3)
                with meta_cols[0]:
                    st.write(f"**Ekspresi:** {metadata.get('ekspresi', 'unknown')}")
                with meta_cols[1]:
                    st.write(f"**Sudut:** {metadata.get('sudut', 'unknown')}")
                with meta_cols[2]:
                    st.write(f"**Pencahayaan:** {metadata.get('pencahayaan', 'unknown')}")
        
        # Buat baris untuk gambar asli dan hasil normalisasi
        cols = st.columns(2)
        
        # Path lengkap ke gambar asli di direktori output
        full_orig_path = os.path.join(augmented_dataset_dir, orig_path)
        
        # Juga perlu mendapatkan path ke gambar asli dari raw untuk perbandingan
        # Kita perlu mengekstrak informasi subjek, suku, dan nama file
        parts = orig_path.split('/')
        suku = parts[1]
        filename = parts[2]
        
        # Ekstrak nama subjek dari nama file (contoh: Fanza_Sunda_...)
        name_parts = filename.split('_')
        if len(name_parts) >= 2:
            subjek = name_parts[0]
            raw_suku = name_parts[1]
            # Path ke raw image
            raw_path = os.path.join(input_dir, raw_suku, subjek, filename)
        else:
            # Jika nama file tidak sesuai pola, gunakan gambar dari output saja
            raw_path = full_orig_path
        
        # Path ke gambar normalisasi
        norm_path = os.path.join(augmented_dataset_dir, norm_info['normalized'])
        
        # Periksa apakah raw image tersedia
        if os.path.exists(raw_path):
            try:
                raw_image = Image.open(raw_path)
                with cols[0]:
                    st.write("Gambar Raw Asli:")
                    st.image(raw_image, use_column_width=True)
            except Exception as e:
                with cols[0]:
                    st.error(f"Tidak dapat menampilkan gambar raw: {e}")
        
        # Tampilkan gambar normalisasi
        try:
            if os.path.exists(norm_path):
                norm_image = Image.open(norm_path)
                with cols[1]:
                    st.write("Hasil Normalisasi (500x500):")
                    st.image(norm_image, use_column_width=True)
            else:
                with cols[1]:
                    st.error(f"File normalisasi tidak ditemukan: {norm_path}")
        except Exception as e:
            st.error(f"Error saat menampilkan gambar normalisasi: {e}")
        
        # Tambahkan garis pemisah
        st.markdown("---")

def display_augmentation_comparison(augmentation_stats, augmented_dataset_dir, input_dir):
    """
    Tampilkan perbandingan gambar asli, normalisasi, dan hasil augmentasi
    """
    if not augmentation_stats:
        st.info("Tidak ada data augmentasi. Silakan proses dataset terlebih dahulu.")
        return
    
    st.subheader("Perbandingan Gambar Asli dan Hasil Augmentasi")
    
    # Filter berdasarkan suku
    available_suku = list(augmentation_stats['per_suku'].keys())
    if not available_suku:
        st.warning("Tidak ada data suku yang tersedia")
        return
    
    selected_suku = st.selectbox("Pilih Suku", available_suku, key="aug_suku")
    
    # Filter berdasarkan split
    split_options = ["train", "validation", "test"]
    selected_split = st.selectbox("Pilih Split", split_options, key="aug_split")
    
    # Dapatkan semua gambar asli untuk suku dan split yang dipilih
    original_images = {}
    
    # Periksa apakah ada data augmentasi atau tidak
    if 'augmentation_map' in augmentation_stats:
        for orig_path, aug_info in augmentation_stats['augmentation_map'].items():
            if aug_info['suku'] == selected_suku and aug_info['split'] == selected_split:
                # Filter gambar dengan suku yang dipilih
                # Hanya tampilkan gambar yang memiliki augmentasi (biasanya hanya data training)
                if 'augmented_versions' in aug_info and len(aug_info['augmented_versions']) > 0:
                    original_images[orig_path] = {
                        'normalized': aug_info.get('normalized', None),
                        'augmented_versions': aug_info['augmented_versions'],
                        'face_detected': aug_info.get('face_detected', True),
                        'metadata': aug_info.get('metadata', {})
                    }
    
    # Jika tidak ada gambar untuk kombinasi suku dan split yang dipilih
    if not original_images:
        st.warning(f"Tidak ada gambar dengan augmentasi untuk suku {selected_suku} di split {selected_split}")
        if selected_split != "train":
            st.info("Augmentasi biasanya hanya dilakukan pada data training.")
        return
    
    # Opsi untuk menampilkan hanya gambar dengan wajah terdeteksi
    show_only_detected = st.checkbox("Tampilkan hanya gambar dengan wajah terdeteksi", value=True, key="aug_detect_check")
    
    # Filter berdasarkan ekspresi (jika ada dalam metadata)
    ekspresi_options = set()
    for img_info in original_images.values():
        if 'metadata' in img_info and 'ekspresi' in img_info['metadata']:
            ekspresi = img_info['metadata']['ekspresi']
            if ekspresi != 'unknown':
                ekspresi_options.add(ekspresi)
    
    selected_ekspresi = None
    if ekspresi_options:
        ekspresi_options = list(ekspresi_options)
        ekspresi_options.insert(0, "Semua Ekspresi")
        selected_ekspresi = st.selectbox("Filter berdasarkan Ekspresi", ekspresi_options, key="aug_ekspresi")
    
    # Filter gambar berdasarkan pilihan
    filtered_images = {}
    for path, info in original_images.items():
        # Filter berdasarkan deteksi wajah
        if show_only_detected and not info.get('face_detected', True):
            continue
        
        # Filter berdasarkan ekspresi
        if selected_ekspresi and selected_ekspresi != "Semua Ekspresi":
            if 'metadata' not in info or 'ekspresi' not in info['metadata'] or info['metadata']['ekspresi'] != selected_ekspresi:
                continue
        
        filtered_images[path] = info
    
    # Jika tidak ada gambar yang memenuhi kriteria filter
    if not filtered_images:
        st.warning(f"Tidak ada gambar yang memenuhi kriteria filter")
        return
    
    # Tampilkan pilihan gambar
    image_options = list(filtered_images.keys())
    
    # Dapatkan nama-nama gambar saja (tanpa path)
    display_options = [os.path.basename(img_path) for img_path in image_options]
    
    # Jika ada terlalu banyak gambar, batasi tampilan menggunakan pagination
    items_per_page = 3
    total_pages = max(1, (len(image_options) + items_per_page - 1) // items_per_page)
    
    page = st.number_input("Halaman", min_value=1, max_value=total_pages, value=1, key="aug_page")
    
    start_idx = (page - 1) * items_per_page
    end_idx = min(start_idx + items_per_page, len(image_options))
    
    st.write(f"Menampilkan {end_idx - start_idx} dari {len(image_options)} gambar")
    
    # Tampilkan gambar yang dipilih
    for i in range(start_idx, end_idx):
        orig_path = image_options[i]
        img_info = filtered_images[orig_path]
        aug_paths = img_info['augmented_versions']
        
        # Tampilkan judul untuk gambar ini
        st.markdown(f"### Gambar {i+1}: {display_options[i]}")
        
        # Status deteksi wajah
        face_detected = img_info.get('face_detected', True)
        if face_detected:
            st.success("✅ Wajah terdeteksi dan dinormalisasi")
        else:
            st.warning("⚠️ Wajah tidak terdeteksi, gambar hanya diubah ukuran")
        
        # Tampilkan metadata gambar jika ada
        if 'metadata' in img_info:
            metadata = img_info['metadata']
            if metadata:
                meta_cols = st.columns(3)
                with meta_cols[0]:
                    st.write(f"**Ekspresi:** {metadata.get('ekspresi', 'unknown')}")
                with meta_cols[1]:
                    st.write(f"**Sudut:** {metadata.get('sudut', 'unknown')}")
                with meta_cols[2]:
                    st.write(f"**Pencahayaan:** {metadata.get('pencahayaan', 'unknown')}")
        
        # Buat baris untuk gambar asli, normalisasi, dan augmentasi
        cols = st.columns(3)
        
        # Path lengkap ke gambar asli di direktori output
        full_orig_path = os.path.join(augmented_dataset_dir, orig_path)
        
        # Juga perlu mendapatkan path ke gambar asli dari raw untuk perbandingan
        # Kita perlu mengekstrak informasi subjek, suku, dan nama file
        parts = orig_path.split('/')
        suku = parts[1]
        filename = parts[2]
        
        # Ekstrak nama subjek dari nama file (contoh: Fanza_Sunda_...)
        name_parts = filename.split('_')
        if len(name_parts) >= 2:
            subjek = name_parts[0]
            raw_suku = name_parts[1]
            # Path ke raw image
            raw_path = os.path.join(input_dir, raw_suku, subjek, filename)
        else:
            # Jika nama file tidak sesuai pola, gunakan gambar dari output saja
            raw_path = full_orig_path
        
        # Periksa apakah raw image tersedia
        if os.path.exists(raw_path):
            try:
                raw_image = Image.open(raw_path)
                with cols[0]:
                    st.write("Gambar Raw Asli:")
                    st.image(raw_image, use_column_width=True)
            except Exception as e:
                with cols[0]:
                    st.error(f"Tidak dapat menampilkan gambar raw: {e}")
        
        # Tampilkan gambar normalisasi
        if 'normalized' in img_info and img_info['normalized']:
            try:
                norm_path = os.path.join(augmented_dataset_dir, img_info['normalized'])
                if os.path.exists(norm_path):
                    norm_image = Image.open(norm_path)
                    with cols[1]:
                        st.write("Hasil Normalisasi:")
                        st.image(norm_image, use_column_width=True)
                else:
                    with cols[1]:
                        st.error(f"File normalisasi tidak ditemukan")
            except Exception as e:
                with cols[1]:
                    st.error(f"Error saat menampilkan gambar normalisasi")
        
        try:
            # Tampilkan contoh augmentasi (ambil yang pertama)
            if aug_paths and len(aug_paths) > 0:
                # Path lengkap ke gambar augmentasi
                full_aug_path = os.path.join(augmented_dataset_dir, aug_paths[0])
                if os.path.exists(full_aug_path):
                    aug_image = Image.open(full_aug_path)
                    with cols[2]:
                        st.write(f"Contoh Augmentasi:")
                        st.image(aug_image, use_column_width=True)
                else:
                    with cols[2]:
                        st.error(f"File augmentasi tidak ditemukan")
        except Exception as e:
            with cols[2]:
                st.error(f"Error saat menampilkan gambar augmentasi")
        
        # Tampilkan semua hasil augmentasi
        st.write("Semua Hasil Augmentasi:")
        
        # Gunakan grid untuk menampilkan semua hasil augmentasi
        num_cols = 3  # Jumlah kolom dalam grid
        num_rows = (len(aug_paths) + num_cols - 1) // num_cols  # Jumlah baris yang dibutuhkan
        
        for row in range(num_rows):
            aug_cols = st.columns(num_cols)
            for col in range(num_cols):
                idx = row * num_cols + col
                if idx < len(aug_paths):
                    aug_path = aug_paths[idx]
                    # Path lengkap ke gambar augmentasi
                    full_aug_path = os.path.join(augmented_dataset_dir, aug_path)
                    if os.path.exists(full_aug_path):
                        aug_image = Image.open(full_aug_path)
                        with aug_cols[col]:
                            st.write(f"Augmentasi #{idx+1}:")
                            st.image(aug_image, use_column_width=True)
                            # Tampilkan nama file
                            st.caption(os.path.basename(aug_path))
        
        # Tambahkan garis pemisah
        st.markdown("---")

# def validate_dataset_structure(input_dir):
#     """
#     Validasi struktur dataset: dataset/raw/<subjek>/<suku>/<file>.jpg
#     """
#     if not os.path.exists(input_dir):
#         st.error(f"Direktori tidak ditemukan: {input_dir}")
#         return False

#     total_subjects = 0
#     issue_detected = False
#     for subject in os.listdir(input_dir):
#         subject_path = os.path.join(input_dir, subject)
#         if not os.path.isdir(subject_path):
#             continue

#         for suku in os.listdir(subject_path):
#             suku_path = os.path.join(subject_path, suku)
#             if not os.path.isdir(suku_path):
#                 continue

#             images = [f for f in os.listdir(suku_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
#             if len(images) < 1:
#                 st.warning(f"Tidak ada gambar ditemukan di {suku_path}")
#                 issue_detected = True
#             else:
#                 total_subjects += 1

#     st.info(f"Total Subjek Terdeteksi: {total_subjects}")
#     return not issue_detected


import os
import streamlit as st

def validate_dataset_structure(base_dir):
    """
    Validasi struktur dataset:
    dataset/raw/<SUKU>/<SUBJEK>/<SUBJEK>_<SUKU>_<...>.jpg
    """
    st.header("📁 Validasi Struktur Dataset")

    errors = []
    warnings = []

    raw_dir = os.path.join(base_dir, 'raw')
    if not os.path.exists(raw_dir):
        errors.append(f"Direktori 'raw' tidak ditemukan: {raw_dir}")
        return False

    for suku in os.listdir(raw_dir):
        suku_path = os.path.join(raw_dir, suku)
        if not os.path.isdir(suku_path):
            continue

        for subjek in os.listdir(suku_path):
            subjek_path = os.path.join(suku_path, subjek)
            if not os.path.isdir(subjek_path):
                continue

            images = [
                f for f in os.listdir(subjek_path)
                if f.lower().endswith(('.jpg', '.jpeg', '.png'))
            ]

            if len(images) < 4:
                warnings.append(f"Subjek '{subjek}' dari suku '{suku}' hanya memiliki {len(images)} gambar (minimal 4).")

            for img in images:
                expected_prefix = f"{subjek}_{suku}_"
                if not img.startswith(expected_prefix):
                    errors.append(f"Format nama file tidak sesuai: {img}. Seharusnya diawali dengan '{expected_prefix}'")
                else:
                    name_no_ext = os.path.splitext(img)[0]
                    parts = name_no_ext.split("_")
                    if len(parts) != 6:
                        warnings.append(f"Format label file tidak lengkap: {img} (harus 6 bagian dipisah '_')")

    # Tampilkan hasil
    if errors:
        st.error("❌ Kesalahan Struktur Dataset:")
        for e in errors:
            st.error(f"- {e}")
    else:
        st.success("✅ Tidak ada kesalahan fatal dalam struktur dataset.")

    if warnings:
        st.warning("⚠️ Peringatan Struktur Dataset:")
        for w in warnings:
            st.warning(f"- {w}")

    return len(errors) == 0


def dataset_info_section():
    """
    Bagian informasi dataset
    """
    st.sidebar.header("Informasi Dataset")
    
    # Pilih direktori untuk divalidasi
    base_dir = st.sidebar.text_input(
        "Direktori Base Dataset", 
        value="dataset",
        help="Path ke direktori utama dataset"
    )
    
    # Tombol validasi
    if st.sidebar.button("Validasi Struktur Dataset"):
        validate_dataset_structure(base_dir)

def browse_raw_images():
    """
    Browser gambar raw
    """
    st.header("Browser Gambar Raw")
    
    # Pilih direktori raw
    raw_dir = st.text_input("Direktori Raw", value="dataset/raw")
    
    if not os.path.exists(raw_dir):
        st.error(f"Direktori tidak ditemukan: {raw_dir}")
        return
    
    # List subjek
    subjek_list = [d for d in os.listdir(raw_dir) if os.path.isdir(os.path.join(raw_dir, d))]
    
    if not subjek_list:
        st.warning("Tidak ada subjek yang ditemukan")
        return
    
    # Pilih subjek
    subjek = st.selectbox("Pilih Suku", subjek_list)
    
    # List suku untuk subjek yang dipilih
    subjek_path = os.path.join(raw_dir, subjek)
    suku_list = [d for d in os.listdir(subjek_path) if os.path.isdir(os.path.join(subjek_path, d))]
    
    if not suku_list:
        st.warning(f"Tidak ada suku untuk subjek {subjek}")
        return
    
    # Pilih suku
    suku = st.selectbox("Pilih Subjek", suku_list)
    
    # Tampilkan gambar
    suku_path = os.path.join(subjek_path, suku)
    images = [img for img in os.listdir(suku_path) if img.lower().endswith(('.png', '.jpg', '.jpeg', '.JPG'))]
    
    if not images:
        st.warning(f"Tidak ada gambar untuk subjek {subjek} dari suku {suku}")
        return
    
    # Tampilkan gambar dalam grid
    cols = st.columns(3)  # 3 kolom
    
    for i, img_name in enumerate(images):
        img_path = os.path.join(suku_path, img_name)
        img = Image.open(img_path)
        
        col_idx = i % 3
        with cols[col_idx]:
            st.image(img, caption=img_name, use_column_width=True)

if __name__ == "__main__":
    # Konfigurasi halaman Streamlit
    st.set_page_config(
        page_title="Face Ethnicity Recognition Dataset Preprocessing",
        page_icon=":camera:",
        layout="wide"
    )
    
    # Tambahkan navigasi di sidebar
    menu = st.sidebar.radio(
        "Menu Preprocessing",
        [
            "Preprocessing Dataset", 
            "Validasi Dataset",
            "Browser Gambar Raw"
        ]
    )
    
    # Tampilkan halaman sesuai pilihan
    if menu == "Preprocessing Dataset":
        main()
    elif menu == "Validasi Dataset":
        dataset_info_section()
    elif menu == "Browser Gambar Raw":
        browse_raw_images()

