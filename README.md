# Sistem Pengenalan Etnis Wajah Indonesia

## Pendahuluan

Sistem Pengenalan Etnis Wajah merupakan solusi teknologi komputer vision yang mengintegrasikan tiga algoritma canggih untuk analisis dan klasifikasi wajah: Haar Cascade Classifier untuk deteksi wajah, Jaringan Siamese untuk perbandingan kesamaan wajah, dan Jaringan Saraf Konvolusional (CNN) dengan Transfer Learning untuk klasifikasi etnis.

## Algoritma Utama

### 1. Deteksi Wajah: Haar Cascade Classifier

**Konsep Algoritma:**
- Dikembangkan oleh Paul Viola dan Michael Jones pada 2001
- Metode deteksi objek berbasis fitur Haar-like
- Menggunakan classifier bertingkat dengan konsep AdaBoost

**Cara Kerja:**
- Membagi gambar menjadi sub-jendela berukuran tetap
- Menghitung fitur Haar yang merepresentasikan perbedaan intensitas piksel
- Menggunakan classifier bertingkat untuk mengurangi waktu komputasi
- Menghasilkan kotak pembatas (bounding box) di sekitar wajah terdeteksi

**Kelebihan:**
- Komputasi cepat
- Akurasi baik untuk deteksi wajah frontal
- Membutuhkan sedikit memori

### 2. Perbandingan Kesamaan Wajah: Jaringan Siamese

**Konsep Algoritma:**
- Arsitektur jaringan saraf dengan cabang kembar
- Menggunakan bobot yang sama untuk membandingkan pasangan gambar
- Bertujuan memetakan gambar ke ruang vektor fitur dimensional rendah

**Cara Kerja:**
- Dua cabang jaringan dengan bobot yang sama
- Menerima dua gambar masukan
- Mengekstraksi fitur dari masing-masing gambar
- Menghitung jarak antara representasi fitur
- Menggunakan metrik jarak Euclidean atau Cosine

**Metrik Perbandingan:**
- Euclidean Distance: Jarak langsung antara dua titik vektor
- Cosine Similarity: Mengukur kosinus sudut antara dua vektor

### 3. Klasifikasi Etnis: CNN dengan Transfer Learning

**Konsep Algoritma:**
- Menggunakan arsitektur MobileNetV2 yang sudah dilatih
- Memanfaatkan transfer learning untuk adaptasi ke dataset etnis
- Menambahkan lapisan klasifikasi khusus untuk etnis Indonesia

**Cara Kerja:**
- Membekukan lapisan dasar MobileNetV2
- Menambahkan lapisan fully connected
- Fine-tuning pada dataset etnis Jawa, Sunda, Melayu
- Menggunakan teknik data augmentation untuk meningkatkan generalisasi

**Teknik Transfer Learning:**
- Feature Extraction: Menggunakan representasi fitur dari model pra-latih
- Fine-tuning: Melatih ulang sebagian lapisan dengan dataset spesifik

## Prasyarat Sistem

### Perangkat Keras
- Prosesor: Intel/AMD 64-bit dengan dukungan SSE4.2 atau lebih baru
- RAM: Minimal 8 GB (16 GB direkomendasikan)
- Ruang Penyimpanan: Minimal 5 GB 
- (Opsional) GPU CUDA-compatible untuk akselerasi pelatihan

### Perangkat Lunak
- Python 3.8 - 3.10 (Python 3.11 belum sepenuhnya kompatibel)
- pip (versi terbaru)
- git
- Virtual environment (venv/conda)

## Langkah Instalasi Terperinci

### 1. Persiapan Lingkungan Python

#### Windows
```powershell
# Buka Command Prompt atau PowerShell
# Pastikan Python sudah terinstal

# Periksa versi Python
python --version

# Buat direktori proyek
mkdir face-ethnicity-recognition
cd face-ethnicity-recognition

# Buat virtual environment
python -m venv venv

# Aktifkan virtual environment
venv\Scripts\activate
```

#### Linux/macOS
```bash
# Buka terminal
# Pastikan Python sudah terinstal

# Periksa versi Python
python3 --version

# Buat direktori proyek
mkdir face-ethnicity-recognition
cd face-ethnicity-recognition

# Buat virtual environment
python3 -m venv venv

# Aktifkan virtual environment
source venv/bin/activate
```

### 2. Kloning Repositori

```bash
git clone https://github.com/farhannm/face-ethnicity-recognition.git
cd face-ethnicity-recognition
```

### 3. Instalasi Dependensi

```bash
# Upgrade pip ke versi terbaru
pip install --upgrade pip

# Instal dependensi utama
pip install -r requirements.txt

# Instal tambahan untuk computer vision
pip install opencv-python-headless
pip install tensorflow
pip install streamlit
```

### 4. Persiapan Struktur Direktori

```bash
# Buat direktori yang diperlukan
mkdir -p models/weights
mkdir -p dataset/raw
mkdir -p dataset/final_dataset
mkdir -p dataset/augmented_dataset
```

### 5. Jalankan Aplikasi

```bash
# Jalankan aplikasi Streamlit
streamlit run app.py
```

## Pemecahan Masalah Umum

### Kesalahan Instalasi TensorFlow
```bash
# Jika mengalami masalah dengan TensorFlow
pip uninstall tensorflow
pip install tensorflow==2.13.0

# Untuk pengguna Mac M1/M2
pip install tensorflow-macos
```

### Kesalahan Dependensi
```bash
# Update pip dan dependensi
pip install --upgrade pip
pip install --upgrade -r requirements.txt
```

## Verifikasi Instalasi

```bash
# Periksa instalasi
python -c "import tensorflow as tf; print(tf.__version__)"
python -c "import cv2; print(cv2.__version__)"
streamlit hello
```

**Catatan Penting**: 
- Sistem bersifat eksperimental
- Akurasi tidak mutlak 100%
- Pertimbangkan aspek etika penggunaan
