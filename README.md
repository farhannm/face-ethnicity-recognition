# Sistem Pengenalan Etnis Wajah Indonesia

## Fitur dan Algoritma

### Main (Utama)

#### 1. Face Similarity
Memungkinkan sistem untuk mengidentifikasi dan membandingkan wajah untuk menentukan apakah dua gambar wajah yang berbeda berasal dari orang yang sama.

**Teknologi:**
- FaceNet Pre-train Models

#### 2. Deteksi Suku/Etnis
Mengklasifikasikan wajah seseorang ke dalam kategori suku/etnis berdasarkan fitur wajah menggunakan teknik computer vision.

**Teknologi:**
- MTCNN (deteksi wajah)
- FaceNet Pre-train Models
- Classifiers (KNN dan SVM)

### Pendamping

#### 1. Gender Detection
Memungkinkan sistem untuk mengidentifikasi jenis kelamin dari seseorang berdasarkan citra wajahnya. Sistem menganalisis struktur wajah untuk mengklasifikasikan gender secara real-time, lengkap dengan tingkat kepercayaan (confidence score).

**Teknologi:**
- Convolutional Neural Network (CNN) dengan pre-train model DeepFace

#### 2. Age Estimation
Memperkirakan usia seseorang berdasarkan karakteristik visual dari wajahnya. Sistem tidak hanya melihat ciri-ciri yang mencolok seperti kerutan atau tekstur kulit, tetapi juga memanfaatkan fitur-fitur halus yang diperoleh dari lapisan-lapisan model deep learning untuk menghasilkan estimasi usia yang lebih akurat.

**Teknologi:**
- Convolutional Neural Network (CNN) dengan pre-train model DeepFace

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

# Buat virtual environment
python3 -m venv venv

# Aktifkan virtual environment
source venv/bin/activate
```

### 2. Clone Repositori

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
```

### 4. Jalankan Aplikasi

```bash
# Jalankan aplikasi Streamlit
streamlit run app.py
```

**Catatan Penting**: 
- Sistem bersifat eksperimental
- Akurasi tidak mutlak 100%
- Pertimbangkan aspek etika penggunaan

## Anggota Kelompok

- Farhan Maulana - 231511044
- Indah Ratu Pramudita - 2311511050
- Nazla Kayla - 231511057

**JTK 2023 - Politeknik Negeri Bandung**
