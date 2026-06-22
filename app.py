import streamlit as st
import cv2
import numpy as np
import os
import time
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

# ==========================================
# KONFIGURASI HALAMAN
# ==========================================
st.set_page_config(
    page_title="Identifikasi Kemiripan Wajah", 
    page_icon="🕵️‍♂️", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

IMG_SIZE = (100, 100)
DATASET_PATH = "dataset"

# --- FUNGSI PREPROCESSING & FACE DETECTION ---
def detect_and_crop_face(image_bytes, return_visual=False):
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        return (None, None) if return_visual else None
        
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    
    if len(faces) == 0:
        return (None, None) if return_visual else None
        
    x, y, w, h = faces[0]
    face_crop = gray[y:y+h, x:x+w]
    face_resized = cv2.resize(face_crop, IMG_SIZE)
    
    face_normalized = face_resized / 255.0
    
    if return_visual:
        return face_normalized.flatten(), face_resized
    return face_normalized.flatten()

# --- FUNGSI LOAD DATABASE ---
def load_dataset(dataset_path):
    X, labels, img_paths = [], [], []
    if not os.path.exists(dataset_path):
        return None, None, None
        
    for person_name in os.listdir(dataset_path):
        person_folder = os.path.join(dataset_path, person_name)
        if not os.path.isdir(person_folder):
            continue
        for filename in os.listdir(person_folder):
            if filename.lower().endswith((".jpg", ".jpeg", ".png")):
                image_path = os.path.join(person_folder, filename)
                with open(image_path, "rb") as f:
                    file_bytes = f.read()
                vector = detect_and_crop_face(file_bytes)
                if vector is not None:
                    X.append(vector)
                    labels.append(person_name) # Menyimpan nama folder sebagai identitas
                    img_paths.append(image_path)
    return np.array(X), np.array(labels), img_paths

# ==========================================
# ANTARMUKA WEB
# ==========================================
st.markdown("<h1 style='text-align: center;'>👶 ➡️ 🧑 Identifikasi Wajah (Kelompok)</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: gray;'>Sistem Pengenalan Wajah dengan PCA & SVD (Mendukung Banyak Orang)</p>", unsafe_allow_html=True)
st.markdown("---")

# ==========================================
# TAHAP 1: DATA LATIH (DATABASE KELOMPOK)
# ==========================================
st.header("🗂️ A. Tahap Data Latih (Database Wajah)")

if st.button("🚀 Eksekusi Tahap Data Latih", use_container_width=True):
    with st.status("⚙️ Menjalankan langkah-langkah Data Latih...", expanded=True) as status:
        st.write("✔️ 1. Mengumpulkan gambar wajah dari folder dataset kelompok...")
        time.sleep(0.5)
        X, labels, img_paths = load_dataset(DATASET_PATH)
        
        if X is not None and len(X) > 0:
            st.write("✔️ 2. Mendeteksi dan crop area wajah (Haar Cascade)...")
            st.write("✔️ 3. Mengubah ke grayscale...")
            st.write("✔️ 4. Resize ke ukuran yang sama (100x100)...")
            st.write("✔️ 5. Flatten menjadi vektor 1D...")
            time.sleep(0.5)
            
            st.write(f"✔️ 6. Bentuk matriks data X (Ukuran: {X.shape})...")
            
            # Menyimpan label (nama teman) ke dalam memory sistem
            st.session_state['labels'] = labels
            
            if len(X) == 1:
                st.session_state['use_pca'] = False
                st.session_state['X_fit'] = X 
                st.write("⚠️ *Hanya 1 foto terdeteksi. Melewati PCA.*")
            else:
                st.write("✔️ 7. Lakukan perhitungan PCA menggunakan SVD...")
                st.session_state['use_pca'] = True
                n_comp = min(50, len(X) - 1)
                pca = PCA(n_components=n_comp)
                X_pca = pca.fit_transform(X)
                
                st.session_state['pca'] = pca
                st.session_state['X_fit'] = X_pca
                
            st.write("✔️ 8. Simpan model PCA dan representasi wajah database tersebut.")
            time.sleep(0.5)
            status.update(label="Tahap Data Latih Selesai!", state="complete", expanded=False)
            
            # Menampilkan daftar orang yang berhasil direkam
            unique_names = np.unique(labels)
            st.success(f"Berhasil merekam {len(X)} wajah dari {len(unique_names)} orang: {', '.join(unique_names)}")
        else:
            status.update(label="Gagal mendeteksi dataset!", state="error", expanded=True)
            st.error("Folder kosong atau wajah tidak ditemukan.")

st.markdown("<br>", unsafe_allow_html=True)

# ==========================================
# TAHAP 2: DATA UJI & KEPUTUSAN IDENTIFIKASI
# ==========================================
st.header("🔍 B. Tahap Data Uji (Proses Mengidentifikasi)")

col_left, col_right = st.columns([1, 1.5])

with col_left:
    st.write("**Masukkan gambar wajah baru (Foto Masa Kini):**")
    img_uji = st.file_uploader("Unggah foto...", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
    threshold = st.slider("Ambang Batas (Threshold)", 0.0, 1.0, 0.75, 0.05)

with col_right:
    if img_uji:
        if 'X_fit' not in st.session_state:
            st.error("⚠️ Selesaikan 'Tahap Data Latih' terlebih dahulu di atas!")
        else:
            with st.status("⚙️ Menjalankan langkah-langkah Data Uji...", expanded=True) as status_uji:
                st.write("✔️ 1-5. Memproses (Deteksi, Crop, Grayscale, Resize, Flatten)...")
                
                face_test, visual_crop = detect_and_crop_face(img_uji.read(), return_visual=True)
                time.sleep(0.5)

                if face_test is None:
                    status_uji.update(label="Wajah Tidak Terdeteksi!", state="error")
                    st.error("Wajah tidak terdeteksi pada foto.")
                else:
                    st.write("✔️ 6. Proyeksikan vektor wajah baru ke ruang PCA...")
                    X_fit = st.session_state['X_fit']
                    labels = st.session_state['labels']
                    
                    if st.session_state.get('use_pca', False):
                        pca = st.session_state['pca']
                        face_test_processed = pca.transform(face_test.reshape(1, -1))
                    else:
                        face_test_processed = face_test.reshape(1, -1)
                        
                    st.write("✔️ 7. Hitung nilai similarity (kemiripan) dengan seluruh database...")
                    similarities = cosine_similarity(face_test_processed, X_fit)[0]
                    
                    # LOGIKA BARU: Mengambil indeks dengan nilai tertinggi untuk mengetahui nama pemilik wajah
                    best_index = np.argmax(similarities)
                    best_sim = similarities[best_index]
                    best_label = labels[best_index]
                    
                    st.write("✔️ 8. Ambil keputusan akhir berdasarkan threshold.")
                    time.sleep(0.5)
                    status_uji.update(label="Tahap Data Uji Selesai!", state="complete", expanded=True)

            # --- TAMPILAN KEPUTUSAN AKHIR ---
            st.markdown("### 📋 Hasil Identifikasi")
            
            with st.expander("👁️ Lihat Bukti Preprocessing Wajah", expanded=True):
                st.image(visual_crop, caption="Wajah Uji (Cropped)", width=150)
            
            st.metric(label="Skor Kemiripan Tertinggi", value=f"{best_sim:.4f}")
            
            # Keputusan Akhir
            if best_sim >= threshold:
                st.success(f"### 🟢 KEPUTUSAN: WAJAH DIKENALI\nSistem mengidentifikasi bahwa foto masa kini ini sangat mirip dengan **{best_label.upper()}** (Skor melewati ambang batas {threshold}).")
            else:
                st.error(f"### 🔴 KEPUTUSAN: TIDAK DIKENAL\nSistem tidak dapat mengenali wajah ini atau perubahan strukturnya terlalu drastis (Skor $<$ Threshold {threshold}).")
    else:
        st.info("👈 Silakan unggah foto masa kini di sebelah kiri untuk memulai pengujian.")