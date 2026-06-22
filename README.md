# 🧮 Sistem Kemiripan Wajah Berbasis PCA & SVD menggunakan Streamlit

Aplikasi web interaktif untuk mendeteksi kemiripan wajah (1 vs 1) dan mengidentifikasi wajah berdasarkan database foto yang tersedia. Aplikasi ini menggunakan **Haar Cascade** untuk deteksi wajah awal, serta kombinasi **Principal Component Analysis (PCA)** dan **Singular Value Decomposition (SVD)** untuk reduksi dimensi fitur wajah.

## 🚀 Fitur Utama
- **Auto Face Detection & Crop:** Memotong area wajah secara otomatis menggunakan OpenCV Haar Cascade sebelum diekstrak fiturnya.
- **Deteksi Kemiripan (1 vs 1):** Membandingkan dua gambar wajah secara langsung menggunakan *Cosine Similarity*.
- **Identifikasi Wajah:** Mencari dan mencocokkan wajah input dengan database wajah yang terdaftar di sistem.
- **Antarmuka Modern:** Dibangun menggunakan Streamlit sehingga mudah digunakan dan interaktif.

## 📁 Struktur Project
```text
deteksi-wajah-pca/
├── .streamlit/
│   └── config.toml
├── dataset/
│   └── [nama_orang]/
├── app.py
├── requirements.txt
└── README.md