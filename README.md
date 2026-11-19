# 🌱 Klasifikasi Penyakit Daun Kacang — Panduan Menjalankan di Lokal

Aplikasi web untuk mengklasifikasikan daun kacang ke dalam tiga kelas —
`angular_leaf_spot`, `bean_rust`, `healthy` — menggunakan model transfer
learning EfficientNetB0 dan DenseNet121, dengan frontend [Streamlit](https://streamlit.io).

## Struktur proyek

```
bean-leaf-classifier/
├── app.py                 # Aplikasi Streamlit (frontend + inference)
├── requirements.txt       # Dependensi Python
├── notebooks/
│   └── Klasifikasi_Penyakit_Daun_Kacang.ipynb
└── models/                 # Harus diisi manual (lihat langkah 2)
    ├── efficientnetb0.keras
    ├── densenet121.keras
    └── class_names.json
```

## Prasyarat

- **Python 3.11 atau 3.12** (TensorFlow belum mendukung Python 3.13 ke atas).
  Cek versi Python Anda dengan:
  ```
  python --version
  ```
  Jika Python 3.13+, install Python 3.11/3.12 dari [python.org](https://www.python.org/downloads/windows/)
  tanpa perlu menghapus versi yang sudah ada (bisa berdampingan).

## Langkah 1 — Jalankan notebook training (jika belum punya model)

Buka `notebooks/Klasifikasi_Penyakit_Daun_Kacang.ipynb` di Google Colab
(disarankan pakai GPU T4), jalankan semua cell sampai selesai. Di akhir
notebook akan ter-download `bean_leaf_models_bundle.zip`.

## Langkah 2 — Taruh model ke folder `models/`

Ekstrak isi `bean_leaf_models_bundle.zip` ke folder `models/` pada project
ini. **Jangan ubah nama file** — `app.py` memuat file dengan nama persis:
`efficientnetb0.keras`, `densenet121.keras`, `class_names.json`.

## Langkah 3 — Buat virtual environment

Dari folder root project, jalankan:

```bash
# Jika punya beberapa versi Python di Windows, pilih versi spesifik:
py -3.12 -m venv venv

# Aktifkan venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # macOS/Linux
```

Pastikan prompt terminal menampilkan `(venv)` di depan, dan `python --version`
menunjukkan 3.11.x atau 3.12.x.

## Langkah 4 — Install dependensi

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

Isi `requirements.txt` yang sudah teruji kompatibel:

```
streamlit==1.39.0
tensorflow==2.17.0
pillow==10.4.0
numpy==1.26.4
pyarrow==15.0.2
```

> Catatan: `numpy` sengaja tetap di versi 1.x karena TensorFlow 2.17.0 belum
> mendukung NumPy 2.0, dan `pyarrow` di-pin ke 15.0.2 agar tetap kompatibel
> dengan NumPy 1.x (versi `pyarrow` terbaru mewajibkan NumPy ≥2.0).

## Langkah 5 — Jalankan aplikasi

```bash
streamlit run app.py
```

Browser akan otomatis terbuka ke `http://localhost:8501`. Upload gambar daun
kacang (.jpg/.jpeg/.png), pilih arsitektur model di sidebar, dan lihat hasil
prediksi beserta grafik probabilitas tiap kelas.

## Troubleshooting umum

| Error | Penyebab | Solusi |
|---|---|---|
| `No matching distribution found for tensorflow==2.17.0` | Versi Python tidak didukung TensorFlow (mis. 3.13+) | Gunakan Python 3.11/3.12 (lihat Prasyarat) |
| `ImageMixin.image() got an unexpected keyword argument 'use_container_width'` | Versi Streamlit terpasang lebih lama dari yang mendukung parameter ini | Pastikan `app.py` memakai `use_column_width=True`, bukan `use_container_width=True` |
| `ImportError: pyarrow requires NumPy 2.0 or newer, found 1.26.4` | Versi `pyarrow` terbaru butuh NumPy 2.0+, padahal TensorFlow 2.17.0 belum mendukung NumPy 2.0 | Pin `pyarrow==15.0.2` di `requirements.txt` (bukan menaikkan NumPy) |
| `Cannot install numpy==2.0.2 and tensorflow-intel==2.17.0 because these package versions have conflicting dependencies` | Mencoba memaksa NumPy 2.0 padahal TensorFlow 2.17.0 memerlukan NumPy <2.0 | Kembalikan `numpy==1.26.4`, atasi konflik `pyarrow` seperti baris di atas |
| Model error / file not found di folder `models/` | File `.keras` belum ditaruh atau nama file salah | Ulangi Langkah 2, pastikan nama file persis sesuai yang dimuat `app.py` |

## Sitasi dataset

Makerere AI Lab, "Bean disease dataset," January 2020.
https://github.com/AI-Lab-Makerere/ibean/