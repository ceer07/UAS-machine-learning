# 🌱 Klasifikasi Penyakit Daun Kacang

Aplikasi web untuk mengklasifikasikan daun kacang ke dalam tiga kelas —
`angular_leaf_spot`, `bean_rust`, `healthy` — menggunakan transfer learning
EfficientNetB0 dan DenseNet121 (dataset [Beans, Makerere AI Lab](https://github.com/AI-Lab-Makerere/ibean)).

Notebook training ada di `notebooks/Klasifikasi_Penyakit_Daun_Kacang.ipynb`.
Aplikasi frontend dibangun dengan [Streamlit](https://streamlit.io).

## Struktur proyek

```
bean-leaf-classifier/
├── app.py                 # Aplikasi Streamlit (frontend + inference)
├── requirements.txt       # Dependensi Python
├── .gitignore
├── README.md
├── notebooks/
│   └── Klasifikasi_Penyakit_Daun_Kacang.ipynb
└── models/                 # Diisi manual, lihat langkah 2 di bawah
    ├── efficientnetb0.keras
    ├── densenet121.keras
    ├── class_names.json
    ├── metrics.json
    └── model_manifest.json
```

## 1. Jalankan notebook training

Buka notebook di Google Colab (GPU T4 disarankan), jalankan sampai selesai.
Di akhir notebook akan ter-download `bean_leaf_models_bundle.zip`.

## 2. Siapkan folder `models/`

Ekstrak isi `bean_leaf_models_bundle.zip` ke folder `models/` di project ini.
**Jangan ubah nama file** — `app.py` memuat file dengan nama persis:
`efficientnetb0.keras`, `densenet121.keras`, `class_names.json`.

## 3. Jalankan lokal (opsional, untuk tes sebelum deploy)

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Buka `http://localhost:8501` di browser.

## 4. Push ke GitHub

```bash
git init
git add .
git commit -m "Initial commit: bean leaf disease classifier"
git branch -M main
git remote add origin https://github.com/<username>/<repo-name>.git
git push -u origin main
```

### ⚠️ Model besar → gunakan Git LFS

File `.keras` biasanya berukuran puluhan MB. GitHub menolak file di atas
100 MB dan akan memperingatkan di atas 50 MB. Jika model Anda mendekati
batas itu, aktifkan Git LFS **sebelum** commit model:

```bash
git lfs install
git lfs track "models/*.keras"
git add .gitattributes
git add models/*.keras
git commit -m "Track model files with Git LFS"
```

Alternatif lain: upload model ke [Hugging Face Hub](https://huggingface.co/models)
atau Google Drive, lalu modifikasi `app.py` agar men-download model saat
pertama kali dijalankan (menggunakan `huggingface_hub.hf_hub_download` atau
`gdown`), sehingga model tidak perlu ikut masuk ke repo GitHub sama sekali.
Ini juga membantu kalau repo jadi terlalu besar untuk free tier Streamlit Cloud.

## 5. Deploy ke Streamlit Community Cloud

1. Buka [share.streamlit.io](https://share.streamlit.io) dan login dengan akun GitHub.
2. Klik **New app** → pilih repository dan branch (`main`).
3. Isi **Main file path**: `app.py`.
4. Klik **Deploy**.

Build pertama akan memakan waktu beberapa menit karena menginstall TensorFlow.

### Catatan keterbatasan resource (free tier)

Free tier Streamlit Community Cloud punya batas RAM ~1 GB. Memuat dua model
TensorFlow (EfficientNetB0 + DenseNet121) sekaligus bisa berat. Beberapa opsi
jika aplikasi lambat/crash:

- Muat model hanya saat dipilih di dropdown (`app.py` sudah melakukan ini
  lewat `st.cache_resource`, jadi model yang tidak dipilih tidak ikut dimuat).
- Gunakan `tensorflow-cpu` (lebih ringan dari `tensorflow` penuh) — sudah
  diset di `requirements.txt`.
- Konversi model ke TensorFlow Lite (`.tflite`) untuk inference yang lebih
  ringan, jika performa masih jadi masalah.

## Model & Metrik

Lihat `models/metrics.json` dan `models/model_comparison.csv` (hasil dari
notebook) untuk perbandingan akurasi, precision, recall, F1 antara
EfficientNetB0 dan DenseNet121.

## Sitasi dataset

Makerere AI Lab, "Bean disease dataset," January 2020.
https://github.com/AI-Lab-Makerere/ibean/
