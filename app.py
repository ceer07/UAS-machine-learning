"""
Aplikasi Streamlit - Klasifikasi Penyakit Daun Kacang
Menggunakan model EfficientNetB0 / DenseNet121 hasil transfer learning
(lihat notebook Klasifikasi_Penyakit_Daun_Kacang.ipynb).

Catatan preprocessing (harus sama persis dengan training):
- Gambar di-resize ke 224x224 (bilinear).
- Nilai piksel dibiarkan pada rentang 0..255 (float32), KARENA preprocessing
  (scaling EfficientNet, atau Rescaling -1..1 untuk DenseNet) sudah menjadi
  bagian dari arsitektur model yang disimpan. Jangan dinormalisasi lagi di sini.
"""

import json
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

# ---------------------------------------------------------------------------
# Konfigurasi
# ---------------------------------------------------------------------------
MODELS_DIR = Path(__file__).parent / "models"
IMG_SIZE = (224, 224)
DEFAULT_CLASS_NAMES = ["angular_leaf_spot", "bean_rust", "healthy"]

MODEL_FILES = {
    "EfficientNetB0": "efficientnetb0.keras",
    "DenseNet121": "densenet121.keras",
}

CLASS_LABELS_ID = {
    "angular_leaf_spot": "Bercak Daun Sudut (Angular Leaf Spot)",
    "bean_rust": "Karat Daun (Bean Rust)",
    "healthy": "Sehat (Healthy)",
}

st.set_page_config(
    page_title="Klasifikasi Penyakit Daun Kacang",
    page_icon="🌱",
    layout="centered",
)


# ---------------------------------------------------------------------------
# Loader (di-cache supaya model tidak dimuat ulang setiap interaksi)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_class_names():
    path = MODELS_DIR / "class_names.json"
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return DEFAULT_CLASS_NAMES


@st.cache_resource(show_spinner=True)
def load_model(model_filename: str):
    model_path = MODELS_DIR / model_filename
    if not model_path.exists():
        return None
    return tf.keras.models.load_model(model_path)


def preprocess_image(image: Image.Image) -> np.ndarray:
    image = image.convert("RGB").resize(IMG_SIZE)
    array = np.array(image, dtype=np.float32)  # tetap 0..255
    return np.expand_dims(array, axis=0)


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
def main():
    class_names = load_class_names()

    st.title("🌱 Klasifikasi Penyakit Daun Kacang")
    st.write(
        "Unggah foto daun kacang untuk memprediksi apakah daun **sehat**, "
        "terkena **bercak daun sudut (angular leaf spot)**, atau "
        "**karat daun (bean rust)** — menggunakan model transfer learning "
        "EfficientNetB0 / DenseNet121 (dataset Beans, Makerere AI Lab)."
    )

    with st.sidebar:
        st.header("Pengaturan")
        model_choice = st.selectbox("Pilih arsitektur model", list(MODEL_FILES.keys()))
        st.caption(
            "Kedua model dilatih dengan konfigurasi sama (ImageNet weights, "
            "fine-tuning 30 layer terakhir). Lihat `models/metrics.json` "
            "untuk perbandingan akurasi."
        )

    model = load_model(MODEL_FILES[model_choice])

    if model is None:
        st.error(
            f"File model `{MODEL_FILES[model_choice]}` tidak ditemukan di folder "
            f"`models/`. Jalankan notebook training sampai selesai, unduh "
            f"`bean_leaf_models_bundle.zip`, lalu ekstrak isinya ke folder `models/` "
            f"pada project ini (jangan ubah nama file)."
        )
        st.stop()

    uploaded_file = st.file_uploader(
        "Unggah gambar daun (.jpg / .jpeg / .png)", type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is None:
        st.info("Silakan unggah gambar daun kacang terlebih dahulu.")
        return

    image = Image.open(uploaded_file)
    st.image(image, caption="Gambar yang diunggah", use_column_width=True)

    with st.spinner("Memproses prediksi..."):
        batch = preprocess_image(image)
        probabilities = model.predict(batch, verbose=0)[0]

    predicted_idx = int(np.argmax(probabilities))
    predicted_label = class_names[predicted_idx]
    confidence = float(probabilities[predicted_idx])
    display_label = CLASS_LABELS_ID.get(predicted_label, predicted_label)

    if predicted_label == "healthy":
        st.success(f"Prediksi: **{display_label}** ({confidence:.2%} keyakinan)")
    else:
        st.warning(f"Prediksi: **{display_label}** ({confidence:.2%} keyakinan)")

    st.subheader("Probabilitas tiap kelas")
    prob_dict = {
        CLASS_LABELS_ID.get(name, name): float(prob)
        for name, prob in zip(class_names, probabilities)
    }
    st.bar_chart(prob_dict)

    st.caption(
        "⚠️ Hasil prediksi bersifat eksperimental (proyek akademik) dan tidak "
        "menggantikan diagnosis ahli pertanian/agronomi."
    )


if __name__ == "__main__":
    main()
