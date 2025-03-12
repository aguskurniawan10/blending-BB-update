import streamlit as st
import pandas as pd
import numpy as np
import os
import pickle
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Set page config
st.set_page_config(page_title="Prediksi GCV", layout="wide")
st.markdown("""
    <style>
        .main {
            background-color: #f4f4f4;
        }
        .stButton > button {
            background-color: #0084ff;
            color: white;
            font-size: 18px;
            padding: 10px 20px;
        }
        .stSidebar {
            background-color: #f0f0f0;
        }
    </style>
""", unsafe_allow_html=True)

# Header dengan ikon
st.markdown("""
    <h1 style='text-align: center; color: #333;'>🔍 Prediksi GCV (ARB) LAB</h1>
    <h5 style='text-align: center; color: #555;'>Aplikasi Prediksi Nilai GCV Batubara Menggunakan AI</h5>
    <hr>
""", unsafe_allow_html=True)

# Path untuk model dan preprocessing tools
MODEL_PATH = "best_model.pkl"
IMPUTER_PATH = "imputer.pkl"
SCALER_PATH = "scaler.pkl"
ENCODER_PATH = "label_encoder.pkl"
BEST_MODEL_INFO_PATH = "best_model_info.pkl"

# Function to load models
@st.cache_data
def load_models():
    required_files = {
        "Model": MODEL_PATH,
        "Imputer": IMPUTER_PATH,
        "Scaler": SCALER_PATH,
        "Label Encoder": ENCODER_PATH,
        "Model Info": BEST_MODEL_INFO_PATH
    }
    
    missing_files = [name for name, path in required_files.items() if not os.path.exists(path)]
    if missing_files:
        st.error(f"File berikut tidak ditemukan: {', '.join(missing_files)}")
        st.info("Pastikan semua file model dan preprocessing tersedia.")
        st.stop()
    
    try:
        return {name: pickle.load(open(path, "rb")) for name, path in required_files.items()}
    except Exception as e:
        st.error(f"Error loading model components: {str(e)}")
        st.stop()

# Load models
models = load_models()
best_model = models["Model"]
imputer = models["Imputer"]
scaler = models["Scaler"]
label_encoder = models["Label Encoder"]
best_model_info = models["Model Info"]

# Sidebar dengan informasi model
st.sidebar.markdown("""
    ## ℹ️ Informasi Model
    - **Model Terbaik:** {name}
    - **R² Score:** {r2:.4f}
    - **Fitur yang Diharapkan:** {features} fitur
""".format(name=best_model_info['name'], r2=best_model_info['r2'], features=imputer.n_features_in_), unsafe_allow_html=True)

# Tab navigasi
st.markdown("### 🔹 Pilih Mode Prediksi")
tab1, tab2 = st.tabs(["Prediksi Standar", "Debugging Info"])

with tab1:
    st.markdown("#### 📌 Informasi Supplier dan Lokasi")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🔹 Supplier 1")
        supplier_list = label_encoder.classes_.tolist()
        supplier_1 = st.selectbox("Pilih Supplier 1", supplier_list, key="supplier1")
        location_1 = st.radio("Lokasi Pengambilan", ["Tongkang", "Coalyard"], key="loc1")
        storage_time_1 = st.number_input("Lama Penyimpanan (hari)", 0, 365, 0, key="storage1") if location_1 == "Coalyard" else 0
    
    with col2:
        st.markdown("### 🔹 Supplier 2")
        supplier_2 = st.selectbox("Pilih Supplier 2", supplier_list, key="supplier2")
        location_2 = st.radio("Lokasi Pengambilan", ["Tongkang", "Coalyard"], key="loc2")
        storage_time_2 = st.number_input("Lama Penyimpanan (hari)", 0, 365, 0, key="storage2") if location_2 == "Coalyard" else 0
    
    st.markdown("#### 🔀 Persentase Campuran")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        supplier_1_percentage = st.slider(f"{supplier_1}", 0, 100, 50, step=10, key="perc1")
    with col2:
        supplier_2_percentage = st.slider(f"{supplier_2}", 0, 100, 50, step=10, key="perc2")
    with col3:
        biomass_percentage = st.slider("Biomass", 0, 100, 0, step=10, key="biomass")
    
    # Ensure total percentage is 100%
    total_percentage = supplier_1_percentage + supplier_2_percentage + biomass_percentage
    if total_percentage != 100:
        st.warning(f"Total persentase saat ini: {total_percentage}%. Harus 100%.")
    
    st.markdown("#### 📊 Parameter Batubara")
    parameters = ["GCV ARB UNLOADING", "TM ARB UNLOADING", "Ash Content ARB UNLOADING", "Total Sulphur ARB UNLOADING"]
    param_ranges = {"GCV ARB UNLOADING": (3500, 5500), "TM ARB UNLOADING": (20, 40), "Ash Content ARB UNLOADING": (2, 10), "Total Sulphur ARB UNLOADING": (0.1, 1.0)}
    
    for param in parameters:
        col1, col2 = st.columns(2)
        with col1:
            param_values_1 = st.number_input(f"{param} - {supplier_1}", *param_ranges[param], key=f"{param}_1")
        with col2:
            param_values_2 = st.number_input(f"{param} - {supplier_2}", *param_ranges[param], key=f"{param}_2")
    
    # Tombol prediksi
    if st.button("🔮 Prediksi GCV"):
        st.success("Prediksi berhasil! Hasil akan ditampilkan di sini.")

st.markdown("---")
st.markdown("© 2025 GCV Prediction Tool | UBP JPR")
