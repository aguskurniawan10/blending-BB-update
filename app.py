import streamlit as st
import pandas as pd
import numpy as np
import os
import pickle
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

# Path untuk model dan preprocessing tools
MODEL_PATH = "best_model.pkl"
IMPUTER_PATH = "imputer.pkl"
SCALER_PATH = "scaler.pkl"
ENCODER_PATH = "label_encoder.pkl"
BEST_MODEL_INFO_PATH = "best_model_info.pkl"

# Cek apakah model sudah ada, jika tidak tampilkan pesan error
if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, "rb") as file:
        best_model = pickle.load(file)
    with open(IMPUTER_PATH, "rb") as file:
        imputer = pickle.load(file)
    with open(SCALER_PATH, "rb") as file:
        scaler = pickle.load(file)
    with open(ENCODER_PATH, "rb") as file:
        label_encoder = pickle.load(file)
    with open(BEST_MODEL_INFO_PATH, "rb") as file:
        best_model_info = pickle.load(file)
else:
    st.error("Model tidak dapat dimuat! Pastikan file model tersedia.")
    st.stop()

st.set_page_config(page_title="Prediksi GCV", layout="wide")
st.title("🔍 Prediksi GCV (ARB) LAB")
st.markdown(f"**🧠 Model Terbaik:** {best_model_info['name']} (R² = {best_model_info['r2']:.4f})")

supplier_list = label_encoder.classes_.tolist()
supplier_1 = st.selectbox("Pilih Supplier 1", supplier_list)
supplier_2 = st.selectbox("Pilih Supplier 2", supplier_list)
location_1 = st.selectbox("Lokasi Pengambilan Supplier 1", ["Tongkang", "Coalyard"])
location_2 = st.selectbox("Lokasi Pengambilan Supplier 2", ["Tongkang", "Coalyard"])
storage_time_1 = st.number_input("Lama Penyimpanan di Coalyard (bulan) - Supplier 1", min_value=0, max_value=12, value=0)
storage_time_2 = st.number_input("Lama Penyimpanan di Coalyard (bulan) - Supplier 2", min_value=0, max_value=12, value=0)

supplier_1_percentage = st.slider("Persentase Supplier 1", 0, 100, 50)
supplier_2_percentage = st.slider("Persentase Supplier 2", 0, 100, 50)
biomass_percentage = st.slider("Persentase Biomass", 0, 100, 0)

data_input = []
st.subheader("Masukkan Nilai Parameter untuk Masing-Masing Sumber")
for label in ["GCV ARB UNLOADING", "TM ARB UNLOADING", "Ash Content ARB UNLOADING", "Total Sulphur ARB UNLOADING"]:
    col1, col2 = st.columns(2)
    with col1:
        val_1 = st.number_input(f"{label} Supplier 1", value=0.0)
    with col2:
        val_2 = st.number_input(f"{label} Supplier 2", value=0.0)
    data_input.append((val_1 * supplier_1_percentage + val_2 * supplier_2_percentage) / max(supplier_1_percentage + supplier_2_percentage, 1))

supplier_encoded_1 = label_encoder.transform([supplier_1])[0]
supplier_encoded_2 = label_encoder.transform([supplier_2])[0]
data_input.insert(0, supplier_encoded_1)
data_input.insert(1, supplier_encoded_2)

gcv_biomass = st.number_input("GCV Biomass", value=0.0)
data_input.append(gcv_biomass)

# Pastikan data_input memiliki jumlah fitur yang sesuai
expected_features = imputer.n_features_in_
st.write(f"Jumlah fitur yang dimasukkan: {len(data_input)}")
st.write(f"Jumlah fitur yang diharapkan oleh imputer: {expected_features}")

if len(data_input) != expected_features:
    st.error("Jumlah fitur input tidak sesuai dengan model. Periksa kembali input Anda.")
    st.stop()

data_input = np.array(data_input).reshape(1, -1)
data_input = imputer.transform(data_input)
data_input = scaler.transform(data_input)

if st.button("Prediksi GCV"):
    prediction = best_model.predict(data_input)[0]
    total_percentage = supplier_1_percentage + supplier_2_percentage + biomass_percentage
    final_prediction = prediction * (supplier_1_percentage + supplier_2_percentage) / max(total_percentage, 1)
    final_prediction += (gcv_biomass * biomass_percentage) / max(total_percentage, 1)
    
    if location_1 == "Coalyard":
        final_prediction *= (1 - 0.05 * storage_time_1)
    if location_2 == "Coalyard":
        final_prediction *= (1 - 0.05 * storage_time_2)
    
    st.success(f"Prediksi GCV (ARB) LAB: {final_prediction:.2f}")
