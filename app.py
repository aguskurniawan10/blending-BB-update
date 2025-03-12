import streamlit as st
import pandas as pd
import numpy as np
import os
import pickle
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score

# Path untuk model dan preprocessing tools
MODEL_PATH = "best_model.pkl"
IMPUTER_PATH = "imputer.pkl"
SCALER_PATH = "scaler.pkl"
ENCODER_PATH = "label_encoder.pkl"
BEST_MODEL_INFO_PATH = "best_model_info.pkl"

# Fungsi untuk melatih dan menyimpan model
def train_and_save_model():
    df = pd.read_excel("https://github.com/aguskurniawan10/prediksiNKLabUBPJPR/raw/main/DATA%20PREDIKSI%20NK%20LAB%202025.xlsx")
    df.columns = df.columns.str.strip()

    required_columns = ['Suppliers', 'GCV ARB UNLOADING', 'TM ARB UNLOADING', 
                        'Ash Content ARB UNLOADING', 'Total Sulphur ARB UNLOADING', 'GCV (ARB) LAB']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Kolom '{col}' tidak ditemukan dalam dataset!")

    label_encoder = LabelEncoder()
    df['Suppliers'] = label_encoder.fit_transform(df['Suppliers'])

    X = df[['Suppliers', 'GCV ARB UNLOADING', 'TM ARB UNLOADING', 
            'Ash Content ARB UNLOADING', 'Total Sulphur ARB UNLOADING']]
    y = df['GCV (ARB) LAB']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    imputer = SimpleImputer(strategy='median')
    X_train_imputed = imputer.fit_transform(X_train)
    X_test_imputed = imputer.transform(X_test)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imputed)
    X_test_scaled = scaler.transform(X_test_imputed)

    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=1.0),
        'Lasso Regression': Lasso(alpha=1.0),
        'Decision Tree': DecisionTreeRegressor(random_state=42),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Support Vector Regression': SVR(kernel='rbf')
    }

    best_model = None
    best_score = float('-inf')
    best_model_name = ""
    
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        r2 = r2_score(y_test, y_pred)
        if r2 > best_score:
            best_score = r2
            best_model = model
            best_model_name = name

    with open(MODEL_PATH, "wb") as file:
        pickle.dump(best_model, file)
    with open(IMPUTER_PATH, "wb") as file:
        pickle.dump(imputer, file)
    with open(SCALER_PATH, "wb") as file:
        pickle.dump(scaler, file)
    with open(ENCODER_PATH, "wb") as file:
        pickle.dump(label_encoder, file)
    with open(BEST_MODEL_INFO_PATH, "wb") as file:
        pickle.dump({"name": best_model_name, "r2": best_score}, file)

# Cek apakah model sudah ada, jika tidak maka latih ulang
if not os.path.exists(MODEL_PATH):
    st.warning("Model belum ditemukan! Melatih ulang model...")
    train_and_save_model()

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

if st.button("Prediksi GCV"):
    prediction = best_model.predict([data_input])[0]
    if location_1 == "Coalyard":
        prediction *= (1 - 0.05 * storage_time_1)
    if location_2 == "Coalyard":
        prediction *= (1 - 0.05 * storage_time_2)
    st.success(f"Prediksi GCV (ARB) LAB: {prediction:.2f}")
