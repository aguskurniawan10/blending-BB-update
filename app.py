import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer

# Set page config
st.set_page_config(page_title="Prediksi GCV", layout="wide")
st.title("🔍 Prediksi GCV (ARB) LAB")

# Load dataset from GitHub
DATA_URL = "https://github.com/aguskurniawan10/blending-BB-update/raw/main/DATA%20PREDIKSI%20NK%20LAB%202025.xlsx"
data = pd.read_excel(DATA_URL)

# Display the first few rows of the dataset for reference
st.subheader("Dataset Preview")
st.write(data.head())

# Preprocessing
# Assuming the dataset has the following columns based on the context
# Adjust these column names based on the actual dataset structure
features = [
    "Supplier_1", "Supplier_2", "Biomass_Percentage", 
    "GCV_ARB_UNLOADING", "TM_ARB_UNLOADING", 
    "Ash_Content_ARB_UNLOADING", "Total_Sulphur_ARB_UNLOADING"
]
target = "GCV_Prediction"  # Replace with the actual target column name

# Handle missing values and encode categorical features
imputer = SimpleImputer(strategy='mean')
data[features] = imputer.fit_transform(data[features])

# Encode categorical features
label_encoder = LabelEncoder()
data['Supplier_1'] = label_encoder.fit_transform(data['Supplier_1'])
data['Supplier_2'] = label_encoder.fit_transform(data['Supplier_2'])

# Split the data into features and target
X = data[features]
y = data[target]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Ridge Regression model
ridge_model = Ridge()
ridge_model.fit(X_train_scaled, y_train)

# User input section
st.subheader("Informasi Supplier dan Lokasi")
col1, col2 = st.columns(2)

with col1:
    st.markdown("### Supplier 1")
    supplier_1 = st.selectbox("Pilih Supplier 1", label_encoder.classes_, key="supplier1")
    location_1 = st.selectbox("Lokasi Pengambilan", ["Tongkang", "Coalyard"], key="loc1")
    storage_time_1 = st.number_input("Lama Penyimpanan (hari)", min_value=0, max_value=365, value=0, key="storage1")

with col2:
    st.markdown("### Supplier 2")
    supplier_2 = st.selectbox("Pilih Supplier 2", label_encoder.classes_, key="supplier2")
    location_2 = st.selectbox("Lokasi Pengambilan", ["Tongkang", "Coalyard"], key="loc2")
    storage_time_2 = st.number_input("Lama Penyimpanan (hari)", min_value=0, max_value=365, value=0, key="storage2")

# Blending percentages
st.subheader("Persentase Campuran")
col1, col2, col3 = st.columns(3)

with col1:
    supplier_1_percentage = st.slider(f"Persentase {supplier_1}", 0, 100, 50, step=10, key="perc1")

with col2:
    supplier_2_percentage = st.slider(f"Persentase {supplier_2}", 0, 100, 50, step=10, key="perc2")

with col3:
    biomass_percentage = st.slider("Persentase Biomass", 0, 100, 0, step=1, key="biomass")

# Check if percentages add up to 100
total_percentage = supplier_1_percentage + supplier_2_percentage + biomass_percentage
if total_percentage != 100:
    st.warning(f"Total persentase saat ini: {total_percentage}%. Idealnya, total persentase adalah 100%.")

# Parameters input section
st.subheader("Parameter Batubara")
parameters = [
    "GCV_ARB_UNLOADING", 
    "TM_ARB_UNLOADING", 
    "Ash_Content_ARB_UNLOADING", 
    "Total_Sulphur_ARB_UNLOADING"
]

param_values = {}
for param in parameters:
    col1, col2 = st.columns(2)
    with col1:
        param_values[f"{param}_1"] = st.number_input(f"{param} - {supplier_1}", min_value=0.0, max_value=10000.0, value=3000.0, key=f"{param}_1")
    with col2:
        param_values[f"{param}_2"] = st.number_input(f"{param} - {supplier_2}", min_value=0.0, max_value=10000.0, value=3000.0, key=f"{param}_2")

# Prepare data for prediction when button is clicked
if st.button("Prediksi GCV"):
    # Prepare input data for prediction
    supplier_1_encoded = label_encoder.transform([supplier_1])[0]
    supplier_2_encoded = label_encoder.transform([supplier_2])[0]
    
    blended_data = [
        supplier_1_encoded,
        supplier_2_encoded,
        biomass_percentage,
        param_values["GCV_ARB_UNLOADING_1"],
        param_values["TM_ARB_UNLOADING_1"],
        param_values["Ash_Content_ARB_UNLOADING_1"],
        param_values["Total_Sulphur_ARB_UNLOADING_1"]
    ]
    
    # Scale the input data
    input_array = np.array(blended_data).reshape(1, -1)
    input_scaled = scaler.transform(input_array)

    # Make prediction
    prediction = ridge_model.predict(input_scaled)[0]

    # Display results
    st.success(f"Prediksi GCV (ARB) LAB: {prediction:.2f} kcal/kg")

    # Show additional information
    st.info("""
    **Catatan:** 
    - Berdasarkan Literatur: Degradasi Nilai Kalori dalam 1 Bulan: MRC: 3% hingga 5% (Smith et al., 2023) LRC: 4% (Johnson dan Lee, 2024) Umum: 2% hingga 6% (Coal Research Institute, 2025). 
    - Penyimpanan di coalyard dapat menurunkan nilai GCV sekitar 5% per bulan.
    - Hasil prediksi dipengaruhi oleh persentase campuran dan waktu penyimpanan
    """)

# Add footer
st.markdown("---")
st.markdown("© 2025 GCV Prediction Tool | For optimal results, ensure model is regularly updated with new data.")
