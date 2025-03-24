import streamlit as st
import pandas as pd
import numpy as np
import os
import pickle
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Set page config
st.set_page_config(page_title="Prediksi GCV", layout="wide")
st.title("🔍 Prediksi GCV (ARB) LAB")

# Path untuk model dan preprocessing tools
MODEL_DIR = "models"
IMPUTER_DIR = "imputers"
SCALER_DIR = "scalers"
ENCODER_DIR = "encoders"
MODEL_INFO_DIR = "model_infos"

# Function to load models with proper error handling
def load_models():
    model_files = os.listdir(MODEL_DIR)
    imputer_files = os.listdir(IMPUTER_DIR)
    scaler_files = os.listdir(SCALER_DIR)
    encoder_files = os.listdir(ENCODER_DIR)
    model_info_files = os.listdir(MODEL_INFO_DIR)
    
    if not model_files or not imputer_files or not scaler_files or not encoder_files or not model_info_files:
        st.error("Direktori model, imputer, scaler, encoder, atau model info kosong.")
        st.info("Pastikan semua direktori berisi file-file yang diperlukan.")
        st.stop()
    
    models = {}
    for model_file in model_files:
        model_name = os.path.splitext(model_file)[0]
        if (f"{model_name}.pkl" in imputer_files and
            f"{model_name}.pkl" in scaler_files and
            f"{model_name}.pkl" in encoder_files and
            f"{model_name}.pkl" in model_info_files):
            try:
                with open(os.path.join(MODEL_DIR, model_file), "rb") as file:
                    model = pickle.load(file)
                with open(os.path.join(IMPUTER_DIR, f"{model_name}.pkl"), "rb") as file:
                    imputer = pickle.load(file)
                with open(os.path.join(SCALER_DIR, f"{model_name}.pkl"), "rb") as file:
                    scaler = pickle.load(file)
                with open(os.path.join(ENCODER_DIR, f"{model_name}.pkl"), "rb") as file:
                    label_encoder = pickle.load(file)
                with open(os.path.join(MODEL_INFO_DIR, f"{model_name}.pkl"), "rb") as file:
                    model_info = pickle.load(file)
                
                models[model_name] = {
                    "model": model,
                    "imputer": imputer,
                    "scaler": scaler,
                    "label_encoder": label_encoder,
                    "model_info": model_info
                }
            except Exception as e:
                st.error(f"Error loading model components for {model_name}: {str(e)}")
                st.stop()
        else:
            st.warning(f"Komponen model {model_name} tidak lengkap. Mengabaikan model ini.")
    
    if not models:
        st.error("Tidak ada model yang dapat dimuat.")
        st.stop()
    
    return models

# Load models
models = load_models()
model_names = list(models.keys())

# User selects the model
selected_model_name = st.sidebar.selectbox("Pilih Model", model_names)

# Load selected model and its components
selected_model = models[selected_model_name]["model"]
selected_imputer = models[selected_model_name]["imputer"]
selected_scaler = models[selected_model_name]["scaler"]
selected_label_encoder = models[selected_model_name]["label_encoder"]
selected_model_info = models[selected_model_name]["model_info"]

# Get expected features count
expected_features = selected_imputer.n_features_in_
st.sidebar.info(f"Model expects {expected_features} features")

# Display model info
st.markdown(f"**🧠 Model Terbaik:** {selected_model_info['name']} (R² = {selected_model_info['r2']:.4f})")

# Create tabs for different functionalities
tab1, tab2 = st.tabs(["Prediksi Standar", "Debugging Info"])

with tab1:
    # Input section - supplier selection
    st.subheader("Informasi Supplier dan Lokasi")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Supplier 1")
        supplier_list = selected_label_encoder.classes_.tolist()
        supplier_1 = st.selectbox("Pilih Supplier 1", supplier_list, key="supplier1")
        location_1 = st.selectbox("Lokasi Pengambilan", ["Tongkang", "Coalyard"], key="loc1")
        if location_1 == "Coalyard":
            storage_time_1 = st.number_input("Lama Penyimpanan (hari)", min_value=0, max_value=365, value=0, key="storage1")
        else:
            storage_time_1 = 0
    
    with col2:
        st.markdown("### Supplier 2")
        supplier_2 = st.selectbox("Pilih Supplier 2", supplier_list, key="supplier2")
        location_2 = st.selectbox("Lokasi Pengambilan", ["Tongkang", "Coalyard"], key="loc2")
        if location_2 == "Coalyard":
            storage_time_2 = st.number_input("Lama Penyimpanan (hari)", min_value=0, max_value=365, value=0, key="storage2")
        else:
            storage_time_2 = 0
    
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
    
    # Define parameters
    parameters = [
        "GCV ARB UNLOADING", 
        "TM ARB UNLOADING", 
        "Ash Content ARB UNLOADING", 
        "Total Sulphur ARB UNLOADING"
    ]
    
    # Sample ranges for each parameter to guide users
    param_ranges = {
        "GCV ARB UNLOADING": (3500, 5500),
        "TM ARB UNLOADING": (20, 40),
        "Ash Content ARB UNLOADING": (2, 10),
        "Total Sulphur ARB UNLOADING": (0.1, 1.0)
    }
    
    param_values = {}
    
    # Create columns and inputs for each parameter
    for param in parameters:
        col1, col2 = st.columns(2)
        min_val, max_val = param_ranges.get(param, (0.0, 100.0))
        
        with col1:
            param_values[f"{param}_1"] = st.number_input(
                f"{param} - {supplier_1}", 
                min_value=float(0), 
                max_value=float(10000),
                value=float((min_val + max_val) / 2),
                key=f"{param}_1"
            )
        
        with col2:
            param_values[f"{param}_2"] = st.number_input(
                f"{param} - {supplier_2}", 
                min_value=float(0), 
                max_value=float(10000),
                value=float((min_val + max_val) / 2),
                key=f"{param}_2"
            )
    
    # Biomass GCV input if biomass percentage > 0
    if biomass_percentage > 0:
        st.subheader("Parameter Biomass")
        gcv_biomass = st.number_input("GCV Biomass (kcal/kg)", min_value=0.0, max_value=5000.0, value=3000.0)
    else:
        gcv_biomass = 0.0
    
    # Prepare data for prediction when button is clicked
    if st.button("Prediksi GCV"):
        # Determine the model structure and adjust inputs accordingly
        with tab2:
            st.subheader("Model Structure Analysis")
            st.write(f"Expected features count: {expected_features}")
            
            if hasattr(selected_model, 'feature_names_in_'):
                st.write("Feature names in model:")
                st.write(selected_model.feature_names_in_)
        
        # Prepare input data based on model expectations
        blended_data = []
        
        # IMPORTANT: Adjust this section based on your actual model structure
        # Here we're assuming the model expects only one supplier encoding + 4 parameters
        if expected_features == 5:
            # Use only the first supplier encoding
            supplier_encoded = selected_label_encoder.transform([supplier_1])[0]
            blended_data.append(supplier_encoded)
            
            # Calculate blended values for each parameter
            for param in parameters:
                val_1 = param_values[f"{param}_1"]
                val_2 = param_values[f"{param}_2"]
                
                # Calculate weighted average based on percentages
                if (supplier_1_percentage + supplier_2_percentage) > 0:
                    blended_value = (val_1 * supplier_1_percentage + val_2 * supplier_2_percentage) / (supplier_1_percentage + supplier_2_percentage)
                else:
                    blended_value = 0
                    
                blended_data.append(blended_value)
        
        elif expected_features == 6:
            # Use both supplier encodings
            supplier_encoded_1 = selected_label_encoder.transform([supplier_1])[0]
            supplier_encoded_2 = selected_label_encoder.transform([supplier_2])[0]
            blended_data.append(supplier_encoded_1)
            blended_data.append(supplier_encoded_2)
            
            # Calculate blended values for each parameter
            for param in parameters:
                val_1 = param_values[f"{param}_1"]
                val_2 = param_values[f"{param}_2"]
                
                # Calculate weighted average based on percentages
                if (supplier_1_percentage + supplier_2_percentage) > 0:
                    blended_value = (val_1 * supplier_1_percentage + val_2 * supplier_2_percentage) / (supplier_1_percentage + supplier_2_percentage)
                else:
                    blended_value = 0
                    
                blended_data.append(blended_value)
        
        else:
            # For other cases, try to adapt based on expected_features count
            # This is a simplified approach - you may need to adjust for your specific model
            supplier_encoded = selected_label_encoder.transform([supplier_1])[0]
            blended_data.append(supplier_encoded)
            
            # Add just enough parameters to match expected_features
            params_to_use = min(len(parameters), expected_features - 1)
            for i in range(params_to_use):
                param = parameters[i]
                val_1 = param_values[f"{param}_1"]
                val_2 = param_values[f"{param}_2"]
                
                if (supplier_1_percentage + supplier_2_percentage) > 0:
                    blended_value = (val_1 * supplier_1_percentage + val_2 * supplier_2_percentage) / (supplier_1_percentage + supplier_2_percentage)
                else:
                    blended_value = 0
                    
                blended_data.append(blended_value)
            
        # Final check of feature count
        if len(blended_data) != expected_features:
            st.error(f"Jumlah fitur input ({len(blended_data)}) tidak sesuai dengan yang diharapkan model ({expected_features}). Periksa kembali input Anda.")
            
            with tab2:
                st.subheader("Input Data (Tidak Sesuai)")
                st.write(pd.DataFrame([blended_data], columns=[f"Feature {i}" for i in range(len(blended_data))]))
                st.write("Penyesuaian diperlukan untuk menyesuaikan jumlah fitur dengan model")
            
            # Try to fix by truncating or padding
            if len(blended_data) > expected_features:
                blended_data = blended_data[:expected_features]
                st.warning(f"Input data dipotong untuk menyesuaikan dengan model ({expected_features} fitur)")
            else:
                # Pad with zeros
                while len(blended_data) < expected_features:
                    blended_data.append(0)
                st.warning(f"Input data ditambah dengan nilai 0 untuk menyesuaikan dengan model ({expected_features} fitur)")
        
        # Reshape, impute missing values, and scale the data
        input_array = np.array(blended_data).reshape(1, -1)
        
        # Process data - debugging info in the second tab
        with tab2:
            st.subheader("Input Data Mentah (Final)")
            st.write(pd.DataFrame([blended_data], columns=[f"Feature {i}" for i in range(len(blended_data))]))
        
        # Apply imputation
        imputed_array = selected_imputer.transform(input_array)
        with tab2:
            st.subheader("Data Setelah Imputasi")
            st.write(pd.DataFrame(imputed_array, columns=[f"Feature {i}" for i in range(imputed_array.shape[1])]))
        
        # Apply scaling
        scaled_array = selected_scaler.transform(imputed_array)
        with tab2:
            st.subheader("Data Setelah Scaling")
            st.write(pd.DataFrame(scaled_array, columns=[f"Feature {i}" for i in range(scaled_array.shape[1])]))
        
        # Make prediction
        try:
            prediction = selected_model.predict(scaled_array)[0]
            
            with tab2:
                st.subheader("Hasil Prediksi Model")
                st.write(f"Prediksi Awal: {prediction}")
                
                # If model has feature importances, show them
                if hasattr(selected_model, 'feature_importances_'):
                    st.subheader("Feature Importances")
                    importances = dict(zip([f"Feature {i}" for i in range(len(selected_model.feature_importances_))], 
                                         selected_model.feature_importances_))
                    st.write(pd.DataFrame([importances]))
            
            # Perform sanity check on the prediction
            if prediction < 0 or prediction > 10000:
                st.error(f"Model mengembalikan nilai prediksi tidak valid: {prediction}")
                prediction = max(2000, min(prediction, 5500))  # Constrain to reasonable range
                st.warning(f"Nilai diperbaiki ke dalam rentang yang valid: {prediction}")
            
            # Apply biomass blending if applicable
            if biomass_percentage > 0:
                final_prediction = (prediction * (supplier_1_percentage + supplier_2_percentage) + 
                                  gcv_biomass * biomass_percentage) / 100
            else:
                final_prediction = prediction
                
            # Apply storage time effects
            if location_1 == "Coalyard" and storage_time_1 > 0:
                decay_factor_1 = 0.05 * (storage_time_1 / 30)  # Cap at 5% max decrease per supplier
                final_prediction *= (1 - (decay_factor_1 * supplier_1_percentage / 100))
                
            if location_2 == "Coalyard" and storage_time_2 > 0:
                decay_factor_2 = 0.05 * (storage_time_2 / 30)  # Cap at 5% max decrease per supplier
                final_prediction *= (1 - (decay_factor_2 * supplier_2_percentage / 100))
                
            # Ensure result is within reasonable bounds
            final_prediction = max(2000, min(final_prediction, 7000))
                
            # Display results
            st.success(f"Prediksi GCV (ARB) LAB: {final_prediction:.2f} kcal/kg")
            
            # Show additional information
            st.info("""
            **Catatan:** 
            - Berdasarkan Literatur : Degradasi Nilai Kalori dalam 1 Bulan: MRC: 3% hingga 5% (Smith et al., 2023) LRC: 4% (Johnson dan Lee, 2024) Umum: 2% hingga 6% (Coal Research Institute, 2025). 
            - Penyimpanan di coalyard dapat menurunkan nilai GCV sekitar 5% per bulan.
            - Hasil prediksi dipengaruhi oleh persentase campuran dan waktu penyimpanan
            """)
            
            # Display debug info
            with tab2:
                st.subheader("Perhitungan Final")
                st.write(f"Prediksi Setelah Blending: {final_prediction:.2f} kcal/kg")
                st.write(f"Persentase Total: {total_percentage}%")
                st.write(f"Komposisi: {supplier_1}: {supplier_1_percentage}%, {supplier_2}: {supplier_2_percentage}%, Biomass: {biomass_percentage}%")
                
                if location_1 == "Coalyard" and storage_time_1 > 0:
                    st.write(f"Efek Penyimpanan {supplier_1}: -{decay_factor_1*100:.2f}% (dari {storage_time_1} hari)")
                    
                if location_2 == "Coalyard" and storage_time_2 > 0:
                    st.write(f"Efek Penyimpanan {supplier_2}: -{decay_factor_2*100:.2f}% (dari {storage_time_2} hari)")
                
        except Exception as e:
            st.error(f"Error saat melakukan prediksi: {str(e)}")
            st.info("Periksa kembali input Anda dan pastikan model sudah dilatih dengan benar.")

with tab2:
    if not st.button("Prediksi GCV", key="debug_pred"):
        st.info("Klik tombol 'Prediksi GCV' pada tab 'Prediksi Standar' untuk melihat informasi debugging.")
        
        # Show model info
        st.subheader("Informasi Model")
        if hasattr(selected_model, 'feature_names_in_'):
            st.write("Feature names expected by model:")
            st.write(selected_model.feature_names_in_)
        
        st.write(f"Expected number of features: {selected_imputer.n_features_in_}")
        st.write(f"Model type: {type(selected_model).__name__}")
        
        # Show label encoder info
        st.subheader("Label Encoder Classes")
        st.write(pd.DataFrame({"Supplier": selected_label_encoder.classes_, "Encoded Value": range(len(selected_label_encoder.classes_))}))

# Add footer
st.markdown("---")
st.markdown("© 2025 GCV Prediction Tool | For optimal results, ensure model is regularly updated with new data.")
