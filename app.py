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
MODEL_PATH = "best_model.pkl"
IMPUTER_PATH = "imputer.pkl"
SCALER_PATH = "scaler.pkl"
ENCODER_PATH = "label_encoder.pkl"
BEST_MODEL_INFO_PATH = "best_model_info.pkl"

# Function to load models with proper error handling
def load_models():
    required_files = {
        "Model": MODEL_PATH,
        "Imputer": IMPUTER_PATH,
        "Scaler": SCALER_PATH,
        "Label Encoder": ENCODER_PATH,
        "Model Info": BEST_MODEL_INFO_PATH
    }
    
    missing_files = []
    for name, path in required_files.items():
        if not os.path.exists(path):
            missing_files.append(name)
    
    if missing_files:
        st.error(f"File-file berikut tidak ditemukan: {', '.join(missing_files)}")
        st.info("Pastikan semua file model dan preprocessing tersedia di direktori yang sama dengan aplikasi.")
        st.stop()
    
    model_components = {}
    try:
        with open(MODEL_PATH, "rb") as file:
            model_components["best_model"] = pickle.load(file)
        with open(IMPUTER_PATH, "rb") as file:
            model_components["imputer"] = pickle.load(file)
        with open(SCALER_PATH, "rb") as file:
            model_components["scaler"] = pickle.load(file)
        with open(ENCODER_PATH, "rb") as file:
            model_components["label_encoder"] = pickle.load(file)
        with open(BEST_MODEL_INFO_PATH, "rb") as file:
            model_components["best_model_info"] = pickle.load(file)
        return model_components
    except Exception as e:
        st.error(f"Error loading model components: {str(e)}")
        st.stop()

# Load models
models = load_models()
best_model = models["best_model"]
imputer = models["imputer"]
scaler = models["scaler"]
label_encoder = models["label_encoder"]
best_model_info = models["best_model_info"]

# Get expected features count
expected_features = imputer.n_features_in_
st.sidebar.info(f"Model expects {expected_features} features")

# Display model info
st.markdown(f"**🧠 Model Terbaik:** {best_model_info['name']} (R² = {best_model_info['r2']:.4f})")

# Create tabs for different functionalities
tab1, tab2 = st.tabs(["Prediksi Standar", "Debugging Info"])

with tab1:
    # Input section - supplier selection
    st.subheader("Informasi Supplier dan Lokasi")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Supplier 1")
        supplier_list = label_encoder.classes_.tolist()
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
        supplier_1_percentage = st.slider(f"Persentase {supplier_1}", 0, 100, 50, key="perc1")
    
    with col2:
        supplier_2_percentage = st.slider(f"Persentase {supplier_2}", 0, 100, 50, key="perc2")
    
    with col3:
        biomass_percentage = st.slider("Persentase Biomass", 0, 100, 0, key="biomass")
    
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
            
            if hasattr(best_model, 'feature_names_in_'):
                st.write("Feature names in model:")
                st.write(best_model.feature_names_in_)
        
        # Prepare input data based on model expectations
        blended_data = []
        
        # IMPORTANT: Adjust this section based on your actual model structure
        # Here we're assuming the model expects only one supplier encoding + 4 parameters
        if expected_features == 5:
            # Use only the first supplier encoding
            supplier_encoded = label_encoder.transform([supplier_1])[0]
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
            supplier_encoded_1 = label_encoder.transform([supplier_1])[0]
            supplier_encoded_2 = label_encoder.transform([supplier_2])[0]
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
            supplier_encoded = label_encoder.transform([supplier_1])[0]
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
        imputed_array = imputer.transform(input_array)
        with tab2:
            st.subheader("Data Setelah Imputasi")
            st.write(pd.DataFrame(imputed_array, columns=[f"Feature {i}" for i in range(imputed_array.shape[1])]))
        
        # Apply scaling
        scaled_array = scaler.transform(imputed_array)
        with tab2:
            st.subheader("Data Setelah Scaling")
            st.write(pd.DataFrame(scaled_array, columns=[f"Feature {i}" for i in range(scaled_array.shape[1])]))
        
        # Make prediction
        try:
            prediction = best_model.predict(scaled_array)[0]
            
            with tab2:
                st.subheader("Hasil Prediksi Model")
                st.write(f"Prediksi Awal: {prediction}")
                
                # If model has feature importances, show them
                if hasattr(best_model, 'feature_importances_'):
                    st.subheader("Feature Importances")
                    importances = dict(zip([f"Feature {i}" for i in range(len(best_model.feature_importances_))], 
                                         best_model.feature_importances_))
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
                decay_factor_1 = min(0.05, 0.05 * (storage_time_1 / 30))  # Cap at 5% max decrease per supplier
                final_prediction *= (1 - (decay_factor_1 * supplier_1_percentage / 100))
                
            if location_2 == "Coalyard" and storage_time_2 > 0:
                decay_factor_2 = min(0.05, 0.05 * (storage_time_2 / 30))  # Cap at 5% max decrease per supplier
                final_prediction *= (1 - (decay_factor_2 * supplier_2_percentage / 100))
                
            # Ensure result is within reasonable bounds
            final_prediction = max(2000, min(final_prediction, 7000))
                
            # Display results
            st.success(f"Prediksi GCV (ARB) LAB: {final_prediction:.2f} kcal/kg")
            
            # Show additional information
            st.info("""
            **Catatan:** 
            - Hasil prediksi dipengaruhi oleh persentase campuran dan waktu penyimpanan.
            - Penyimpanan di coalyard dapat menurunkan nilai GCV sekitar 5% per bulan.
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
        if hasattr(best_model, 'feature_names_in_'):
            st.write("Feature names expected by model:")
            st.write(best_model.feature_names_in_)
        
        st.write(f"Expected number of features: {imputer.n_features_in_}")
        st.write(f"Model type: {type(best_model).__name__}")
        
        # Show label encoder info
        st.subheader("Label Encoder Classes")
        st.write(pd.DataFrame({"Supplier": label_encoder.classes_, "Encoded Value": range(len(label_encoder.classes_))}))

# Add footer
st.markdown("---")
st.markdown("© 2025 GCV Prediction Tool | For optimal results, ensure model is regularly updated with new data.")
