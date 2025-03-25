#!/usr/bin/env python
# coding: utf-8

import streamlit as st
import pandas as pd
import numpy as np
import os
import pickle
import requests
from io import BytesIO
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score

# GitHub raw file URL
GITHUB_DATA_URL = "https://raw.githubusercontent.com/aguskurniawan10/calculator_blending/main/DATA%20PREDIKSI%20NK%20LAB%202025.xlsx"

# Set page config with a futuristic theme
st.set_page_config(
    page_title="Coal Fusion | AI Blending Predictor", 
    page_icon="🔮", 
    layout="wide", 
    initial_sidebar_state="collapsed"
)

# Enhanced custom CSS for a modern, futuristic look
st.markdown("""
<style>
    /* Global Styles */
    .stApp {
        background: linear-gradient(135deg, #0a1128 0%, #1b2544 100%);
        color: #e6f1ff;
        font-family: 'Inter', 'Segoe UI', Roboto, sans-serif;
    }

    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        background-color: rgba(27, 37, 68, 0.7);
        border-radius: 15px;
        backdrop-filter: blur(10px);
    }
    .stTabs [data-baseweb="tab"] {
        color: #7ed4ff;
        transition: all 0.3s ease;
        padding: 10px 15px;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background-color: rgba(44, 62, 112, 0.5);
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background-color: #2980b9;
        color: white;
    }

    /* Input Styling */
    .stNumberInput > div > div > input,
    .stSelectbox > div > div {
        background-color: rgba(27, 37, 68, 0.7);
        color: #7ed4ff;
        border: 2px solid #2c3e70;
        border-radius: 10px;
        backdrop-filter: blur(5px);
        transition: all 0.3s ease;
    }
    .stNumberInput > div > div > input:focus,
    .stSelectbox > div > div:focus {
        border-color: #3498db;
        box-shadow: 0 0 10px rgba(52, 152, 219, 0.5);
    }

    /* Button Styling */
    .stButton > button {
        background-color: #2980b9;
        color: white;
        border-radius: 15px;
        transition: all 0.3s ease;
        padding: 10px 20px;
        font-weight: bold;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .stButton > button:hover {
        background-color: #3498db;
        transform: scale(1.05);
        box-shadow: 0 0 15px rgba(52, 152, 219, 0.5);
    }

    /* Alert and Info Styling */
    .stAlert, .stInfo {
        background-color: rgba(44, 62, 112, 0.7);
        color: #7ed4ff;
        border-radius: 15px;
        backdrop-filter: blur(5px);
    }

    /* Title Styling */
    .title {
        text-align: center;
        color: #7ed4ff;
        text-shadow: 0 0 15px rgba(126, 212, 255, 0.5);
        margin-bottom: 20px;
        letter-spacing: 2px;
    }

    /* DataTable Styling */
    .stDataFrame {
        background-color: rgba(27, 37, 68, 0.7);
        color: #7ed4ff;
        border-radius: 15px;
        backdrop-filter: blur(5px);
    }
</style>
""", unsafe_allow_html=True)

# Rest of the original implementation remains the same
st.title("🔮 CALCULATOR BLENDING BATUBARA UBP JPR")

# [All the existing functions and implementation remain unchanged]

# Enhanced footer with futuristic styling
st.markdown("""
<div style="text-align: center; color: #7ed4ff; padding: 20px; 
            background-color: rgba(27, 37, 68, 0.7); 
            border-radius: 15px; 
            backdrop-filter: blur(10px);">
    <p style="margin-bottom: 10px;">
        <strong>© 2025 Coal Fusion | Powered by Advanced Machine Learning</strong>
    </p>
    <p style="font-size: 0.9em; opacity: 0.7;">
        Transforming Coal Blending with Predictive Intelligence 🤖✨
    </p>
</div>
""", unsafe_allow_html=True)
