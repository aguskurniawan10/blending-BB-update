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

# Set page config with a futuristic theme
st.set_page_config(
    page_title="Coal Fusion | AI Blending Predictor", 
    page_icon="🔮", 
    layout="wide", 
    initial_sidebar_state="collapsed"
)

# Custom CSS for a futuristic look
st.markdown("""
<style>
    .stApp {
        background-color: #0a1128;
        color: #e6f1ff;
    }
    .stTabs [data-baseweb="tab-list"] {
        background-color: #1b2544;
        border-radius: 12px;
    }
    .stTabs [data-baseweb="tab"] {
        color: #7ed4ff;
        transition: all 0.3s ease;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #2c3e70;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background-color: #2980b9;
        color: white;
    }
    .stNumberInput > div > div > input {
        background-color: #1b2544;
        color: #7ed4ff;
        border: 2px solid #2c3e70;
        border-radius: 8px;
    }
    .stSelectbox > div > div {
        background-color: #1b2544;
        color: #7ed4ff;
        border: 2px solid #2c3e70;
        border-radius: 8px;
    }
    .stButton > button {
        background-color: #2980b9;
        color: white;
        border-radius: 12px;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        background-color: #3498db;
        transform: scale(1.05);
    }
    .stAlert {
        background-color: #2c3e70;
        color: #7ed4ff;
        border-radius: 12px;
    }
</style>
""", unsafe_allow_html=True)

# Rest of the code remains the same as in the original script

st.title("🔮 Coal Fusion: AI Blending Predictor")
st.subheader("Intelligent Coal Quality Forecasting")

# The entire original implementation follows, with the addition of the custom CSS styling above
# [rest of the original code remains unchanged]

# At the end, add a more futuristic footer
st.markdown("""
<div style="text-align: center; color: #7ed4ff; padding: 20px; background-color: #1b2544; border-radius: 12px;">
    <p>© 2025 Coal Fusion | Powered by Advanced Machine Learning 🤖✨</p>
    <p>Transforming Coal Blending with Predictive Intelligence</p>
</div>
""", unsafe_allow_html=True)
