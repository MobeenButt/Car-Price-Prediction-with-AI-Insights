# app.py
import streamlit as st 
import pandas as pd
import joblib
import google.generativeai as genai
from dotenv import load_dotenv
import os
from pathlib import Path

# Page config
st.set_page_config(page_title="Car Price Predictor", layout="wide")

# Load environment variables
load_dotenv()

# Configure Gemini API
try:
    api_key = os.getenv('GOOGLE_API_KEY')
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not found in environment variables")
    
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-2.5-flash-preview-05-20')
except Exception as e:
    st.error(f"Error configuring Gemini API: {str(e)}")
    model = None

# Check and load model files
try:
    rf_model = joblib.load('car_price_model.joblib')
    scaler = joblib.load('scaler.joblib')
    feature_names = joblib.load('feature_names.joblib')
except FileNotFoundError:
    st.error("Model files not found. Please run model_training.py first.")
    st.stop()

def predict_price(features):
    """Make prediction using the trained model"""
    # Ensure all required features are present
    missing_features = set(feature_names) - set(features.columns)
    for feature in missing_features:
        features[feature] = 0
    
    # Reorder columns to match training data
    features = features[feature_names]
    
    features_scaled = scaler.transform(features)
    prediction = rf_model.predict(features_scaled)
    return prediction[0]

def get_gemini_response(prompt):
    """Get insights from Gemini AI"""
    try:
        if not model:
            return "AI insights unavailable - API configuration error"
        
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error getting AI insights: {str(e)}"

# UI Elements
st.title('🚗 Car Price Prediction with AI Insights')
st.markdown("---")

# Input form with two columns
col1, col2 = st.columns(2)

with col1:
    st.subheader("Vehicle Specifications")
    kilometers = st.number_input('Kilometers Driven', min_value=0)
    mileage = st.number_input('Mileage (kmpl)', min_value=0.0)
    engine = st.number_input('Engine (CC)', min_value=0)
    power = st.number_input('Power (bhp)', min_value=0.0)
    seats = st.number_input('Seats', min_value=2, max_value=10)

with col2:
    st.subheader("Additional Details")
    age = st.number_input('Age (years)', min_value=0)
    fuel_type = st.selectbox('Fuel Type', ['Petrol', 'Diesel', 'CNG', 'LPG'])
    transmission = st.selectbox('Transmission', ['Manual', 'Automatic'])
    owner_type = st.selectbox('Owner Type', ['First', 'Second', 'Third', 'Fourth & Above'])

if st.button('Predict Price', type='primary'):
    # Prepare features with all possible dummy columns
    base_features = {
        'Kilometers_Driven': [kilometers],
        'Mileage': [mileage],
        'Engine': [engine],
        'Power': [power],
        'Seats': [seats],
        'Age': [age]
    }
    
    # Create dummy columns for categorical variables
    fuel_types = ['Petrol', 'Diesel', 'CNG', 'LPG']
    transmissions = ['Manual', 'Automatic']
    owner_types = ['First', 'Second', 'Third', 'Fourth & Above']
    
    # Add fuel type dummies
    for ft in fuel_types:
        base_features[f'Fuel_Type_{ft}'] = [1 if fuel_type == ft else 0]
    
    # Add transmission dummies
    for trans in transmissions:
        base_features[f'Transmission_{trans}'] = [1 if transmission == trans else 0]
    
    # Add owner type dummies
    for ot in owner_types:
        base_features[f'Owner_Type_{ot}'] = [1 if owner_type == ot else 0]
    
    # Create DataFrame
    features = pd.DataFrame(base_features)
    
    # Make prediction
    predicted_price = predict_price(features)
    
    # Show prediction
    st.markdown("---")
    st.subheader("Prediction Results")
    st.success(f'💰 Predicted Price: ₹{predicted_price:.2f} Lakhs')
    
    # Get Gemini insights with error handling
    prompt = f"""
    As an automotive expert, analyze this car's details and price:
    
    Specifications:
    - Kilometers driven: {kilometers:,} km
    - Mileage: {mileage} kmpl
    - Engine: {engine} CC
    - Power: {power} bhp
    - Age: {age} years
    - Fuel Type: {fuel_type}
    - Transmission: {transmission}
    - Owner Type: {owner_type}
    
    Predicted Price: ₹{predicted_price:.2f} Lakhs
    
    Provide a brief professional analysis covering:
    1. Price justification
    2. Vehicle condition assessment
    3. Key value factors
    4. Potential concerns
    Keep it concise and factual.
    """
    
    with st.spinner('Getting AI insights...'):
        ai_response = get_gemini_response(prompt)
        st.markdown("---")
        st.subheader("🤖 AI Expert Analysis")
        st.write(ai_response)
