# model_training.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os

try:
    # Load data
    print("Loading training data...")
    df = pd.read_csv('train-data.csv')
    print(f"Loaded {len(df)} training samples")

    # Data preprocessing
    def preprocess_data(df):
        # Drop unnecessary columns
        df = df.drop(['Name', 'Location', 'New_Price'], axis=1)
        
        # Convert Mileage to numeric
        df['Mileage'] = df['Mileage'].str.extract('(\d+\.?\d*)').astype(float)
        
        # Convert Engine CC to numeric
        df['Engine'] = df['Engine'].str.extract('(\d+)').astype(float)
        
        # Convert Power bhp to numeric
        df['Power'] = df['Power'].str.extract('(\d+\.?\d*)').astype(float)
        
        # Convert Year to age
        current_year = 2024
        df['Age'] = current_year - df['Year']
        df = df.drop('Year', axis=1)
        
        # Handle categorical variables
        df = pd.get_dummies(df, columns=['Fuel_Type', 'Transmission', 'Owner_Type'])
        
        return df

    print("Preprocessing data...")
    df_processed = preprocess_data(df)
    
    # Remove the 'Unnamed: 0' column if it exists
    if 'Unnamed: 0' in df_processed.columns:
        df_processed = df_processed.drop('Unnamed: 0', axis=1)

    # Split features and target
    X = df_processed.drop('Price', axis=1)
    y = df_processed['Price']

    # Store feature names
    feature_names = X.columns.tolist()
    
    # Save feature names
    joblib.dump(feature_names, 'feature_names.joblib')

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Training set size: {len(X_train)}, Test set size: {len(X_test)}")

    # Scale features
    print("Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train model
    print("Training Random Forest model...")
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train_scaled, y_train)

    # Evaluate model
    y_pred = rf_model.predict(X_test_scaled)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"\nModel Performance:")
    print(f"Mean Squared Error: {mse:.2f}")
    print(f"R² Score: {r2:.2f}")

    # Save model and scaler
    print("\nSaving model and scaler...")
    joblib.dump(rf_model, 'car_price_model.joblib')
    joblib.dump(scaler, 'scaler.joblib')
    print("Model training completed successfully!")

except Exception as e:
    print(f"Error during model training: {str(e)}")