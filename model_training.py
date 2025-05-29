# model_training.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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
    
    # Print initial data info
    print("\nInitial data info:")
    print(df[['Mileage', 'Engine', 'Power']].describe())
    
    # Data preprocessing
    def preprocess_data(df):
        """Preprocess the data for model training"""
        # Create a copy to avoid modifying original data
        df = df.copy()
        
        # Drop unnecessary columns
        df = df.drop(['Name', 'Location', 'New_Price'], axis=1)
        
        # Convert Mileage to numeric (handle both km/kg and kmpl)
        print("\nUnique Mileage values before processing:")
        print(df['Mileage'].unique())
        df['Mileage'] = df['Mileage'].str.extract(r'(\d+\.?\d*)').astype(float)
        
        # Convert Engine CC to numeric
        print("\nUnique Engine values before processing:")
        print(df['Engine'].unique())
        df['Engine'] = df['Engine'].str.extract(r'(\d+)').astype(float)
        
        # Convert Power bhp to numeric (handle null values)
        print("\nUnique Power values before processing:")
        print(df['Power'].unique())
        df['Power'] = df['Power'].replace('null bhp', np.nan)
        df['Power'] = df['Power'].str.extract(r'(\d+\.?\d*)').astype(float)
        df['Power'] = df['Power'].fillna(df['Power'].median())
        
        # Convert Year to age
        current_year = 2024
        df['Age'] = current_year - df['Year']
        df = df.drop('Year', axis=1)
        
        # Handle categorical variables
        df = pd.get_dummies(df, columns=['Fuel_Type', 'Transmission', 'Owner_Type'])
        
        return df

    print("\nPreprocessing data...")
    df_processed = preprocess_data(df)
    
    print("\nProcessed data info:")
    print(df_processed.describe())
    
    # Remove the 'Unnamed: 0' column if it exists
    if 'Unnamed: 0' in df_processed.columns:
        df_processed = df_processed.drop('Unnamed: 0', axis=1)

    # Check for any remaining missing values
    if df_processed.isnull().sum().any():
        print("\nWarning: Missing values detected after preprocessing:")
        print(df_processed.isnull().sum()[df_processed.isnull().sum() > 0])
    
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

    # Create visualizations directory
    if not os.path.exists('visualizations'):
        os.makedirs('visualizations')

    # Correlation Heatmap using processed data
    plt.figure(figsize=(12, 8))
    numeric_cols = ['Price', 'Kilometers_Driven', 'Mileage', 'Engine', 'Power', 'Age']
    sns.heatmap(df_processed[numeric_cols].corr(), annot=True)
    plt.title('Feature Correlation Heatmap')
    plt.tight_layout()
    plt.savefig('visualizations/correlation_heatmap.png')
    plt.close()

    # Feature Importance Plot
    importances = pd.DataFrame({
        'feature': feature_names,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)

    plt.figure(figsize=(12, 6))
    sns.barplot(data=importances.head(10), x='importance', y='feature')
    plt.title('Top 10 Most Important Features')
    plt.tight_layout()
    plt.savefig('visualizations/feature_importance.png')
    plt.close()

    # Actual vs Predicted Plot
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Price')
    plt.ylabel('Predicted Price')
    plt.title('Actual vs Predicted Prices')
    plt.tight_layout()
    plt.savefig('visualizations/actual_vs_predicted.png')
    plt.close()

    print("Model training and visualization completed successfully!")

except Exception as e:
    print(f"Error during model training: {str(e)}")
    print("\nFull error details:")
    import traceback
    print(traceback.format_exc())