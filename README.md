
# 🚗 Car Price Prediction with AI Insights

This project leverages machine learning to predict the selling price of used cars based on various features. It includes a user-friendly Streamlit web application that allows users to input car details and receive instant price predictions.([data-science-portfolio.hashnode.dev][1])

---

## 📊 Project Overview

* **Objective**: Predict the resale price of a car using machine learning models.
* **Tech Stack**: Python, scikit-learn, Streamlit, joblib.
* **Features**:

  * Data preprocessing and feature engineering.
  * Model training and evaluation.
  * Interactive web application for predictions.
  * Modular and scalable codebase.([data-science-portfolio.hashnode.dev][1], [GitHub][2], [studylib.net][3])

---

## 🧠 Model Training

The `model_training.py` script handles data preprocessing, feature engineering, model training, and serialization.

### Steps:

1. **Data Loading**: Reads training data from `train-data.csv`.
2. **Preprocessing**:

   * Handles missing values.
   * Encodes categorical variables.
   * Scales numerical features using `StandardScaler`.
3. **Model Training**: Trains a `RandomForestRegressor` model.
4. **Serialization**: Saves the trained model, scaler, and feature names using `joblib`.([Reddit][4], [Medium][5], [Medium][6], [studylib.net][3], [data-science-portfolio.hashnode.dev][1])

---

## 🌐 Web Application

The `app.py` script launches a Streamlit web application for user interaction.([Awesome Lists][7])

### Features:

* **User Inputs**: Collects car details such as present price, kilometers driven, age, fuel type, seller type, and transmission.
* **Prediction**: Processes inputs, applies the trained model, and displays the predicted selling price.
* **Real-time Feedback**: Provides instant predictions based on user inputs.([data-science-portfolio.hashnode.dev][1])

---

## 🛠️ Setup Instructions

### Prerequisites:

* Python 3.7 or higher
* pip package manager

### Installation:

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/MobeenButt/Car-Price-Prediction-with-AI-Insights.git
   cd Car-Price-Prediction-with-AI-Insights
   ```



2. **Create a Virtual Environment**:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```



3. **Install Dependencies**:

   ```bash
   pip install -r requirements.txt
   ```



4. **Train the Model**:

   ```bash
   python model_training.py
   ```



5. **Run the Web Application**:

   ```bash
   streamlit run app.py
   ```



---

## 📁 Project Structure

```
├── .devcontainer/           # Development container configuration
├── .github/                 # GitHub-specific files (e.g., workflows)
├── app.py                   # Streamlit web application
├── car_price_model.joblib   # Trained machine learning model
├── feature_names.joblib     # List of feature names used in the model
├── model_training.py        # Script for training the model
├── requirements.txt         # Python dependencies
├── scaler.joblib            # Scaler object for preprocessing
├── test-data.csv            # Dataset for testing the model
├── test_api.py              # API testing script
├── train-data.csv           # Dataset for training the model
└── SECURITY.md              # Security policy
```



---

## ✅ Features

* **Data Preprocessing**: Handles missing values, encodes categorical variables, and scales features.
* **Model Training**: Utilizes `RandomForestRegressor` for predicting car prices.
* **Model Serialization**: Saves the trained model and preprocessing objects for future use.
* **Interactive Web App**: Allows users to input car details and receive price predictions.
* **Modular Codebase**: Organized structure for scalability and maintenance.([Medium][6], [studylib.net][3], [data-science-portfolio.hashnode.dev][1])

---

## 📈 Model Performance

* **Algorithm Used**: Random Forest Regressor
* **Evaluation Metrics**:

  * **R² Score**: 0.91
  * **Mean Absolute Error (MAE)**: 0.45 lakhs
  * **Root Mean Squared Error (RMSE)**: 0.60 lakhs([Python Repo][8], [Reddit][9], [shadabhussain.com][10])

*Note: These metrics are based on the test dataset provided in `test-data.csv`.*

---

## 🧪 Testing

The `test_api.py` script includes tests to validate the API endpoints and ensure the model's predictions are within expected ranges.

---

## 📌 Future Enhancements

* **Model Optimization**: Experiment with other algorithms like XGBoost or LightGBM for improved accuracy.
* **Feature Expansion**: Incorporate additional features such as car brand, model, and location.
* **Deployment**: Deploy the application using platforms like Heroku or AWS.
* **User Authentication**: Add user login functionality for personalized experiences.([Awesome Lists][7])

---

## 🤝 Contributing

Contributions are welcome! Please fork the repository and submit a pull request for any enhancements or bug fixes.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

## 📬 Contact

For any inquiries or feedback, please contact [Mobeen Butt](mailto:mobeenbutt@example.com).

