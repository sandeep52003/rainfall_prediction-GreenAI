🌧️ Rainfall Prediction using Machine Learning ☁️
This project predicts rainfall using machine learning techniques based on historical weather data. The dataset includes meteorological parameters such as pressure, humidity, temperature, cloud cover, and wind speed. The model uses Random Forest Classifier for classification, with data preprocessing, hyperparameter tuning, and model evaluation.
🚀 Features
✅ Data Preprocessing: Handles missing values and standardizes weather data.
✅ Feature Engineering: Removes unnecessary columns and optimizes feature selection.
✅ Machine Learning Model: Implements Random Forest Classifier for rainfall prediction.
✅ Hyperparameter Tuning: Uses GridSearchCV for optimizing model parameters.
✅ Cross-Validation: Ensures robustness using cross_val_score.
✅ Model Saving & Deployment: Saves the trained model using Pickle for future predictions.
🛠️ Technologies Used
•	Python 🐍
•	Pandas & NumPy for data handling
•	Matplotlib & Seaborn for data visualization
•	Scikit-learn for machine learning
•	Pickle for model saving
📂 Dataset Overview
The dataset includes 366 entries with the following 11 features after preprocessing:
Column Name	Data Type	Description
pressure	float64	Atmospheric pressure (hPa)
maxtemp	float64	Maximum temperature (°C)
temparature	float64	Average daily temperature (°C)
mintemp	float64	Minimum temperature (°C)
dewpoint	float64	Dew point temperature (°C)
humidity	int64	Humidity percentage (%)
cloud	int64	Cloud cover percentage (%)
rainfall	object	Target variable (Rain/No Rain)
sunshine	float64	Sunshine duration (hours)
winddirection	float64	Wind direction (°)
windspeed	float64	Wind speed (km/h)

📌 Data Preprocessing Steps
✔ Removed extra spaces in column names
✔ Dropped unnecessary columns (day)
✔ Handled missing values in winddirection and windspeed using mode imputation
📊 Results
The Random Forest Classifier achieves high accuracy in predicting rainfall. Model performance is evaluated using:
•	Confusion Matrix
•	Classification Report
•	Accuracy Score
🎯 Future Enhancements
🔹 Deploy the model using Flask/Django for real-time predictions
🔹 Integrate live weather API for dynamic forecasting
🔹 Experiment with Deep Learning models for improved accuracy
________________________________________
