import webbrowser
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import joblib
import json
import numpy as np
import uvicorn
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# Initialize FastAPI app
app = FastAPI()

# Define paths
model_path = "rainfall_prediction_model.joblib"
feature_names_path = "rainfall_prediction_model_features.json"

# Function to train the model (if not found)
def train_model():
    print("🔄 Training new model...")

    # Simulated dataset (Replace with real data)
    data = {
        "pressure": np.random.uniform(900, 1100, 1000),
        "dewpoint": np.random.uniform(-10, 25, 1000),
        "humidity": np.random.uniform(10, 100, 1000),
        "cloud": np.random.uniform(0, 100, 1000),
        "sunshine": np.random.uniform(0, 12, 1000),
        "winddirection": np.random.uniform(0, 360, 1000),
        "windspeed": np.random.uniform(0, 30, 1000),
        "rainfall": np.random.choice([0, 1], 1000)  # 0 = No Rain, 1 = Rain
    }
    
    df = pd.DataFrame(data)
    X = df.drop(columns=["rainfall"])
    y = df["rainfall"]

    # Standardize data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Save model and feature names
    joblib.dump(model, model_path)
    with open(feature_names_path, "w") as f:
        json.dump(list(X.columns), f)

    print("✅ Model trained and saved successfully!")

# Load or train model
if os.path.exists(model_path) and os.path.exists(feature_names_path):
    try:
        model = joblib.load(model_path)
        with open(feature_names_path, "r") as f:
            feature_names = json.load(f)
        if not hasattr(model, "predict"):
            raise ValueError("Loaded object is not a valid scikit-learn model")
        print("✅ Model and feature names loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        train_model()
        model = joblib.load(model_path)
        with open(feature_names_path, "r") as f:
            feature_names = json.load(f)
else:
    train_model()
    model = joblib.load(model_path)
    with open(feature_names_path, "r") as f:
        feature_names = json.load(f)

# Define input data model
class WeatherInput(BaseModel):
    pressure: float
    dewpoint: float
    humidity: float
    cloud: float
    sunshine: float
    winddirection: float
    windspeed: float

# Serve the enhanced HTML page
@app.get("/", response_class=HTMLResponse)
async def serve_home():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Rainfall Prediction</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                text-align: center;
                background: linear-gradient(to right, #6dd5ed, #2193b0);
                margin: 0;
                padding: 20px;
                color: white;
            }
            .container {
                background: rgba(255, 255, 255, 0.9);
                padding: 20px;
                border-radius: 10px;
                box-shadow: 0px 0px 15px rgba(0, 0, 0, 0.2);
                max-width: 400px;
                margin: auto;
                color: black;
            }
            h1 {
                color: #333;
            }
            form {
                display: flex;
                flex-direction: column;
            }
            label {
                font-weight: bold;
                margin-top: 10px;
            }
            input {
                padding: 8px;
                margin-top: 5px;
                border-radius: 5px;
                border: 1px solid #ccc;
            }
            button {
                margin-top: 15px;
                padding: 10px;
                background: #28a745;
                color: white;
                border: none;
                cursor: pointer;
                border-radius: 5px;
                font-size: 16px;
            }
            button:hover {
                background: #218838;
            }
            #result {
                font-size: 1.3em;
                margin-top: 20px;
                font-weight: bold;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Rainfall Prediction System</h1>
            <form id="prediction-form">
                <label>Pressure:</label><input type="number" step="any" id="pressure" required><br>
                <label>Dew Point:</label><input type="number" step="any" id="dewpoint" required><br>
                <label>Humidity:</label><input type="number" step="any" id="humidity" required><br>
                <label>Cloud Cover:</label><input type="number" step="any" id="cloud" required><br>
                <label>Sunshine:</label><input type="number" step="any" id="sunshine" required><br>
                <label>Wind Direction:</label><input type="number" step="any" id="winddirection" required><br>
                <label>Wind Speed:</label><input type="number" step="any" id="windspeed" required><br>
                <button type="submit">Predict</button>
            </form>
            <p id="result"></p>
        </div>
        <script>
            document.getElementById("prediction-form").addEventListener("submit", async function (event) {
                event.preventDefault();
                const data = {
                    pressure: parseFloat(document.getElementById("pressure").value),
                    dewpoint: parseFloat(document.getElementById("dewpoint").value),
                    humidity: parseFloat(document.getElementById("humidity").value),
                    cloud: parseFloat(document.getElementById("cloud").value),
                    sunshine: parseFloat(document.getElementById("sunshine").value),
                    winddirection: parseFloat(document.getElementById("winddirection").value),
                    windspeed: parseFloat(document.getElementById("windspeed").value)
                };
                const response = await fetch("/predict", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify(data)
                });
                const result = await response.json();
                document.getElementById("result").innerText = "Prediction: " + result.prediction;
            });
        </script>
    </body>
    </html>
    """

# Prediction API
@app.post("/predict")
async def predict_rainfall(data: WeatherInput):
    input_data = np.array([[data.pressure, data.dewpoint, data.humidity, data.cloud, data.sunshine, data.winddirection, data.windspeed]])
    prediction = model.predict(input_data)
    return {"prediction": "Rainfall" if prediction[0] == 1 else "No Rainfall"}

# Run FastAPI
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
    webbrowser.open("http://127.0.0.1:8000")
