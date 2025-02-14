import joblib
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# Define paths
model_path = "rainfall_prediction_model.joblib"
feature_names_path = "rainfall_prediction_model_features.json"

# Create a dummy dataset (replace with real data if available)
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

# Convert to DataFrame
df = pd.DataFrame(data)

# Separate features and target
X = df.drop(columns=["rainfall"])
y = df["rainfall"]

# Standardize data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model and feature names properly
joblib.dump(model, model_path)
with open(feature_names_path, "w") as f:
    json.dump(list(X.columns), f)

print("✅ Model trained and saved successfully!")
