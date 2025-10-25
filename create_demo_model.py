# create_demo_model.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib

# Create synthetic demo data
np.random.seed(42)
n_samples = 1000

data = {
    'Age': np.random.randint(20, 65, n_samples),
    'Sleep Duration': np.random.uniform(4, 10, n_samples),
    'Quality of Sleep': np.random.randint(1, 11, n_samples),
    'Physical Activity Level': np.random.randint(0, 121, n_samples),
    'Stress Level': np.random.randint(1, 11, n_samples),
    'Heart Rate': np.random.randint(60, 101, n_samples),
    'Daily Steps': np.random.randint(2000, 15001, n_samples),
    'Systolic_BP': np.random.randint(100, 141, n_samples),
    'Diastolic_BP': np.random.randint(60, 91, n_samples),
    'Gender_encoded': np.random.randint(0, 3, n_samples),
    'Occupation_encoded': np.random.randint(0, 20, n_samples),
    'BMI Category_encoded': np.random.randint(0, 4, n_samples),
    'Age_Group_encoded': np.random.randint(0, 4, n_samples),
    'Sleep_Quality_Category_encoded': np.random.randint(0, 3, n_samples),
    'Stress_Level_Category_encoded': np.random.randint(0, 3, n_samples)
}

# Create target based on rules
conditions = [
    (data['Stress Level'] >= 7) & (data['Quality of Sleep'] <= 5),
    (data['Sleep Duration'] <= 5) & (data['Stress Level'] >= 6),
    (data['BMI Category_encoded'] >= 2) & (data['Age'] > 40),
    (data['Physical Activity Level'] < 20) & (data['Quality of Sleep'] <= 6)
]
choices = ['Insomnia', 'Insomnia', 'Sleep Apnea', 'Insomnia']

df = pd.DataFrame(data)
df['Sleep Disorder'] = np.select(conditions, choices, default='None')

# Prepare features and target
feature_columns = [col for col in data.keys()]
X = df[feature_columns]
y = df['Sleep Disorder']

# Encode target
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_scaled, y_encoded)

# Save model artifacts
model_artifacts = {
    'model': model,
    'scaler': scaler,
    'label_encoder': label_encoder,
    'feature_columns': feature_columns
}

joblib.dump(model_artifacts, 'best_sleep_disorder_model.pkl')
print("✅ Demo model created and saved as 'best_sleep_disorder_model.pkl'")