from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import joblib

app = Flask(__name__)

# Load model & scaler
model = joblib.load("../model/model.save")
scaler = joblib.load("../model/scaler.save")

# Load dataset
df = pd.read_csv("../dataset/pmsm_temperature_data.csv")

@app.route('/')
def home():
    profiles = df['profile_id'].unique()
    return render_template("index.html", profiles=profiles)

@app.route('/load_sample', methods=['POST'])
def load_sample():
    profile_id = int(request.form['profile'])

    # Get one row from selected profile
    sample = df[df['profile_id'] == profile_id].iloc[0]

    return render_template(
        "index.html",
        profiles=df['profile_id'].unique(),
        sample=sample
    )

@app.route('/predict', methods=['POST'])
def predict():

    inputs = [
        float(request.form['ambient']),
        float(request.form['coolant']),
        float(request.form['u_d']),
        float(request.form['u_q']),
        float(request.form['motor_speed']),
        float(request.form['i_d']),
        float(request.form['i_q'])
    ]

    final = np.array([inputs])
    prediction = model.predict(final)

    return render_template(
        "index.html",
        prediction_text=f"Predicted Temperature: {prediction[0]:.2f}",
        inputs=inputs
    )


if __name__ == "__main__":
    app.run(debug=True)
