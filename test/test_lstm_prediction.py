import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import joblib
import tensorflow as tf

def safe_transform_label_encoder(le, values):
    # Map unseen labels to a default value (e.g., most frequent or 0)
    known_labels = set(le.classes_)
    mapped = []
    for v in values:
        if v in known_labels:
            mapped.append(v)
        else:
            mapped.append(le.classes_[0])  # fallback to first known class
    return le.transform(mapped)

def test_lstm_prediction(cnn_output_file="cnn_output_for_lstm.csv", seq_length=5):
    # Load LSTM model and preprocessing objects
    try:
        lstm_model = load_model('lstm_traffic_model.h5', custom_objects={'mse': tf.keras.losses.MeanSquaredError()})
        scaler = joblib.load('scaler.save')
        le_weather = joblib.load('le_weather.save')
    except Exception as e:
        print(f"Error loading model or preprocessing objects: {e}")
        return

    # Load CNN output CSV
    try:
        df = pd.read_csv(cnn_output_file)
    except Exception as e:
        print(f"Error reading CNN output file {cnn_output_file}: {e}")
        return

    # Encode Weather Conditions safely
    try:
        df['Weather_Encoded'] = safe_transform_label_encoder(le_weather, df['Weather Conditions'])
    except Exception as e:
        print(f"Error encoding weather conditions: {e}")
        df['Weather_Encoded'] = 0

    # Sort by Route and Timestamp
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df = df.sort_values(['Route', 'Timestamp'])

    features = ['Traffic Intensity', 'Weather_Encoded', 'Accident Reports']

    # Try routes until one has sufficient data length
    routes = df['Route'].unique()
    for route in routes:
        group = df[df['Route'] == route]
        data = group[features].values
        if len(data) <= seq_length:
            print(f"Route {route} skipped due to insufficient data length ({len(data)}).")
            continue
        data_scaled = scaler.transform(data)

        # Create sequences
        X = []
        for i in range(len(data_scaled) - seq_length):
            X.append(data_scaled[i:i+seq_length])
        if len(X) == 0:
            print(f"No sequences created for route {route} due to insufficient data length.")
            continue
        X = np.array(X)

        # Predict using LSTM
        try:
            preds_scaled = lstm_model.predict(X)
            print(f"Route: {route}")
            print(f"Predictions shape: {preds_scaled.shape}")
            print("Sample predictions (scaled):", preds_scaled[:5])
            break  # test one route only
        except Exception as e:
            print(f"Prediction error for route {route}: {e}")
            continue
    else:
        print("No route with sufficient data length found for prediction.")

if __name__ == "__main__":
    test_lstm_prediction()
