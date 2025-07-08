import os
import csv
import time
import threading
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model

# Load LSTM model and preprocessing objects
try:
    lstm_model = load_model('./models/lstm_model.h5', compile=False)
    le_origin = joblib.load('./models/le_origin.pkl')
    le_destination = joblib.load('./models/le_destination.pkl')
    le_route = joblib.load('./models/le_route.pkl')
    le_weather = joblib.load('./models/le_weather.pkl')
    scaler_continuous = joblib.load('./models/scaler_continuous.pkl')
except Exception as e:
    print(f"Error loading LSTM model or preprocessing objects: {e}")
    lstm_model = None
    le_origin = None
    le_destination = None
    le_route = None
    le_weather = None
    scaler_continuous = None

seq_length = 10

def prepare_input_from_cnn_output(cnn_df):
    # Sort by Date and Time columns instead of Timestamp
    cnn_df = cnn_df.sort_values(['Date', 'Time']).tail(seq_length)
    origin_encoded = le_origin.transform(cnn_df['Origin'])
    destination_encoded = le_destination.transform(cnn_df['Destination'])
    return np.column_stack((origin_encoded, destination_encoded)).astype(np.float32)

def run_lstm_predictions_on_cnn_output(cnn_output_file="./data/cnn_output_for_lstm.csv", output_file="./data/lstm_predictions_output.csv"):
    if lstm_model is None or le_origin is None or le_destination is None or le_route is None or le_weather is None or scaler_continuous is None:
        print("LSTM model or preprocessing objects not loaded. Skipping LSTM predictions.")
        return

    import os
    file_exists = os.path.exists(output_file)

    # Open file in append mode if exists, else write mode and write header
    with open(output_file, 'a' if file_exists else 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'Date', 'Time', 'Day of the Week', 'Origin', 'Destination', 'Route',
            'Weather Conditions', 'Accident Reports', 'Traffic Intensity', 'Distance'
        ])
        if not file_exists:
            writer.writeheader()

        buffer_df = pd.DataFrame()
        try:
            for chunk in pd.read_csv(cnn_output_file, chunksize=1, encoding='utf-8'):
                buffer_df = pd.concat([buffer_df, chunk], ignore_index=True)
                if len(buffer_df) < seq_length:
                    continue

                input_seq_batch = np.expand_dims(prepare_input_from_cnn_output(buffer_df), axis=0)
                route_pred, weather_pred, cont_pred = lstm_model.predict(input_seq_batch)

                # decode
                route_labels = np.array([np.argmax(route_pred[0])])
                weather_labels = np.array([np.argmax(weather_pred[0])])
                route_decoded = le_route.inverse_transform(route_labels)
                weather_decoded = le_weather.inverse_transform(weather_labels)
                cont_unscaled = scaler_continuous.inverse_transform(cont_pred[0].reshape(1, -1))

                # Adjust Traffic Intensity and Distance by dividing by 10
                adjusted_traffic_intensity = cont_unscaled[0, 1] / 10.0
                adjusted_distance = cont_unscaled[0, 2] / 10.0

                # write first-step prediction
                t = datetime.now() + timedelta(minutes=10)
                row = {
                    'Date': t.strftime('%Y-%m-%d'),
                    'Time': t.strftime('%H:%M:%S'),
                    'Day of the Week': t.strftime('%A'),
                    'Origin': buffer_df.iloc[-1]['Origin'],
                    'Destination': buffer_df.iloc[-1]['Destination'],
                    'Route': route_decoded[0],
                    'Weather Conditions': weather_decoded[0],
                    'Accident Reports': cont_unscaled[0, 0],
                    'Traffic Intensity': adjusted_traffic_intensity,
                    'Distance': adjusted_distance
                }
                writer.writerow(row)
        except Exception as e:
            print(f"Error during LSTM prediction: {e}")

def update_graph_weights_with_lstm_predictions(graph, lstm_predictions_file="./data/lstm_predictions_output.csv", prediction_horizon_minutes=10):
    """
    Update graph edge weights based on LSTM predicted traffic intensity for the selected prediction horizon.
    The LSTM predictions CSV should have columns: Route, Predicted Traffic Intensity, Timestamp, Weather Conditions, Accident Reports.
    Only predictions within the next prediction_horizon_minutes are considered.
    """
    if not os.path.exists(lstm_predictions_file):
        return graph  # No LSTM predictions, return original graph

    try:
        pred_df = pd.read_csv(lstm_predictions_file, encoding='utf-8')
    except Exception:
        return graph  # On error, return original graph

    # Create 'Timestamp' column if missing by combining 'Date' and 'Time'
    if 'Timestamp' not in pred_df.columns and 'Date' in pred_df.columns and 'Time' in pred_df.columns:
        pred_df['Timestamp'] = pd.to_datetime(pred_df['Date'] + ' ' + pred_df['Time'])

    # Filter predictions within the prediction horizon from now
    now = datetime.now()
    horizon_end = now + timedelta(minutes=prediction_horizon_minutes)
    pred_df['Timestamp'] = pd.to_datetime(pred_df['Timestamp'])
    pred_df = pred_df[(pred_df['Timestamp'] >= now) & (pred_df['Timestamp'] <= horizon_end)]

    # Aggregate predicted traffic intensity per route (mean)
    route_pred_map = pred_df.groupby('Route')['Traffic Intensity'].mean().to_dict()

    # Update graph weights for edges matching route names
    for origin in graph:
        for destination in graph[origin]:
            weight, route_name = graph[origin][destination]
            if route_name in route_pred_map:
                predicted_traffic = route_pred_map[route_name]
                # Adjust weight with predicted traffic intensity factor
                new_weight = weight + (predicted_traffic * 10)
                graph[origin][destination] = (new_weight, route_name)
            else:
                # No prediction for this route, keep original weight
                graph[origin][destination] = (weight, route_name)

    return graph

# Global variable to cache live traffic data
live_traffic_data = {
    "vehicles_in_roi": None,
    "congestion_level": None,
    "timestamp": None
}

# Function to read live traffic data from CSV periodically
def read_live_traffic_data(csv_file="traffic_live_data.csv"):
    global live_traffic_data
    try:
        if os.path.exists(csv_file):
            df = pd.read_csv(csv_file, encoding='utf-8')
            if not df.empty:
                # Read the last row (latest data)
                last_row = df.iloc[-1]
                live_traffic_data["vehicles_in_roi"] = last_row.get("vehicles_in_roi", None)
                live_traffic_data["congestion_level"] = last_row.get("congestion_level", None)
                live_traffic_data["timestamp"] = last_row.get("timestamp", None)
    except Exception as e:
        pass

# Start a background thread to update live traffic data every 5 seconds
def start_live_data_thread():
    def update_loop():
        while True:
            read_live_traffic_data()
            time.sleep(5)
    thread = threading.Thread(target=update_loop, daemon=True)
    thread.start()

def predict_congestion_and_time(route_data, time_adjustment=0):
    """
    Predict congestion level and travel time using LSTM model if available,
    otherwise fallback to rule-based prediction.
    Uses live traffic data from Traffic_Video.py if available.
    """
    # Use live traffic data if available
    if live_traffic_data["vehicles_in_roi"] is not None:
        traffic_intensity_live = float(live_traffic_data["vehicles_in_roi"])
    else:
        traffic_intensity_live = int(route_data['Traffic Intensity']) + time_adjustment

    if lstm_model is None or scaler_continuous is None or le_weather is None:
        # Fallback to rule-based prediction
        traffic_intensity = traffic_intensity_live
        if traffic_intensity < 30:
            congestion = 'Low'
            predicted_time = 15
        elif 30 <= traffic_intensity < 60:
            congestion = 'Medium'
            predicted_time = 30
        else:
            congestion = 'High'
            predicted_time = 60
        return congestion, predicted_time

    # Prepare input features for LSTM prediction
    try:
        # Encode weather condition
        weather_encoded = le_weather.transform([route_data['Weather Conditions']])[0]
    except Exception:
        weather_encoded = 0  # default if encoding fails

    # Create feature array: Traffic Intensity, Weather Encoded, Accident Reports
    features = np.array([[traffic_intensity_live,
                          weather_encoded,
                          route_data['Accident Reports']]], dtype=float)

    # Scale features
    features_scaled = scaler_continuous.transform(features)

    # Create sequence input for LSTM (repeat last known sequence)
    seq_length = 10
    X_input = np.tile(features_scaled, (seq_length, 1))
    X_input = X_input.reshape((1, seq_length, features.shape[1]))

    # Predict scaled traffic intensity
    pred_scaled = lstm_model.predict(X_input)
    # Handle possible 3D output shape (1, 1, 1) or (1, 1)
    if len(pred_scaled.shape) == 3:
        pred_scaled = pred_scaled.squeeze(-1)  # shape (1, 1)
    pred_scaled = pred_scaled.reshape(-1, 1)  # ensure 2D shape (samples, features)

    zeros_pad = np.zeros((pred_scaled.shape[0], features.shape[1]-1))
    pred_traffic_intensity = scaler_continuous.inverse_transform(
        np.hstack([pred_scaled, zeros_pad])
    )[0,0]

    # Map predicted traffic intensity to congestion and time
    if pred_traffic_intensity < 30:
        congestion = 'Low'
        predicted_time = 15
    elif 30 <= pred_traffic_intensity < 60:
        congestion = 'Medium'
        predicted_time = 30
    else:
        congestion = 'High'
        predicted_time = 60

    return congestion, predicted_time

# Start live data thread on import
start_live_data_thread()