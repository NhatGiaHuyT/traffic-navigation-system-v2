import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model
from datetime import datetime, timedelta
import csv

# ---- load only what you need ----
model                = load_model('lstm_model.h5', compile=False)
le_origin            = joblib.load('le_origin.pkl')
le_destination       = joblib.load('le_destination.pkl')
le_route             = joblib.load('le_route.pkl')
le_weather           = joblib.load('le_weather.pkl')
scaler_continuous    = joblib.load('scaler_continuous.pkl')

meta = joblib.load('lstm_meta.pkl')
seq_length           = meta['seq_length']
prediction_steps     = meta['prediction_steps']

def prepare_input_from_cnn_output(cnn_df):
    # Sort by Date and Time columns instead of Timestamp
    cnn_df = cnn_df.sort_values(['Date', 'Time']).tail(seq_length)
    origin_encoded      = le_origin.transform(cnn_df['Origin'])
    destination_encoded = le_destination.transform(cnn_df['Destination'])
    return np.column_stack((origin_encoded, destination_encoded)).astype(np.float32)

def test_lstm_prediction_with_cnn_input_line_by_line(
        cnn_output_csv='output_lstm.csv',
        output_csv='../data/lstm_predictions_output.csv'):

    # write header
    with open(output_csv, 'w', newline='') as f:
        writer = csv.DictWriter(
            f, fieldnames=[
                'Date','Time','Day of the Week','Origin','Destination',
                'Route','Weather Conditions','Accident Reports',
                'Traffic Intensity','Distance'
            ])
        writer.writeheader()

    buffer_df = pd.DataFrame()
    for chunk in pd.read_csv(cnn_output_csv, chunksize=1):
        buffer_df = pd.concat([buffer_df, chunk], ignore_index=True)
        if len(buffer_df) < seq_length:
            continue

        input_seq_batch = np.expand_dims(
            prepare_input_from_cnn_output(buffer_df), axis=0
        )
        route_pred, weather_pred, cont_pred = model.predict(input_seq_batch)

        # decode
        route_labels   = np.array([np.argmax(route_pred[0])])
        weather_labels = np.array([np.argmax(weather_pred[0])])
        route_decoded  = le_route.inverse_transform(route_labels)
        weather_decoded= le_weather.inverse_transform(weather_labels)
        cont_unscaled  = scaler_continuous.inverse_transform(cont_pred[0].reshape(1, -1))

        # write first‐step
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
            'Traffic Intensity': cont_unscaled[0, 1],
            'Distance': cont_unscaled[0, 2]
        }
        with open(output_csv, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            writer.writerow(row)

if __name__ == "__main__":
    test_lstm_prediction_with_cnn_input_line_by_line()
