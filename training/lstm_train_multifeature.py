import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from catboost import CatBoostClassifier, CatBoostRegressor

# --- LSTM imports ---
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense
from tensorflow.keras.utils import to_categorical

# -----------------------------------------------------------------------------
# 1) Load & preprocess full DataFrame (for CatBoost and LSTM training)
# -----------------------------------------------------------------------------
df = pd.read_csv("district5_complex_routes_with_datetime.csv")
df['Datetime'] = pd.to_datetime(df['Date'] + " " + df['Time'])
df = df.sort_values('Datetime').reset_index(drop=True)

# Encode target labels for CatBoost
df['Route_Code']   = df['Route'].astype('category').cat.codes
df['Weather_Code'] = df['Weather Conditions'].astype('category').cat.codes

# Time features
df['hour']       = df['Datetime'].dt.hour
df['dow']        = df['Datetime'].dt.dayofweek
df['month']      = df['Datetime'].dt.month
df['day']        = df['Datetime'].dt.day
df['is_weekend'] = (df['dow'] >= 5).astype(int)

# Lag features
for lag in (1,2,3):
    df[f'Accident_L{lag}'] = df['Accident Reports'].shift(lag)
    df[f'Traffic_L{lag}']  = df['Traffic Intensity'].shift(lag)
    df[f'Distance_L{lag}'] = df['Distance'].shift(lag)
    df[f'Route_L{lag}']    = df['Route_Code'].shift(lag)
    df[f'Weather_L{lag}']  = df['Weather_Code'].shift(lag)

# Drop NA rows introduced by lagging
df.dropna(inplace=True)
# -----------------------------------------------------------------------------
# 2) Train CatBoost models
# -----------------------------------------------------------------------------
CAT_FEATS = [
    'Origin','Destination','hour','dow','month','day','is_weekend'
] + [f'Route_L{l}' for l in (1,2,3)] + [f'Weather_L{l}' for l in (1,2,3)]
NUM_FEATS = [
    'Accident Reports','Traffic Intensity','Distance'
] + [f'Accident_L{l}' for l in (1,2,3)] \
  + [f'Traffic_L{l}' for l in (1,2,3)] \
  + [f'Distance_L{l}' for l in (1,2,3)]

# Cast cats to str
for c in CAT_FEATS:
    df[c] = df[c].astype(str)

# Split
X         = df[CAT_FEATS + NUM_FEATS]
y_route   = df['Route_Code']
y_weather = df['Weather_Code']
y_cont    = df[['Accident Reports','Traffic Intensity','Distance']]

X_tr, X_te, r_tr, r_te, w_tr, w_te, c_tr, c_te = train_test_split(
    X, y_route, y_weather, y_cont,
    test_size=0.2, random_state=42
)

# Route classifier
clf_route = CatBoostClassifier(
    iterations=500, learning_rate=0.05, depth=6,
    eval_metric='Accuracy', cat_features=CAT_FEATS, verbose=100
)
clf_route.fit(X_tr, r_tr, eval_set=(X_te, r_te), early_stopping_rounds=50)

# Weather classifier
clf_weather = CatBoostClassifier(
    iterations=300, learning_rate=0.1, depth=4,
    eval_metric='Accuracy', cat_features=CAT_FEATS, verbose=100
)
clf_weather.fit(X_tr, w_tr, eval_set=(X_te, w_te), early_stopping_rounds=30)

# Continuous regressor
reg_cont = CatBoostRegressor(
    iterations=300, learning_rate=0.1, depth=6,
    loss_function='MultiRMSE', cat_features=CAT_FEATS, verbose=100
)
reg_cont.fit(X_tr, c_tr, eval_set=(X_te, c_te), early_stopping_rounds=30)

# Save CatBoost artifacts
clf_route.save_model('cat_route.cbm')
clf_weather.save_model('cat_weather.cbm')
reg_cont.save_model('cat_cont.cbm')
joblib.dump((CAT_FEATS, NUM_FEATS), 'feature_lists.pkl')
print("CatBoost models + feature_lists.pkl saved.")
# -----------------------------------------------------------------------------
# 3) Prepare & train a simple single‐step LSTM
# -----------------------------------------------------------------------------
# 3a) Fit encoders & scaler on entire history
le_origin      = LabelEncoder().fit(df['Origin'])
le_destination = LabelEncoder().fit(df['Destination'])
le_route       = LabelEncoder().fit(df['Route'])
le_weather     = LabelEncoder().fit(df['Weather Conditions'])
scaler_continuous = StandardScaler().fit(
    df[['Accident Reports','Traffic Intensity','Distance']]
)

# Hyperparameters
seq_length       = 10
prediction_steps = 1

# Raw arrays
orig_arr    = le_origin.transform(df['Origin'])
dest_arr    = le_destination.transform(df['Destination'])
route_arr   = df['Route_Code'].to_numpy()
weather_arr = df['Weather_Code'].to_numpy()
cont_arr    = df[['Accident Reports','Traffic Intensity','Distance']].to_numpy()

# Build training sequences
X_seq = []
y_route_seq = []
y_weather_seq = []
y_cont_seq = []

for i in range(seq_length, len(df) - prediction_steps + 1):
    # input = last seq_length of origin & dest
    seq_o = orig_arr[i-seq_length:i]
    seq_d = dest_arr[i-seq_length:i]
    X_seq.append(np.stack([seq_o, seq_d], axis=1))
    # targets at step i (single‐step ahead)
    y_route_seq.append(route_arr[i])
    y_weather_seq.append(weather_arr[i])
    y_cont_seq.append(cont_arr[i])

X_seq = np.array(X_seq, dtype=np.float32)                 # shape (N, seq_length, 2)
y_route_seq   = to_categorical(y_route_seq, num_classes=len(le_route.classes_))
y_weather_seq = to_categorical(y_weather_seq, num_classes=len(le_weather.classes_))
y_cont_seq    = np.array(y_cont_seq, dtype=np.float32)    # shape (N, 3)

# 3b) Build multi‐output LSTM
inp = Input(shape=(seq_length, 2), name='lstm_input')
x   = LSTM(64)(inp)

route_out   = Dense(len(le_route.classes_),   activation='softmax', name='route_out')(x)
weather_out = Dense(len(le_weather.classes_), activation='softmax', name='weather_out')(x)
cont_out    = Dense(3,                        activation='linear', name='cont_out')(x)

model = Model(inputs=inp, outputs=[route_out, weather_out, cont_out])
model.compile(
    optimizer='adam',
    loss={
        'route_out':   'categorical_crossentropy',
        'weather_out': 'categorical_crossentropy',
        'cont_out':    'mse'
    },
    metrics={
        'route_out':   'accuracy',
        'weather_out': 'accuracy',
        'cont_out':    'mse'
    }
)

# 3c) Train
model.fit(
    X_seq,
    {'route_out': y_route_seq,
     'weather_out': y_weather_seq,
     'cont_out': y_cont_seq},
    epochs=20,
    batch_size=32,
    validation_split=0.1,
)

# 3d) Save LSTM + encoders + scaler + meta
model.save('lstm_model.h5')
joblib.dump(le_origin,      'le_origin.pkl')
joblib.dump(le_destination, 'le_destination.pkl')
joblib.dump(le_route,       'le_route.pkl')
joblib.dump(le_weather,     'le_weather.pkl')
joblib.dump(scaler_continuous, 'scaler_continuous.pkl')
joblib.dump(
    {'seq_length': seq_length, 'prediction_steps': prediction_steps},
    'lstm_meta.pkl'
)
print("LSTM model + encoders + scaler + meta saved.")
