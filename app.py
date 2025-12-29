import pandas as pd
import numpy as np
from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from dotenv import load_dotenv
import os
import joblib
import pickle
import datetime
import requests

app = Flask(__name__)

# configuraion
load_dotenv()
API_KEY = os.getenv("OPENWEATHER_API_KEY")
CITY = "New York"

if not API_KEY:
    raise ValueError("No API Key found")

print("Load the model and scalers")
model = load_model("gru_model.keras")
scaler_x = joblib.load("scale_X.pkl") 
scaler_y = joblib.load("scaler_y.pkl") 
cheat_sheet = pd.read_csv("station_averages.csv")

# Load Column Names
with open("col_X_names.pkl", "rb") as f:
    cols_numeric = pickle.load(f) 

with open("oh_col_names.pkl", "rb") as f:
    cols_onehot = pickle.load(f)  

print("Get the website")

# website
def get_live_weather():
    url = f"http://api.openweathermap.org/data/2.5/weather?q={CITY}&appid={API_KEY}&units=imperial"
    try:
        data = requests.get(url).json()
        temp = data['main']['temp']
        precip = data.get('rain', {}).get('1h', 0.0)
        return temp, precip
    except:
        print("Unavailable Weather API")
        return 65.0, 0.0

def get_historical_data(station, weekday, hour):
    if hour < 0: 
        hour = 23
        weekday = weekday - 1 if weekday > 0 else 6

    match = cheat_sheet[
        (cheat_sheet['station'] == station) & 
        (cheat_sheet['weekday'] == weekday) & 
        (cheat_sheet['hour'] == hour)
    ]
    
    if not match.empty:
        return match.iloc[0]['entry_per_hour'], match.iloc[0]['borough']
    else:
        return 500, "Manhattan" 

# route
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    station_clean = data['station'] 
    req_hour = int(data['hour'])

    now = datetime.datetime.now()
    weekday = now.weekday()
    month = now.month
    temp, precip = get_live_weather()

    lag_1, borough = get_historical_data(station_clean, weekday, req_hour - 1)
    
    prev_day = weekday - 1 if weekday > 0 else 6
    lag_24, _ = get_historical_data(station_clean, prev_day, req_hour)
    lag_week, _ = get_historical_data(station_clean, weekday, req_hour)

    df_numeric = pd.DataFrame(columns=cols_numeric)
    df_numeric.loc[0] = 0 

    df_numeric['hour_sin'] = np.sin(2 * np.pi * req_hour / 24)
    df_numeric['hour_cos'] = np.cos(2 * np.pi * req_hour / 24)
    df_numeric['weekday_sin'] = np.sin(2 * np.pi * weekday / 7)
    df_numeric['weekday_cos'] = np.cos(2 * np.pi * weekday / 7)
    df_numeric['month_sin'] = np.sin(2 * np.pi * month / 12)
    df_numeric['month_cos'] = np.cos(2 * np.pi * month / 12)

    df_numeric['hr_lag'] = lag_1
    df_numeric['day_lag'] = lag_24
    df_numeric['week_lag'] = lag_week
    df_numeric['temp'] = temp
    df_numeric['precip'] = precip

    df_numeric['isweekend'] = 1 if weekday >= 5 else 0
    df_numeric['is_rush_hr'] = 1 if (7<=req_hour<=10) or (17<=req_hour<=20) else 0
    df_numeric['is_holiday'] = 0 
    df_numeric['is_raining'] = 1 if precip > 0 else 0

    try:
        X_numeric_scaled = scaler_x.transform(df_numeric)
    except Exception as e:
        return jsonify({'error': f"Scaler Error: {str(e)}"})

    df_onehot = pd.DataFrame(columns=cols_onehot)
    df_onehot.loc[0] = 0.0 

    sta_col = f"sta_{station_clean}"
    boro_col = f"boro_{borough}"
    
    if sta_col in df_onehot.columns: df_onehot[sta_col] = 1.0
    if boro_col in df_onehot.columns: df_onehot[boro_col] = 1.0
    
    X_onehot_raw = df_onehot.values
    input_final = np.hstack([X_numeric_scaled, X_onehot_raw])
    input_reshaped = input_final.reshape(1, 1, input_final.shape[1])

    pred_scaled = model.predict(input_reshaped, verbose=0)
    pred_real = scaler_y.inverse_transform(pred_scaled)
    
    result = int(pred_real[0][0])
    if result < 0: result = 0
        
    return jsonify({
        'crowd_count': result,
        'weather': f"{temp}°F"
    })

if __name__ == '__main__':
    app.run(debug=True)

