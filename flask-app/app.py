from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import os
import requests

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from realtime_crypto import RealTimeCrypto
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 

from tensorflow.keras.models import load_model
import joblib

app = Flask(__name__)

# Load model and scaler once
model = load_model('C:/Projects/bitcoin-predictor/flask-app/model.keras')
scaler = joblib.load('C:/Projects/bitcoin-predictor/flask-app/scaler.pkl')

# Define the feature columns in order (must match training!)
FEATURES = ['High', 'Low', 'Close', 'SMA_21']
SEQUENCE_LENGTH = 30

MAE = 2229  # You can update this based on evaluation

@app.route("/", methods=["GET"])
def index():
    # Get current BTC price
    tracker = RealTimeCrypto()
    bitcoin = tracker.get_coin("bitcoin")
    current_price = float(bitcoin.get_price())
    print(f"Current BTC Price: {current_price}")  # Debugging line to check current price
    
    res = requests.get('https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval=1d&limit=30')
    data = res.json()
    
    # Get these values from the data: 'High', 'Low', 'Close', 'SMA_21' and store them in a DataFrame
    df = pd.DataFrame(data, columns=['Open Time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close Time', 'Quote Asset Volume', 'Number of Trades', 'Taker Buy Base Asset Volume', 'Taker Buy Quote Asset Volume', 'Ignore'])
    df = df[['High', 'Low', 'Close']].astype(float)
    df['SMA_21'] = df['Close'].rolling(window=21).mean()
    df = df.dropna()
    print(df.tail())  # Debugging line to check the last few rows of the DataFrame
    

    # Scale and prepare sequence
    scaled = scaler.transform(df)
    X = np.expand_dims(scaled, axis=0)

    # Predict
    predicted_scaled = model.predict(X)
    high_index = FEATURES.index('High')
    predicted_high = scaler.inverse_transform(
        [[0]*high_index + [predicted_scaled[0][0]] + [0]*(len(FEATURES)-high_index-1)]
    )[0][high_index]

    # Determine signal
    delta = predicted_high - current_price
    if delta > MAE:
        signal = "📈 Likely UP"
    elif delta < -MAE:
        signal = "📉 Likely DOWN"
    else:
        signal = "🔍 Uncertain (within margin of error)"

    return render_template(
        "index.html",
        prediction=f"${predicted_high:,.2f}",
        current_price=f"${current_price:,.2f}",
        signal=signal
    )



if __name__ == "__main__":
    app.run(debug=True)