import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import layers
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
import datetime

# === 1. Take Inputs from User ===
stock = input("Enter Stock Ticker (e.g., GOOG, AAPL, TSLA): ").upper()
future_days = int(input("Enter number of future days to predict: "))

# === 2. Download Stock Data ===
start_date = '2012-01-01'
end_date = datetime.datetime.now().strftime('%Y-%m-%d')
sequence_length = 100
train_ratio = 0.8

print(f"\nDownloading data for {stock}...")
df = yf.download(stock, start=start_date, end=end_date)
df = df[['Close']].dropna()

# === 3. Plot Historical MAs ===
plt.figure(figsize=(10,5))
df['MA100'] = df['Close'].rolling(100).mean()
df['MA200'] = df['Close'].rolling(200).mean()
plt.plot(df['Close'], label='Close Price')
plt.plot(df['MA100'], label='100-Day MA', color='r')
plt.plot(df['MA200'], label='200-Day MA', color='b')
plt.title(f'{stock} Closing Price & Moving Averages')
plt.legend()
plt.show()

# === 4. Preprocess ===
data = df['Close'].values.reshape(-1, 1)
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

train_size = int(len(scaled_data) * train_ratio)
train_data = scaled_data[:train_size]

# === 5. Create Sequences ===
def create_sequences(data, seq_len):
    X, y = [], []
    for i in range(seq_len, len(data)):
        X.append(data[i-seq_len:i])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

X_train, y_train = create_sequences(train_data, sequence_length)

# === 6. Build Model ===
model = Sequential([
    LSTM(50, activation='relu', return_sequences=True, input_shape=(X_train.shape[1], 1)),
    Dropout(0.2),
    LSTM(60, activation='relu', return_sequences=True),
    Dropout(0.3),
    LSTM(80, activation='relu', return_sequences=True),
    Dropout(0.4),
    LSTM(120, activation='relu'),
    Dropout(0.5),
    Dense(1)
])
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)

# === 7. Predict Future (Recursive) ===
last_sequence = scaled_data[-sequence_length:]
predicted_scaled = []

for _ in range(future_days):
    X_input = last_sequence.reshape(1, sequence_length, 1)
    next_scaled = model.predict(X_input, verbose=0)[0][0]
    predicted_scaled.append(next_scaled)
    last_sequence = np.append(last_sequence[1:], [[next_scaled]], axis=0)

# === 8. Inverse Scale & Show ===
predicted_prices = scaler.inverse_transform(np.array(predicted_scaled).reshape(-1, 1))
future_dates = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=future_days)

# Show predictions as numbers
print(f"\n📈 Predicted Closing Prices for {stock}:\n")
for date, price in zip(future_dates, predicted_prices):
    print(f"{date.date()} ➤ ₹{price[0]:.2f}")

# === 9. Plot Prediction ===
plt.figure(figsize=(10,5))
plt.plot(future_dates, predicted_prices, marker='o', color='orange', label='Future Prediction')
plt.title(f"📊 {stock} Stock Price Forecast for Next {future_days} Days")
plt.xlabel("Date")
plt.ylabel("Predicted Price (₹)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# === 10. Save Model ===
model.save(f"stock_lstm_model.h5")
