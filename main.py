from flask import Flask, render_template, request, jsonify
from chatbot_engine import process_user_input  # Chatbot logic
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64
import numpy as np
import pandas as pd
import yfinance as yf
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime

app = Flask(__name__)

# ==================== ROUTES =======================

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    user_message = request.json["message"]
    bot_response = process_user_input(user_message)
    return jsonify({"response": bot_response})


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        symbol = data.get("symbol", "AAPL").upper()
        days = int(data.get("days", 7))

        # === 1. Load stock data
        end_date = datetime.now().strftime('%Y-%m-%d')
        df = yf.download(symbol, start="2012-01-01", end=end_date)
        df = df[["Close"]].dropna()

        # === 2. Preprocess
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(df)
        sequence_len = 100
        last_sequence = scaled_data[-sequence_len:]

        # === 3. Load model (trained & saved beforehand)
        model = load_model("stock_lstm_model.h5")

        # === 4. Predict future prices recursively
        predicted_scaled = []
        seq = last_sequence.copy()

        for _ in range(days):
            x = seq.reshape(1, sequence_len, 1)
            pred = model.predict(x)[0][0]
            predicted_scaled.append(pred)
            seq = np.append(seq[1:], [[pred]], axis=0)

        predicted_prices = scaler.inverse_transform(np.array(predicted_scaled).reshape(-1, 1)).flatten()

        # === 5. Detect correct currency
        if symbol.endswith(".NS") or symbol.endswith(".BO") or symbol in ["^NSEI", "^BSESN"]:
            currency_symbol = "₹"  # Indian stocks (NSE, BSE)
        else:
            currency_symbol = "$"  # All others assume USD

        # === 6. Generate base64 chart
        future_dates = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=days)
        plt.figure(figsize=(8, 4))
        plt.plot(future_dates, predicted_prices, marker='o', color='orange')
        plt.title(f"{symbol} Stock Price Forecast - Next {days} Days")
        plt.xlabel("Date")
        plt.ylabel(f"Predicted Price ({currency_symbol})")
        plt.grid(True)

        img = io.BytesIO()
        plt.tight_layout()
        plt.savefig(img, format='png')
        plt.close()
        img.seek(0)
        chart_data = base64.b64encode(img.read()).decode("utf-8")

        # === 7. Format prediction output
        predictions = "\n".join(
            [f"{date.strftime('%Y-%m-%d')}: {currency_symbol}{price:.2f}" for date, price in zip(future_dates, predicted_prices)]
        )

        return jsonify({
            "predictions": predictions,
            "chart": f"data:image/png;base64,{chart_data}"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ==================== RUN =======================
if __name__ == "__main__":
    app.run(debug=True)
