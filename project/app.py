from flask import Flask, render_template, request, jsonify
import pandas as pd
from prophet import Prophet

app = Flask(__name__)

# Load and process bus data
def load_data():
    try:
        df = pd.read_csv("bus_data.csv")
        df.columns = ["ds", "y"]  # Prophet requires "ds" (date) and "y" (value)
        df["ds"] = pd.to_datetime(df["ds"])

        # Convert 'y' (arrival time) from HH:MM to total minutes
        df["y"] = df["y"].apply(lambda x: int(x.split(":")[0]) * 60 + int(x.split(":")[1]))

        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

# Train Prophet model
def train_model(df):
    model = Prophet()
    model.fit(df)
    return model
# Predict bus arrivals
def predict_future(model, periods=10, freq="5min"):
    future = model.make_future_dataframe(periods=periods, freq=freq)
    forecast = model.predict(future)

    # Convert predicted minutes back to HH:MM
    forecast["yhat_readable"] = forecast["yhat"].apply(lambda x: f"{int(x // 60):02}:{int(x % 60):02}")
    return forecast

# Detect bus bunching
def detect_bunching(forecast, threshold=3):
    forecast["diff"] = forecast["yhat"].diff().abs()
    bunching_buses = forecast[forecast["diff"] < threshold]
    return not bunching_buses.empty

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        df = load_data()
        
        if df is None:
            print('1')
            
            return jsonify({"error": "Data loading failed"}), 200   

        model = train_model(df)
        forecast = predict_future(model)

        bus_bunching_detected = detect_bunching(forecast)

 # Prepare the response as a table-friendly JSON
        results = []
        for _, row in forecast.iterrows():
            results.append({"Time": str(row["ds"]), "Predicted Arrival": row["yhat_readable"]})
        data =jsonify({"bunching": bus_bunching_detected, "predictions": results})
        print(data)
        return jsonify({"bunching": bus_bunching_detected, "predictions": results}), 200

    except Exception as e:
        print(3)
        print(f"Error in prediction route: {e}")
        return jsonify({"error": f"An error occurred: {e}"}), 200

if __name__ == "__main__":
    app.run(debug=True, port=5000,host="0.0.0.0")  