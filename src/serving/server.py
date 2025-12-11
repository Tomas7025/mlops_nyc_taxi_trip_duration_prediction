from flask import Flask, request, jsonify
import pickle
import numpy as np
import pandas as pd
import joblib
import os
import mlflow
from mlflow.tracking import MlflowClient

from sklearn.pipeline import Pipeline
# from transformers import columnDropperTransformer,pickupDataTransformer,oneHotEncodingTransformer,standarScalingTransformer,missingDataTransformer, lowerCaseTransformer
from prometheus_client import Histogram, Counter, Gauge, generate_latest, CONTENT_TYPE_LATEST

# Allow all hosts to connect to Mlflow
os.environ["MLFLOW_ALLOWED_HOSTS"] = "*"

# Configure MLflow tracking URI and authentication
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI","http://10.17.0.162:5000"))
app = Flask(__name__)


# Standard Metrics (Manual)
HTTP_REQUEST_COUNT = Counter(
    'http_requests_total',
    'Total HTTP Requests',
    ['method', 'endpoint', 'status_code']
)

HTTP_REQUEST_LATENCY = Histogram(
    'http_request_duration_seconds',
    'HTTP Request latency',
    ['method', 'endpoint']
)

# Custom Metrics
# PREDICTION_DISTRIBUTION = Histogram(
#     'model_prediction_fare', 
#     'Distribution of predicted fare amounts', 
#     buckets=[0, 5, 10, 15, 20, 25, 30, 40, 50, 100]
# )
PREDICTION_DISTRIBUTION = Histogram(
    'model_prediction_duration_minutes',
    'Distribution of predicted trip duration',
    # Buckets ajustados para minutos (viagens curtas e longas)
    buckets=[0, 2, 4, 6, 8, 10, 12, 15, 20, 25, 30, 45, 60, 90, 120]
)

# Monitorizar a Hora do Pedido (Temporal Drift)
INPUT_PICKUP_HOUR = Histogram(
    'model_input_pickup_hour',
    'Distribution of pickup hours (0-23)',
    buckets=[0, 4, 8, 12, 16, 20, 24] # Buckets de 4 em 4 horas (Madrugada, Manhã, Tarde, Noite)
)

PREDICTION_COUNTER = Counter('model_predictions_total', 'Total number of predictions', ['vendor_id'])
# LAST_PREDICTION_VALUE = Gauge('model_last_prediction_fare', 'The last predicted fare amount')
LAST_PREDICTION_VALUE = Gauge('model_last_prediction_duration_minutes', 'The last predicted trip duration in minutes')

# Input Drift Metrics
INPUT_TRIP_DISTANCE = Histogram(
    'model_input_trip_distance',
    'Distribution of input trip distances',
    buckets=[0, 1, 2, 5, 10, 20, 50]
)

INPUT_PASSENGER_COUNT = Histogram(
    'model_input_passenger_count',
    'Distribution of input passenger counts',
    buckets=[0, 1, 2, 3, 4, 5, 6, 10]
)

INPUT_PICKUP_LOCATION = Histogram(
    'model_input_pulocation_id',
    'Distribution of Pickup Location IDs',
    buckets=[0, 50, 100, 150, 200, 265] # IDs vão de 1 a 263 em NY
)


CAT_COLS = ['VendorID','RatecodeID','payment_type']
NUM_COLS = ['passenger_count','trip_distance','PULocationID','DOLocationID','fare_amount','extra','mta_tax','tip_amount','tolls_amount','improvement_surcharge','total_amount']
INPUT_COLS = ['VendorID', 'tpep_pickup_datetime', 'passenger_count', 'trip_distance', 'PULocationID', 'DOLocationID']

MODEL_NAME = os.getenv('MLFLOW_MODEL_NAME')
if not MODEL_NAME:
    raise EnvironmentError("Missing required env var: MLFLOW_MODEL_NAME")

MODEL_ALIAS = os.getenv("MODEL_ALIAS")

# Try to load model and pipeline once on startup
try:
    app.config["MODEL"] = mlflow.sklearn.load_model(
        model_uri=f"models:/{MODEL_NAME}@{MODEL_ALIAS}"
    )
    
    # Get run_id to load pipeline artifact
    client = MlflowClient()
    alias_info = client.get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS)
    run_id = alias_info.run_id
    
    local_path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="preprocessing/pipeline.pkl")
    app.config["PIPELINE"] = joblib.load(local_path)
    
    print("Model and pipeline loaded successfully at startup.")
except Exception as e:
    app.config["MODEL"] = None
    app.config["PIPELINE"] = None
    print(f"Could not load model/pipeline at startup: {e}")
    print("App will start without a model. You can load it later using /reload.")

# #not being used rn
# def load_model(model_name):
#     path = "models/" + model_name + ".pkl"
#     with open(path, "rb") as f:
#         model = pickle.load(f)
#     return model


#Endpoints

@app.route('/metrics')
def metrics():
    return generate_latest(), 200, {'Content-Type': CONTENT_TYPE_LATEST}

@app.route("/model-info", methods=["GET"])
def model_info():
    """Return current model alias version + run ID hash"""
    try:
        client = MlflowClient()
        alias_info = client.get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS)
        return jsonify(
            model_name=MODEL_NAME,
            alias=MODEL_ALIAS,
            version=alias_info.version,
            run_id=alias_info.run_id,  # Commit SHA is stored in run_id (if you passed it)
        )
    except Exception as e:
        return jsonify(error=str(e)), 500
    
@app.route("/health", methods=["GET"])
def health():
    """Simple health check."""
    return jsonify(status="healthy", model_loaded=app.config["MODEL"] is not None)

@app.route("/predict", methods=["POST"])
def predict():
    """Make predictions using the model stored in app config."""
    with HTTP_REQUEST_LATENCY.labels(method='POST', endpoint='/predict').time():
        model = app.config["MODEL"]
        pipeline = app.config.get("PIPELINE")

        if model is None or pipeline is None:
            HTTP_REQUEST_COUNT.labels(method='POST', endpoint='/predict', status_code=503).inc()
            return jsonify(error="Model or pipeline not loaded. Please call /reload first."), 503
        
        try:
            data = request.get_json()
            df = pd.DataFrame(data["data"])

            # Handle single sample (list of values) vs batch (list of lists)
            # if user_input and isinstance(user_input[0], (int, float, str)):
            #     user_input = [user_input]

            # Convert input to DataFrame
            # df = pd.DataFrame(user_input, columns=INPUT_COLS)
            
            # Capture VendorIDs for metrics
            vendor_ids = df['VendorID'].tolist() if 'VendorID' in df.columns else []
            
            # Log Input Metrics
            if 'trip_distance' in df.columns:
                for val in df['trip_distance']:
                    INPUT_TRIP_DISTANCE.observe(float(val))
            
            if 'passenger_count' in df.columns:
                for val in df['passenger_count']:
                    INPUT_PASSENGER_COUNT.observe(float(val))

            if 'PULocationID' in df.columns:
                for val in df['PULocationID']:
                    INPUT_PICKUP_LOCATION.observe(float(val))

            # Ensure datetime column is actually datetime (required by pickupDataTransformer)
            df['tpep_pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime'])
    
            # Log pickup hour for temporal drift monitoring
            hours = df['tpep_pickup_datetime'].dt.hour
            for hr in hours:
                INPUT_PICKUP_HOUR.observe(float(hr))

            df['pickup_hour'] = df['tpep_pickup_datetime'].dt.hour
            df['pickup_day_of_week'] = df['tpep_pickup_datetime'].dt.dayofweek
            df['pickup_month'] = df['tpep_pickup_datetime'].dt.month
            df['pickup_day'] = df['tpep_pickup_datetime'].dt.day
            df.drop(['tpep_pickup_datetime'],axis=1,inplace=True)

            # Transform
            X = pipeline.transform(df)

            predictions = model.predict(X).tolist()

            # Log custom metrics
            for i, pred in enumerate(predictions):
                PREDICTION_DISTRIBUTION.observe(pred)
                LAST_PREDICTION_VALUE.set(pred)
                
                v_id = str(vendor_ids[i]) if i < len(vendor_ids) else 'unknown'
                PREDICTION_COUNTER.labels(vendor_id=v_id).inc()
                
                print(f"Logged metric for prediction: {pred}, VendorID: {v_id}", flush=True)

            HTTP_REQUEST_COUNT.labels(method='POST', endpoint='/predict', status_code=200).inc()
            return jsonify({
                "predictions": predictions
            })

        except Exception as e:
            print(f"Error during prediction: {e}", flush=True)
            HTTP_REQUEST_COUNT.labels(method='POST', endpoint='/predict', status_code=400).inc()
            return jsonify({"error": str(e)}), 400

@app.route("/reload", methods=["GET"])
def reload_model():
    """Reload model from MLflow and store in Flask app config."""
    try:
        # Reload model
        model = mlflow.sklearn.load_model(model_uri=f"models:/{MODEL_NAME}@{MODEL_ALIAS}")
        app.config["MODEL"] = model
        
        # Reload pipeline
        client = MlflowClient()
        alias_info = client.get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS)
        run_id = alias_info.run_id
        
        local_path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="preprocessing/pipeline.pkl")
        app.config["PIPELINE"] = joblib.load(local_path)
        
        return jsonify(message="Model and pipeline reloaded successfully.")
    except Exception as e:
        return jsonify(error=f"Failed to load model/pipeline: {e}"), 500


def main():
    app.run(host="0.0.0.0", port=8080, debug=True)


if __name__ == "__main__":
    main()

#curl -X POST http://127.0.0.1:5000/predict \
#     -H "Content-Type: application/json" \
#     -d '{
#           "model": "random_forest",
#           "inputs": [5.1, 3.5, 1.4, 0.2]
#         }'