from flask import Flask, request, jsonify
import pickle
import numpy as np
import pandas as pd
import joblib
import os
import mlflow
from mlflow.tracking import MlflowClient

from sklearn.pipeline import Pipeline
from transformers import columnDropperTransformer,pickupDataTransformer,oneHotEncodingTransformer,standarScalingTransformer,missingDataTransformer, lowerCaseTransformer

# Allow all hosts to connect to Mlflow
os.environ["MLFLOW_ALLOWED_HOSTS"] = "*"

# Configure MLflow tracking URI and authentication
mlflow.set_tracking_uri("http://10.17.0.162:5000")
# mlflow.set_tracking_uri("http://127.0.0.1:5000")
app = Flask(__name__)

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
    model = app.config["MODEL"]
    pipeline = app.config.get("PIPELINE")

    if model is None or pipeline is None:
        return jsonify(error="Model or pipeline not loaded. Please call /reload first."), 503
    
    try:
        data = request.get_json()
        df = pd.DataFrame(data["data"])

        # Handle single sample (list of values) vs batch (list of lists)
        # if user_input and isinstance(user_input[0], (int, float, str)):
        #     user_input = [user_input]

        # Convert input to DataFrame
        # df = pd.DataFrame(user_input, columns=INPUT_COLS)
        
        # Ensure datetime column is actually datetime (required by pickupDataTransformer)
        df['tpep_pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime'])
 
        df['pickup_hour'] = df['tpep_pickup_datetime'].dt.hour
        df['pickup_day_of_week'] = df['tpep_pickup_datetime'].dt.dayofweek
        df['pickup_month'] = df['tpep_pickup_datetime'].dt.month
        df['pickup_day'] = df['tpep_pickup_datetime'].dt.day
        df.drop(['tpep_pickup_datetime'],axis=1,inplace=True)

        # Transform
        X = pipeline.transform(df)

        predictions = model.predict(X).tolist()

        return jsonify({
            "predictions": predictions
        })

    except Exception as e:
        print(f"Error during prediction: {e}", flush=True)
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