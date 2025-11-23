from flask import Flask, request, jsonify
import pickle
import numpy as np
import os
import mlflow
from mlflow.tracking import MlflowClient

from sklearn.pipeline import Pipeline
from src.transformers import columnDropperTransformer,pickupDataTransformer,oneHotEncodingTransformer,standarScalingTransformer,missingDataTransformer, lowerCaseTransformer

# Allow all hosts to connect to Mlflow
os.environ["MLFLOW_ALLOWED_HOSTS"] = "*"

# Configure MLflow tracking URI and authentication
mlflow.set_tracking_uri("http://10.17.0.162:5000")
app = Flask(__name__)

CAT_COLS = ['vendor_id','rate_code','payment_type']
NUM_COLS = ['passenger_count','trip_distance','pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude','fare_amount','surcharge','mta_tax','tip_amount','tolls_amount','total_amount']

MODEL_NAME = os.getenv('MLFLOW_MODEL_NAME')
if not MODEL_NAME:
    raise EnvironmentError("Missing required env var: MLFLOW_MODEL_NAME")

MODEL_ALIAS = os.getenv("MODEL_ALIAS")

# Try to load model once on startup
try:
    app.config["MODEL"] = mlflow.pyfunc.load_model(
        model_uri=f"models:/{MODEL_NAME}@{MODEL_ALIAS}"
    )
    print("Model loaded successfully at startup.")
except Exception as e:
    app.config["MODEL"] = None
    print(f"Could not load model at startup: {e}")
    print("App will start without a model. You can load it later using /reload.")

#not being used rn
def load_model(model_name):
    path = "models/" + model_name + ".pkl"
    with open(path, "rb") as f:
        model = pickle.load(f)
    return model


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
    if model is None:
        return jsonify(error="Model not loaded. Please call /reload first."), 503
    
    try:
        data = request.get_json()

        user_input = data["inputs"]

        X = np.array(user_input)
        baseline_pipeline = Pipeline(steps=[('dropper',columnDropperTransformer(['store_and_fwd_flag','dropoff_datetime'])),
									('lower_case',lowerCaseTransformer()),
									('data_transformer',pickupDataTransformer()),
									('one_hot_encoding',oneHotEncodingTransformer(CAT_COLS)),
									('numerical_scaling',standarScalingTransformer(NUM_COLS))])

        baseline_pipeline.fit(X)
        X = baseline_pipeline.transform(X)

        #if the user sends only one sample
        if X.ndim == 1:
            X = X.reshape(1, -1)

        predictions = model.predict(X).tolist()

        return jsonify({
            "predictions": predictions
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/reload", methods=["GET"])
def reload_model():
    """Reload model from MLflow and store in Flask app config."""
    try:
        model = mlflow.pyfunc.load_model(model_uri=f"models:/{MODEL_NAME}@{MODEL_ALIAS}")
        app.config["MODEL"] = model
        return jsonify(message="Model reloaded successfully.")
    except Exception as e:
        return jsonify(error=f"Failed to load model: {e}"), 500


def main():
    app.run(host="0.0.0.0", port=5000, debug=True)


if __name__ == "__main__":
    main()

#curl -X POST http://127.0.0.1:5000/predict \
#     -H "Content-Type: application/json" \
#     -d '{
#           "model": "random_forest",
#           "inputs": [5.1, 3.5, 1.4, 0.2]
#         }'
