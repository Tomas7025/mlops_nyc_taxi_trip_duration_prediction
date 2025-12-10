import mlflow
import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error, mean_squared_error,r2_score, mean_absolute_error
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,StandardScaler

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor

# from transformers import columnDropperTransformer,pickupDataTransformer,oneHotEncodingTransformer,standarScalingTransformer,missingDataTransformer, lowerCaseTransformer

import pickle
# Se estiveres dentro da VM:
# mlflow.set_tracking_uri("http://10.17.0.162:5000")

# Ou se estiveres a usar túnel SSH (fora da VM, mas na VPN):

# mlflow.set_experiment("nyc_taxi_project")

CAT_COLS = ['VendorID']
NUM_COLS = ['passenger_count','trip_distance']

def preprocess_data(data):
	# data = data.head(1000000) # ! <- limitado a 1000000

	data['tpep_pickup_datetime'] = pd.to_datetime(data['tpep_pickup_datetime'])
	data['tpep_dropoff_datetime'] = pd.to_datetime(data['tpep_dropoff_datetime'])

	data['RatecodeID'] = pd.to_numeric(data['RatecodeID'], downcast='integer')
	data['duration'] = (data['tpep_dropoff_datetime'] - data['tpep_pickup_datetime']).dt.total_seconds() / 60

	data = data.drop(['tpep_dropoff_datetime', 'RatecodeID', 'store_and_fwd_flag', 'payment_type', 'fare_amount', 'extra', 'mta_tax', 'tip_amount', 'tolls_amount', 'improvement_surcharge', 'total_amount','congestion_surcharge','airport_fee'],axis=1)

	data['pickup_hour'] = data['tpep_pickup_datetime'].dt.hour
	data['pickup_day_of_week'] = data['tpep_pickup_datetime'].dt.dayofweek
	data['pickup_month'] = data['tpep_pickup_datetime'].dt.month
	data['pickup_day'] = data['tpep_pickup_datetime'].dt.day
	data.drop(['tpep_pickup_datetime'],axis=1,inplace=True)

	return data

def save_model(model):
	#to support multiple models
	model_name = model.__class__.__name__
	try:
		with open("models/" + model_name + ".pkl", "wb") as f:
			pickle.dump(model, f)
			print("Model " + str(model_name) + " saved")
	except Exception as e:
		print("Exception: " + str(e))

def calculate_metrics(y_test, y_pred):
	mse = mean_squared_error(y_test, y_pred)
	mae = mean_absolute_error(y_test, y_pred)
	rmse = root_mean_squared_error(y_test,y_pred)
	r2score = r2_score(y_test,y_pred)
	
	print(f"MSE: {str(round(mse,3))}")
	print(f"MAE: {str(round(mae, 3))}")
	print(f"RMSE: {str(round(rmse,3))}")
	print(f"R2SCORE: {str(round(r2score,3))}")

	mlflow.log_metric("MSE", mse)
	mlflow.log_metric("MAE", mae)
	mlflow.log_metric("RMSE", rmse)
	mlflow.log_metric("R2_SCORE", r2score)

	return {
		"MSE": mse,
		"MAE": mae,
		"RMSE": rmse,
		"R2SCORE": r2score
	}

def main():
	# Get env vars that come from github actions 
	import os
	COMMIT_SHA = "test_sha"#os.getenv('COMMIT_SHA')
	if not COMMIT_SHA:
		raise EnvironmentError("Missing required env var: COMMIT_SHA")

	MODEL_NAME = "best_model"#os.getenv('MLFLOW_MODEL_NAME')
	if not MODEL_NAME:
		raise EnvironmentError("Missing required env var: MLFLOW_MODEL_NAME")

	EXP_NAME = "test_best_model"#os.getenv('MLFLOW_EXPERIMENT_NAME')
	if not EXP_NAME:
		raise EnvironmentError("Missing required env var: MLFLOW_EXPERIMENT_NAME")
	
	# === mlflow init configs
	
	# Configure MLflow to use local server
	mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://10.17.0.167:5000")
	# mlflow.set_tracking_uri(mlflow_uri) 
	# mlflow_uri="10.17.0.162:5000"
	mlflow.set_tracking_uri(mlflow_uri)
	
	# Set experiment
	mlflow.set_experiment(EXP_NAME)

	with mlflow.start_run() as run:
		# === Read dataset ===
		# data = pd.read_parquet('data/raw/yellow_tripdata_2011-01.parquet')
		data = pd.read_parquet('data/preprocessed/final_subsampled_dataset.parquet')
		data = preprocess_data(data)
			
		# === Pipeline definition ===
		preprocessor = ColumnTransformer(
			transformers=[
				# (name, transformer_object, columns_to_apply_to)
				('scaler', StandardScaler(), NUM_COLS),
				('onehot', OneHotEncoder(sparse_output=False), CAT_COLS)
			],
			remainder='passthrough' # Keep all other columns un-transformed
		)
		baseline_pipeline = Pipeline(steps=[('pre_processor',preprocessor)])
		
		# === Data split ===
		y = data['duration']
		X = data.drop(['duration'],axis=1)	
		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

		baseline_pipeline.fit(X_train)
		X_train = baseline_pipeline.transform(X_train)
		X_test = baseline_pipeline.transform(X_test)

		# Comentei por agora para testar com tudo
		#
		# # === Compute correlations ===
		# data_train = X_train.copy()
		# data_train['duration'] = y_train.values
		# print(y_train.head(100))
		# print(X_train.head(100))
		# print(data_train.head(100))
		# corr = data_train.corr()['duration'].dropna().sort_values(key=abs, ascending=False)
		# print("\n📊 Feature correlations with target:")
		# print(corr.head(20))

		# # === Select top correlated features ===
		# top_features = corr.index[1:5+1].tolist()  # skip target itself
		# X_train = X_train[top_features]
		# X_test = X_test[top_features]

		best_error = 0
		best_run_id = None
		best_model = None
		models = [LinearRegression(), RandomForestRegressor(n_estimators=50, random_state=42), HistGradientBoostingRegressor(random_state=42)]
		for model in models:
			print("Training with: " + str(model.__class__.__name__))
			model.fit(X_train,y_train)

			joblib.dump(baseline_pipeline, "pipeline.pkl")
			mlflow.log_artifact("pipeline.pkl", artifact_path="preprocessing")
			signature = mlflow.models.infer_signature(X_train, model.predict(X_train))
			mlflow.sklearn.log_model(model, name=MODEL_NAME, signature=signature, input_example=X_train[:5])

			y_pred = model.predict(X_test)

			metrics = calculate_metrics(y_test, y_pred)
			if best_error == 0 or metrics['MSE'] < best_error:
				best_error = metrics['MSE']
				best_run_id = run.info.run_id
				best_model = model.__class__.__name__ 

		print(f"\nBest model run_id: {best_run_id} with MSE: {best_error}")
		print("Best model name:" + str(best_model))
		model_uri = f"runs:/{best_run_id}/{MODEL_NAME}"
		print(f"\nmodel_uri: {model_uri}")
			

	try:
		registered_model = mlflow.register_model(model_uri, MODEL_NAME)
		print(f"Registered model version: {registered_model.version}")
		
		# Promote to Production (required for Flask app to load it)
		from mlflow.tracking import MlflowClient
		# Assign alias (e.g., production)
		client = MlflowClient()
		client.set_registered_model_alias(
			name=MODEL_NAME,
			alias="staging",  # or "staging", "canary", etc.
			version=registered_model.version,
		)

		# Create alias for commit SHA (truncate for readability)
		client.set_registered_model_alias(
			name=MODEL_NAME,
			alias=COMMIT_SHA,
			version=registered_model.version,
		)

		print(f"✓ Model version {registered_model.version} promoted to Production")
		
	except Exception as e:
		print(f"ERROR: Failed to register/promote model: {e}")
		print(f"Model URI: {model_uri}")
		print(f"MLflow tracking URI: {os.getenv('MLFLOW_TRACKING_URI')}")
		raise  # Re-raise to fail the CI pipeline if registration fails
 


if __name__ == '__main__':
	main()
