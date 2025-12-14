import mlflow
import pandas as pd
import joblib
import numpy as np
import json
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error, mean_squared_error,r2_score, mean_absolute_error
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,StandardScaler

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer

# from transformers import columnDropperTransformer,pickupDataTransformer,oneHotEncodingTransformer,standarScalingTransformer,missingDataTransformer, lowerCaseTransformer

import pickle
from sklearn import set_config
set_config(transform_output="pandas")

# Se estiveres dentro da VM:
# mlflow.set_tracking_uri("http://10.17.0.162:5000")

# Ou se estiveres a usar túnel SSH (fora da VM, mas na VPN):

# mlflow.set_experiment("nyc_taxi_project")

CAT_COLS = ['VendorID']
NUM_COLS = ['passenger_count','trip_distance']

def preprocess_data(data):
	#data = data.head(1000000) # ! <- limitado a 1000000

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

	return {
		"MSE": mse,
		"MAE": mae,
		"RMSE": rmse,
		"R2SCORE": r2score
	}

def main():
	# Get env vars that come from github actions 
	import os
	COMMIT_SHA = os.getenv('COMMIT_SHA',"test_sha")
	if not COMMIT_SHA:
		raise EnvironmentError("Missing required env var: COMMIT_SHA")

	MODEL_NAME = os.getenv('MLFLOW_MODEL_NAME',"best_model")
	if not MODEL_NAME:
		raise EnvironmentError("Missing required env var: MLFLOW_MODEL_NAME")

	EXP_NAME = os.getenv('MLFLOW_EXPERIMENT_NAME',"test_best_model")
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
		#data = pd.read_parquet('data/raw/yellow_tripdata_2011-01.parquet')
		data = pd.read_parquet('data/preprocessed/final_subsampled_dataset.parquet')
		validation_data = pd.read_parquet('data/preprocessed/validation_subsampled_dataset.parquet')
		#data = preprocess_data(data)
			
		# === Pipeline definition ===
		#treat 0 trip_distance as missing values
		data.loc[data['trip_distance'] == 0, 'trip_distance'] = np.nan
		preprocessor = ColumnTransformer(
			transformers=[
				(
					'num',
					Pipeline([
						('imputer', SimpleImputer(strategy='mean')),       # <-- Imputes trip_distance
						('scaler', StandardScaler())
					]),
					NUM_COLS
				),

				(
					'cat',
					Pipeline([
						('imputer', SimpleImputer(strategy='most_frequent')),
						('onehot', OneHotEncoder(sparse_output=False))
					]),
					CAT_COLS
				)
			],
			remainder='passthrough'
		)
		baseline_pipeline = Pipeline(steps=[('pre_processor',preprocessor)])
		
		# === Data split ===
		y_train = data['duration']
		X_train = data.drop(['duration'],axis=1)	
		y_test = validation_data['duration']
		X_test = validation_data.drop(['duration'],axis=1)	
		#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

		baseline_pipeline.fit(X_train)
		X_train = baseline_pipeline.transform(X_train)
		X_test = baseline_pipeline.transform(X_test)
		
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

		#Best parameters: {'learning_rate': 0.1, 'l2_regularization': 0.1, 'max_leaf_nodes': 15, 'max_iter': 400}
		learning_rate_list = [0.01, 0.1]
		l2_regularization_list = [0.01, 0.1]
		max_leaf_nodes_list = [15, 31]
		max_iter_list = [200, 400]
		best_error = 0
		best_metrics = None
		best_param = None 
		best_model = None
		
		#Linear
		model = LinearRegression()
		print("Training with LinearRegression")
		model.fit(X_train, y_train)
		y_pred = model.predict(X_test)
		metrics = calculate_metrics(y_test, y_pred)
		if best_error == 0 or metrics['MSE'] < best_error:
			best_error = metrics['MSE']
			best_model = model
			best_metrics = metrics

		print("Best error so far:" + str(best_error))

		#HistGradientBoosting
		#random seed everytime for reproducibility
		seed = np.random.randint(1000000)
		for learning_rate in learning_rate_list:
			for l2_regularization in l2_regularization_list:
				for max_leaf_nodes in max_leaf_nodes_list:
					for max_iter in max_iter_list:
						model= HistGradientBoostingRegressor(learning_rate=learning_rate, l2_regularization=l2_regularization, max_leaf_nodes=max_leaf_nodes, max_iter=max_iter, random_state=seed)
						print(f"Training with: {learning_rate}, {l2_regularization}, {max_leaf_nodes}, {max_iter}")
						model.fit(X_train, y_train)
						y_pred = model.predict(X_test)
						metrics = calculate_metrics(y_test, y_pred)
						if best_error == 0 or metrics['MSE'] < best_error:
							best_error = metrics['MSE']
							best_model = model
							best_param = {
								"learning_rate": learning_rate,
								"l2_regularization": l2_regularization,
								"max_leaf_nodes": max_leaf_nodes,
								"max_iter": max_iter,
								"seed": seed
								}
							best_metrics = metrics
						
		mlflow.log_metric("MSE", best_metrics['MSE'])
		mlflow.log_metric("MAE", best_metrics['MAE'])
		mlflow.log_metric("RMSE", best_metrics['RMSE'])
		mlflow.log_metric("R2_SCORE", best_metrics['R2_SCORE'])
		
		print(f"Best parameters: {best_param} best error: {best_error}")
		json.dump(best_param, open("best_params.json","w"))
		mlflow.log_artifact("best_params.json", artifact_path="model_params")
		joblib.dump(baseline_pipeline, "pipeline.pkl")
		mlflow.log_artifact("pipeline.pkl", artifact_path="preprocessing")
		signature = mlflow.models.infer_signature(X_train, best_model.predict(X_train))
		mlflow.sklearn.log_model(best_model, name=MODEL_NAME, signature=signature, input_example=X_train[:5])

		
		print(f"\nBest model run_id: {run.info.run_id} with MSE: {best_error}")
		print("Best model name:" + str(best_model))
		model_uri = f"runs:/{run.info.run_id}/{MODEL_NAME}"
		print(f"\nmodel_uri: {model_uri}")
			

	try:
		registered_model = mlflow.register_model(model_uri, MODEL_NAME)
		print(f"Registered model version: {registered_model.version}")
		
		#promote to Production
		from mlflow.tracking import MlflowClient
		#assign alias
		client = MlflowClient()
		client.set_registered_model_alias(
			name=MODEL_NAME,
			alias="staging", 
			version=registered_model.version,
		)

		#create alias for commit SHA
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
		raise  #re-raise to fail the CI pipeline if registration fails
 


if __name__ == '__main__':
	main()
