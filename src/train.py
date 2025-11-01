import mlflow
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error, mean_squared_error,r2_score, mean_absolute_error

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor

from transformers import columnDropperTransformer,pickupDataTransformer,oneHotEncodingTransformer,standarScalingTransformer,missingDataTransformer, lowerCaseTransformer

# Se estiveres dentro da VM:
mlflow.set_tracking_uri("http://10.17.0.162:5000")

# Ou se estiveres a usar túnel SSH (fora da VM, mas na VPN):
# mlflow.set_tracking_uri("http://127.0.0.1:5000")

mlflow.set_experiment("nyc_taxi_project")

CAT_COLS = ['vendor_id','rate_code','payment_type']
NUM_COLS = ['passenger_count','trip_distance','pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude','fare_amount','surcharge','mta_tax','tip_amount','tolls_amount','total_amount']

def preprocess_data(data):
	data = data.head(1000000) # ! <- ta a ser limitado em 1000000

	data['pickup_datetime'] = pd.to_datetime(data['pickup_datetime'])
	data['dropoff_datetime'] = pd.to_datetime(data['dropoff_datetime'])

	data['rate_code'] = pd.to_numeric(data['rate_code'], downcast='integer')
	data['duration'] = (data['dropoff_datetime'] - data['pickup_datetime']).dt.total_seconds() / 60

	return data

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
	with mlflow.start_run() as run:
		# === Read dataset ===
		data = pd.read_parquet('data\\raw\\yellow_tripdata_2010-01.parquet')
		data = preprocess_data(data)
			
		# === Pipeline definition ===
		baseline_pipeline = Pipeline(steps=[('dropper',columnDropperTransformer(['store_and_fwd_flag','dropoff_datetime'])),
									('lower_case',lowerCaseTransformer()),
									('data_transformer',pickupDataTransformer()),
									('one_hot_encoding',oneHotEncodingTransformer(CAT_COLS)),
									('numerical_scaling',standarScalingTransformer(NUM_COLS))])
		
		# === Data split ===
		y = data['duration']
		X = data.drop(['duration'],axis=1)	
		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

		# 
		X_train, y_train = missingDataTransformer().transform(X_train,y_train)
		print(X_train.index.equals(y_train.index))
		X_test, y_test = missingDataTransformer().transform(X_test,y_test)

		baseline_pipeline.fit(X_train)
		X_train = baseline_pipeline.transform(X_train)
		X_test = baseline_pipeline.transform(X_test)
		print(X_train.index.equals(y_train.index))

		# === Compute correlations ===
		data_train = X_train.copy()
		data_train['duration'] = y_train.values
		print(y_train.head(100))
		print(X_train.head(100))
		print(data_train.head(100))
		corr = data_train.corr()['duration'].dropna().sort_values(key=abs, ascending=False)
		print("\n📊 Feature correlations with target:")
		print(corr.head(20))

		# === Select top correlated features ===
		top_features = corr.index[1:5+1].tolist()  # skip target itself
		X_train = X_train[top_features]
		X_test = X_test[top_features]

		print(X_train)

			
		print('Linear Regressor')

		model = LinearRegression()

		model.fit(X_train,y_train)

		signature = mlflow.models.infer_signature(X_train, model.predict(X_train))
		mlflow.sklearn.log_model(model, signature=signature, input_example=X_train[:5])

		y_pred = model.predict(X_test)

		calculate_metrics(y_test, y_pred)



if __name__ == '__main__':
	main()
