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

    # === Read dataset ===
    #data = pd.read_parquet('data/raw/yellow_tripdata_2011-01.parquet')
    data = pd.read_parquet('data/preprocessed/final_subsampled_dataset.parquet')
    #data = preprocess_data(data)
        
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
    learning_rate_list = [0.001, 0.01, 0.1]
    l2_regularization_list = [0, 0.01, 0.1]
    max_leaf_nodes_list = [15, 31, 63]
    max_iter_list = [100, 200, 400]
    best_param = None 
    
    for learning_rate in learning_rate_list:
        for l2_regularization in l2_regularization_list:
            for max_leaf_nodes in max_leaf_nodes_list:
                for max_iter in max_iter_list:
                    model= HistGradientBoostingRegressor(learning_rate=learning_rate, l2_regularization=l2_regularization, max_leaf_nodes=max_leaf_nodes, max_iter=max_iter, random_state=42)
                    print(f"Training with: {learning_rate}, {l2_regularization}, {max_leaf_nodes}, {max_iter}")
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    metrics = calculate_metrics(y_test, y_pred)
                    if best_error == 0 or metrics['MSE'] < best_error:
                        best_error = metrics['MSE']
                        #best_run_id = run.info.run_id
                        best_model = model.__class__.__name__ 
                        best_param = {
                            "learning_rate": learning_rate,
                            "l2_regularization": l2_regularization,
                            "max_leaf_nodes": max_leaf_nodes,
                            "max_iter": max_iter
                            }
                    print(f"Best model so far: {best_model} with MSE: {best_error}")    
    print(f"Best parameters: {best_param}") 


if __name__ == '__main__':
	main()
