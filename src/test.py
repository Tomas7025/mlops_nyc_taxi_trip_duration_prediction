import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error,r2_score
from sklearn.metrics import mean_squared_error, mean_absolute_error


#from data_report import missing_values_report
from transformers import columnDropperTransformer,pickupDataTransformer,oneHotEncodingTransformer,standarScalingTransformer,missingDataTransformer, lowerCaseTransformer

CAT_COLS = ['vendor_id','rate_code','payment_type']
NUM_COLS = ['passenger_count','trip_distance','pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude','fare_amount','surcharge','mta_tax','tip_amount','tolls_amount','total_amount']

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
		"mse": mse,
		"mae": mae,
		"rmse": rmse,
		"r2score": r2score
	}

def main():

	data = pd.read_parquet('data\\raw\\yellow_tripdata_2010-01.parquet')

	data = data.head(1000000)

	data['pickup_datetime'] = pd.to_datetime(data['pickup_datetime'])
	data['dropoff_datetime'] = pd.to_datetime(data['dropoff_datetime'])

	data['rate_code'] = pd.to_numeric(data['rate_code'], downcast='integer')

	data['duration'] = (data['dropoff_datetime'] - data['pickup_datetime']).dt.total_seconds() / 60

	print(data.dtypes)

	for col in CAT_COLS:
		print(data[col].unique())

	# missing_columns = missing_values_report(data)

	# for col in missing_columns:
	# 	print(data[col].dropna().unique())

	baseline_pipeline = Pipeline(steps=[('dropper',columnDropperTransformer(['store_and_fwd_flag','dropoff_datetime'])),
									 ('lower_case',lowerCaseTransformer()),
									 ('data_transformer',pickupDataTransformer()),
									 ('one_hot_encoding',oneHotEncodingTransformer(CAT_COLS)),
									 ('numerical_scaling',standarScalingTransformer(NUM_COLS))])

	y = data['duration']
	X = data.drop(['duration'],axis=1)	
	
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

	X_train,y_train = missingDataTransformer().transform(X_train,y_train)
	print(X_train.index.equals(y_train.index))
	X_test,y_test = missingDataTransformer().transform(X_test,y_test)
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


	print(X_train.index.equals(y_train.index))
	print(X_test.index.equals(y_test.index))

	print('Linear Regressor')

	model = LinearRegression()

	model.fit(X_train,y_train)

	y_pred = model.predict(X_test)

	calculate_metrics(y_test, y_pred)

	print('Random Forest')

	rf = RandomForestRegressor(n_estimators=50,max_depth=6)

	rf.fit(X_train,y_train)

	y_pred = rf.predict(X_test)
	calculate_metrics(y_test, y_pred)

	print('HistGradientBoostingRegressor')
	model = HistGradientBoostingRegressor()
	model.fit(X_train, y_train)

	y_pred = model.predict(X_test)

	calculate_metrics(y_test, y_pred)
	
	return


if __name__ == '__main__':
	main()