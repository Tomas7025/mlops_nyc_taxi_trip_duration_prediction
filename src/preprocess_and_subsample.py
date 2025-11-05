import pandas as pd
import os

def drop_columns(data,columns):
	existing = [c for c in columns if c in data.columns]
	return data.drop(existing,axis=1)

def normalize_column_names(data):
	'''
	Falta tratar das coords de pickup/dropoff
	'''
	rename_map = {
		'tpep_pickup_datetime': 'pickup_datetime',
		'tpep_dropoff_datetime': 'dropoff_datetime',
		'VendorID': 'vendor_id',
		'RatecodeID': 'rate_code',
		'extra': 'surcharge',
	}

	data.rename(columns=rename_map, inplace=True)
	return data


def preprocess(data):
	data = normalize_column_names(data)
	print('---Column names normalized')

	data['pickup_datetime'] = pd.to_datetime(data['pickup_datetime'])
	data['dropoff_datetime'] = pd.to_datetime(data['dropoff_datetime'])

	data['rate_code'] = pd.to_numeric(data['rate_code'], downcast='integer')

	data['duration'] = (data['dropoff_datetime'] - data['pickup_datetime']).dt.total_seconds() / 60

	data = drop_columns(data,['store_and_fwd_flag','dropoff_datetime','airport_fee','congestion_surcharge','improvement_surcharge'])
	print('---Columns Dropped')

	data['pickup_hour'] = data['pickup_datetime'].dt.hour
	data['pickup_day_of_week'] = data['pickup_datetime'].dt.dayofweek
	data['pickup_month'] = data['pickup_datetime'].dt.month
	data['pickup_day'] = data['pickup_datetime'].dt.day
	data.drop(['pickup_datetime'],axis=1,inplace=True)
	print('---PickUp Date Split')
	return data


def main():
	'''
	Read one file, apply month/day/hour preprocessing, subsample (5%) based on it and add to final dataset
	Repeat for all files	
	'''

	for file in os.listdir('data/raw'):
		if file.endswith('.parquet'):
			filename, _ = os.path.splitext(os.path.basename(file))
			print(filename)

			data = pd.read_parquet('data\\raw\\'+file)

			print('Reading Done')

			data = preprocess(data)

			print('Preprocessing Done')

			data = (
				data.groupby(['pickup_month', 'pickup_day', 'pickup_hour'], group_keys=False)
				.apply(lambda g: g.sample(frac=0.05))
				.reset_index(drop=True)
			)

			print('Subsampling done')

			data.to_parquet('data/preprocessed/'+filename+'_subsampled_by_hour.parquet')

			print('File Saved')

			data = []


	return

if __name__ == '__main__':
	main()