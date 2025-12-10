import pandas as pd
import os

def drop_columns(data,columns):
	existing = [c for c in columns if c in data.columns]
	return data.drop(existing,axis=1)

'''
Já não é preciso, mas podemos falar no relatório, já que tiinhamos feito

def normalize_column_names(data):

	rename_map = {
		'tpep_pickup_datetime': 'pickup_datetime',
		'tpep_dropoff_datetime': 'dropoff_datetime',
		'VendorID': 'vendor_id',
		'RatecodeID': 'rate_code',
		'extra': 'surcharge',
	}

	data.rename(columns=rename_map, inplace=True)
	return data

'''

def preprocess(data):
	data['tpep_pickup_datetime'] = pd.to_datetime(data['tpep_pickup_datetime'])
	data['tpep_dropoff_datetime'] = pd.to_datetime(data['tpep_dropoff_datetime'])

	data['RatecodeID'] = pd.to_numeric(data['RatecodeID'], downcast='integer')

	data['duration'] = (data['tpep_dropoff_datetime'] - data['tpep_pickup_datetime']).dt.total_seconds() / 60

	data = drop_columns(data,['store_and_fwd_flag','tpep_dropoff_datetime','congestion_surcharge','airport_fee'])
	print('---Columns Dropped')

	data['pickup_hour'] = data['tpep_pickup_datetime'].dt.hour
	data['pickup_day_of_week'] = data['tpep_pickup_datetime'].dt.dayofweek
	data['pickup_month'] = data['tpep_pickup_datetime'].dt.month
	data['pickup_day'] = data['tpep_pickup_datetime'].dt.day
	data.drop(['tpep_pickup_datetime'],axis=1,inplace=True)
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

			data = pd.read_parquet('data/raw/'+file)

			print('Reading Done')

			data = preprocess(data)

			print('Preprocessing Done')

			data = (
				data.groupby(['pickup_month', 'pickup_day', 'pickup_hour'], group_keys=False)
				.apply(lambda g: g.sample(frac=0.025))
				.reset_index(drop=True)
			)

			print('Subsampling done')

			# garanteeing the output directory exists
			if not os.path.exists('data/preprocessed/'):
				os.makedirs('data/preprocessed/')

			data.to_parquet('data/preprocessed/'+filename+'_subsampled_by_hour.parquet')

			print('File Saved')

			data = []


	return

if __name__ == '__main__':
	main()
	parquet_files = [f for f in os.listdir('data/preprocessed') if f.endswith('.parquet')]
	# join all files into a single dataframe and save as a single parquet file
	df_list = [pd.read_parquet(os.path.join('data/preprocessed', f)) for f in parquet_files]
	final_df = pd.concat(df_list, ignore_index=True)
	# make a little debug print
	print(f'Final dataframe shape: {final_df.shape}')
	print(final_df.head())
	print(final_df.info())

	final_df.to_parquet('data/preprocessed/final_subsampled_dataset.parquet')