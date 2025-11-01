import pandas as pd
from sklearn.preprocessing import OneHotEncoder,StandardScaler

class pickupDataTransformer():
	def __init__(self):
		self.columns='pickup_datetime'

	def transform(self,X:pd.DataFrame,y=None):
		X['pickup_hour'] = X['pickup_datetime'].dt.hour
		X['pickup_day_of_week'] = X['pickup_datetime'].dt.dayofweek
		X['pickup_month'] = X['pickup_datetime'].dt.month
		X['pickup_day'] = X['pickup_datetime'].dt.day
		X.drop(['pickup_datetime'],axis=1,inplace=True)
		return X

	def fit(self, X, y=None):
		return self

'''
From https://stackoverflow.com/questions/68402691/adding-dropping-column-instance-into-a-pipeline
'''
class columnDropperTransformer():
	def __init__(self,columns):
		self.columns=columns

	def transform(self,X,y=None):
		return X.drop(self.columns,axis=1)

	def fit(self, X, y=None):
		return self
		

class oneHotEncodingTransformer():
	def __init__(self,columns):
		self.encoder = OneHotEncoder(sparse_output=False)
		self.columns = columns
		
	def transform(self, X, y=None):
		X_transformed = X.copy()
		encoded = pd.DataFrame(
			self.encoder.transform(X[self.columns]),
			columns=self.encoder.get_feature_names_out(self.columns),
			index=X.index
		)

		X_transformed = X_transformed.drop(columns=self.columns)
		X_transformed = pd.concat([X_transformed, encoded], axis=1)

		return X_transformed
		
	def fit(self,X,y=None):
		self.encoder.fit(X[self.columns])
		return self
		
class standarScalingTransformer():
	def __init__(self,columns):
		self.scaler = StandardScaler()
		self.columns = columns
		
	def transform(self, X, y=None):
		X_transformed = X.copy()
		
		scaled_values = self.scaler.transform(X[self.columns])
		X_transformed[self.columns] = scaled_values

		return X_transformed
		
	def fit(self,X,y=None):
		self.scaler.fit(X[self.columns])
		return self


class missingDataTransformer():
	def __init__(self):
		pass
		
	def transform(self, X, y=None):
		mask1 = (X["dropoff_longitude"] != 0) & (X["dropoff_latitude"] != 0)
		X = X.loc[mask1]
		y = y.loc[X.index]

		# Second mask
		mask2 = (X["pickup_longitude"] != 0) & (X["pickup_latitude"] != 0)
		X = X.loc[mask2]
		y = y.loc[X.index]
		return X,y
		
	def fit(self,X,y=None):
		return self

class lowerCaseTransformer():
	def __init__(self):
		pass
		
	def transform(self, X, y=None):
		X = X.rename(columns=lambda x: x.strip().lower() if isinstance(x, str) else x)
		return X
		
	def fit(self,X,y=None):
		return self


