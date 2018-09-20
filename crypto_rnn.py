import pandas as pd
from sklearn import preprocessing
from collections import deque
import random
import numpy as np
import time

import tensorflow as tf 
from tensorflow.python.keras.models import Sequential 
from tensorflow.python.keras.layers import Dense, Dropout, LSTM, BatchNormalization
# from tensorflow.python.keras.callbacks import TensorBoard

periods_back = 60 #from which to use data
periods_forward = 3 #to predict
ratio_to_predict = 'LTC'
n_epochs = 10
batch_size = 64
name = f"{periods_back}-SEQ-{periods_forward}-PRED-{int(time.time())}"

currencies = ['BCH', 'BTC', 'ETH', 'LTC']

def get_data():
	main_df = pd.DataFrame()
	for currency in currencies:
		df = pd.read_csv(f'crypto_data/{currency}-USD.csv', names=[
			'time', 'low', 'high', 'open', f'{currency}_close', f'{currency}_volume'])
		df = df[['time', f'{currency}_close', f'{currency}_volume']]
		df.set_index('time', inplace = True)
		if main_df.empty == True:
			main_df = df
		else:
			main_df = main_df.join(df, how = 'inner')
	main_df.fillna('ffill', inplace = True)
	main_df.dropna(inplace = True)
	return main_df

def classify(current, future):
	if float(future) > float(current):
		return 1
	else:
		return 0

def create_targets(df):
	df[f'{ratio_to_predict}_future'] = main_df[f'{ratio_to_predict}_close'].shift(-periods_forward)
	df[f'{ratio_to_predict}_target'] = list(map(classify, main_df[f'{ratio_to_predict}_close'], main_df[f'{ratio_to_predict}_future']))
	df.drop(f'{ratio_to_predict}_future', axis = 1, inplace = True)

def create_validation_df(df):
	times = sorted(df.index.values)
	last_5pct = times[-int(0.05*len(times))]
	validation_df = df[(df.index >= last_5pct)]
	df = df[(df.index < last_5pct)]
	return df, validation_df

def preprocess_df(df):
	for col in df.columns:
		if col != f'{ratio_to_predict}_target':
			print(col)
			df[col] = df[col].pct_change()
			df.replace([np.inf, -np.inf], np.nan, inplace = True)
			df.dropna(inplace = True)
			df[col] = preprocessing.scale(df[col].values)
	df.dropna(inplace = True)

	sequential_data = []
	prev_days = deque(maxlen=periods_back) #Creates the double ended queue

	for i in df.values:  # iterate over the values
		prev_days.append([n for n in i[:-1]])  # store all but the target
		if len(prev_days) == periods_back:  # make sure we have 60 sequences!
			sequential_data.append([np.array(prev_days), i[-1]])  # append those bad boys!

	random.shuffle(sequential_data)

	X = []
	y = []

	for seq, target in sequential_data:
		X.append(seq)
		y.append(target)

	return np.array(X), np.array(y)

main_df = get_data()

create_targets(main_df)

print(main_df.head())
print(main_df.values.shape)

main_df, validation_main_df = create_validation_df(main_df)

X_train, y_train = preprocess_df(main_df)
X_test, y_test = preprocess_df(validation_main_df)

print(y_train)

model = Sequential()

model.add(LSTM(128, activation ='tanh', return_sequences=True, input_shape=(X_train.shape[1:])))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(LSTM(128, activation ='tanh', return_sequences=True))
model.add(Dropout(0.1))
model.add(BatchNormalization())


model.add(LSTM(128, activation ='tanh'))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(Dense(32, activation ='relu'))
model.add(Dropout(0.2))

model.add(Dense(2, activation = 'softmax'))

model.compile(loss='sparse_categorical_crossentropy',
	optimizer = tf.keras.optimizers.Adam(lr=0.001, decay=1e-6),
	metrics = ['accuracy'])

# tensorboard = TensorBoard(log_dir=f'logs/{name}')

history = model.fit(
	X_train, y_train,
	batch_size = batch_size,
	epochs = n_epochs,
	validation_data = (X_test, y_test),
	class_weight = 'auto'
	# callbacks = [tensorboard]
	)

model.save('model.h5')