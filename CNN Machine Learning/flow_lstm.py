from math import sqrt
from numpy import concatenate
import matplotlib.pyplot as plt
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import LSTM
from keras.layers import Flatten
from keras.layers import TimeDistributed
from sklearn.metrics import r2_score, mean_absolute_error
import numpy as np
from keras.optimizers import SGD, Adam
from keras import metrics
from sklearn.preprocessing import StandardScaler

# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	df = DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg
 
# load dataset
flow = read_csv('soyanggangdam_input08_19.csv', header=0, index_col=0)
data=flow[['inflow','precipitation','temp_max','temp_min','humidity','wind','solar']]
values = data.values

#전체 데이터 시계열그래프 그리기
'''
# specify columns to plot
groups = [0, 1,2,3,4,5,6]
i = 1
# plot each column
plt.figure()
for group in groups:
	plt.subplot(len(groups), 1, i)
	plt.plot(values[:, group],color='darkblue')
	plt.title(data.columns[group], y=0.5, loc='right')
	i += 1
plt.xlabel("Julian day")
plt.show()
'''

# ensure all data is float
values = values.astype('float32')
# normalize features
#scaler = MinMaxScaler(feature_range=(0, 1))
scaler=StandardScaler()
scaled = scaler.fit_transform(values)
 

# specify the number of lag hours
n_hours = 2
n_features = 7
# frame as supervised learning
reframed = series_to_supervised(scaled, n_hours, 1)

values = reframed.values
n_train_hours = 12021
n_validation=13146
train = values[:n_train_hours, :]
validation = values[n_train_hours:n_validation, :]
test = values[n_validation:, :]

 # split into input and outputs
n_obs = n_hours * n_features
train_X, train_y = train[:, :n_obs], train[:, -n_features]
val_X, val_y = validation[:, :n_obs], validation[:, -n_features]
test_X, test_y = test[:, :n_obs], test[:, -n_features]

# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], n_hours, n_features))
val_X = val_X.reshape((val_X.shape[0], n_hours, n_features))
test_X = test_X.reshape((test_X.shape[0], n_hours, n_features))

# design network
model = Sequential()
model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dense(256, kernel_initializer='normal', activation = "relu"))
model.add(Dropout(0.1))
model.add(Dense(512, kernel_initializer='normal', activation = "relu"))
model.add(Dense(512, kernel_initializer='normal', activation = "relu"))
model.add(Dense(64, kernel_initializer='normal', activation = "relu"))
model.add(Dense(1))
model.compile(loss='mean_squared_error', 
              optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8)) 


# fit network
history = model.fit(train_X, train_y, epochs=100, batch_size=72, validation_data=(val_X, val_y), verbose=1, shuffle=False)
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.legend()
plt.title("LSTM")
plt.ylabel("Loss")
plt.xlabel("Epochs")
plt.grid()
plt.show()

# make a prediction
yhat = model.predict(test_X)
test_X = test_X.reshape((test_X.shape[0], n_hours*n_features))

# invert scaling for forecast
inv_yhat = concatenate((yhat, test_X[:, -6:]), axis=1)   
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:,0]

# invert scaling for actual
test_y = test_y.reshape((len(test_y), 1))
inv_y = concatenate((test_y, test_X[:, -6:]), axis=1)   
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:,0]


# calculate RMSE
rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
print("NSE: %.3f" % r2_score(inv_y, inv_yhat))
print('RMSE: %.3f' % rmse)
print("MAE: %.3f" % mean_absolute_error(inv_y, inv_yhat))
print("R: %.3f" % (np.corrcoef(inv_y, inv_yhat)[0,1]))
print("R2: %.3f" % (np.corrcoef(inv_y, inv_yhat)[0,1])**2)
pre={'y':inv_y,'y_hat':inv_yhat}
d_result=DataFrame(data=pre)
d_result.to_csv("c_lstm.csv")

x=np.arange(0,len(inv_y),1)
plt.clf() 
plt.plot(x,pre['y'], label='Observation',color='orangered')
plt.plot(x,pre['y_hat'], label='LSTM',color='darkblue', linestyle='--', marker='o',markersize=3)
#plt.title('Long Short Term Memory',fontsize=13)
plt.xlabel("Julian day")
plt.ylabel("Dam Inflow (m$^3$/s)")
plt.legend()
plt.show()
plt.savefig('flow_lstm.png')

