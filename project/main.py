import numpy as np
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

rcParams['figure.figsize'] = 22, 10

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

anomaly_df = pd.read_csv('content/spx.csv', parse_dates=['date'], index_col='date')

# Splitting the Dataset into Training & Testing
train, test = train_test_split(anomaly_df, test_size=0.05, shuffle=False)

# Preparing the Data
scaler = StandardScaler()
scaler = scaler.fit(train[['close']])

train['close'] = scaler.transform(train[['close']])
test['close'] = scaler.transform(test[['close']])


def create_dataset(X, y, time_steps=1):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        v = X.iloc[i:(i + time_steps)].values
        Xs.append(v)
        ys.append(y.iloc[i + time_steps])
    return np.array(Xs), np.array(ys)


TIME_STEPS = 30

# reshape to [samples, time_steps, n_features]

X_train, y_train = create_dataset(train[['close']], train.close, TIME_STEPS)
X_test, y_test = create_dataset(test[['close']], test.close, TIME_STEPS)

# Create the Model
model = keras.Sequential()

# encoder
model.add(keras.layers.LSTM(
    units=64,
    input_shape=(X_train.shape[1], X_train.shape[2])
))
model.add(keras.layers.Dropout(rate=0.2))

# decoder
model.add(keras.layers.RepeatVector(n=X_train.shape[1]))
model.add(keras.layers.LSTM(units=64, return_sequences=True))
model.add(keras.layers.Dropout(rate=0.2))
model.add(keras.layers.TimeDistributed(keras.layers.Dense(units=X_train.shape[2])))
model.compile(loss='mae', optimizer='adam')
model.summary()

history = model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=32,
    validation_split=0.1,
    shuffle=False
)

fig = plt.figure()
ax = fig.add_subplot()
ax.plot(history.history['loss'], label='train')
ax.plot(history.history['val_loss'], label='test')
ax.legend()


# Defining the Anomaly Value

# Calculation of the loss between the predicted and the actual closing price data:
X_train_pred = model.predict(X_train)
train_mae_loss = np.mean(np.abs(X_train_pred - X_train), axis=1)

# We will then plot the loss distribution to decide on the threshold for our anomaly detection.
fig = plt.figure(figsize=(20,10))
sns.set(style="darkgrid")
ax = fig.add_subplot()
sns.displot(train_mae_loss, bins=50, kde=True)
ax.set_title('Loss Distribution Training Set ', fontweight ='bold')

# Calculate the Mean Absolute Error
X_test_pred = model.predict(X_test)
test_mae_loss = np.mean(np.abs(X_test_pred - X_test), axis=1)
THRESHOLD = 0.65
test_score_df = pd.DataFrame(index=test[TIME_STEPS:].index)
test_score_df['loss'] = test_mae_loss
test_score_df['threshold'] = THRESHOLD
test_score_df['anomaly'] = test_score_df.loss > test_score_df.threshold
test_score_df['close'] = test[TIME_STEPS:].close

fig = plt.figure()
ax = fig.add_subplot()
ax.plot(test_score_df.index, test_score_df.loss, label='loss')
ax.plot(test_score_df.index, test_score_df.threshold, label='threshold')
ax.legend()

anomalies = test_score_df[test_score_df.anomaly == True]
anomalies.head()

fig = plt.figure()
ax = fig.add_subplot()
ax.plot(test[TIME_STEPS:].index,
  scaler.inverse_transform(test[TIME_STEPS:].close.values.reshape(1,-1)).reshape(-1), label='close price')
ax.plot(anomalies.index,
  scaler.inverse_transform(anomalies.close.values.reshape(1,-1)).reshape(-1), label='anomaly price')
ax.legend()

