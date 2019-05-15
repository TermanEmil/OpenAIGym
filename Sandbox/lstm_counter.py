import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout


def split_sequence(sequence, n_steps):
    x, y = list(), list()
    for i in range(len(sequence)):
        end_ix = i + n_steps

        if end_ix > len(sequence) - 1:
            break

        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        x.append(seq_x)
        y.append(seq_y)
    return np.array(x), np.array(y)


# raw_seq = [i for i in range(100, 900, 10)]
raw_seq = [i for i in range(0, 10)]
n_steps = 3

x, y = split_sequence(raw_seq, n_steps)
print(x)

# reshape from [samples, timesteps] into [samples, timesteps, features]
n_features = 1
x = x.reshape((x.shape[0], x.shape[1], n_features))

model = Sequential()

model.add(LSTM(64, activation='relu', input_shape=(n_steps, n_features)))

model.add(Dense(124, activation='relu'))
model.add(Dropout(0.1))

model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')


model.fit(x, y, epochs=200, verbose=1)

x_input = np.array([8, 9, 10])
x_input = x_input.reshape((1, n_steps, n_features))
yhat = model.predict(x_input, verbose=0)
print(yhat)