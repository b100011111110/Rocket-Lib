import os
import sys
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

sys.path.append('build')
import rocket

# Disable Shuffle
os.environ['ROCKET_SHUFFLE'] = '0'
os.environ['ROCKET_SEED'] = '42'
tf.keras.utils.set_random_seed(42)

def to_rocket(arr):
    tensors = []
    # If arr is 3D (samples, seq_len, features)
    if len(arr.shape) == 3:
        for sample in arr:
            t = rocket.Tensor(sample.shape[0], sample.shape[1])
            for i in range(sample.shape[0]):
                for j in range(sample.shape[1]):
                    t.set_val(i, j, float(sample[i, j]))
            tensors.append(t)
    elif len(arr.shape) == 2:
        for row in arr:
            t = rocket.Tensor(1, row.shape[0])
            for i, val in enumerate(row):
                t.set_val(0, i, float(val))
            tensors.append(t)
    return tensors

def test_rnn():
    print("\n--- Testing RNNLayer ---")
    batch_size = 32
    seq_len = 100
    input_dim = 5
    hidden_dim = 16

    X = np.random.randn(batch_size, seq_len, input_dim).astype(np.float32)
    y = np.random.randn(batch_size, 1).astype(np.float32)

    X_rocket = to_rocket(X)
    y_rocket = to_rocket(y)

    r_model = rocket.Model()
    inp = rocket.InputLayer()
    rnn = rocket.RNNLayer(input_dim, hidden_dim, seq_len, False)
    dense = rocket.DenseLayer(hidden_dim, 1)

    r_model.add(inp, [])
    r_model.add(rnn, [inp])
    r_model.add(dense, [rnn])
    r_model.setInputOutputLayers([inp], [dense])
    r_model.compile(rocket.MSE(), rocket.Adam(lr=0.01))

    import time
    start_time = time.time()
    r_model.train(X_rocket, y_rocket, X_rocket, y_rocket, 100, batch_size)
    rocket_time = time.time() - start_time
    print(f"Rocket RNN Training Time: {rocket_time:.4f} seconds")

    print("\nTraining Keras RNN for 100 epochs...")
    k_model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(seq_len, input_dim)),
        tf.keras.layers.SimpleRNN(hidden_dim, activation='tanh', return_sequences=False),
        tf.keras.layers.Dense(1)
    ])
    k_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), loss='mse')
    
    start_time = time.time()
    k_model.fit(X, y, epochs=100, batch_size=batch_size, verbose=0)
    keras_time = time.time() - start_time
    print(f"Keras RNN Training Time:  {keras_time:.4f} seconds")
    print(f"Speedup vs Keras:         {keras_time / rocket_time:.2f}x\n")

def test_lstm():
    print("\n--- Testing LSTMLayer (Stacked) ---")
    batch_size = 32
    seq_len = 100
    input_dim = 5
    hidden_dim = 16

    X = np.random.randn(batch_size, seq_len, input_dim).astype(np.float32)
    y = np.random.randn(batch_size, hidden_dim).astype(np.float32)

    X_rocket = to_rocket(X)
    y_rocket = to_rocket(y)

    r_model = rocket.Model()
    inp = rocket.InputLayer()
    lstm1 = rocket.LSTMLayer(input_dim, hidden_dim, seq_len, True)
    drop = rocket.DropoutLayer(0.2)
    lstm2 = rocket.LSTMLayer(hidden_dim, hidden_dim, seq_len, False)

    r_model.add(inp, [])
    r_model.add(lstm1, [inp])
    r_model.add(drop, [lstm1])
    r_model.add(lstm2, [drop])
    r_model.setInputOutputLayers([inp], [lstm2])
    r_model.compile(rocket.MSE(), rocket.Adam(lr=0.01))

    import time
    start_time = time.time()
    r_model.train(X_rocket, y_rocket, X_rocket, y_rocket, 100, batch_size)
    rocket_time = time.time() - start_time
    print(f"Rocket Pure Stacked LSTM Training Time: {rocket_time:.4f} seconds")

    print("\nTraining Keras Pure Stacked LSTM for 100 epochs...")
    k_model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(shape=(seq_len, input_dim)),
        tf.keras.layers.LSTM(hidden_dim, return_sequences=True),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.LSTM(hidden_dim, return_sequences=False)
    ])
    k_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), loss='mse')
    
    start_time = time.time()
    k_model.fit(X, y, epochs=100, batch_size=batch_size, verbose=0)
    keras_time = time.time() - start_time
    print(f"Keras Pure Stacked LSTM Training Time:  {keras_time:.4f} seconds")
    print(f"Speedup vs Keras:                      {keras_time / rocket_time:.2f}x\n")

if __name__ == "__main__":
    test_rnn()
    test_lstm()
