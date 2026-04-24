import os
import sys
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

sys.path.append('build')
import rocket

os.environ['ROCKET_SHUFFLE'] = '0'
os.environ['ROCKET_SEED'] = '42'
tf.keras.utils.set_random_seed(42)

def to_rocket(arr):
    tensors = []
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

def verify_rnn():
    print("\n--- Verifying Stacked RNN Correctness ---")
    batch_size = 2
    seq_len = 5
    input_dim = 3
    hidden_dim = 4

    X = np.random.randn(batch_size, seq_len, input_dim).astype(np.float32)
    y = np.random.randn(batch_size, 1).astype(np.float32)

    X_rocket = to_rocket(X)

    # Keras Model
    k_model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(shape=(seq_len, input_dim)),
        tf.keras.layers.SimpleRNN(hidden_dim, activation='tanh', return_sequences=True),
        tf.keras.layers.SimpleRNN(hidden_dim, activation='tanh', return_sequences=False),
        tf.keras.layers.Dense(1)
    ])
    
    # Rocket Model
    r_model = rocket.Model()
    inp = rocket.InputLayer()
    rnn1 = rocket.RNNLayer(input_dim, hidden_dim, seq_len, True)
    rnn2 = rocket.RNNLayer(hidden_dim, hidden_dim, seq_len, False)
    dense = rocket.DenseLayer(hidden_dim, 1)

    r_model.add(inp, [])
    r_model.add(rnn1, [inp])
    r_model.add(rnn2, [rnn1])
    r_model.add(dense, [rnn2])
    r_model.setInputOutputLayers([inp], [dense])
    r_model.compile(rocket.MSE(), rocket.Adam(lr=0.01))
    
    # Sync Weights
    kw1 = k_model.layers[0].get_weights()
    for i in range(input_dim):
        for j in range(hidden_dim):
            rnn1.weights_ih.set_val(i, j, float(kw1[0][i, j]))
    for i in range(hidden_dim):
        for j in range(hidden_dim):
            rnn1.weights_hh.set_val(i, j, float(kw1[1][i, j]))
            rnn1.biases.set_val(0, j, float(kw1[2][j]))

    kw2 = k_model.layers[1].get_weights()
    for i in range(hidden_dim):
        for j in range(hidden_dim):
            rnn2.weights_ih.set_val(i, j, float(kw2[0][i, j]))
            rnn2.weights_hh.set_val(i, j, float(kw2[1][i, j]))
            rnn2.biases.set_val(0, j, float(kw2[2][j]))
            
    kw3 = k_model.layers[2].get_weights()
    for i in range(hidden_dim):
        for j in range(1):
            dense.weights.set_val(i, j, float(kw3[0][i, j]))
    dense.biases.set_val(0, 0, float(kw3[1][0]))

    # Forward Pass Compare
    k_preds = k_model.predict(X, verbose=0)
    
    r_preds = []
    for x in X_rocket:
        out = r_model.predict([x])[0]
        r_preds.append(out)
    
    diff_max = 0
    for b in range(batch_size):
        r_val = r_preds[b].get_val(0, 0)
        k_val = k_preds[b, 0]
        diff_max = max(diff_max, abs(r_val - k_val))
        
    print(f"Max Absolute Error (Forward Pass): {diff_max:.6f}")
    if diff_max < 1e-4:
        print("✅ Stacked RNN correctness verified.")
    else:
        print("❌ Stacked RNN output diverges from Keras.")

def reorder_lstm_weights(w):
    # Keras is [i, f, c, o] -> Rocket is [i, f, o, g]
    hidden_dim = w.shape[1] // 4
    i = w[:, :hidden_dim]
    f = w[:, hidden_dim:2*hidden_dim]
    c = w[:, 2*hidden_dim:3*hidden_dim]
    o = w[:, 3*hidden_dim:]
    return np.concatenate([i, f, o, c], axis=1)

def verify_lstm():
    print("\n--- Verifying Stacked LSTM Correctness ---")
    batch_size = 2
    seq_len = 5
    input_dim = 3
    hidden_dim = 4

    X = np.random.randn(batch_size, seq_len, input_dim).astype(np.float32)
    y = np.random.randn(batch_size, 1).astype(np.float32)

    X_rocket = to_rocket(X)

    # Keras Model
    k_model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(shape=(seq_len, input_dim)),
        tf.keras.layers.LSTM(hidden_dim, return_sequences=True),
        tf.keras.layers.LSTM(hidden_dim, return_sequences=False),
        tf.keras.layers.Dense(1)
    ])
    
    # Rocket Model
    r_model = rocket.Model()
    inp = rocket.InputLayer()
    lstm1 = rocket.LSTMLayer(input_dim, hidden_dim, seq_len, True)
    lstm2 = rocket.LSTMLayer(hidden_dim, hidden_dim, seq_len, False)
    dense = rocket.DenseLayer(hidden_dim, 1)

    r_model.add(inp, [])
    r_model.add(lstm1, [inp])
    r_model.add(lstm2, [lstm1])
    r_model.add(dense, [lstm2])
    r_model.setInputOutputLayers([inp], [dense])
    r_model.compile(rocket.MSE(), rocket.Adam(lr=0.01))
    
    # Sync Weights
    kw1 = k_model.layers[0].get_weights()
    W1 = reorder_lstm_weights(kw1[0])
    U1 = reorder_lstm_weights(kw1[1])
    b1 = reorder_lstm_weights(kw1[2].reshape(1, -1)).flatten()
    for i in range(input_dim):
        for j in range(4 * hidden_dim):
            lstm1.weights_ih.set_val(i, j, float(W1[i, j]))
    for i in range(hidden_dim):
        for j in range(4 * hidden_dim):
            lstm1.weights_hh.set_val(i, j, float(U1[i, j]))
            lstm1.biases.set_val(0, j, float(b1[j]))

    kw2 = k_model.layers[1].get_weights()
    W2 = reorder_lstm_weights(kw2[0])
    U2 = reorder_lstm_weights(kw2[1])
    b2 = reorder_lstm_weights(kw2[2].reshape(1, -1)).flatten()
    for i in range(hidden_dim):
        for j in range(4 * hidden_dim):
            lstm2.weights_ih.set_val(i, j, float(W2[i, j]))
            lstm2.weights_hh.set_val(i, j, float(U2[i, j]))
            lstm2.biases.set_val(0, j, float(b2[j]))
            
    kw3 = k_model.layers[2].get_weights()
    for i in range(hidden_dim):
        for j in range(1):
            dense.weights.set_val(i, j, float(kw3[0][i, j]))
    dense.biases.set_val(0, 0, float(kw3[1][0]))

    # Forward Pass Compare
    k_preds = k_model.predict(X, verbose=0)
    
    r_preds = []
    for x in X_rocket:
        out = r_model.predict([x])[0]
        r_preds.append(out)
    
    diff_max = 0
    for b in range(batch_size):
        r_val = r_preds[b].get_val(0, 0)
        k_val = k_preds[b, 0]
        diff_max = max(diff_max, abs(r_val - k_val))
        
    print(f"Max Absolute Error (Forward Pass): {diff_max:.6f}")
    if diff_max < 1e-4:
        print("✅ Stacked LSTM correctness verified.")
    else:
        print("❌ Stacked LSTM output diverges from Keras.")

if __name__ == "__main__":
    verify_rnn()
    verify_lstm()
