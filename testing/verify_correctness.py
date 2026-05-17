import os
import sys
import numpy as np

sys.path.append('build')
import rocket

try:
    import torch
    import torch.nn as nn
except ImportError:
    print("Please install PyTorch: pip install torch")
    sys.exit(1)

os.environ['ROCKET_SHUFFLE'] = '0'
os.environ['ROCKET_SEED'] = '42'
torch.manual_seed(42)
np.random.seed(42)

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

class PyTorchRNN(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.rnn1 = nn.RNN(input_dim, hidden_dim, num_layers=1, batch_first=True)
        self.rnn2 = nn.RNN(hidden_dim, hidden_dim, num_layers=1, batch_first=True)
        self.dense = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x1, _ = self.rnn1(x)
        x2, _ = self.rnn2(x1)
        out = self.dense(x2[:, -1, :])
        return out

def verify_rnn():
    print("\n--- Verifying Stacked RNN Correctness ---")
    batch_size = 2
    seq_len = 5
    input_dim = 3
    hidden_dim = 4

    X = np.random.randn(batch_size, seq_len, input_dim).astype(np.float32)
    X_rocket = to_rocket(X)

    # PyTorch Model
    pt_model = PyTorchRNN(input_dim, hidden_dim)
    
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
    
    # Sync Weights from PyTorch to Rocket-Lib
    # rnn1
    kw1_ih = pt_model.rnn1.weight_ih_l0.detach().numpy().T
    kw1_hh = pt_model.rnn1.weight_hh_l0.detach().numpy().T
    kw1_b = pt_model.rnn1.bias_ih_l0.detach().numpy() + pt_model.rnn1.bias_hh_l0.detach().numpy()
    
    for i in range(input_dim):
        for j in range(hidden_dim):
            rnn1.weights_ih.set_val(i, j, float(kw1_ih[i, j]))
    for i in range(hidden_dim):
        for j in range(hidden_dim):
            rnn1.weights_hh.set_val(i, j, float(kw1_hh[i, j]))
            rnn1.biases.set_val(0, j, float(kw1_b[j]))

    # rnn2
    kw2_ih = pt_model.rnn2.weight_ih_l0.detach().numpy().T
    kw2_hh = pt_model.rnn2.weight_hh_l0.detach().numpy().T
    kw2_b = pt_model.rnn2.bias_ih_l0.detach().numpy() + pt_model.rnn2.bias_hh_l0.detach().numpy()
    
    for i in range(hidden_dim):
        for j in range(hidden_dim):
            rnn2.weights_ih.set_val(i, j, float(kw2_ih[i, j]))
            rnn2.weights_hh.set_val(i, j, float(kw2_hh[i, j]))
            rnn2.biases.set_val(0, j, float(kw2_b[j]))
            
    # dense
    kw3_w = pt_model.dense.weight.detach().numpy().T
    kw3_b = pt_model.dense.bias.detach().numpy()
    for i in range(hidden_dim):
        for j in range(1):
            dense.weights.set_val(i, j, float(kw3_w[i, j]))
    dense.biases.set_val(0, 0, float(kw3_b[0]))

    # Forward Pass Compare
    pt_model.eval()
    with torch.no_grad():
        pt_preds = pt_model(torch.tensor(X)).numpy()
    
    r_preds = []
    for x in X_rocket:
        out = r_model.predict([x])[0]
        r_preds.append(out)
    
    diff_max = 0
    for b in range(batch_size):
        r_val = r_preds[b].get_val(0, 0)
        k_val = pt_preds[b, 0]
        diff_max = max(diff_max, abs(r_val - k_val))
        
    print(f"Max Absolute Error (Forward Pass): {diff_max:.6f}")
    if diff_max < 1e-4:
        print("✅ Stacked RNN correctness verified.")
    else:
        print("❌ Stacked RNN output diverges from PyTorch.")

def reorder_lstm_weights(w, axis=0):
    # PyTorch is [i, f, g, o] -> Rocket is [i, f, o, g]
    chunks = np.split(w, 4, axis=axis)
    return np.concatenate([chunks[0], chunks[1], chunks[3], chunks[2]], axis=axis)

class PyTorchLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.lstm1 = nn.LSTM(input_dim, hidden_dim, num_layers=1, batch_first=True)
        self.lstm2 = nn.LSTM(hidden_dim, hidden_dim, num_layers=1, batch_first=True)
        self.dense = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x1, _ = self.lstm1(x)
        x2, _ = self.lstm2(x1)
        out = self.dense(x2[:, -1, :])
        return out

def verify_lstm():
    print("\n--- Verifying Stacked LSTM Correctness ---")
    batch_size = 2
    seq_len = 5
    input_dim = 3
    hidden_dim = 4

    X = np.random.randn(batch_size, seq_len, input_dim).astype(np.float32)
    X_rocket = to_rocket(X)

    # PyTorch Model
    pt_model = PyTorchLSTM(input_dim, hidden_dim)
    
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
    # lstm1
    w_ih = reorder_lstm_weights(pt_model.lstm1.weight_ih_l0.detach().numpy(), axis=0).T
    w_hh = reorder_lstm_weights(pt_model.lstm1.weight_hh_l0.detach().numpy(), axis=0).T
    b_total = reorder_lstm_weights(pt_model.lstm1.bias_ih_l0.detach().numpy() + pt_model.lstm1.bias_hh_l0.detach().numpy(), axis=0)
    
    for i in range(input_dim):
        for j in range(4 * hidden_dim):
            lstm1.weights_ih.set_val(i, j, float(w_ih[i, j]))
    for i in range(hidden_dim):
        for j in range(4 * hidden_dim):
            lstm1.weights_hh.set_val(i, j, float(w_hh[i, j]))
            lstm1.biases.set_val(0, j, float(b_total[j]))

    # lstm2
    w2_ih = reorder_lstm_weights(pt_model.lstm2.weight_ih_l0.detach().numpy(), axis=0).T
    w2_hh = reorder_lstm_weights(pt_model.lstm2.weight_hh_l0.detach().numpy(), axis=0).T
    b2_total = reorder_lstm_weights(pt_model.lstm2.bias_ih_l0.detach().numpy() + pt_model.lstm2.bias_hh_l0.detach().numpy(), axis=0)
    
    for i in range(hidden_dim):
        for j in range(4 * hidden_dim):
            lstm2.weights_ih.set_val(i, j, float(w2_ih[i, j]))
            lstm2.weights_hh.set_val(i, j, float(w2_hh[i, j]))
            lstm2.biases.set_val(0, j, float(b2_total[j]))
            
    # dense
    kw3_w = pt_model.dense.weight.detach().numpy().T
    kw3_b = pt_model.dense.bias.detach().numpy()
    for i in range(hidden_dim):
        for j in range(1):
            dense.weights.set_val(i, j, float(kw3_w[i, j]))
    dense.biases.set_val(0, 0, float(kw3_b[0]))

    # Forward Pass Compare
    pt_model.eval()
    with torch.no_grad():
        pt_preds = pt_model(torch.tensor(X)).numpy()
    
    r_preds = []
    for x in X_rocket:
        out = r_model.predict([x])[0]
        r_preds.append(out)
    
    diff_max = 0
    for b in range(batch_size):
        r_val = r_preds[b].get_val(0, 0)
        k_val = pt_preds[b, 0]
        diff_max = max(diff_max, abs(r_val - k_val))
        
    print(f"Max Absolute Error (Forward Pass): {diff_max:.6f}")
    if diff_max < 1e-4:
        print("✅ Stacked LSTM correctness verified.")
    else:
        print("❌ Stacked LSTM output diverges from PyTorch.")

if __name__ == "__main__":
    verify_rnn()
    verify_lstm()
