import os
import sys
import numpy as np
import time

sys.path.append('build')
import rocket

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
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
        self.rnn = nn.RNN(input_dim, hidden_dim, batch_first=True)
        self.dense = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        _, h = self.rnn(x)
        return self.dense(h.squeeze(0))

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

    start_time = time.time()
    r_model.train(X_rocket, y_rocket, X_rocket, y_rocket, 100, batch_size)
    rocket_time = time.time() - start_time
    print(f"Rocket RNN Training Time: {rocket_time:.4f} seconds")

    print("\nTraining PyTorch RNN for 100 epochs...")
    pt_model = PyTorchRNN(input_dim, hidden_dim)
    optimizer = optim.Adam(pt_model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    
    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32)
    dataset = torch.utils.data.TensorDataset(X_t, y_t)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

    start_time = time.time()
    pt_model.train()
    for epoch in range(100):
        for batch_X, batch_y in loader:
            optimizer.zero_grad()
            out = pt_model(batch_X)
            loss = criterion(out, batch_y)
            loss.backward()
            optimizer.step()
    pytorch_time = time.time() - start_time
    print(f"PyTorch RNN Training Time:  {pytorch_time:.4f} seconds")
    print(f"Speedup vs PyTorch:         {pytorch_time / rocket_time:.2f}x\n")

class PyTorchLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.lstm1 = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.drop = nn.Dropout(0.2)
        self.lstm2 = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)

    def forward(self, x):
        x, _ = self.lstm1(x)
        x = self.drop(x)
        _, (h, _) = self.lstm2(x)
        return h.squeeze(0)

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

    start_time = time.time()
    r_model.train(X_rocket, y_rocket, X_rocket, y_rocket, 100, batch_size)
    rocket_time = time.time() - start_time
    print(f"Rocket Pure Stacked LSTM Training Time: {rocket_time:.4f} seconds")

    print("\nTraining PyTorch Pure Stacked LSTM for 100 epochs...")
    pt_model = PyTorchLSTM(input_dim, hidden_dim)
    optimizer = optim.Adam(pt_model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    
    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32)
    dataset = torch.utils.data.TensorDataset(X_t, y_t)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

    start_time = time.time()
    pt_model.train()
    for epoch in range(100):
        for batch_X, batch_y in loader:
            optimizer.zero_grad()
            out = pt_model(batch_X)
            loss = criterion(out, batch_y)
            loss.backward()
            optimizer.step()
    pytorch_time = time.time() - start_time
    print(f"PyTorch Pure Stacked LSTM Training Time:  {pytorch_time:.4f} seconds")
    print(f"Speedup vs PyTorch:                      {pytorch_time / rocket_time:.2f}x\n")

if __name__ == "__main__":
    test_rnn()
    test_lstm()
