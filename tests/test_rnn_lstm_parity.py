import os
import sys
import numpy as np
import time
import re

# Ensure we can find the built rocket module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "build")))
try:
    import rocket
except ImportError as e:
    print(f"Failed to import rocket module: {e}")
    sys.exit(1)

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
    arr_float = arr.astype(np.float32)
    if len(arr_float.shape) == 3:
        for sample in arr_float:
            tensors.append(rocket.Tensor(sample))
    elif len(arr_float.shape) == 2:
        for row in arr_float:
            tensors.append(rocket.Tensor(row.reshape(1, -1)))
    return tensors


class Word2Vec:
    def __init__(self, sentences, vector_size, window=5, min_count=1, workers=4):
        self.wv = {}
        rng = np.random.RandomState(42)
        for sentence in sentences:
            for word in sentence:
                if word not in self.wv:
                    vec = rng.normal(size=vector_size).astype(np.float32)
                    norm = np.linalg.norm(vec)
                    self.wv[word] = vec / (norm + 1e-8)


# =====================================================================
# RNN / LSTM Performance Tests
# =====================================================================

class PyTorchRNN(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.rnn = nn.RNN(input_dim, hidden_dim, batch_first=True)
        self.dense = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        _, h = self.rnn(x)
        return self.dense(h.squeeze(0))


def test_rnn():
    print("\n" + "="*40)
    print(" 1. Testing RNNLayer Performance")
    print("="*40)
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
    print("\n" + "="*40)
    print(" 2. Testing LSTMLayer (Stacked) Performance")
    print("="*40)
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


# =====================================================================
# RNN / LSTM Mathematical Correctness Verification
# =====================================================================

class PyTorchStackedRNN(nn.Module):
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
    print("\n" + "="*40)
    print(" 3. Verifying Stacked RNN Mathematical Parity")
    print("="*40)
    batch_size = 2
    seq_len = 5
    input_dim = 3
    hidden_dim = 4

    X = np.random.randn(batch_size, seq_len, input_dim).astype(np.float32)
    X_rocket = to_rocket(X)

    pt_model = PyTorchStackedRNN(input_dim, hidden_dim)
    
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
    
    # Sync weights
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

    kw2_ih = pt_model.rnn2.weight_ih_l0.detach().numpy().T
    kw2_hh = pt_model.rnn2.weight_hh_l0.detach().numpy().T
    kw2_b = pt_model.rnn2.bias_ih_l0.detach().numpy() + pt_model.rnn2.bias_hh_l0.detach().numpy()
    
    for i in range(hidden_dim):
        for j in range(hidden_dim):
            rnn2.weights_ih.set_val(i, j, float(kw2_ih[i, j]))
            rnn2.weights_hh.set_val(i, j, float(kw2_hh[i, j]))
            rnn2.biases.set_val(0, j, float(kw2_b[j]))
            
    kw3_w = pt_model.dense.weight.detach().numpy().T
    kw3_b = pt_model.dense.bias.detach().numpy()
    for i in range(hidden_dim):
        for j in range(1):
            dense.weights.set_val(i, j, float(kw3_w[i, j]))
    dense.biases.set_val(0, 0, float(kw3_b[0]))

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
        sys.exit(1)


class PyTorchStackedLSTM(nn.Module):
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


def reorder_lstm_weights(w, axis=0):
    chunks = np.split(w, 4, axis=axis)
    return np.concatenate([chunks[0], chunks[1], chunks[3], chunks[2]], axis=axis)


def verify_lstm():
    print("\n" + "="*40)
    print(" 4. Verifying Stacked LSTM Mathematical Parity")
    print("="*40)
    batch_size = 2
    seq_len = 5
    input_dim = 3
    hidden_dim = 4

    X = np.random.randn(batch_size, seq_len, input_dim).astype(np.float32)
    X_rocket = to_rocket(X)

    pt_model = PyTorchStackedLSTM(input_dim, hidden_dim)
    
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
    
    # Sync weights
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

    w2_ih = reorder_lstm_weights(pt_model.lstm2.weight_ih_l0.detach().numpy(), axis=0).T
    w2_hh = reorder_lstm_weights(pt_model.lstm2.weight_hh_l0.detach().numpy(), axis=0).T
    b2_total = reorder_lstm_weights(pt_model.lstm2.bias_ih_l0.detach().numpy() + pt_model.lstm2.bias_hh_l0.detach().numpy(), axis=0)
    
    for i in range(hidden_dim):
        for j in range(4 * hidden_dim):
            lstm2.weights_ih.set_val(i, j, float(w2_ih[i, j]))
            lstm2.weights_hh.set_val(i, j, float(w2_hh[i, j]))
            lstm2.biases.set_val(0, j, float(b2_total[j]))
            
    kw3_w = pt_model.dense.weight.detach().numpy().T
    kw3_b = pt_model.dense.bias.detach().numpy()
    for i in range(hidden_dim):
        for j in range(1):
            dense.weights.set_val(i, j, float(kw3_w[i, j]))
    dense.biases.set_val(0, 0, float(kw3_b[0]))

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
        sys.exit(1)


# =====================================================================
# Spam Classification Tests & Benchmarks
# =====================================================================

def generate_spam_data(num_samples=1000):
    spam_keywords = ["free", "win", "cash", "prize", "urgent", "click", "offer", "guaranteed", "buy", "cheap"]
    ham_keywords = ["hello", "meeting", "tomorrow", "mom", "dinner", "ok", "thanks", "schedule", "call", "later"]
    
    vocab = spam_keywords + ham_keywords
    
    np.random.seed(42)
    embeddings = {}
    for i, w in enumerate(vocab):
        vec = np.random.randn(8) * 0.1
        vec[i % 8] += 0.5
        embeddings[w] = vec
        
    X_text = []
    Y_labels = []
    
    for _ in range(num_samples):
        is_spam = np.random.rand() > 0.5
        seq = []
        if is_spam:
            length = np.random.randint(4, 8)
            seq = [np.random.choice(spam_keywords) for _ in range(length)]
            Y_labels.append([1.0])
        else:
            length = np.random.randint(4, 8)
            seq = [np.random.choice(ham_keywords) for _ in range(length)]
            Y_labels.append([0.0])
            
        while len(seq) < 8:
            seq.append(np.random.choice(vocab))
            
        X_text.append(seq[:8])
        
    return X_text, np.array(Y_labels, dtype=np.float32), embeddings, vocab


def text_to_tensor(X_text, embeddings, seq_len, emb_dim):
    X = np.zeros((len(X_text), seq_len, emb_dim), dtype=np.float32)
    for i, seq in enumerate(X_text):
        for j, w in enumerate(seq):
            if j < seq_len:
                X[i, j, :] = embeddings.get(w, np.zeros(emb_dim))
    return X


def test_spam_classification():
    print("\n" + "="*40)
    print(" 5. Running Synthetic Spam Classification Benchmark")
    print("="*40)
    X_text, Y_np, embeddings, vocab = generate_spam_data(1000)
    
    seq_len = 8
    emb_dim = 8
    hidden_dim = 16
    
    X_np = text_to_tensor(X_text, embeddings, seq_len, emb_dim)
    
    X_train = to_rocket(X_np[:800])
    Y_train = to_rocket(Y_np[:800])
    X_test = to_rocket(X_np[800:])
    Y_test = to_rocket(Y_np[800:])
    
    model = rocket.Model()
    inp = rocket.InputLayer()
    lstm1 = rocket.LSTMLayer(emb_dim, hidden_dim, seq_len, True)
    drop = rocket.DropoutLayer(0.3)
    lstm2 = rocket.LSTMLayer(hidden_dim, hidden_dim, seq_len, False)
    dense_out = rocket.DenseLayer(hidden_dim, 1)
    act_out = rocket.ActivationLayer(rocket.Sigmoid())
    
    model.add(inp, [])
    model.add(lstm1, [inp])
    model.add(drop, [lstm1])
    model.add(lstm2, [drop])
    model.add(dense_out, [lstm2])
    model.add(act_out, [dense_out])
    
    model.setInputOutputLayers([inp], [act_out])
    model.compile(rocket.BCE(), rocket.Adam(lr=0.01))
    
    print("Training Model for 20 Epochs...")
    model.train(X_train, Y_train, X_test, Y_test, 20, 32)
    
    print("\n--- Example Inference ---")
    spam_sample = ["urgent", "free", "cash", "prize", "click", "urgent", "win", "now"]
    ham_sample = ["hello", "mom", "dinner", "tomorrow", "thanks", "later", "call", "ok"]
    
    spam_np = text_to_tensor([spam_sample], embeddings, seq_len, emb_dim)
    ham_np = text_to_tensor([ham_sample], embeddings, seq_len, emb_dim)
    
    pred_spam = model.predict(to_rocket(spam_np))[0].get_val(0, 0)
    pred_ham = model.predict(to_rocket(ham_np))[0].get_val(0, 0)
    
    print(f"Spam Message: {' '.join(spam_sample)}")
    print(f"Prediction: {pred_spam:.4f} (Spam? {pred_spam > 0.5})")
    print(f"Ham Message: {' '.join(ham_sample)}")
    print(f"Prediction: {pred_ham:.4f} (Spam? {pred_ham > 0.5})")


def load_dataset():
    spam_texts = []
    ham_texts = []
    filepath = 'tests/data/sms.tsv' if os.path.exists('tests/data/sms.tsv') else 'SMSSpamCollection'
    with open(filepath, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                label = parts[0]
                text = re.sub(r'[^a-zA-Z ]', '', parts[1].lower())
                tokens = text.split()
                if label == 'spam' and len(spam_texts) < 1000:
                    spam_texts.append(tokens)
                elif label == 'ham' and len(ham_texts) < 1000:
                    ham_texts.append(tokens)
    
    texts = spam_texts + ham_texts
    labels = [1.0] * len(spam_texts) + [0.0] * len(ham_texts)
    return texts, np.array(labels, dtype=np.float32)


def test_spam_benchmark():
    print("\n" + "="*40)
    print(" 6. Running Real SMS Spam Collection Benchmark")
    print("="*40)
    texts, labels = load_dataset()
    num_samples = len(texts)
    print(f"Loaded {num_samples} messages.")

    print("Generating Word2Vec Embeddings (size=32)...")
    w2v = Word2Vec(sentences=texts, vector_size=32)
    emb_dim = 32
    seq_len = 100
    
    print("Preparing 3D Tensors...")
    X_np = np.zeros((num_samples, seq_len, emb_dim), dtype=np.float32)
    for i, text in enumerate(texts):
        valid_words = [word for word in text if word in w2v.wv][:seq_len]
        start_idx = seq_len - len(valid_words)
        for j, word in enumerate(valid_words):
            X_np[i, start_idx + j, :] = w2v.wv[word]
    
    indices = np.arange(num_samples)
    np.random.shuffle(indices)
    X_np = X_np[indices]
    labels = labels[indices].reshape(-1, 1)
    
    split = int(0.8 * num_samples)
    X_train_np, X_test_np = X_np[:split], X_np[split:]
    Y_train_np, Y_test_np = labels[:split], labels[split:]
    
    print(f"Training on {len(X_train_np)} samples, testing on {len(X_test_np)} samples.")

    r_model = rocket.Model()
    inp = rocket.InputLayer()
    lstm1 = rocket.LSTMLayer(emb_dim, 32, seq_len, True)
    lstm2 = rocket.LSTMLayer(32, 32, seq_len, False)
    dense = rocket.DenseLayer(32, 1)
    act = rocket.ActivationLayer(rocket.Sigmoid())
    
    r_model.add(inp, [])
    r_model.add(lstm1, [inp])
    r_model.add(lstm2, [lstm1])
    r_model.add(dense, [lstm2])
    r_model.add(act, [dense])
    r_model.setInputOutputLayers([inp], [act])
    r_model.compile(rocket.BCE(), rocket.Adam(lr=0.003))
    
    X_train_rocket = to_rocket(X_train_np)
    Y_train_rocket = to_rocket(Y_train_np)
    X_test_rocket = to_rocket(X_test_np)
    Y_test_rocket = to_rocket(Y_test_np)
    
    start_time = time.time()
    r_model.train(X_train_rocket, Y_train_rocket, X_test_rocket, Y_test_rocket, epochs=15, batch_size=128)
    rocket_time = time.time() - start_time
    print(f"Rocket-Lib Training Time: {rocket_time:.4f} seconds")

    class PyTorchLSTMModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(input_size=emb_dim, hidden_size=32, num_layers=2, batch_first=True)
            self.dense = nn.Linear(32, 1)
            self.sigmoid = nn.Sigmoid()

        def forward(self, x):
            out, _ = self.lstm(x)
            out = out[:, -1, :]
            out = self.sigmoid(self.dense(out))
            return out

    pt_model = PyTorchLSTMModel()
    optimizer = optim.Adam(pt_model.parameters(), lr=0.003)
    criterion = nn.BCELoss()

    X_train_t = torch.tensor(X_train_np, dtype=torch.float32)
    Y_train_t = torch.tensor(Y_train_np, dtype=torch.float32)
    dataset = torch.utils.data.TensorDataset(X_train_t, Y_train_t)
    loader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True)

    start_time = time.time()
    pt_model.train()
    for epoch in range(15):
        total_loss = 0.0
        for batch_X, batch_y in loader:
            optimizer.zero_grad()
            out = pt_model(batch_X)
            loss = criterion(out, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"PyTorch Epoch {epoch+1}/15 - Loss: {total_loss/len(loader):.4f}")
    pytorch_time = time.time() - start_time
    print(f"PyTorch Training Time: {pytorch_time:.4f} seconds")

    def evaluate(preds, actual):
        preds = np.array(preds).flatten()
        actual = np.array(actual).flatten()
        preds_binary = (preds > 0.5).astype(np.float32)
        
        tp = np.sum((preds_binary == 1) & (actual == 1))
        tn = np.sum((preds_binary == 0) & (actual == 0))
        fp = np.sum((preds_binary == 1) & (actual == 0))
        fn = np.sum((preds_binary == 0) & (actual == 1))
        
        accuracy = (tp + tn) / len(actual) if len(actual) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return accuracy, precision, recall, f1

    # Weight synchronization to evaluate parity
    w_ih = reorder_lstm_weights(pt_model.lstm.weight_ih_l0.detach().numpy(), axis=0).T
    w_hh = reorder_lstm_weights(pt_model.lstm.weight_hh_l0.detach().numpy(), axis=0).T
    b_total = reorder_lstm_weights(pt_model.lstm.bias_ih_l0.detach().numpy() + pt_model.lstm.bias_hh_l0.detach().numpy(), axis=0)
    for i in range(emb_dim):
        for j in range(4 * 32):
            lstm1.weights_ih.set_val(i, j, float(w_ih[i, j]))
    for i in range(32):
        for j in range(4 * 32):
            lstm1.weights_hh.set_val(i, j, float(w_hh[i, j]))
            lstm1.biases.set_val(0, j, float(b_total[j]))

    w2_ih = reorder_lstm_weights(pt_model.lstm.weight_ih_l1.detach().numpy(), axis=0).T
    w2_hh = reorder_lstm_weights(pt_model.lstm.weight_hh_l1.detach().numpy(), axis=0).T
    b2_total = reorder_lstm_weights(pt_model.lstm.bias_ih_l1.detach().numpy() + pt_model.lstm.bias_hh_l1.detach().numpy(), axis=0)
    for i in range(32):
        for j in range(4 * 32):
            lstm2.weights_ih.set_val(i, j, float(w2_ih[i, j]))
            lstm2.weights_hh.set_val(i, j, float(w2_hh[i, j]))
            lstm2.biases.set_val(0, j, float(b2_total[j]))

    kw3_w = pt_model.dense.weight.detach().numpy().T
    kw3_b = pt_model.dense.bias.detach().numpy()
    for i in range(32):
        for j in range(1):
            dense.weights.set_val(i, j, float(kw3_w[i, j]))
    dense.biases.set_val(0, 0, float(kw3_b[0]))

    print("\n[ Rocket-Lib Evaluation ]")
    r_preds = []
    for x in X_test_rocket:
        res = r_model.predict([x])
        r_preds.append(res[0].get_val(0, 0))
    
    r_metrics = evaluate(r_preds, Y_test_np)
    print(f"Accuracy:  {r_metrics[0]:.4f}")
    print(f"Precision: {r_metrics[1]:.4f}")
    print(f"Recall:    {r_metrics[2]:.4f}")
    print(f"F1-Score:  {r_metrics[3]:.4f}")

    print("\n[ PyTorch Evaluation ]")
    pt_model.eval()
    X_test_t = torch.tensor(X_test_np, dtype=torch.float32)
    with torch.no_grad():
        pt_preds = pt_model(X_test_t).numpy()
    pt_metrics = evaluate(pt_preds, Y_test_np)
    print(f"Accuracy:  {pt_metrics[0]:.4f}")
    print(f"Precision: {pt_metrics[1]:.4f}")
    print(f"Recall:    {pt_metrics[2]:.4f}")
    print(f"F1-Score:  {pt_metrics[3]:.4f}")

    print("\n[ Correctness Check ]")
    acc_diff = abs(r_metrics[0] - pt_metrics[0])
    print(f"Accuracy Difference: {acc_diff * 100:.2f}%")
    if acc_diff <= 0.02:
        print("PASS: Rocket-Lib is within 2% of PyTorch accuracy.")
    else:
        print("FAIL: Rocket-Lib accuracy difference exceeds 2%.")
        sys.exit(1)

    print(f"\nSpeedup vs PyTorch: {pytorch_time / rocket_time:.2f}x")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Rocket-Lib RNN/LSTM Correctness & Parity Tests")
    parser.add_argument(
        "--test", 
        type=str, 
        choices=["rnn_perf", "lstm_perf", "rnn_correct", "lstm_correct", "spam_synthetic", "spam_real", "all"], 
        default="all",
        help="Specify which test to execute (default: all)"
    )
    args = parser.parse_args()

    if args.test == "all":
        test_rnn()
        test_lstm()
        verify_rnn()
        verify_lstm()
        test_spam_classification()
        test_spam_benchmark()
    elif args.test == "rnn_perf":
        test_rnn()
    elif args.test == "lstm_perf":
        test_lstm()
    elif args.test == "rnn_correct":
        verify_rnn()
    elif args.test == "lstm_correct":
        verify_lstm()
    elif args.test == "spam_synthetic":
        test_spam_classification()
    elif args.test == "spam_real":
        test_spam_benchmark()
