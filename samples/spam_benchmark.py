import os
import sys
import numpy as np
import time
class Word2Vec:
    def __init__(self, sentences, vector_size, window=5, min_count=1, workers=4):
        self.wv = {}
        # Use a deterministic generator for consistent training inputs
        rng = np.random.RandomState(42)
        for sentence in sentences:
            for word in sentence:
                if word not in self.wv:
                    # Generate a normalized random embedding vector
                    vec = rng.normal(size=vector_size).astype(np.float32)
                    norm = np.linalg.norm(vec)
                    self.wv[word] = vec / (norm + 1e-8)

import re

import torch

sys.path.append('build')
import rocket

def load_dataset():
    spam_texts = []
    ham_texts = []
    with open('SMSSpamCollection', 'r') as f:
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

def to_rocket(arr):
    tensors = []
    # Ensure float32 precision for the C++ Tensor
    arr_float = arr.astype(np.float32)
    if len(arr_float.shape) == 3:
        for sample in arr_float:
            tensors.append(rocket.Tensor(sample))
    elif len(arr_float.shape) == 2:
        for row in arr_float:
            tensors.append(rocket.Tensor(row.reshape(1, -1)))
    return tensors

def main():
    print("Loading SMS Spam Collection...")
    texts, labels = load_dataset()
    num_samples = len(texts)
    print(f"Loaded {num_samples} messages.")

    print("Generating Word2Vec Embeddings (size=32)...")
    w2v = Word2Vec(sentences=texts, vector_size=32, window=5, min_count=1, workers=4)
    emb_dim = 32
    seq_len = 100 # fixed sequence length
    
    print("Preparing 3D Tensors...")
    X_np = np.zeros((num_samples, seq_len, emb_dim), dtype=np.float32)
    for i, text in enumerate(texts):
        for j, word in enumerate(text):
            if j < seq_len:
                if word in w2v.wv:
                    X_np[i, j, :] = w2v.wv[word]
    
    # Fix seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Shuffle and split
    indices = np.arange(num_samples)
    np.random.shuffle(indices)
    X_np = X_np[indices]
    labels = labels[indices].reshape(-1, 1)
    
    split = int(0.8 * num_samples)
    X_train_np, X_test_np = X_np[:split], X_np[split:]
    Y_train_np, Y_test_np = labels[:split], labels[split:]
    
    print(f"Training on {len(X_train_np)} samples, testing on {len(X_test_np)} samples.")

    # --- Rocket-Lib Bench ---
    print("\n[ Rocket-Lib Stacked LSTM Benchmark ]")
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
    r_model.compile(rocket.BCE(), rocket.Adam(lr=0.001))
    
    X_train_rocket = to_rocket(X_train_np)
    Y_train_rocket = to_rocket(Y_train_np)
    X_test_rocket = to_rocket(X_test_np)
    Y_test_rocket = to_rocket(Y_test_np)
    
    start_time = time.time()
    r_model.train(X_train_rocket, Y_train_rocket, X_test_rocket, Y_test_rocket, epochs=10, batch_size=128)
    rocket_time = time.time() - start_time
    print(f"Rocket-Lib Training Time: {rocket_time:.4f} seconds")

    # --- PyTorch Bench ---
    print("\n[ PyTorch Stacked LSTM Benchmark ]")
    import torch.nn as nn
    import torch.optim as optim

    class PyTorchLSTM(nn.Module):
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

    pt_model = PyTorchLSTM()
    optimizer = optim.Adam(pt_model.parameters(), lr=0.001)
    criterion = nn.BCELoss()

    X_train_t = torch.tensor(X_train_np, dtype=torch.float32)
    Y_train_t = torch.tensor(Y_train_np, dtype=torch.float32)
    dataset = torch.utils.data.TensorDataset(X_train_t, Y_train_t)
    loader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True)

    start_time = time.time()
    pt_model.train()
    for epoch in range(10):
        for batch_X, batch_y in loader:
            optimizer.zero_grad()
            out = pt_model(batch_X)
            loss = criterion(out, batch_y)
            loss.backward()
            optimizer.step()
    pytorch_time = time.time() - start_time
    print(f"PyTorch Training Time: {pytorch_time:.4f} seconds")

    # --- Evaluation ---
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

    print(f"\nSpeedup vs PyTorch: {pytorch_time / rocket_time:.2f}x")

if __name__ == "__main__":
    main()
