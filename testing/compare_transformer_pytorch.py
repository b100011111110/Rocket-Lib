import os
import sys
import time
import string
import numpy as np

sys.path.append('build')
import rocket

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
except ImportError:
    print("Please install PyTorch: pip install torch")
    sys.exit(1)

def load_spam_data(filepath, max_samples=1500):
    X_text = []
    Y_labels = []
    vocab_set = set()
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= max_samples:
                break
            parts = line.strip().split('\t')
            if len(parts) != 2:
                continue
                
            label, text = parts
            
            text = text.lower()
            for p in string.punctuation:
                text = text.replace(p, ' ')
            tokens = text.split()
            
            if not tokens:
                continue
                
            vocab_set.update(tokens)
            X_text.append(tokens)
            Y_labels.append([1.0] if label == 'spam' else [0.0])
            
    vocab = list(vocab_set)
    vocab.append("<PAD>")
    vocab.append("<UNK>")
    
    np.random.seed(42)
    embeddings = {}
    for w in vocab:
        embeddings[w] = np.random.randn(8) * 0.1
        
    return X_text, np.array(Y_labels, dtype=np.float32), embeddings, vocab

def text_to_tensor(X_text, embeddings, seq_len, emb_dim):
    X = np.zeros((len(X_text), seq_len, emb_dim), dtype=np.float32)
    unk_emb = embeddings.get("<UNK>", np.zeros(emb_dim))
    for i, seq in enumerate(X_text):
        for j, w in enumerate(seq):
            if j < seq_len:
                X[i, j, :] = embeddings.get(w, unk_emb)
    return X

def to_rocket(arr):
    tensors = []
    if len(arr.shape) == 3:
        for sample in arr:
            t = rocket.Tensor(sample.astype(np.float32))
            tensors.append(t)
    elif len(arr.shape) == 2:
        for row in arr:
            t = rocket.Tensor(row.reshape(1, -1).astype(np.float32))
            tensors.append(t)
    return tensors

def build_rocket_model(emb_dim, seq_len, ff_dim, lr=0.001):
    model = rocket.Model()
    
    inp = rocket.InputLayer()
    enc1 = rocket.TransformerEncoderLayer(emb_dim, seq_len, ff_dim)
    enc2 = rocket.TransformerEncoderLayer(emb_dim, seq_len, ff_dim)
    enc3 = rocket.TransformerEncoderLayer(emb_dim, seq_len, ff_dim)
    
    pool = rocket.RNNLayer(emb_dim, emb_dim, seq_len, False)
    dense_out = rocket.DenseLayer(emb_dim, 1)
    act_out = rocket.ActivationLayer(rocket.Sigmoid())
    
    model.add(inp, [])
    model.add(enc1, [inp])
    model.add(enc2, [enc1])
    model.add(enc3, [enc2])
    model.add(pool, [enc3])
    model.add(dense_out, [pool])
    model.add(act_out, [dense_out])
    
    model.setInputOutputLayers([inp], [act_out])
    model.compile(rocket.BCE(), rocket.Adam(lr=lr))
    return model

class SimpleTransformerEncoderBlock(nn.Module):
    def __init__(self, emb_dim, ff_dim):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim=emb_dim, num_heads=1, dropout=0.0, batch_first=True)
        self.norm1 = nn.LayerNorm(emb_dim, eps=1e-5)
        self.ffn = nn.Sequential(
            nn.Linear(emb_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, emb_dim)
        )
        self.norm2 = nn.LayerNorm(emb_dim, eps=1e-5)

    def forward(self, x):
        attn_out, _ = self.mha(x, x, x)
        x = self.norm1(x + attn_out)
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        return x

class PyTorchTransformerSeq(nn.Module):
    def __init__(self, emb_dim, ff_dim):
        super().__init__()
        self.enc1 = SimpleTransformerEncoderBlock(emb_dim, ff_dim)
        self.enc2 = SimpleTransformerEncoderBlock(emb_dim, ff_dim)
        self.enc3 = SimpleTransformerEncoderBlock(emb_dim, ff_dim)
        self.rnn = nn.RNN(input_size=emb_dim, hidden_size=emb_dim, batch_first=True)
        self.dense_out = nn.Linear(emb_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.enc1(x)
        x = self.enc2(x)
        x = self.enc3(x)
        _, h = self.rnn(x)
        out = self.sigmoid(self.dense_out(h.squeeze(0)))
        return out

def build_pytorch_model(emb_dim, seq_len, ff_dim, lr=0.001):
    model = PyTorchTransformerSeq(emb_dim, ff_dim)
    return model

def main():
    print("Loading Real SMS Spam Collection Dataset...")
    X_text, Y_np, embeddings, vocab = load_spam_data("samples/data/sms.tsv", 4000)
    
    seq_len = 16
    emb_dim = 8
    ff_dim = 16
    epochs = 100
    
    X_np = text_to_tensor(X_text, embeddings, seq_len, emb_dim)
    
    split_idx = int(len(X_np) * 0.8)
    X_train_np = X_np[:split_idx]
    Y_train_np = Y_np[:split_idx]
    X_test_np = X_np[split_idx:]
    Y_test_np = Y_np[split_idx:]
    
    X_train_rk = to_rocket(X_train_np)
    Y_train_rk = to_rocket(Y_train_np)
    X_test_rk = to_rocket(X_test_np)
    Y_test_rk = to_rocket(Y_test_np)
    
    print("\n==============================================")
    print("      Training PyTorch Transformer Encoder")
    print("==============================================")
    pytorch_model = build_pytorch_model(emb_dim, seq_len, ff_dim, lr=0.005)
    optimizer = optim.Adam(pytorch_model.parameters(), lr=0.005)
    criterion = nn.BCELoss()
    
    X_train_t = torch.tensor(X_train_np, dtype=torch.float32)
    Y_train_t = torch.tensor(Y_train_np, dtype=torch.float32)
    dataset = torch.utils.data.TensorDataset(X_train_t, Y_train_t)
    loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
    
    pt_start = time.time()
    pytorch_model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_X, batch_y in loader:
            optimizer.zero_grad()
            out = pytorch_model(batch_X)
            loss = criterion(out, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} - loss: {epoch_loss/len(loader):.4f}")
    pt_end = time.time()
    
    pytorch_model.eval()
    X_test_t = torch.tensor(X_test_np, dtype=torch.float32)
    with torch.no_grad():
        pt_preds = pytorch_model(X_test_t).numpy()
    pytorch_acc = np.mean((pt_preds > 0.5) == Y_test_np)
    
    print("\n==============================================")
    print("      Training Rocket Transformer Encoder")
    print("==============================================")
    rocket_model = build_rocket_model(emb_dim, seq_len, ff_dim, lr=0.005)
    
    r_start = time.time()
    rocket_model.train(X_train_rk, Y_train_rk, X_test_rk, Y_test_rk, epochs, 32)
    r_end = time.time()
    
    correct = 0
    total = len(X_test_rk)
    for i in range(total):
        pred = rocket_model.predict([X_test_rk[i]])[0].get_val(0, 0)
        y_true = Y_test_np[i][0]
        if (pred > 0.5) == (y_true > 0.5):
            correct += 1
    rocket_acc = correct / total
    
    print("\n==============================================")
    print("                COMPARISON")
    print("==============================================")
    print(f"{'Metric':<20} | {'PyTorch':<15} | {'Rocket':<15}")
    print("-" * 55)
    print(f"{'Test Accuracy':<20} | {pytorch_acc*100:13.2f}% | {rocket_acc*100:13.2f}%")
    print(f"{'Training Time':<20} | {pt_end - pt_start:13.2f}s | {r_end - r_start:13.2f}s")
    print("==============================================\n")
    
if __name__ == "__main__":
    main()
