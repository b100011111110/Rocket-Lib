import os
import sys
import time
import numpy as np
import pandas as pd

try:
    import rocket
except ImportError:
    print("Error: Could not import 'rocket'. Make sure PYTHONPATH is set to the build directory.")
    sys.exit(1)

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
except ImportError:
    print("Please install PyTorch: pip install torch")
    sys.exit(1)

def load_and_preprocess_emotion(csv_path='samples/data/emotion.csv', max_samples=4000, seq_len=64, vocab_size=100):
    print("Loading Emotion Dataset...")
    df = pd.read_csv(csv_path)
    
    texts = df['text'].values[:max_samples]
    labels = df['label'].values[:max_samples]
    
    # Get unique characters
    chars = sorted(list(set("".join(texts))))
    char_to_ix = {ch: i+1 for i, ch in enumerate(chars)} # 0 is for padding
    ix_to_char = {i+1: ch for i, ch in enumerate(chars)}
    
    # Ensure we don't exceed vocab_size
    actual_vocab = len(chars) + 1
    if actual_vocab > vocab_size:
        print(f"Warning: Actual vocabulary ({actual_vocab}) exceeds vocab_size ({vocab_size}). Some characters will be mapped to 0.")
    
    X = np.zeros((len(texts), seq_len, vocab_size), dtype=np.float32)
    Y = np.zeros((len(texts), 6), dtype=np.float32) # 6 emotion classes
    
    for i, text in enumerate(texts):
        # One-hot encode the characters
        for j, ch in enumerate(text[:seq_len]):
            idx = char_to_ix.get(ch, 0)
            if idx < vocab_size:
                X[i, j, idx] = 1.0
            else:
                X[i, j, 0] = 1.0
        
        # Target
        label = int(labels[i])
        if 0 <= label < 6:
            Y[i, label] = 1.0
            
    # Split into train and test
    split_idx = int(0.8 * len(texts))
    X_train, X_test = X[:split_idx], X[split_idx:]
    Y_train, Y_test = Y[:split_idx], Y[split_idx:]
    
    print(f"Vocab Size: {vocab_size}")
    print(f"Training samples: {len(X_train)}, Testing samples: {len(X_test)}")
    
    return X_train, Y_train, X_test, Y_test, char_to_ix, ix_to_char

class TransformerEncoderBlock(nn.Module):
    def __init__(self, emb_dim, num_heads, ff_dim):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim=emb_dim, num_heads=num_heads, dropout=0.1, batch_first=True)
        self.norm1 = nn.LayerNorm(emb_dim, eps=1e-5)
        self.ffn = nn.Sequential(
            nn.Linear(emb_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, emb_dim)
        )
        self.norm2 = nn.LayerNorm(emb_dim, eps=1e-5)
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)

    def forward(self, x):
        attn_out, _ = self.mha(x, x, x)
        x = self.norm1(x + self.dropout1(attn_out))
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout2(ffn_out))
        return x

class PyTorchTransformerEncoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, num_heads, ff_dim, num_classes):
        super().__init__()
        self.proj = nn.Linear(vocab_size, emb_dim)
        self.enc1 = TransformerEncoderBlock(emb_dim, num_heads, ff_dim)
        self.enc2 = TransformerEncoderBlock(emb_dim, num_heads, ff_dim)
        self.dense_out = nn.Linear(emb_dim, num_classes)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = self.proj(x)
        x = self.enc1(x)
        x = self.enc2(x)
        # Global average pooling
        x = torch.mean(x, dim=1)
        out = self.softmax(self.dense_out(x))
        return out

def build_pytorch_model(vocab_size, seq_len, num_classes=6, emb_dim=32, ff_dim=64, lr=0.005, num_heads=4):
    model = PyTorchTransformerEncoder(vocab_size, emb_dim, num_heads, ff_dim, num_classes)
    return model

def build_rocket_model(vocab_size, seq_len, num_classes=6, emb_dim=32, ff_dim=64, lr=0.001, num_heads=4):
    model = rocket.Model()
    
    inp = rocket.InputLayer()
    
    # Project one-hot
    emb_proj = rocket.DenseLayer(vocab_size, emb_dim)
    
    # 2 Layers of Transformer Encoder
    enc1 = rocket.TransformerMHEncoderLayer(emb_dim, seq_len, ff_dim, num_heads, 0.1)
    enc2 = rocket.TransformerMHEncoderLayer(emb_dim, seq_len, ff_dim, num_heads, 0.1)
    
    pool = rocket.GlobalAveragePooling1DLayer(seq_len, emb_dim)
    
    # Output projection to classes
    dense_out = rocket.DenseLayer(emb_dim, num_classes)
    act_out = rocket.ActivationLayer(rocket.Softmax())
    
    model.add(inp, [])
    model.add(emb_proj, [inp])
    model.add(enc1, [emb_proj])
    model.add(enc2, [enc1])
    model.add(pool, [enc2])
    model.add(dense_out, [pool])
    model.add(act_out, [dense_out])
    
    model.setInputOutputLayers([inp], [act_out])
    model.compile(rocket.CCE(), rocket.Adam(lr=lr, eps=1e-7))
    return model

def to_rocket(np_array):
    tensors = []
    for i in range(len(np_array)):
        t = rocket.Tensor(np_array[i])
        tensors.append(t)
    return tensors

def evaluate_rocket(model, X_test_tensors, Y_test_np):
    correct = 0
    total = len(X_test_tensors)
    for i in range(total):
        pred = model.predict([X_test_tensors[i]])[0]
        pred_arr = np.array(pred).flatten()
        pred_idx = np.argmax(pred_arr)
        true_idx = np.argmax(Y_test_np[i])
        if pred_idx == true_idx:
            correct += 1
    return correct / total

def evaluate_pytorch(model, X_test, Y_test):
    model.eval()
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    with torch.no_grad():
        preds = model(X_test_t).numpy()
    pred_indices = np.argmax(preds, axis=-1)
    true_indices = np.argmax(Y_test, axis=-1)
    return np.mean(pred_indices == true_indices)

def main():
    seq_len = 32
    num_samples = 5000
    vocab_size = 100
    emb_dim = 32
    ff_dim = 64
    num_heads = 4
    epochs = 20
    batch_size = 32
    num_classes = 6
    
    X_train, Y_train, X_test, Y_test, char_to_ix, ix_to_char = load_and_preprocess_emotion(
        max_samples=num_samples, seq_len=seq_len, vocab_size=vocab_size)
    
    print("\n==============================================")
    print("      Training PyTorch Transformer Encoder")
    print("==============================================")
    pytorch_model = build_pytorch_model(vocab_size, seq_len, num_classes, emb_dim, ff_dim, lr=0.005, num_heads=num_heads)
    optimizer = optim.Adam(pytorch_model.parameters(), lr=0.005)
    criterion = nn.CrossEntropyLoss()
    
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    Y_train_t = torch.tensor(Y_train, dtype=torch.float32)
    dataset = torch.utils.data.TensorDataset(X_train_t, Y_train_t)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
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
        print(f"Epoch {epoch+1}/{epochs} - loss: {epoch_loss/len(loader):.4f}")
    pt_end = time.time()
    
    print("\n==============================================")
    print("      Training Rocket Transformer Encoder")
    print("==============================================")
    rocket_model = build_rocket_model(vocab_size, seq_len, num_classes, emb_dim, ff_dim, lr=0.001, num_heads=num_heads)
    
    rocket_X_train = to_rocket(X_train)
    rocket_Y_train = to_rocket(Y_train.reshape(-1, 1, num_classes))
    rocket_X_test = to_rocket(X_test)
    rocket_Y_test = to_rocket(Y_test.reshape(-1, 1, num_classes))
    
    r_start = time.time()
    rocket_model.train(rocket_X_train, rocket_Y_train, rocket_X_test, rocket_Y_test, epochs, batch_size)
    r_end = time.time()
    
    # Evaluate
    pytorch_acc = evaluate_pytorch(pytorch_model, X_test, Y_test)
    rocket_acc = evaluate_rocket(rocket_model, rocket_X_test, Y_test)
    
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
