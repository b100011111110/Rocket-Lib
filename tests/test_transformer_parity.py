import os
import sys
import time
import string
import urllib.request
import numpy as np
import pandas as pd

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

# Ensure determinism
os.environ["ROCKET_SEED"] = "42"
np.random.seed(42)


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


# =====================================================================
# 1. Transformer Encoder Test (on Emotion Dataset)
# =====================================================================

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
        x = torch.mean(x, dim=1)
        out = self.softmax(self.dense_out(x))
        return out


def load_and_preprocess_emotion(csv_path='tests/data/emotion.csv', max_samples=4000, seq_len=64, vocab_size=100):
    print("Loading Emotion Dataset...")
    df = pd.read_csv(csv_path)
    
    texts = df['text'].values[:max_samples]
    labels = df['label'].values[:max_samples]
    
    chars = sorted(list(set("".join(texts))))
    char_to_ix = {ch: i+1 for i, ch in enumerate(chars)}
    ix_to_char = {i+1: ch for i, ch in enumerate(chars)}
    
    actual_vocab = len(chars) + 1
    if actual_vocab > vocab_size:
        print(f"Warning: Actual vocabulary ({actual_vocab}) exceeds vocab_size ({vocab_size}). Some characters will be mapped to 0.")
    
    X = np.zeros((len(texts), seq_len, vocab_size), dtype=np.float32)
    Y = np.zeros((len(texts), 6), dtype=np.float32)
    
    for i, text in enumerate(texts):
        for j, ch in enumerate(text[:seq_len]):
            idx = char_to_ix.get(ch, 0)
            if idx < vocab_size:
                X[i, j, idx] = 1.0
            else:
                X[i, j, 0] = 1.0
        label = int(labels[i])
        if 0 <= label < 6:
            Y[i, label] = 1.0
            
    split_idx = int(0.8 * len(texts))
    X_train, X_test = X[:split_idx], X[split_idx:]
    Y_train, Y_test = Y[:split_idx], Y[split_idx:]
    
    print(f"Vocab Size: {vocab_size}")
    print(f"Training samples: {len(X_train)}, Testing samples: {len(X_test)}")
    return X_train, Y_train, X_test, Y_test, char_to_ix, ix_to_char


def test_transformer_encoder():
    print("\n" + "="*40)
    print(" 1. Training Transformer Encoder (Emotion)")
    print("="*40)
    
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
    
    print("\nTraining PyTorch Transformer Encoder...")
    pytorch_model = PyTorchTransformerEncoder(vocab_size, emb_dim, num_heads, ff_dim, num_classes)
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
    
    print("\nTraining Rocket Transformer Encoder...")
    rocket_model = rocket.Model()
    inp = rocket.InputLayer()
    emb_proj = rocket.DenseLayer(vocab_size, emb_dim)
    enc1 = rocket.TransformerMHEncoderLayer(emb_dim, seq_len, ff_dim, num_heads, 0.1)
    enc2 = rocket.TransformerMHEncoderLayer(emb_dim, seq_len, ff_dim, num_heads, 0.1)
    pool = rocket.GlobalAveragePooling1DLayer(seq_len, emb_dim)
    dense_out = rocket.DenseLayer(emb_dim, num_classes)
    act_out = rocket.ActivationLayer(rocket.Softmax())
    
    rocket_model.add(inp, [])
    rocket_model.add(emb_proj, [inp])
    rocket_model.add(enc1, [emb_proj])
    rocket_model.add(enc2, [enc1])
    rocket_model.add(pool, [enc2])
    rocket_model.add(dense_out, [pool])
    rocket_model.add(act_out, [dense_out])
    
    rocket_model.setInputOutputLayers([inp], [act_out])
    rocket_model.compile(rocket.CCE(), rocket.Adam(lr=0.001, eps=1e-7))
    
    rocket_X_train = to_rocket(X_train)
    rocket_Y_train = to_rocket(Y_train.reshape(-1, 1, num_classes))
    rocket_X_test = to_rocket(X_test)
    rocket_Y_test = to_rocket(Y_test.reshape(-1, 1, num_classes))
    
    r_start = time.time()
    rocket_model.train(rocket_X_train, rocket_Y_train, rocket_X_test, rocket_Y_test, epochs, batch_size)
    r_end = time.time()
    
    pytorch_acc = np.mean(np.argmax(pytorch_model(torch.tensor(X_test)).detach().numpy(), axis=-1) == np.argmax(Y_test, axis=-1))
    
    correct = 0
    for i in range(len(rocket_X_test)):
        pred = rocket_model.predict([rocket_X_test[i]])[0]
        pred_idx = np.argmax(np.array(pred).flatten())
        true_idx = np.argmax(Y_test[i])
        if pred_idx == true_idx:
            correct += 1
    rocket_acc = correct / len(rocket_X_test)
    
    print("\n" + "="*45)
    print(" ENCODER COMPARISON")
    print("="*45)
    print(f"{'Metric':<20} | {'PyTorch':<15} | {'Rocket':<15}")
    print("-" * 55)
    print(f"{'Test Accuracy':<20} | {pytorch_acc*100:13.2f}% | {rocket_acc*100:13.2f}%")
    print(f"{'Training Time':<20} | {pt_end - pt_start:13.2f}s | {r_end - r_start:13.2f}s")
    print("="*45)


# =====================================================================
# 2. Transformer Decoder Test (on Tiny Shakespeare)
# =====================================================================

class CausalTransformerDecoderBlock(nn.Module):
    def __init__(self, emb_dim, num_heads, ff_dim, seq_len):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim=emb_dim, num_heads=num_heads, dropout=0.0, batch_first=True)
        self.norm1 = nn.LayerNorm(emb_dim, eps=1e-5)
        self.ffn = nn.Sequential(
            nn.Linear(emb_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, emb_dim)
        )
        self.norm2 = nn.LayerNorm(emb_dim, eps=1e-5)
        self.seq_len = seq_len

    def forward(self, x):
        mask = nn.Transformer.generate_square_subsequent_mask(x.size(1), device=x.device)
        attn_out, _ = self.mha(x, x, x, attn_mask=mask, is_causal=True)
        x = self.norm1(x + attn_out)
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        return x


class PyTorchTransformerDecoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, num_heads, ff_dim, seq_len):
        super().__init__()
        self.proj = nn.Linear(vocab_size, emb_dim)
        self.dec1 = CausalTransformerDecoderBlock(emb_dim, num_heads, ff_dim, seq_len)
        self.dec2 = CausalTransformerDecoderBlock(emb_dim, num_heads, ff_dim, seq_len)
        self.dense_out = nn.Linear(emb_dim, vocab_size)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = self.proj(x)
        x = self.dec1(x)
        x = self.dec2(x)
        out = self.softmax(self.dense_out(x))
        return out


def prepare_tinyshakespeare(seq_len=32, num_samples=2000):
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    filepath = "tests/data/input.txt"
    os.makedirs("tests/data", exist_ok=True)
    if not os.path.exists(filepath):
        print("Downloading Tiny Shakespeare...")
        urllib.request.urlretrieve(url, filepath)

    with open(filepath, 'r', encoding='utf-8') as f:
        text = f.read()

    chars = sorted(list(set(text)))
    target_vocab = 200
    if len(chars) < target_vocab:
        for i in range(target_vocab - len(chars)):
            chars.append(chr(1000 + i))
            
    vocab_size = len(chars)
    char_to_ix = {ch: i for i, ch in enumerate(chars)}
    
    X_list = []
    Y_list = []
    for i in range(min(num_samples, len(text) - seq_len - 1)):
        chunk = text[i:i+seq_len]
        target = text[i+1:i+seq_len+1]
        X_list.append([char_to_ix[ch] for ch in chunk])
        Y_list.append([char_to_ix[ch] for ch in target])
        
    return np.array(X_list), np.array(Y_list), vocab_size, chars


def to_one_hot_sequence(X, vocab_size, seq_len):
    num_samples = X.shape[0]
    out = np.zeros((num_samples, seq_len, vocab_size), dtype=np.float32)
    for i in range(num_samples):
        for j in range(seq_len):
            out[i, j, X[i, j]] = 1.0
    return out


def generate_text_pytorch(model, start_text, chars, char_to_ix, seq_len, num_generate=50):
    text = start_text
    vocab_size = len(chars)
    model.eval()
    for _ in range(num_generate):
        input_seq = text[-seq_len:] if len(text) >= seq_len else text.rjust(seq_len)
        X_test = np.zeros((1, seq_len, vocab_size), dtype=np.float32)
        for i, ch in enumerate(input_seq):
            if ch in char_to_ix:
                X_test[0, i, char_to_ix[ch]] = 1.0
            else:
                X_test[0, i, 0] = 1.0
        X_test_t = torch.tensor(X_test, dtype=torch.float32)
        with torch.no_grad():
            pred = model(X_test_t).numpy()[0]
        last_idx = seq_len - 1
        pred_class = np.argmax(pred[last_idx])
        text += chars[pred_class]
    return text


def generate_text(model, start_text, chars, char_to_ix, seq_len, num_generate=50):
    text = start_text
    vocab_size = len(chars)
    for _ in range(num_generate):
        input_seq = text[-seq_len:] if len(text) >= seq_len else text.rjust(seq_len)
        X_test = np.zeros((1, seq_len, vocab_size), dtype=np.float32)
        for i, ch in enumerate(input_seq):
            if ch in char_to_ix:
                X_test[0, i, char_to_ix[ch]] = 1.0
            else:
                X_test[0, i, 0] = 1.0
        X_rk = to_rocket(X_test)[0]
        pred = model.predict([X_rk])[0]
        last_idx = seq_len - 1
        max_p = -1
        pred_class = 0
        for v in range(vocab_size):
            p = pred.get_val(last_idx, v)
            if p > max_p:
                max_p = p
                pred_class = v
        text += chars[pred_class]
    return text


def test_transformer_decoder():
    print("\n" + "="*40)
    print(" 2. Training Transformer Decoder (Shakespeare)")
    print("="*40)
    
    seq_len = 32
    num_samples = 8000
    emb_dim = 32
    ff_dim = 64
    epochs = 20
    batch_size = 64
    
    print("Loading and Preprocessing Tiny Shakespeare...")
    X_idx, Y_idx, vocab_size, chars = prepare_tinyshakespeare(seq_len=seq_len, num_samples=num_samples)
    char_to_ix = {ch: i for i, ch in enumerate(chars)}
    
    X_np = to_one_hot_sequence(X_idx, vocab_size, seq_len)
    Y_np = to_one_hot_sequence(Y_idx, vocab_size, seq_len)
    
    split_idx = int(len(X_np) * 0.8)
    X_train_np = X_np[:split_idx]
    Y_train_np = Y_np[:split_idx]
    X_test_np = X_np[split_idx:]
    Y_test_np = Y_np[split_idx:]
    
    X_train_rk = to_rocket(X_train_np)
    Y_train_rk = to_rocket(Y_train_np)
    X_test_rk = to_rocket(X_test_np)
    Y_test_rk = to_rocket(Y_test_np)
    
    print("\nTraining PyTorch Transformer Decoder...")
    pytorch_model = PyTorchTransformerDecoder(vocab_size, emb_dim, num_heads=4, ff_dim=ff_dim, seq_len=seq_len)
    optimizer = optim.Adam(pytorch_model.parameters(), lr=0.005)
    criterion = nn.CrossEntropyLoss()
    
    X_train_t = torch.tensor(X_train_np, dtype=torch.float32)
    Y_train_t = torch.tensor(Y_train_np, dtype=torch.float32)
    dataset = torch.utils.data.TensorDataset(X_train_t, Y_train_t)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    pt_start = time.time()
    pytorch_model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_X, batch_y in loader:
            optimizer.zero_grad()
            out = pytorch_model(batch_X)
            loss = criterion(out.view(-1, vocab_size), batch_y.view(-1, vocab_size))
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs} - loss: {epoch_loss/len(loader):.4f}")
    pt_end = time.time()
    
    pytorch_model.eval()
    pt_preds = pytorch_model(torch.tensor(X_test_np)).detach().numpy()
    pt_acc = np.mean(np.argmax(pt_preds, axis=-1) == np.argmax(Y_test_np, axis=-1))
    
    print("\nTraining Rocket Transformer Decoder...")
    rocket_model = rocket.Model()
    inp = rocket.InputLayer()
    emb_proj = rocket.DenseLayer(vocab_size, emb_dim)
    dec1 = rocket.TransformerMHDecoderLayer(emb_dim, seq_len, ff_dim, 4)
    dec2 = rocket.TransformerMHDecoderLayer(emb_dim, seq_len, ff_dim, 4)
    dense_out = rocket.DenseLayer(emb_dim, vocab_size)
    act_out = rocket.ActivationLayer(rocket.Softmax())
    
    rocket_model.add(inp, [])
    rocket_model.add(emb_proj, [inp])
    rocket_model.add(dec1, [emb_proj])
    rocket_model.add(dec2, [dec1])
    rocket_model.add(dense_out, [dec2])
    rocket_model.add(act_out, [dense_out])
    
    rocket_model.setInputOutputLayers([inp], [act_out])
    rocket_model.compile(rocket.CCE(), rocket.Adam(lr=0.005))
    
    r_start = time.time()
    rocket_model.train(X_train_rk, Y_train_rk, X_test_rk, Y_test_rk, epochs, batch_size)
    r_end = time.time()
    
    correct = 0
    total_chars = len(X_test_rk) * seq_len
    for i in range(len(X_test_rk)):
        pred_tensor = rocket_model.predict([X_test_rk[i]])[0]
        y_true = Y_test_np[i]
        for j in range(seq_len):
            max_p = -1
            pred_class = 0
            true_class = np.argmax(y_true[j])
            for v in range(vocab_size):
                p = pred_tensor.get_val(j, v)
                if p > max_p:
                    max_p = p
                    pred_class = v
            if pred_class == true_class:
                correct += 1
    rocket_acc = correct / total_chars
    
    print("\n" + "="*45)
    print(" DECODER COMPARISON")
    print("="*45)
    print(f"{'Metric':<20} | {'PyTorch':<15} | {'Rocket':<15}")
    print("-" * 55)
    print(f"{'Test Accuracy':<20} | {pt_acc*100:13.2f}% | {rocket_acc*100:13.2f}%")
    print(f"{'Training Time':<20} | {pt_end - pt_start:13.2f}s | {r_end - r_start:13.2f}s")
    print("="*45)
    
    prompt = "First Citizen:"
    print(f"\nPrompt: '{prompt}'")
    generated_pytorch = generate_text_pytorch(pytorch_model, prompt, chars, char_to_ix, seq_len, 50)
    print(f"Generated text (PyTorch):\n{generated_pytorch}")
    generated = generate_text(rocket_model, prompt, chars, char_to_ix, seq_len, 50)
    print(f"Generated text (Rocket):\n{generated}")


# =====================================================================
# 3. Transformer Sequence + RNN Test (on SMS Dataset)
# =====================================================================

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


def text_to_tensor_seq(X_text, embeddings, seq_len, emb_dim):
    X = np.zeros((len(X_text), seq_len, emb_dim), dtype=np.float32)
    unk_emb = embeddings.get("<UNK>", np.zeros(emb_dim))
    for i, seq in enumerate(X_text):
        for j, w in enumerate(seq):
            if j < seq_len:
                X[i, j, :] = embeddings.get(w, unk_emb)
    return X


def test_transformer_seq_rnn():
    print("\n" + "="*40)
    print(" 3. Training Transformer Sequence + RNN (SMS)")
    print("="*40)
    X_text, Y_np, embeddings, vocab = load_spam_data("tests/data/sms.tsv", 4000)
    
    seq_len = 16
    emb_dim = 8
    ff_dim = 16
    epochs = 100
    
    X_np = text_to_tensor_seq(X_text, embeddings, seq_len, emb_dim)
    split_idx = int(len(X_np) * 0.8)
    X_train_np = X_np[:split_idx]
    Y_train_np = Y_np[:split_idx]
    X_test_np = X_np[split_idx:]
    Y_test_np = Y_np[split_idx:]
    
    X_train_rk = to_rocket(X_train_np)
    Y_train_rk = to_rocket(Y_train_np)
    X_test_rk = to_rocket(X_test_np)
    Y_test_rk = to_rocket(Y_test_np)
    
    print("\nTraining PyTorch Transformer Seq RNN...")
    pytorch_model = PyTorchTransformerSeq(emb_dim, ff_dim)
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
    pt_preds = pytorch_model(torch.tensor(X_test_np)).detach().numpy()
    pytorch_acc = np.mean((pt_preds > 0.5) == Y_test_np)
    
    print("\nTraining Rocket Transformer Seq RNN...")
    rocket_model = rocket.Model()
    inp = rocket.InputLayer()
    enc1 = rocket.TransformerEncoderLayer(emb_dim, seq_len, ff_dim)
    enc2 = rocket.TransformerEncoderLayer(emb_dim, seq_len, ff_dim)
    enc3 = rocket.TransformerEncoderLayer(emb_dim, seq_len, ff_dim)
    pool = rocket.RNNLayer(emb_dim, emb_dim, seq_len, False)
    dense_out = rocket.DenseLayer(emb_dim, 1)
    act_out = rocket.ActivationLayer(rocket.Sigmoid())
    
    rocket_model.add(inp, [])
    rocket_model.add(enc1, [inp])
    rocket_model.add(enc2, [enc1])
    rocket_model.add(enc3, [enc2])
    rocket_model.add(pool, [enc3])
    rocket_model.add(dense_out, [pool])
    rocket_model.add(act_out, [dense_out])
    
    rocket_model.setInputOutputLayers([inp], [act_out])
    rocket_model.compile(rocket.BCE(), rocket.Adam(lr=0.005))
    
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
    
    print("\n" + "="*45)
    print(" SEQ + RNN COMPARISON")
    print("="*45)
    print(f"{'Metric':<20} | {'PyTorch':<15} | {'Rocket':<15}")
    print("-" * 55)
    print(f"{'Test Accuracy':<20} | {pytorch_acc*100:13.2f}% | {rocket_acc*100:13.2f}%")
    print(f"{'Training Time':<20} | {pt_end - pt_start:13.2f}s | {r_end - r_start:13.2f}s")
    print("="*45)


# =====================================================================
# 4. Transformer Spam Encoder (on SMS Dataset)
# =====================================================================

def test_transformer_spam_encoder():
    print("\n" + "="*40)
    print(" 4. Training Transformer Spam Encoder Classifier (SMS)")
    print("="*40)
    X_text, Y_np, embeddings, vocab = load_spam_data("tests/data/sms.tsv", 4000)
    
    seq_len = 16
    emb_dim = 8
    
    X_np = text_to_tensor_seq(X_text, embeddings, seq_len, emb_dim)
    split_idx = int(len(X_np) * 0.8)
    X_train = to_rocket(X_np[:split_idx])
    Y_train = to_rocket(Y_np[:split_idx])
    X_test = to_rocket(X_np[split_idx:])
    Y_test = to_rocket(Y_np[split_idx:])
    
    print("Building Rocket-Lib Transformer Encoder Spam Classifier...")
    model = rocket.Model()
    inp = rocket.InputLayer()
    enc1 = rocket.TransformerEncoderLayer(emb_dim, seq_len, 16)
    enc2 = rocket.TransformerEncoderLayer(emb_dim, seq_len, 16)
    enc3 = rocket.TransformerEncoderLayer(emb_dim, seq_len, 16)
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
    model.compile(rocket.BCE(), rocket.Adam(lr=0.001))
    
    print("Training Model for 100 Epochs...")
    model.train(X_train, Y_train, X_test, Y_test, 100, 32)
    
    print("\n--- Example Inference ---")
    spam_sample = ["urgent", "free", "cash", "prize", "claim", "now", "offer"]
    ham_sample = ["hello", "mom", "dinner", "tomorrow", "thanks", "later", "call", "ok"]
    
    spam_np = text_to_tensor_seq([spam_sample], embeddings, seq_len, emb_dim)
    ham_np = text_to_tensor_seq([ham_sample], embeddings, seq_len, emb_dim)
    
    pred_spam = model.predict(to_rocket(spam_np))[0].get_val(0, 0)
    pred_ham = model.predict(to_rocket(ham_np))[0].get_val(0, 0)
    
    print(f"Spam Message: {' '.join(spam_sample)}")
    print(f"Prediction: {pred_spam:.4f} (Spam? {pred_spam > 0.5})")
    print(f"Ham Message: {' '.join(ham_sample)}")
    print(f"Prediction: {pred_ham:.4f} (Spam? {pred_ham > 0.5})")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Rocket-Lib Transformer Encoder/Decoder Parity Tests")
    parser.add_argument(
        "--test", 
        type=str, 
        choices=["encoder", "decoder", "seq", "spam_encoder", "all"], 
        default="all",
        help="Specify which test to execute (default: all)"
    )
    args = parser.parse_args()

    if args.test == "all":
        test_transformer_encoder()
        test_transformer_decoder()
        test_transformer_seq_rnn()
        test_transformer_spam_encoder()
    elif args.test == "encoder":
        test_transformer_encoder()
    elif args.test == "decoder":
        test_transformer_decoder()
    elif args.test == "seq":
        test_transformer_seq_rnn()
    elif args.test == "spam_encoder":
        test_transformer_spam_encoder()
