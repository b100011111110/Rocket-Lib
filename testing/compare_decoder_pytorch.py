import os
import sys
import time
import urllib.request
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

def prepare_tinyshakespeare(seq_len=32, num_samples=2000):
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    filepath = "samples/data/input.txt"
    os.makedirs("samples/data", exist_ok=True)
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

def build_pytorch_model(vocab_size, seq_len, emb_dim=32, ff_dim=64, lr=0.005, num_heads=4):
    model = PyTorchTransformerDecoder(vocab_size, emb_dim, num_heads, ff_dim, seq_len)
    return model

def build_rocket_model(vocab_size, seq_len, emb_dim=32, ff_dim=64, lr=0.005, num_heads=4):
    model = rocket.Model()
    
    inp = rocket.InputLayer()
    
    emb_proj = rocket.DenseLayer(vocab_size, emb_dim)
    
    dec1 = rocket.TransformerMHDecoderLayer(emb_dim, seq_len, ff_dim, num_heads)
    dec2 = rocket.TransformerMHDecoderLayer(emb_dim, seq_len, ff_dim, num_heads)
    
    dense_out = rocket.DenseLayer(emb_dim, vocab_size)
    act_out = rocket.ActivationLayer(rocket.Softmax())
    
    model.add(inp, [])
    model.add(emb_proj, [inp])
    model.add(dec1, [emb_proj])
    model.add(dec2, [dec1])
    model.add(dense_out, [dec2])
    model.add(act_out, [dense_out])
    
    model.setInputOutputLayers([inp], [act_out])
    model.compile(rocket.CCE(), rocket.Adam(lr=lr))
    return model

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

def main():
    seq_len = 32
    num_samples = 8000
    emb_dim = 32
    ff_dim = 64
    epochs = 20
    batch_size = 64
    
    print("Loading and Preprocessing Tiny Shakespeare...")
    X_idx, Y_idx, vocab_size, chars = prepare_tinyshakespeare(seq_len=seq_len, num_samples=num_samples)
    
    char_to_ix = {ch: i for i, ch in enumerate(chars)}
    
    print(f"Vocab Size: {vocab_size}")
    
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
    
    print("\n==============================================")
    print("      Training PyTorch Transformer Decoder")
    print("==============================================")
    pytorch_model = build_pytorch_model(vocab_size, seq_len, emb_dim, ff_dim, lr=0.005)
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
    
    # PyTorch accuracy
    pytorch_model.eval()
    X_test_t = torch.tensor(X_test_np, dtype=torch.float32)
    with torch.no_grad():
        pt_preds = pytorch_model(X_test_t).numpy()
    pt_acc = np.mean(np.argmax(pt_preds, axis=-1) == np.argmax(Y_test_np, axis=-1))
    
    print("\n==============================================")
    print("      Training Rocket Transformer Decoder")
    print("==============================================")
    rocket_model = build_rocket_model(vocab_size, seq_len, emb_dim, ff_dim, lr=0.005)
    
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
    
    print("\n==============================================")
    print("                COMPARISON")
    print("==============================================")
    print(f"{'Metric':<20} | {'PyTorch':<15} | {'Rocket':<15}")
    print("-" * 55)
    print(f"{'Test Accuracy':<20} | {pt_acc*100:13.2f}% | {rocket_acc*100:13.2f}%")
    print(f"{'Training Time':<20} | {pt_end - pt_start:13.2f}s | {r_end - r_start:13.2f}s")
    print("==============================================\n")
    
    prompt = "First Citizen:"
    
    print("\n==============================================")
    print("          TEXT GENERATION DEMO (PyTorch)")
    print("==============================================")
    print(f"Prompt: '{prompt}'")
    generated_pytorch = generate_text_pytorch(pytorch_model, prompt, chars, char_to_ix, seq_len, 100)
    print(f"Generated text:\n{generated_pytorch}")
    print("==============================================\n")
    
    print("\n==============================================")
    print("          TEXT GENERATION DEMO (Rocket)")
    print("==============================================")
    print(f"Prompt: '{prompt}'")
    generated = generate_text(rocket_model, prompt, chars, char_to_ix, seq_len, 100)
    print(f"Generated text:\n{generated}")
    print("==============================================\n")

if __name__ == "__main__":
    main()
