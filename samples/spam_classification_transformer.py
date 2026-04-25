import os
import sys
import numpy as np

sys.path.append('build')
import rocket

import string

def load_spam_data(filepath, max_samples=2000):
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
            
            # Simple tokenization
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
    # Simple random word embeddings for the real vocabulary
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

def main():
    print("Loading Real SMS Spam Collection Dataset...")
    X_text, Y_np, embeddings, vocab = load_spam_data("samples/data/sms.tsv", 4000)
    
    seq_len = 16
    emb_dim = 8
    
    X_np = text_to_tensor(X_text, embeddings, seq_len, emb_dim)
    
    split_idx = int(len(X_np) * 0.8)
    X_train = to_rocket(X_np[:split_idx])
    Y_train = to_rocket(Y_np[:split_idx])
    
    X_test = to_rocket(X_np[split_idx:])
    Y_test = to_rocket(Y_np[split_idx:])
    
    print("Building Rocket-Lib Transformer Encoder Spam Classifier...")
    model = rocket.Model()
    
    inp = rocket.InputLayer()
    
    # 3 Layers of Transformer Encoder
    enc1 = rocket.TransformerEncoderLayer(emb_dim, seq_len, 16)
    enc2 = rocket.TransformerEncoderLayer(emb_dim, seq_len, 16)
    enc3 = rocket.TransformerEncoderLayer(emb_dim, seq_len, 16)
    
    pool = rocket.RNNLayer(emb_dim, emb_dim, seq_len, False)
    
    # Binary Classification Head
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
    
    print("Training Model for 10 Epochs...")
    model.train(X_train, Y_train, X_test, Y_test, 100, 32)
    
    print("\n--- Example Inference ---")
    spam_sample = ["urgent", "free", "cash", "prize", "claim", "now", "offer"]
    ham_sample = ["hello", "mom", "dinner", "tomorrow", "thanks", "later", "call", "ok"]
    
    spam_np = text_to_tensor([spam_sample], embeddings, seq_len, emb_dim)
    ham_np = text_to_tensor([ham_sample], embeddings, seq_len, emb_dim)
    
    pred_spam = model.predict(to_rocket(spam_np))[0].get_val(0, 0)
    pred_ham = model.predict(to_rocket(ham_np))[0].get_val(0, 0)
    
    print(f"Spam Message: {' '.join(spam_sample)}")
    print(f"Prediction: {pred_spam:.4f} (Spam? {pred_spam > 0.5})")
    
    print(f"\nHam Message: {' '.join(ham_sample)}")
    print(f"Prediction: {pred_ham:.4f} (Spam? {pred_ham > 0.5})")

if __name__ == "__main__":
    main()
