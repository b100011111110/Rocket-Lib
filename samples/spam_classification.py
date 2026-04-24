import os
import sys
import numpy as np

sys.path.append('build')
import rocket

def generate_spam_data(num_samples=1000):
    spam_keywords = ["free", "win", "cash", "prize", "urgent", "click", "offer", "guaranteed", "buy", "cheap"]
    ham_keywords = ["hello", "meeting", "tomorrow", "mom", "dinner", "ok", "thanks", "schedule", "call", "later"]
    
    vocab = spam_keywords + ham_keywords
    
    np.random.seed(42)
    # Simple word embeddings
    embeddings = {}
    for i, w in enumerate(vocab):
        vec = np.random.randn(8) * 0.1
        vec[i % 8] += 0.5 # Add some deterministic feature to help learning
        embeddings[w] = vec
        
    X_text = []
    Y_labels = []
    
    for _ in range(num_samples):
        is_spam = np.random.rand() > 0.5
        seq = []
        if is_spam:
            # Generate spam message
            length = np.random.randint(4, 8)
            seq = [np.random.choice(spam_keywords) for _ in range(length)]
            Y_labels.append([1.0])
        else:
            # Generate ham message
            length = np.random.randint(4, 8)
            seq = [np.random.choice(ham_keywords) for _ in range(length)]
            Y_labels.append([0.0])
            
        # Pad sequence to max length (let's say 8)
        while len(seq) < 8:
            seq.append(np.random.choice(vocab)) # add noise padding
            
        X_text.append(seq[:8])
        
    return X_text, np.array(Y_labels, dtype=np.float32), embeddings, vocab

def text_to_tensor(X_text, embeddings, seq_len, emb_dim):
    X = np.zeros((len(X_text), seq_len, emb_dim), dtype=np.float32)
    for i, seq in enumerate(X_text):
        for j, w in enumerate(seq):
            if j < seq_len:
                X[i, j, :] = embeddings[w]
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
    print("Generating Spam Classification Dataset with Embeddings...")
    X_text, Y_np, embeddings, vocab = generate_spam_data(1000)
    
    seq_len = 8
    emb_dim = 8
    hidden_dim = 16
    
    X_np = text_to_tensor(X_text, embeddings, seq_len, emb_dim)
    
    X_train = to_rocket(X_np[:800])
    Y_train = to_rocket(Y_np[:800])
    
    X_test = to_rocket(X_np[800:])
    Y_test = to_rocket(Y_np[800:])
    
    print("Building Rocket-Lib Stacked LSTM Spam Classifier...")
    model = rocket.Model()
    
    inp = rocket.InputLayer()
    
    # Encoder
    lstm1 = rocket.LSTMLayer(emb_dim, hidden_dim, seq_len, True)
    drop = rocket.DropoutLayer(0.3)
    lstm2 = rocket.LSTMLayer(hidden_dim, hidden_dim, seq_len, False) # Output: (batch, hidden_dim)
    
    # Binary Classification Head
    dense_out = rocket.DenseLayer(hidden_dim, 1)
    act_out = rocket.ActivationLayer(rocket.Sigmoid())
    
    model.add(inp, [])
    model.add(lstm1, [inp])
    model.add(drop, [lstm1])
    model.add(lstm2, [drop])
    model.add(dense_out, [lstm2])
    model.add(act_out, [dense_out])
    
    model.setInputOutputLayers([inp], [act_out])
    
    # Using Binary Cross Entropy
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
    
    print(f"\nHam Message: {' '.join(ham_sample)}")
    print(f"Prediction: {pred_ham:.4f} (Spam? {pred_ham > 0.5})")

if __name__ == "__main__":
    main()
