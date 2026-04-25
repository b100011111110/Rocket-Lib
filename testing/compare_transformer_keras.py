import os
import sys
import time
import string
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

sys.path.append('build')
import rocket

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

def build_keras_model(emb_dim, seq_len, ff_dim, lr=0.001):
    inputs = tf.keras.Input(shape=(seq_len, emb_dim))
    
    x = inputs
    for _ in range(3):
        # Attention
        att_output = tf.keras.layers.MultiHeadAttention(num_heads=1, key_dim=emb_dim)(x, x)
        add1 = tf.keras.layers.Add()([x, att_output])
        norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-5)(add1)
        
        # FFN
        ff1 = tf.keras.layers.Dense(ff_dim, activation="relu")(norm1)
        ff2 = tf.keras.layers.Dense(emb_dim)(ff1)
        add2 = tf.keras.layers.Add()([norm1, ff2])
        x = tf.keras.layers.LayerNormalization(epsilon=1e-5)(add2)
        
    # RNN pooling
    x = tf.keras.layers.SimpleRNN(emb_dim, activation='tanh')(x)
    
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
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
    print("       Training Keras Transformer Encoder")
    print("==============================================")
    keras_model = build_keras_model(emb_dim, seq_len, ff_dim, lr=0.005)
    
    k_start = time.time()
    keras_history = keras_model.fit(X_train_np, Y_train_np, epochs=epochs, batch_size=32, validation_data=(X_test_np, Y_test_np), verbose=2)
    k_end = time.time()
    
    keras_loss, keras_acc = keras_model.evaluate(X_test_np, Y_test_np, verbose=0)
    
    print("\n==============================================")
    print("      Training Rocket Transformer Encoder")
    print("==============================================")
    rocket_model = build_rocket_model(emb_dim, seq_len, ff_dim, lr=0.005)
    
    r_start = time.time()
    rocket_model.train(X_train_rk, Y_train_rk, X_test_rk, Y_test_rk, epochs, 32)
    r_end = time.time()
    
    # Calculate rocket accuracy manually for comparison
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
    print(f"{'Metric':<20} | {'Keras':<15} | {'Rocket':<15}")
    print("-" * 55)
    print(f"{'Test Accuracy':<20} | {keras_acc*100:13.2f}% | {rocket_acc*100:13.2f}%")
    print(f"{'Training Time':<20} | {k_end - k_start:13.2f}s | {r_end - r_start:13.2f}s")
    print("==============================================\n")
    
if __name__ == "__main__":
    main()
