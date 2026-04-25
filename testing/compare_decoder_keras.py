import os
import sys
import time
import urllib.request
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

sys.path.append('build')
import rocket

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
    
    # Pad vocab to reach 200 if necessary
    target_vocab = 200
    if len(chars) < target_vocab:
        for i in range(target_vocab - len(chars)):
            chars.append(chr(1000 + i)) # dummy unused characters
            
    vocab_size = len(chars)
    char_to_ix = {ch: i for i, ch in enumerate(chars)}
    
    # We will just take the first N chunks of seq_len
    X_list = []
    Y_list = []
    
    for i in range(min(num_samples, len(text) - seq_len - 1)):
        chunk = text[i:i+seq_len]
        target = text[i+1:i+seq_len+1]
        X_list.append([char_to_ix[ch] for ch in chunk])
        Y_list.append([char_to_ix[ch] for ch in target])
        
    return np.array(X_list), np.array(Y_list), vocab_size, chars

def to_one_hot_sequence(X, vocab_size, seq_len):
    # Shape: (samples, seq_len, vocab_size)
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

def build_keras_model(vocab_size, seq_len, emb_dim=32, ff_dim=64, lr=0.005, num_heads=4):
    inputs = tf.keras.Input(shape=(seq_len, vocab_size))
    
    # Simple embedding by projecting one-hot via dense layer or just using the one-hot as input to dense
    x = tf.keras.layers.Dense(emb_dim)(inputs)
    
    # 2 Layers of Causal Transformer Decoder
    for _ in range(2):
        # Causal Attention
        att_output = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=emb_dim // num_heads)(x, x, use_causal_mask=True)
        add1 = tf.keras.layers.Add()([x, att_output])
        norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-5)(add1)
        
        # FFN
        ff1 = tf.keras.layers.Dense(ff_dim, activation="relu")(norm1)
        ff2 = tf.keras.layers.Dense(emb_dim)(ff1)
        add2 = tf.keras.layers.Add()([norm1, ff2])
        x = tf.keras.layers.LayerNormalization(epsilon=1e-5)(add2)
        
    outputs = tf.keras.layers.Dense(vocab_size, activation="softmax")(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def build_rocket_model(vocab_size, seq_len, emb_dim=32, ff_dim=64, lr=0.005, num_heads=4):
    model = rocket.Model()
    
    inp = rocket.InputLayer()
    
    # Project one-hot to embedding
    emb_proj = rocket.DenseLayer(vocab_size, emb_dim)
    
    # 2 Layers of Transformer Decoder
    dec1 = rocket.TransformerMHDecoderLayer(emb_dim, seq_len, ff_dim, num_heads)
    dec2 = rocket.TransformerMHDecoderLayer(emb_dim, seq_len, ff_dim, num_heads)
    
    # Output projection to vocab
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

def generate_text_keras(model, start_text, chars, char_to_ix, seq_len, num_generate=50):
    text = start_text
    vocab_size = len(chars)
    for _ in range(num_generate):
        # Prepare input
        input_seq = text[-seq_len:] if len(text) >= seq_len else text.rjust(seq_len)
        X_test = np.zeros((1, seq_len, vocab_size), dtype=np.float32)
        for i, ch in enumerate(input_seq):
            if ch in char_to_ix:
                X_test[0, i, char_to_ix[ch]] = 1.0
            else:
                X_test[0, i, 0] = 1.0 # fallback
                
        pred = model.predict(X_test, verbose=0)[0] # (seq_len, vocab_size)
        
        # Get the prediction for the last character
        last_idx = seq_len - 1
        pred_class = np.argmax(pred[last_idx])
        
        text += chars[pred_class]
    return text

def generate_text(model, start_text, chars, char_to_ix, seq_len, num_generate=50):
    text = start_text
    vocab_size = len(chars)
    for _ in range(num_generate):
        # Prepare input
        input_seq = text[-seq_len:] if len(text) >= seq_len else text.rjust(seq_len)
        X_test = np.zeros((1, seq_len, vocab_size), dtype=np.float32)
        for i, ch in enumerate(input_seq):
            if ch in char_to_ix:
                X_test[0, i, char_to_ix[ch]] = 1.0
            else:
                X_test[0, i, 0] = 1.0 # fallback
                
        X_rk = to_rocket(X_test)[0]
        pred = model.predict([X_rk])[0] # (seq_len, vocab_size)
        
        # Get the prediction for the last character
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
    
    # Split
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
    print("       Training Keras Transformer Decoder")
    print("==============================================")
    keras_model = build_keras_model(vocab_size, seq_len, emb_dim, ff_dim, lr=0.005)
    
    k_start = time.time()
    keras_model.fit(X_train_np, Y_train_np, epochs=epochs, batch_size=batch_size, validation_data=(X_test_np, Y_test_np), verbose=2)
    k_end = time.time()
    
    keras_loss, keras_acc = keras_model.evaluate(X_test_np, Y_test_np, verbose=0)
    
    print("\n==============================================")
    print("      Training Rocket Transformer Decoder")
    print("==============================================")
    rocket_model = build_rocket_model(vocab_size, seq_len, emb_dim, ff_dim, lr=0.005)
    
    r_start = time.time()
    rocket_model.train(X_train_rk, Y_train_rk, X_test_rk, Y_test_rk, epochs, batch_size)
    r_end = time.time()
    
    # Calculate rocket accuracy manually for comparison
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
    print(f"{'Metric':<20} | {'Keras':<15} | {'Rocket':<15}")
    print("-" * 55)
    print(f"{'Test Accuracy':<20} | {keras_acc*100:13.2f}% | {rocket_acc*100:13.2f}%")
    print(f"{'Training Time':<20} | {k_end - k_start:13.2f}s | {r_end - r_start:13.2f}s")
    print("==============================================\n")
    
    prompt = "First Citizen:"
    
    print("\n==============================================")
    print("          TEXT GENERATION DEMO (Keras)")
    print("==============================================")
    print(f"Prompt: '{prompt}'")
    generated_keras = generate_text_keras(keras_model, prompt, chars, char_to_ix, seq_len, 100)
    print(f"Generated text:\n{generated_keras}")
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
