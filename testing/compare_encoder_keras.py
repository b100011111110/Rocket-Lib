import os
import sys
import time
import numpy as np
import pandas as pd
import tensorflow as tf

# Suppress TF logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

try:
    import rocket
except ImportError:
    print("Error: Could not import 'rocket'. Make sure PYTHONPATH is set to the build directory.")
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

def convert_to_rocket_tensors(X, Y):
    tensors = []
    for i in range(len(X)):
        # Reshape X to 2D (seq_len, vocab_size)
        x_tensor = rocket.Tensor(X[i])
        # Reshape Y to 2D (1, num_classes)
        y_tensor = rocket.Tensor(Y[i].reshape(1, -1))
        tensors.append((x_tensor, y_tensor))
    return tensors

def build_keras_model(vocab_size, seq_len, num_classes=6, emb_dim=32, ff_dim=64, lr=0.001, num_heads=4):
    inputs = tf.keras.Input(shape=(seq_len, vocab_size))
    
    # Project one-hot
    x = tf.keras.layers.Dense(emb_dim)(inputs)
    
    # 2 Layers of Transformer Encoder (No Causal Mask)
    for _ in range(2):
        # Attention
        att_output = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=emb_dim // num_heads)(x, x)
        att_output = tf.keras.layers.Dropout(0.1)(att_output)
        add1 = tf.keras.layers.Add()([x, att_output])
        norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-5)(add1)
        
        # FFN
        ff1 = tf.keras.layers.Dense(ff_dim, activation="relu")(norm1)
        ff2 = tf.keras.layers.Dense(emb_dim)(ff1)
        ff2 = tf.keras.layers.Dropout(0.1)(ff2)
        add2 = tf.keras.layers.Add()([norm1, ff2])
        x = tf.keras.layers.LayerNormalization(epsilon=1e-5)(add2)
        
    # Global Average Pooling for classification
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    
    # Output projection to classes
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def build_rocket_model(vocab_size, seq_len, num_classes=6, emb_dim=32, ff_dim=64, lr=0.001, num_heads=4):
    model = rocket.Model()
    
    inp = rocket.InputLayer()
    
    # Project one-hot
    emb_proj = rocket.DenseLayer(vocab_size, emb_dim)
    
    # 2 Layers of Transformer Encoder
    enc1 = rocket.TransformerMHEncoderLayer(emb_dim, seq_len, ff_dim, num_heads, 0.1)
    enc2 = rocket.TransformerMHEncoderLayer(emb_dim, seq_len, ff_dim, num_heads, 0.1)
    
    # Average pooling layer (since we don't have GlobalAveragePooling1D natively, we can use Flatten and then dense, or a simple sequence-to-vector layer)
    # Actually we can just flatten the sequence output
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
    model.compile(rocket.CCE(), rocket.Adam(lr=lr, eps=1e-7))  # match Keras epsilon
    return model

def to_rocket(np_array):
    tensors = []
    for i in range(len(np_array)):
        t = rocket.Tensor(np_array[i])
        tensors.append(t)
    return tensors

def evaluate_rocket(model, X_test_tensors, Y_test_np):
    """Evaluate rocket model; X_test_tensors is list of Tensors, Y_test_np is numpy (N, num_classes)."""
    correct = 0
    total = len(X_test_tensors)
    for i in range(total):
        pred = model.predict([X_test_tensors[i]])[0]
        # pred is a Tensor of shape (1, num_classes) — flatten to 1D
        pred_arr = np.array(pred).flatten()
        pred_idx = np.argmax(pred_arr)
        true_idx = np.argmax(Y_test_np[i])
        if pred_idx == true_idx:
            correct += 1
    return correct / total

def main():
    seq_len = 32       # was 64 — halves O(n²) attention cost
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
    print("       Training Keras Transformer Encoder")
    print("==============================================")
    keras_model = build_keras_model(vocab_size, seq_len, num_classes, emb_dim, ff_dim, lr=0.005, num_heads=num_heads)
    
    k_start = time.time()
    keras_model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_test, Y_test), verbose=2)
    k_end = time.time()
    
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
    _, keras_acc = keras_model.evaluate(X_test, Y_test, verbose=0)
    rocket_acc = evaluate_rocket(rocket_model, rocket_X_test, Y_test)  # use numpy Y_test
    
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
