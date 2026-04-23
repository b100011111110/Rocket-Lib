import os
import sys
import time
import numpy as np

# Ensure we can find the built rocket module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "build")))
import rocket

def main():
    print("Building ResNet-style DAG Architecture...")
    
    # 1. Initialize Model
    model = rocket.Model()

    # 2. Define Layers
    input_layer = rocket.InputLayer()
    
    # Entry block
    dense1 = rocket.DenseLayer(20, 64)
    relu1 = rocket.ActivationLayer(rocket.ReLU())

    # --- RESIDUAL BLOCK ---
    # Path A: Weighted computation
    dense_res = rocket.DenseLayer(64, 64)
    
    # Path B: Skip connection (Identity)
    # Note: In Rocket, we can use an ActivationLayer(Linear) as an Identity bypass
    bypass = rocket.ActivationLayer(rocket.Linear())
    
    # Combining Path A and Path B
    # The ReLU layer here will receive the SUM of dense_res and bypass
    relu_sum = rocket.ActivationLayer(rocket.ReLU())
    # ----------------------

    # Output block
    dense_out = rocket.DenseLayer(64, 1)

    # 3. Connect the DAG
    model.add(input_layer, [])
    model.add(dense1, [input_layer])
    model.add(relu1, [dense1])

    # Diverging point
    model.add(dense_res, [relu1]) # Path A
    model.add(bypass, [relu1])    # Path B (Skip connection)

    # Combining point (Rocket sums all inputs in the prev_layers list automatically)
    model.add(relu_sum, [dense_res, bypass])

    model.add(dense_out, [relu_sum])

    # Finalize
    model.setInputOutputLayers([input_layer], [dense_out])
    model.compile(rocket.BCEWithLogits(), rocket.Adam(lr=0.005))
    model.summary()



    # 4. Generate Data
    X = np.random.randn(1000, 20).astype(np.float32)
    y = (np.sum(X**2, axis=1) > 20).astype(np.float32).reshape(-1, 1)

    def to_rocket(arr):
        tensors = []
        for row in arr:
            t = rocket.Tensor(1, row.shape[0])
            for i, val in enumerate(row):
                t.set_val(0, i, float(val))
            tensors.append(t)
        return tensors

    x_train = to_rocket(X)
    y_train = to_rocket(y)

    # 5. Train Rocket
    print("\n[1/2] Training Rocket Residual Model (50 epochs)...")
    os.environ["ROCKET_SEED"] = "42"
    start_rocket = time.time()
    model.train(x_train, y_train, x_train, y_train, epochs=50, batch_size=32)
    time_rocket = time.time() - start_rocket

    # 6. Benchmark Keras
    time_keras = None
    try:
        import tensorflow as tf
        print("\n[2/2] Training Keras Equivalent (50 epochs)...")
        inputs = tf.keras.Input(shape=(20,))
        x = tf.keras.layers.Dense(64)(inputs)
        x = tf.keras.layers.ReLU()(x)
        path_a = tf.keras.layers.Dense(64)(x)
        path_b = x
        x = tf.keras.layers.Add()([path_a, path_b])
        x = tf.keras.layers.ReLU()(x)
        outputs = tf.keras.layers.Dense(1)(x)
        k_model = tf.keras.Model(inputs, outputs)
        k_model.compile(optimizer=tf.keras.optimizers.Adam(0.005), loss=tf.keras.losses.BinaryCrossentropy(from_logits=True))
        
        start_keras = time.time()
        k_model.fit(X, y, epochs=50, batch_size=32, verbose=0)
        time_keras = time.time() - start_keras
    except ImportError:
        print("\nTensorFlow not found. Skipping benchmark.")

    print("\n" + "="*35)
    print(" RESNET PERFORMANCE & ACCURACY")
    print("="*35)
    print(f"Rocket Time: {time_rocket:.4f}s")
    
    # Evaluate Rocket Accuracy
    r_preds = []
    for x in x_train:
        out = model.predict([x])[0]
        r_preds.append(out.get_val(0, 0))
    r_probs = 1.0 / (1.0 + np.exp(-np.array(r_preds)))
    r_acc = np.mean((r_probs > 0.5) == y.flatten())
    print(f"Rocket Accuracy: {r_acc*100:.2f}%")

    if time_keras:
        print(f"Keras Time:  {time_keras:.4f}s")
        k_probs = 1.0 / (1.0 + np.exp(-k_model.predict(X, verbose=0).flatten()))
        k_acc = np.mean((k_probs > 0.5) == y.flatten())
        print(f"Keras Accuracy:  {k_acc*100:.2f}%")
        print("-" * 35)
        print(f"Rocket is {time_keras/time_rocket:.2f}x faster!")
    print("="*35)

if __name__ == "__main__":
    main()
