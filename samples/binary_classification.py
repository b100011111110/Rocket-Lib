import os
import sys
import sys
import time
import numpy as np

# Ensure we can find the built rocket module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "build")))
import rocket

def main():
    # 1. Create Synthetic Data
    print("Generating training data...")
    X = np.random.randn(1000, 20).astype(np.float32)
    # Simple rule: if sum of first 5 features > 0, class 1, else 0
    y = (np.sum(X[:, :5], axis=1) > 0).astype(np.float32).reshape(-1, 1)

    # 2. Build the Rocket Model
    model = rocket.Model()

    # Define layers
    input_layer = rocket.InputLayer()
    dense1 = rocket.DenseLayer(20, 64)
    relu1 = rocket.ActivationLayer(rocket.ReLU())
    dense_out = rocket.DenseLayer(64, 1)

    # Connect the graph
    model.add(input_layer, [])
    model.add(dense1, [input_layer])
    model.add(relu1, [dense1])
    model.add(dense_out, [relu1])

    # Finalize model boundary
    model.setInputOutputLayers([input_layer], [dense_out])

    # 3. Compile with Adam optimizer
    optimizer = rocket.Adam(lr=0.01)
    model.compile(rocket.BCEWithLogits(), optimizer)

    """
    KERAS EQUIVALENT:
    -----------------
    import tensorflow as tf
    k_model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, input_shape=(20,)),
        tf.keras.layers.ReLU(),
        tf.keras.layers.Dense(1)
    ])
    k_model.compile(optimizer=tf.keras.optimizers.Adam(0.01), 
                    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True))
    """

    # 4. Convert numpy to Rocket Tensors
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
    print("\n[1/2] Training Rocket-Lib (50 epochs)...")
    os.environ["ROCKET_SEED"] = "42"
    start_rocket = time.time()
    model.train(x_train, y_train, x_train, y_train, epochs=50, batch_size=32)
    time_rocket = time.time() - start_rocket

    # 6. Benchmark Keras (if available)
    time_keras = None
    try:
        import tensorflow as tf
        print("\n[2/2] Training Keras Equivalent (50 epochs)...")
        k_model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, input_shape=(20,)),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Dense(1)
        ])
        k_model.compile(optimizer=tf.keras.optimizers.Adam(0.01), 
                        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True))
        
        start_keras = time.time()
        k_model.fit(X, y, epochs=50, batch_size=32, verbose=0)
        time_keras = time.time() - start_keras
    except ImportError:
        print("\nTensorFlow not found. Skipping Keras benchmark.")

    # 7. Results Comparison
    print("\n" + "="*35)
    print(" PERFORMANCE & ACCURACY COMPARISON")
    print("="*35)
    print(f"Rocket-Lib Time: {time_rocket:.4f}s")
    
    # Evaluate Rocket Accuracy
    r_preds = []
    for x in x_train:
        out = model.predict([x])[0]
        r_preds.append(out.get_val(0, 0))
    r_probs = 1.0 / (1.0 + np.exp(-np.array(r_preds)))
    r_acc = np.mean((r_probs > 0.5) == y.flatten())
    print(f"Rocket Accuracy: {r_acc*100:.2f}%")

    if time_keras:
        print(f"Keras Time:      {time_keras:.4f}s")
        k_probs = 1.0 / (1.0 + np.exp(-k_model.predict(X, verbose=0).flatten()))
        k_acc = np.mean((k_probs > 0.5) == y.flatten())
        print(f"Keras Accuracy:  {k_acc*100:.2f}%")
        print("-" * 35)
        print(f"Rocket is {time_keras/time_rocket:.2f}x faster!")
    print("="*35)

if __name__ == "__main__":
    main()
