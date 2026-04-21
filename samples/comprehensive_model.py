import os
import sys
import time
import numpy as np

# Ensure we can find the built rocket module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "build")))
import rocket

def main():
    print("Running Comprehensive Rocket-Lib Showcase...")
    
    # 1. Initialize Model
    model = rocket.Model()

    # 2. Layer Definitions
    layers = {
        "input": rocket.InputLayer(),
        "dense1": rocket.DenseLayer(20, 128),
        "relu1": rocket.ActivationLayer(rocket.ReLU()),
        "reg1": rocket.RegularizationLayer(0.001), # L2 Regularization
        "drop1": rocket.DropoutLayer(0.2),                 # 20% Dropout
        "dense2": rocket.DenseLayer(128, 64),
        "relu2": rocket.ActivationLayer(rocket.ReLU()),
        "drop2": rocket.DropoutLayer(0.1),                 # 10% Dropout
        "dense_out": rocket.DenseLayer(64, 1)
    }

    # 3. Graph Assembly (The DAG)
    model.add(layers["input"], [])
    model.add(layers["dense1"], [layers["input"]])
    model.add(layers["relu1"], [layers["dense1"]])
    model.add(layers["reg1"], [layers["relu1"]])
    model.add(layers["drop1"], [layers["reg1"]])
    model.add(layers["dense2"], [layers["drop1"]])
    model.add(layers["relu2"], [layers["dense2"]])
    model.add(layers["drop2"], [layers["relu2"]])
    model.add(layers["dense_out"], [layers["drop2"]])

    # 4. Model Configuration
    model.setInputOutputLayers([layers["input"]], [layers["dense_out"]])
    model.compile(rocket.BCEWithLogits(), rocket.Adam(lr=0.001))

    # 5. Data Preparation
    X = np.random.randn(2000, 20).astype(np.float32)
    y = (np.sum(X[:, :10], axis=1) > 0).astype(np.float32).reshape(-1, 1)

    def to_rocket(arr):
        tensors = []
        for row in arr:
            t = rocket.Tensor(1, row.shape[0])
            for i, v in enumerate(row): t.set_val(0, i, float(v))
            tensors.append(t)
        return tensors

    x_train = to_rocket(X)
    y_train = to_rocket(y)

    # 6. Train Rocket
    print("\n[1/2] Training Rocket Comprehensive Model (50 epochs)...")
    os.environ["ROCKET_SEED"] = "42"
    start_rocket = time.time()
    model.train(x_train, y_train, x_train, y_train, epochs=50, batch_size=32)
    time_rocket = time.time() - start_rocket

    # 7. Benchmark Keras
    time_keras = None
    try:
        import tensorflow as tf
        print("\n[2/2] Training Keras Equivalent (50 epochs)...")
        k_model = tf.keras.Sequential([
            tf.keras.layers.Dense(128, input_shape=(20,), 
                                  activity_regularizer=tf.keras.regularizers.L2(0.001)),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(64),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(1)
        ])
        k_model.compile(optimizer=tf.keras.optimizers.Adam(0.001), 
                        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True))
        
        start_keras = time.time()
        k_model.fit(X, y, epochs=50, batch_size=32, verbose=0)
        time_keras = time.time() - start_keras
    except ImportError:
        print("\nTensorFlow not found. Skipping benchmark.")

    # 8. Results Comparison
    print("\n" + "="*35)
    print(" COMPREHENSIVE PERFORMANCE & ACCURACY")
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
