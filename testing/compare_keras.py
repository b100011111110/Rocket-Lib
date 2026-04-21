import os
import random
import sys
import time

import numpy as np

try:
    build_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "build"))
except NameError:
    build_dir = os.path.abspath(os.path.join(os.getcwd(), "build"))
if build_dir not in sys.path:
    sys.path.append(build_dir)

import rocket

try:
    import tensorflow as tf
except ImportError:
    print("Please install tensorflow: pip install tensorflow")
    sys.exit(1)

try:
    from sklearn.datasets import make_classification
    from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
except ImportError:
    print("Please install scikit-learn: pip install scikit-learn")
    sys.exit(1)

np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)


def to_rocket_tensors(arr):
    tensors = []
    for i in range(arr.shape[0]):
        t = rocket.Tensor(1, arr.shape[1])
        for j in range(arr.shape[1]):
            t.set_val(0, j, float(arr[i, j]))
        tensors.append(t)
    return tensors


def set_dropout_mode(dropout_layers, is_training):
    for layer in dropout_layers:
        layer.set_training(is_training)


def sync_dense_weights(keras_dense_layers, rocket_dense_layers):
    for k_layer, r_layer in zip(keras_dense_layers, rocket_dense_layers):
        weights, biases = k_layer.get_weights()
        for i in range(weights.shape[0]):
            for j in range(weights.shape[1]):
                r_layer.weights.set_val(i, j, float(weights[i, j]))
        for j in range(biases.shape[0]):
            r_layer.biases.set_val(0, j, float(biases[j]))


def make_keras_optimizer(keras_lr):
    legacy_optimizers = getattr(tf.keras.optimizers, "legacy", None)
    if legacy_optimizers is not None:
        try:
            return legacy_optimizers.Adam(learning_rate=keras_lr, epsilon=1e-7)
        except ImportError:
            pass
    return tf.keras.optimizers.Adam(learning_rate=keras_lr, epsilon=1e-7)


def evaluate_binary_predictions(y_true, probs, threshold=0.5):
    y_true = y_true.flatten().astype(np.int32)
    probs = probs.flatten()
    preds = (probs >= threshold).astype(np.int32)

    return {
        "accuracy": float(np.mean(preds == y_true)),
        "bce": float(
            -np.mean(
                y_true * np.log(probs + 1e-8)
                + (1 - y_true) * np.log(1 - probs + 1e-8)
            )
        ),
        "precision": float(precision_score(y_true, preds, zero_division=0)),
        "recall": float(recall_score(y_true, preds, zero_division=0)),
        "f1": float(f1_score(y_true, preds, zero_division=0)),
        "auc": float(roc_auc_score(y_true, probs)),
        "confusion": confusion_matrix(y_true, preds),
    }


def main():
    input_dim = int(os.getenv("ROCKET_INPUT_DIM", "16"))
    epochs = int(os.getenv("ROCKET_EPOCHS", "500"))
    rocket_lr = float(os.getenv("ROCKET_LR", "0.01"))
    keras_lr = float(os.getenv("KERAS_LR", str(rocket_lr)))
    dropout_rate = float(os.getenv("ROCKET_DROPOUT", "0.15"))
    reg_lambda = float(os.getenv("ROCKET_REG_LAMBDA", "0.001"))

    print("Generating synthetic dataset (binary classification with >12 features)...")
    X, y = make_classification(
        n_samples=10000,
        n_features=input_dim,
        n_informative=10,
        n_redundant=4,
        n_repeated=0,
        n_classes=2,
        n_clusters_per_class=2,
        class_sep=1.2,
        flip_y=0.03,
        random_state=42,
    )
    y = y.reshape(-1, 1).astype(np.float32)
    X = X.astype(np.float32)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_test = scaler.transform(X_test).astype(np.float32)

    batch_size = len(X_train) // 64
    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"Dividing train set into 64 parts (batch_size={batch_size})")

    print("\n--- Preparing Rocket tensors ---")
    x_train_rt = to_rocket_tensors(X_train)
    y_train_rt = to_rocket_tensors(y_train)
    x_test_rt = to_rocket_tensors(X_test)
    y_test_rt = to_rocket_tensors(y_test)

    print("\n--- Building Rocket model ---")
    r_model = rocket.Model()

    r_input = rocket.InputLayer()
    r_dense1 = rocket.DenseLayer(input_dim, 64)
    r_relu1 = rocket.ActivationLayer(rocket.ReLU())
    r_reg1 = rocket.RegularizationLayer(reg_lambda, 2)
    r_drop1 = rocket.DropoutLayer(dropout_rate)

    r_dense2 = rocket.DenseLayer(64, 32)
    r_relu2 = rocket.ActivationLayer(rocket.ReLU())
    r_reg2 = rocket.RegularizationLayer(reg_lambda, 2)
    r_drop2 = rocket.DropoutLayer(dropout_rate)

    r_dense3 = rocket.DenseLayer(32, 16)
    r_relu3 = rocket.ActivationLayer(rocket.ReLU())
    r_reg3 = rocket.RegularizationLayer(reg_lambda, 2)
    r_drop3 = rocket.DropoutLayer(dropout_rate)

    r_dense4 = rocket.DenseLayer(16, 1)
    r_out = rocket.ActivationLayer(rocket.Linear())

    rocket_dropout_layers = [r_drop1, r_drop2, r_drop3]
    rocket_dense_layers = [r_dense1, r_dense2, r_dense3, r_dense4]

    r_model.add(r_input, [])
    r_model.add(r_dense1, [r_input])
    r_model.add(r_reg1, [r_dense1])
    r_model.add(r_relu1, [r_reg1])
    r_model.add(r_drop1, [r_relu1])

    r_model.add(r_dense2, [r_drop1])
    r_model.add(r_reg2, [r_dense2])
    r_model.add(r_relu2, [r_reg2])
    r_model.add(r_drop2, [r_relu2])

    r_model.add(r_dense3, [r_drop2])
    r_model.add(r_reg3, [r_dense3])
    r_model.add(r_relu3, [r_reg3])
    r_model.add(r_drop3, [r_relu3])

    r_model.add(r_dense4, [r_drop3])
    r_model.add(r_out, [r_dense4])
    r_model.setInputOutputLayers([r_input], [r_out])

    r_loss = rocket.BCEWithLogits()
    r_opt = rocket.Adam(rocket_lr, 0.9, 0.999, 1e-7)
    r_model.compile(r_loss, r_opt)

    print("\n--- Building Keras model and synchronizing weights ---")
    k_model = tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(shape=(input_dim,)),
            tf.keras.layers.Dense(64, activation=None, name="dense_1"),
            tf.keras.layers.ReLU(name="relu_1"),
            tf.keras.layers.ActivityRegularization(l2=reg_lambda, name="reg_1"),
            tf.keras.layers.Dropout(dropout_rate, name="drop_1"),
            tf.keras.layers.Dense(32, activation=None, name="dense_2"),
            tf.keras.layers.ReLU(name="relu_2"),
            tf.keras.layers.ActivityRegularization(l2=reg_lambda, name="reg_2"),
            tf.keras.layers.Dropout(dropout_rate, name="drop_2"),
            tf.keras.layers.Dense(16, activation=None, name="dense_3"),
            tf.keras.layers.ReLU(name="relu_3"),
            tf.keras.layers.ActivityRegularization(l2=reg_lambda, name="reg_3"),
            tf.keras.layers.Dropout(dropout_rate, name="drop_3"),
            tf.keras.layers.Dense(1, activation=None, name="dense_out"),
        ]
    )
    k_model.compile(
        optimizer=make_keras_optimizer(keras_lr),
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc"),
        ],
    )

    keras_dense_layers = [
        k_model.get_layer("dense_1"),
        k_model.get_layer("dense_2"),
        k_model.get_layer("dense_3"),
        k_model.get_layer("dense_out"),
    ]
    sync_dense_weights(keras_dense_layers, rocket_dense_layers)

    print(
        f"\n--- Training Rocket model for {epochs} epochs "
        f"(batch_size={batch_size}, lr={rocket_lr}) ---"
    )
    set_dropout_mode(rocket_dropout_layers, True)
    start_time = time.time()
    r_model.train(x_train_rt, y_train_rt, x_test_rt, y_test_rt, epochs, batch_size)
    rocket_time = time.time() - start_time
    print(f"Rocket training took {rocket_time:.2f} seconds.")

    print("\n--- Evaluating Rocket model ---")
    set_dropout_mode(rocket_dropout_layers, False)
    r_preds = []
    for x in x_test_rt:
        out_tensor = r_model.predict([x])[0]
        r_preds.append(out_tensor.get_val(0, 0))

    r_preds = np.array(r_preds).flatten()
    r_probs = 1.0 / (1.0 + np.exp(-r_preds))
    r_metrics = evaluate_binary_predictions(y_test, r_probs)
    print(
        "Rocket Test Metrics: "
        f"BCE={r_metrics['bce']:.4f}, "
        f"Accuracy={r_metrics['accuracy']*100:.2f}%, "
        f"Precision={r_metrics['precision']:.4f}, "
        f"Recall={r_metrics['recall']:.4f}, "
        f"F1={r_metrics['f1']:.4f}, "
        f"AUC={r_metrics['auc']:.4f}"
    )
    print(f"Rocket Confusion Matrix:\n{r_metrics['confusion']}")

    print(
        f"\n--- Training Keras model for {epochs} epochs "
        f"(batch_size={batch_size}, lr={keras_lr}) ---"
    )
    start_time = time.time()
    k_model.fit(X_train, y_train, epochs=epochs, verbose=0, batch_size=batch_size)
    keras_time = time.time() - start_time
    print(f"Keras training took {keras_time:.2f} seconds.")

    k_eval = k_model.evaluate(X_test, y_test, verbose=0)
    k_bce = float(k_eval[0])
    k_acc = float(k_eval[1])
    k_precision = float(k_eval[2])
    k_recall = float(k_eval[3])
    k_auc = float(k_eval[4])
    k_probs = 1.0 / (1.0 + np.exp(-k_model.predict(X_test, verbose=0).flatten()))
    k_metrics = evaluate_binary_predictions(y_test, k_probs)
    print(
        "Keras Test Metrics: "
        f"BCE={k_bce:.4f}, "
        f"Accuracy={k_acc*100:.2f}%, "
        f"Precision={k_precision:.4f}, "
        f"Recall={k_recall:.4f}, "
        f"F1={k_metrics['f1']:.4f}, "
        f"AUC={k_auc:.4f}"
    )
    print(f"Keras Confusion Matrix:\n{k_metrics['confusion']}\n")

    print("\n--- Comparison ---")
    print(f"Rocket BCE: {r_metrics['bce']:.4f} | Keras BCE: {k_bce:.4f}")
    print(f"Rocket Acc: {r_metrics['accuracy']*100:.2f}% | Keras Acc: {k_acc*100:.2f}%")
    print(f"Rocket Precision: {r_metrics['precision']:.4f} | Keras Precision: {k_precision:.4f}")
    print(f"Rocket Recall: {r_metrics['recall']:.4f} | Keras Recall: {k_recall:.4f}")
    print(f"Rocket F1: {r_metrics['f1']:.4f} | Keras F1: {k_metrics['f1']:.4f}")
    print(f"Rocket AUC: {r_metrics['auc']:.4f} | Keras AUC: {k_auc:.4f}")
    print(f"Rocket Time: {rocket_time:.2f}s | Keras Time: {keras_time:.2f}s")


if __name__ == "__main__":
    main()
