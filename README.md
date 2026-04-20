# Rocket-Lib

**Rocket-Lib** is a high-performance Artificial Neural Network (ANN) library featuring a Python front-end API accelerated by an optimized C++ computational engine. It is designed for researchers and developers who need mathematical transparency and high execution speed in a lightweight package.

## 🚀 Key Features

*   **Custom C++ Engine:** Ground-up implementation of Tensors, Layers, and Optimizers.
*   **Python Bindings:** Seamless integration via `pybind11` for a modern data-science workflow.
*   **Validated Parity:** Rigorously tested against Keras/TensorFlow to ensure mathematical correctness (BCE Loss, Adam dynamics, etc.).
*   **Deterministic Training:** Support for reproducible experiments via global seeding (`ROCKET_SEED`).
*   **Advanced Components:**
    *   **Layers:** Dense (Fully Connected), Dropout, Activity Regularization (L1/L2).
    *   **Optimizers:** Adam (with bias correction), SGD.
    *   **Activations:** ReLU, Sigmoid, Tanh, Linear.

## 🛠️ Build & Installation

### Prerequisites
*   CMake (>= 3.10)
*   C++17 compatible compiler (GCC/Clang)
*   Python 3.x + development headers

### Build Instructions
```bash
mkdir build && cd build
cmake ../core
make -j$(nproc)
```
This generates the `rocket` Python module in the `build/` directory.

## 💻 Usage Example

```python
import sys
sys.path.append("build")
import rocket

# Initialize model
model = rocket.Model()

# Define architecture
input_layer = rocket.InputLayer()
dense1 = rocket.DenseLayer(16, 64)
relu1 = rocket.ActivationLayer(rocket.ReLU())
drop1 = rocket.DropoutLayer(0.15)
dense_out = rocket.DenseLayer(64, 1)

# Build graph
model.add(input_layer, [])
model.add(dense1, [input_layer])
model.add(relu1, [dense1])
model.add(drop1, [relu1])
model.add(dense_out, [drop1])

# Set I/O
model.setInputOutputLayers([input_layer], [dense_out])

# Compile and train
opt = rocket.Adam(learning_rate=0.005)
loss_fn = rocket.BCEWithLogits()
model.compile(loss_fn, opt)

model.train(x_train, y_train, x_val, y_val, epochs=100, batch_size=128)
```

## 🧪 Validation & Testing

Rocket-Lib includes a comprehensive parity suite to ensure its mathematical engine matches established frameworks.

### Running Comparison Tests
To run the automated comparison against Keras:
```bash
# Requires: tensorflow, scikit-learn
sh tests.sh
```

### Recent Parity Results (10,000 Samples)
| Metric | Rocket-Lib | Keras (Reference) |
| :--- | :--- | :--- |
| **Accuracy** | 97.40% | 97.05% |
| **BCE Loss** | 0.1379 | 0.1358 |
| **F1 Score** | 0.9745 | 0.9707 |
| **AUC** | 0.9867 | 0.9737 |

## 🤝 Contributing
Contributions are welcome! Please ensure all core logic changes are validated using the parity scripts in `testing/`.

## 📄 License
This project is licensed under the MIT License.
