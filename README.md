# Rocket-Lib

**Rocket-Lib** is a high-performance Artificial Neural Network (ANN) library featuring a Python front-end API accelerated by an optimized C++ computational engine.

## 🚀 Key Features

*   **Custom C++ Engine:** Ground-up implementation of Tensors, Layers, and Optimizers.
*   **Graph-Based Architecture:** Supports complex topologies (DAGs) beyond simple sequential stacks.
*   **Mathematical Parity:** Rigorously validated against Keras/TensorFlow (BCE Loss, Adam dynamics).
*   **Deterministic Training:** Reproducible experiments via `ROCKET_SEED`.

## 🛠️ Build & Installation

```bash
mkdir build && cd build
cmake ../core
make -j$(nproc)
```

## 💻 Technical Reference & Usage

### Model Definition Walkthrough
Rocket-Lib uses a **Dependency Map** approach. Every layer is a node in a Directed Acyclic Graph (DAG).

```python
import rocket
model = rocket.Model()

# 1. Define Layers
input_layer = rocket.InputLayer()
dense1 = rocket.DenseLayer(16, 64)
relu1 = rocket.ActivationLayer(rocket.ReLU())

# 2. Connect the Graph (Topological Dependency)
model.add(input_layer, [])
model.add(dense1, [input_layer])
model.add(relu1, [dense1])

# 3. Compile & Train
model.setInputOutputLayers([input_layer], [relu1])
model.compile(rocket.BCEWithLogits(), rocket.Adam(0.005))
model.train(x_train, y_train, x_val, y_val, epochs=120, batch_size=125)
```

### Core Classes
| Class | Role | Design Rationale |
| :--- | :--- | :--- |
| `Tensor` | Data Storage | Uses raw `double*` pointers for SIMD optimization. |
| `Model` | Orchestrator | Uses Kahn's Algorithm for topological sorting. **Note: Current engine is single-threaded.** |
| `Adam` | Optimizer | Implements full bias correction and soft gradient clipping. |
| `BCEWithLogits` | Loss | Combines Sigmoid + BCE into one step to prevent numerical "log(0)" explosions. |

## 📐 Mathematical Specifications

### Adam Update Rule
1. $\hat{m}_t = m_t / (1 - \beta_1^t)$
2. $\hat{v}_t = v_t / (1 - \beta_2^t)$
3. $\theta_{t+1} = \theta_t - \frac{\eta \cdot \hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$ (where $\epsilon=1e-7$)

### Activity Regularization
Implemented as a discrete layer to allow per-layer penalty tuning.
*   **L2 Gradient:** $2 \cdot \lambda \cdot X / \text{batch\_size}$

## 🧪 Validation Results (10,000 Samples)
| **Accuracy** | 97.40% | 97.05% |
| **BCE Loss** | 0.1379 | 0.1358 |
| **F1 Score** | 0.9745 | 0.9707 |
| **Time Taken** | ~45s | ~8s |

> [!NOTE]
> Rocket-Lib is currently a single-threaded CPU implementation. Performance comparisons with Keras reflect the overhead of a highly-optimized, multi-threaded framework versus our lightweight, educational-first C++ engine.

## 🤝 Contributing & Parity
All core engine changes must maintain **Mathematical Parity**. Before submitting a PR, run the validation suite:
```bash
sh tests.sh  # Variance must be < 2%
```

## 📄 License
MIT License.
