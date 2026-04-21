# 🚀 Rocket-Lib: High-Performance Neural Engine

**Rocket-Lib** is a state-of-the-art C++14 Artificial Neural Network (ANN) library optimized for modern CPU architectures. It features a lightweight Python front-end, a cache-contiguity focused computational engine, and a flexible Directed Acyclic Graph (DAG) architecture that rivals industry-standard frameworks in speed and efficiency for small-to-medium scale workloads.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![C++ Standard](https://img.shields.io/badge/C%2B%2B-14-blue.svg)](https://en.cppreference.com/w/cpp/14)
[![Python Version](https://img.shields.io/badge/Python-3.x-green.svg)](https://www.python.org/)

---

## 🌟 Key Features

*   **⚡ Cache-Optimized Engine:** Utilizes **i-k-j loop reordering** for matrix operations, transforming random column-stride access into sequential row-stride scans. This maximizes L1/L2 cache hits and triggers automatic AVX2 vectorization.
*   **🧶 Intelligent Multithreading:** Powered by a persistent **Singleton ThreadPool**. Execution is governed by a **FLOP-count heuristic** (e.g., 100k FLOP threshold), ensuring parallelism only happens when compute outweighs synchronization overhead.
*   **🗺️ Modular DAG Architecture:** Define complex topologies (like ResNet skip connections) using a flexible graph engine. The library automatically resolves execution order via Kahn's Topological Sort.
*   **🧪 Mathematical Parity:** Engineered for bit-perfect convergence with Keras/TensorFlow. Includes numerically stable `BCEWithLogits` (Log-Sum-Exp trick) and a production-grade **Adam** optimizer.
*   **🐍 Python-C++ Interop:** High-performance bindings via `pybind11` provide a zero-copy interface, allowing NumPy-to-Tensor conversion and seamless model training from Python.

---

## 📊 Performance & Convergence

Rocket-Lib is designed to outperform general-purpose frameworks on CPU by focusing on memory hierarchy and minimized framework overhead.

### 500-Epoch Stress Test (10k Samples)
| Metric | Rocket-Lib (Multi-Thread) | Keras (Reference) | Result |
| :--- | :--- | :--- | :--- |
| **Training Time** | **49.34s** | 122.73s | **2.5x Faster** |
| **Accuracy** | **97.25%** | 96.55% | ✅ Parity+ |
| **BCE Loss** | **0.1125** | 0.1753 | ✅ Stable |

### Forward Pass Throughput
| Architecture | Rocket-Lib Speedup | Why? |
| :--- | :--- | :--- |
| **Sequential Dense** | **11.22x** | Cache-aware GEMM + No spawn overhead |
| **ResNet DAG** | **4.27x** | Zero-latency graph traversal |
| **Comprehensive** | **3.64x** | Optimized backward-pass partitioning |

---

## 🛠️ Installation & Build

### Prerequisites
- **CMake** (>= 3.14)
- **C++14 Compiler** (GCC 7+ / Clang 5+)
- **Python 3.x**

### Build Instructions
To achieve the benchmarked speeds, Rocket-Lib must be compiled with hardware-specific optimizations:

```bash
mkdir build && cd build
# CMake automatically sets -O3 -ffast-math -march=native
cmake ../core
make -j$(nproc)
```

> [!IMPORTANT]
> The engine utilizes `-ffast-math` for a significant throughput boost. While this maintains parity for standard neural network training, it trades off strict IEEE 754 compliance for speed.

---

## 🏗️ Technical Highlights

### 1. The i-k-j Kernel
Most naive matrix multiplies suffer from "cache thrashing" due to column-wise strides. Rocket-Lib reorders these loops to ensure that every memory access is a sequential linear walk. This allows the CPU prefetcher to load data before it's even requested.

### 2. Strategic Parallelism
We don't just "add threads." The engine analyzes the workload of each layer:
- **Forward Pass:** Partitioned by output rows.
- **Grad-Weights:** Partitioned by input columns (k-dimension) to avoid locking and false sharing.
- **Thresholding:** Small layers remain single-threaded to avoid the "Thread-Spawn Tax."

### 3. Reproducibility
Set `ROCKET_SEED` in your environment to ensure deterministic results for dropout masks.
```bash
export ROCKET_SEED=42
python samples/binary_classification.py
```


## 💻 Quick Start

Building a model in Rocket-Lib is intuitive. Here is a simple binary classifier:

```python
import rocket
import numpy as np

# 1. Initialize Model
model = rocket.Model()

# 2. Define Layers
input_layer = rocket.InputLayer()
dense1 = rocket.DenseLayer(20, 64)
relu1 = rocket.ActivationLayer(rocket.ReLU())
dense_out = rocket.DenseLayer(64, 1)

# 3. Assemble DAG
model.add(input_layer, [])
model.add(dense1, [input_layer])
model.add(relu1, [dense1])
model.add(dense_out, [relu1])

# 4. Finalize & Compile
model.setInputOutputLayers([input_layer], [dense_out])
model.compile(rocket.BCEWithLogits(), rocket.Adam(lr=0.01))

# 5. Train (Tensors are initialized from data)
model.train(x_train, y_train, x_val, y_val, epochs=50, batch_size=32)
```

---

## 🏗️ Architecture Deep Dive

### The DAG Engine
Unlike sequential frameworks, Rocket-Lib treats every model as a graph of `Layer` nodes. When you `compile()` the model, it performs a **Topological Sort** using Kahn's Algorithm to determine the exact execution order. This allows for:
- **Skip Connections:** Connect any layer to any subsequent layer.
- **Multi-Input/Multi-Output:** Support for branched architectures.

### Cache-Optimized Memory
The C++ core utilizes a custom `Tensor` class designed for cache contiguity. Matrix multiplications are implemented with raw pointer arithmetic to avoid the overhead of standard library abstractions, ensuring that data stays in L1/L2 cache as long as possible.

### ThreadPool Execution
Rocket-Lib identifies compute-heavy operations and dispatches them to a persistent `ThreadPool`. This avoids the latency of spawning threads on-the-fly and allows for fine-grained control over CPU core utilization.

---

## 📚 Technical Reference

### Core Components
- **Layers:** `InputLayer`, `DenseLayer`, `DropoutLayer`, `RegularizationLayer`, `ActivationLayer`.
- **Activations:** `ReLU`, `LeakyReLU`, `Sigmoid`, `Tanh`, `Linear`, `Softplus`, `Softmax`.
- **Losses:** `MSE`, `MAE`, `Huber`, `BCE`, `BCEWithLogits` (Logit-stable), `CCE`.
- **Optimizers:** `Adam` (with bias correction), `RMSprop`, `SGD`.

---

## 🤝 Contributing & Testing

We maintain a strict parity suite to ensure mathematical correctness. To run tests:
```bash
sh tests.sh
```
All changes to the core engine must maintain within 2% variance of the Keras reference implementation.

---

## 📄 License

Rocket-Lib is released under the **MIT License**. See `LICENSE` for details.
