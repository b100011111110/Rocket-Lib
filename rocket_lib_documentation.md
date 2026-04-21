# Rocket-Lib — Usage Guide & API Reference

*Complete documentation for building, training, and evaluating models with Rocket-Lib.*

---

## Table of Contents

1. [Installation & Build](#1-installation--build)
2. [Core Concepts](#2-core-concepts)
3. [Tensors](#3-tensors)
4. [Layers](#4-layers)
5. [Activations](#5-activations)
6. [Loss Functions](#6-loss-functions)
7. [Optimizers](#7-optimizers)
8. [Building a Model](#8-building-a-model)
9. [Training](#9-training)
10. [Inference](#10-inference)
11. [DAG Models (ResNet-style)](#11-dag-models-resnet-style)
12. [Environment Variables](#12-environment-variables)
13. [Full Examples](#13-full-examples)
14. [Known Limitations](#14-known-limitations)

---

## 1. Installation & Build

Rocket-Lib has no runtime Python dependencies beyond NumPy. The C++ core is compiled once and imported as a native module.

**Prerequisites**

- GCC or Clang with C++14 support
- CMake ≥ 3.14
- Python ≥ 3.8
- pybind11 (fetched automatically by CMake)

**Build**

```bash
git clone https://github.com/shrihari-s/Rocket-Lib
cd Rocket-Lib
mkdir build && cd build
cmake ../core
make -j$(nproc)
cd ..
```

This produces a `rocket*.so` shared library inside `build/`. No installation step is required.

**Import**

```python
import sys
sys.path.append('build')   # path to the compiled .so
import rocket
```

For convenience, the provided scripts add this path automatically. When running from the repo root, `build/` is always the correct path.

**Full benchmark suite**

```bash
sh tests.sh
```

**Keras parity comparison only**

```bash
sh run_comparison.sh
```

---

## 2. Core Concepts

Rocket-Lib models are **directed acyclic graphs** of layers. Every layer has exactly one input tensor and one output tensor. Layers are connected explicitly by declaring their predecessor list when calling `model.add()`. This makes both sequential chains and branching ResNet-style architectures expressible with the same API.

Training data is passed as Python lists of `rocket.Tensor` objects — one tensor per sample. The engine handles batching internally.

The execution flow for a training run:

```
Data (NumPy) → Rocket Tensors → model.train() → Forward → Loss → Backward → Optimizer Update
```

---

## 3. Tensors

`rocket.Tensor` is the fundamental data container. Tensors are always two-dimensional (rows × cols). A single training sample is typically a `Tensor(1, feature_dim)`.

### Constructor

```python
t = rocket.Tensor(rows, cols)
```

Allocates a `rows × cols` tensor. Values are initialized to random Xavier-distributed floats by the base constructor.

### Setting and Getting Values

```python
t.set_val(row, col, value)   # set element at (row, col)
t.get_val(row, col)          # get element at (row, col) → float
```

Both methods perform bounds checking and raise `IndexError` on out-of-range access.

### Properties

```python
t.rows         # int
t.cols         # int
t.owns_memory  # bool — whether this tensor manages its own buffer
```

### Converting NumPy Arrays to Tensors

Rocket-Lib does not accept NumPy arrays directly. Use this utility, which is present in all sample scripts:

```python
def to_rocket(numpy_array):
    tensors = []
    for row in numpy_array:
        t = rocket.Tensor(1, row.shape[0])
        for i, val in enumerate(row):
            t.set_val(0, i, float(val))
        tensors.append(t)
    return tensors
```

This produces a list of `Tensor(1, feature_dim)` objects — one per sample.

**Performance note:** This loop runs in Python and is the bottleneck for large datasets. For 10,000 samples with 16 features, ingestion takes a few seconds. The C++ training loop itself is unaffected; ingestion happens once before `model.train()`.

### Converting Tensors Back to NumPy

```python
def tensor_to_numpy(t):
    import numpy as np
    arr = np.zeros((t.rows, t.cols))
    for i in range(t.rows):
        for j in range(t.cols):
            arr[i, j] = t.get_val(i, j)
    return arr
```

---

## 4. Layers

All layers inherit from `rocket.Layer` and expose `forward()` and `backward()` via the pybind11 binding. Layers are instantiated in Python and connected through the `Model` graph.

### InputLayer

```python
layer = rocket.InputLayer()
```

Passes its input tensor directly to its output. Every model must have exactly one `InputLayer` as the graph entry point.

### DenseLayer

```python
layer = rocket.DenseLayer(input_dim, output_dim)
```

Fully-connected linear transformation: `output = input @ weights + biases`.

Weights are initialized with **Xavier/Glorot uniform** initialization using `std::random_device` (non-deterministic). Biases are initialized to zero.

**Accessible members:**

```python
layer.weights        # rocket.Tensor (input_dim × output_dim)
layer.biases         # rocket.Tensor (1 × output_dim)
layer.grad_weights   # rocket.Tensor — populated after backward()
layer.grad_biases    # rocket.Tensor — populated after backward()
layer.grad_input     # rocket.Tensor — populated after backward()
```

**Parallelization:** The forward GEMM is dispatched to the ThreadPool when `input.rows × weights.cols × weights.rows > 100,000`. Below this threshold, it runs single-threaded to avoid sync overhead.

### DropoutLayer

```python
layer = rocket.DropoutLayer(rate=0.5)
```

Applies inverted dropout during training. The `rate` parameter is clamped to `[0, 1)`. Dropout is applied only when `is_training=True`.

```python
layer.set_training(True)   # enable dropout (during training)
layer.set_training(False)  # disable dropout (during evaluation/inference)
```

The random mask is seeded from the `ROCKET_SEED` environment variable. If unset, defaults to seed `42`.

### RegularizationLayer

```python
layer = rocket.RegularizationLayer(lambda_val, type=2)
```

Applies L1 (`type=1`) or L2 (`type=2`) activity regularization. Acts as a pass-through in the forward pass and adds a penalty gradient in the backward pass.

| type | Regularization |
|------|---------------|
| 1    | L1 — gradient is `±lambda` based on sign of activation |
| 2    | L2 — gradient is `2 * lambda * activation / batch_size` |

### ActivationLayer

```python
layer = rocket.ActivationLayer(activation_fn)
```

Wraps any `rocket.Activation` object. Memory for the activation function is managed by pybind11's `keep_alive` mechanism — do not delete the activation object while the layer is alive.

---

## 5. Activations

Activations are passed to `ActivationLayer`. All implement `forward()` and `backward()`.

```python
rocket.ReLU()       # max(0, x)
rocket.Sigmoid()    # 1 / (1 + exp(-x)), with epsilon clipping in backward
rocket.Tanh()       # tanh(x)
rocket.Linear()     # identity — useful as a pass-through or output activation
```

**Example:**

```python
relu_layer = rocket.ActivationLayer(rocket.ReLU())
```

---

## 6. Loss Functions

Loss functions are passed to `model.compile()`. All implement `forward()` (returns scalar loss) and `backward()` (returns gradient tensor).

### MSE — Mean Squared Error

```python
loss = rocket.MSE()
```

Standard mean squared error. Use for regression tasks.

### BCE — Binary Cross-Entropy

```python
loss = rocket.BCE()
```

Expects sigmoid-activated predictions in `[0, 1]`. Uses epsilon clipping at `1e-7` to prevent log(0). Gradients are zeroed at the clipping boundaries.

### BCEWithLogits — Binary Cross-Entropy from Logits

```python
loss = rocket.BCEWithLogits()
```

**Recommended for binary classification.** Accepts raw logits (pre-sigmoid). Uses the Log-Sum-Exp trick in the forward pass for numerical stability:

```
loss = max(x, 0) - x*t + log(1 + exp(-|x|))
```

The backward pass uses a standard sigmoid gradient. This is the loss used in all Rocket-Lib benchmarks.

### CCE — Categorical Cross-Entropy

```python
loss = rocket.CCE()
```

For multi-class classification with softmax outputs.

---

## 7. Optimizers

Optimizers are passed to `model.compile()`. All implement `update(param, grad)`.

### SGD

```python
opt = rocket.SGD(lr=0.01)
```

Standard stochastic gradient descent: `param -= lr * grad`.

### Adam

```python
opt = rocket.Adam(lr=0.001, b1=0.9, b2=0.999, eps=1e-8)
```

Adam with first and second moment estimates and bias correction. A gradient clamp of `±10.0` is applied before the moment update — this is a deliberate stabilization addition, not in the standard Adam derivation.

```
g = clip(grad, -10, 10)
m = β₁ * m + (1 - β₁) * g
v = β₂ * v + (1 - β₂) * g²
param -= lr * (m / (1 - β₁ᵗ)) / (sqrt(v / (1 - β₂ᵗ)) + eps)
```

**Known issue:** `begin_step()` — which increments the internal `step_count` used in bias correction — is currently called twice per batch. This causes the bias correction to converge at double the intended rate. This does not break training but subtly deviates from standard Adam behavior.

### RMSprop

```python
opt = rocket.RMSprop(lr=0.001, rho=0.9, eps=1e-8)
```

RMSprop with adaptive learning rates based on a moving average of squared gradients.

---

## 8. Building a Model

Models are built by instantiating layers and registering them with the graph via `model.add(layer, prev_layers)`.

### Sequential Model

```python
model = rocket.Model()

input_layer  = rocket.InputLayer()
dense1       = rocket.DenseLayer(16, 64)
relu1        = rocket.ActivationLayer(rocket.ReLU())
dense2       = rocket.DenseLayer(64, 32)
relu2        = rocket.ActivationLayer(rocket.ReLU())
output_layer = rocket.DenseLayer(32, 1)

model.add(input_layer,  [])
model.add(dense1,       [input_layer])
model.add(relu1,        [dense1])
model.add(dense2,       [relu1])
model.add(relu2,        [dense2])
model.add(output_layer, [relu2])

model.setInputOutputLayers([input_layer], [output_layer])
model.compile(rocket.BCEWithLogits(), rocket.Adam(lr=0.001))
```

**Rules:**
- `InputLayer` must be added first with an empty predecessor list `[]`
- `model.add(layer, prev_layers)` must be called before `setInputOutputLayers`
- `setInputOutputLayers` declares the graph boundary — inputs and outputs
- `compile` must be called after `setInputOutputLayers`

### Compiling

```python
model.compile(loss, optimizer)
```

Triggers Kahn's topological sort on the registered layers, producing a fixed execution order. Must be called before `train()` or `predict()`.

---

## 9. Training

```python
model.train(x_train, y_train, x_test, y_test, epochs, batch_size=1)
```

| Parameter    | Type              | Description |
|-------------|-------------------|-------------|
| `x_train`   | `list[Tensor]`    | Training inputs, one tensor per sample |
| `y_train`   | `list[Tensor]`    | Training targets, one tensor per sample |
| `x_test`    | `list[Tensor]`    | Validation inputs (passed but not yet evaluated internally) |
| `y_test`    | `list[Tensor]`    | Validation targets |
| `epochs`    | `int`             | Number of full passes over the training set |
| `batch_size`| `int`             | Samples per gradient update (default: 1) |

Each epoch prints:

```
Epoch 1/50 - Loss: 0.4231 - Time: 0.83s
```

**Shuffling** is enabled by default. Set `ROCKET_SHUFFLE=0` to disable.

**Dropout layers** must be set to training mode before calling `train()`:

```python
for layer in dropout_layers:
    layer.set_training(True)
```

### Batch Size Selection

Rocket-Lib's parallelization threshold is 100,000 FLOPs for the forward pass. For a `DenseLayer(16, 64)`, a batch of 1 has `1 × 64 × 16 = 1,024` FLOPs — well below threshold, so it runs single-threaded. A batch of 100 has `100 × 64 × 16 = 102,400` FLOPs — above threshold, parallel dispatch kicks in.

As a rule: **use batch sizes ≥ 32 for models with hidden dims ≥ 64** to benefit from the ThreadPool.

---

## 10. Inference

```python
predictions = model.predict([x])   # returns list[Tensor]
logit = predictions[0].get_val(0, 0)
prob = 1.0 / (1.0 + np.exp(-logit))  # sigmoid if using BCEWithLogits
```

`predict()` takes a list of input tensors (matching the number of input layers) and returns a list of output tensors.

**Always disable dropout before inference:**

```python
for layer in dropout_layers:
    layer.set_training(False)
```

### Batch Inference

```python
r_probs = []
for x in x_test_tensors:
    out = model.predict([x])[0]
    r_probs.append(out.get_val(0, 0))
r_probs = np.array(r_probs)
```

Inference runs one sample at a time in the current implementation. Batched `predict()` is not yet supported.

---

## 11. DAG Models (ResNet-style)

Any layer can have multiple predecessors. When a layer has more than one predecessor, Rocket-Lib **sums** all predecessor outputs element-wise before passing the result to that layer's forward pass. This is how skip connections and residual blocks are expressed.

### Residual Block Example

```python
model = rocket.Model()

input_layer = rocket.InputLayer()
dense1      = rocket.DenseLayer(20, 64)
relu1       = rocket.ActivationLayer(rocket.ReLU())

# Diverging point — two paths from relu1
path_a      = rocket.DenseLayer(64, 64)         # weighted path
path_b      = rocket.ActivationLayer(rocket.Linear())  # identity (skip connection)

# Merging point — relu_merge receives sum of path_a and path_b
relu_merge  = rocket.ActivationLayer(rocket.ReLU())
output      = rocket.DenseLayer(64, 1)

model.add(input_layer, [])
model.add(dense1,      [input_layer])
model.add(relu1,       [dense1])
model.add(path_a,      [relu1])          # Path A
model.add(path_b,      [relu1])          # Path B (skip)
model.add(relu_merge,  [path_a, path_b]) # Sum of both paths
model.add(output,      [relu_merge])

model.setInputOutputLayers([input_layer], [output])
model.compile(rocket.BCEWithLogits(), rocket.Adam(lr=0.005))
```

**Keras equivalent:**

```python
inputs  = tf.keras.Input(shape=(20,))
x       = tf.keras.layers.Dense(64)(inputs)
x       = tf.keras.layers.ReLU()(x)
path_a  = tf.keras.layers.Dense(64)(x)
path_b  = x                               # identity
x       = tf.keras.layers.Add()([path_a, path_b])
x       = tf.keras.layers.ReLU()(x)
outputs = tf.keras.layers.Dense(1)(x)
model   = tf.keras.Model(inputs, outputs)
```

**Execution detail:** For single-predecessor nodes, Rocket-Lib takes a direct path and skips the summation allocation. For multi-predecessor nodes, it materializes a combined tensor. Without the move constructor (currently deleted), the single-predecessor path still performs a deep copy — this is the highest-priority outstanding fix.

---

## 12. Environment Variables

Rocket-Lib's behavior can be modified at runtime through environment variables. Set these before running any training script.

| Variable            | Default | Effect |
|--------------------|---------|--------|
| `ROCKET_SEED`       | `42`    | Seed for Dropout masks and training shuffle. Set to any unsigned integer for reproducibility. |
| `ROCKET_SHUFFLE`    | `1`     | Set to `0` to disable epoch-level sample shuffling. Useful for debugging gradient parity. |
| `ROCKET_EPOCHS`     | `120`   | Default epoch count for `compare_keras.py`. Override without editing source. |
| `ROCKET_LR`         | `0.01`  | Learning rate override for `compare_keras.py`. |
| `ROCKET_DROPOUT`    | `0.15`  | Dropout rate override for `compare_keras.py`. |
| `ROCKET_REG_LAMBDA` | `0.001` | Regularization lambda override for `compare_keras.py`. |
| `ROCKET_INPUT_DIM`  | `16`    | Input feature dimension override for `compare_keras.py`. |

**Example — reproducible run with no shuffling:**

```bash
export ROCKET_SEED=42
export ROCKET_SHUFFLE=0
python samples/binary_classification.py
```

**Note on DenseLayer initialization:** `DenseLayer` uses `std::random_device` internally for Xavier weight initialization, which is non-deterministic. Setting `ROCKET_SEED` controls dropout and shuffling but does not currently fix DenseLayer initialization. Results are stable in practice but not bit-for-bit reproducible across runs.

---

## 13. Full Examples

### Binary Classification (Sequential)

```python
import os, sys, numpy as np
sys.path.append('build')
import rocket

# Data
X = np.random.randn(1000, 20).astype(np.float32)
y = (np.sum(X[:, :5], axis=1) > 0).astype(np.float32).reshape(-1, 1)

# Convert to Rocket Tensors
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

# Build model
model       = rocket.Model()
inp         = rocket.InputLayer()
dense1      = rocket.DenseLayer(20, 64)
relu1       = rocket.ActivationLayer(rocket.ReLU())
dense_out   = rocket.DenseLayer(64, 1)

model.add(inp,       [])
model.add(dense1,    [inp])
model.add(relu1,     [dense1])
model.add(dense_out, [relu1])

model.setInputOutputLayers([inp], [dense_out])
model.compile(rocket.BCEWithLogits(), rocket.Adam(lr=0.01))

# Train
os.environ['ROCKET_SEED'] = '42'
model.train(x_train, y_train, x_train, y_train, epochs=50, batch_size=32)

# Evaluate
preds = []
for x in x_train:
    out = model.predict([x])[0]
    preds.append(out.get_val(0, 0))

probs = 1.0 / (1.0 + np.exp(-np.array(preds)))
acc = np.mean((probs > 0.5) == y.flatten())
print(f"Accuracy: {acc * 100:.2f}%")
```

---

### Multi-Layer with Dropout and Regularization

```python
model       = rocket.Model()
inp         = rocket.InputLayer()
dense1      = rocket.DenseLayer(20, 128)
relu1       = rocket.ActivationLayer(rocket.ReLU())
reg1        = rocket.RegularizationLayer(0.001, 2)   # L2
drop1       = rocket.DropoutLayer(0.2)
dense2      = rocket.DenseLayer(128, 64)
relu2       = rocket.ActivationLayer(rocket.ReLU())
drop2       = rocket.DropoutLayer(0.1)
dense_out   = rocket.DenseLayer(64, 1)

model.add(inp,       [])
model.add(dense1,    [inp])
model.add(relu1,     [dense1])
model.add(reg1,      [relu1])
model.add(drop1,     [reg1])
model.add(dense2,    [drop1])
model.add(relu2,     [dense2])
model.add(drop2,     [relu2])
model.add(dense_out, [drop2])

model.setInputOutputLayers([inp], [dense_out])
model.compile(rocket.BCEWithLogits(), rocket.Adam(lr=0.001))

# Enable dropout for training
for d in [drop1, drop2]:
    d.set_training(True)

model.train(x_train, y_train, x_train, y_train, epochs=50, batch_size=32)

# Disable dropout for inference
for d in [drop1, drop2]:
    d.set_training(False)
```

---

### Synchronizing Weights with Keras (for parity testing)

```python
def sync_weights(keras_dense_layers, rocket_dense_layers):
    for k_layer, r_layer in zip(keras_dense_layers, rocket_dense_layers):
        weights, biases = k_layer.get_weights()
        for i in range(weights.shape[0]):
            for j in range(weights.shape[1]):
                r_layer.weights.set_val(i, j, float(weights[i, j]))
        for j in range(biases.shape[0]):
            r_layer.biases.set_val(0, j, float(biases[j]))
```

Copying Keras-initialized weights into Rocket layers ensures both engines start from identical parameters — useful for gradient parity verification. See `testing/verify_parity.py` for a full example.

---

## 14. Known Limitations

These are current technical constraints, not design goals. Most have straightforward fixes in progress.

**Move constructor is deleted.** `Tensor` move semantics were removed during a refactor. All tensor assignments perform deep copies. The most visible consequence is `ActivationLayer`, which routes through a `memcpy` to avoid double-free:

```cpp
Tensor result = activation_fn->forward(input);
std::memcpy(output.data, result.data, output.rows * output.cols * sizeof(double));
```

Restoring the move constructor would fix this and eliminate copy overhead from the entire DAG single-predecessor path.

**Non-deterministic DenseLayer initialization.** `DenseLayer` uses `std::random_device` for Xavier weight initialization. `ROCKET_SEED` does not affect this. Results are empirically stable across runs, but bit-for-bit reproducibility requires patching `layer.cpp` to use a seeded `std::mt19937`.

**Double initialization sweep.** `DenseLayer` initializes the weight buffer twice: once in the `Tensor` base constructor and once in the Xavier pass. This is a startup-only cost — one redundant sweep of the weight matrix — that consolidating into a single pass would eliminate.

**`begin_step()` double-count in Adam.** Adam's `step_count` is incremented twice per batch. Bias correction converges at double the intended rate. This subtly deviates from standard Adam behavior but does not destabilize training in practice.

**Backward pass BCEWithLogits overflow.** The Log-Sum-Exp trick is applied in the forward pass only. The backward pass uses a raw `std::exp(-logit)` that can overflow for extreme negative logits. Safe for the inputs seen in the benchmark tasks.

**No batched inference.** `model.predict()` processes one sample at a time. Large-scale evaluation requires a Python loop over the test set.

**Epoch-level map reallocation.** `layer_outputs` and `layer_grads` are re-allocated at the start of each epoch. Moving them outside the epoch loop and calling `.clear()` inside it would eliminate this churn on long training runs.

---

*github.com/shrihari-s/Rocket-Lib | Built in C++14 | Tested on Ubuntu 22.04, Intel Core i7-12650H*
