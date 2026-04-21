# From One Core to Many: How Rocket-Lib's Neural Engine Got 11x Faster Without Touching the Math

### Threads alone did almost nothing. Here's what actually worked — and what the code still gets wrong.

---

*By Shri Hari S. | github.com/shrihari-s/Rocket-Lib*

---

## The Embarrassing Middle Chapter

Most performance articles show you the final scoreboard and reverse-engineer a clean story from it. This one won't do that.

Rocket-Lib is a custom C++14 neural engine built from scratch — Row-Major tensor storage, Adam optimizer from first principles, BCEWithLogits loss with a numerical stability trick, the whole thing. The pitch was simple: hand-written C++ should be faster than Python-wrapped TensorFlow on the same laptop CPU.

On simple models, the single-threaded version was holding its own at around **0.8 seconds per epoch**. Not dominant, but credible. The natural next step was adding threads.

The result? **0.70 to 0.75 seconds per epoch.**

A rounding error. On a 16-thread CPU, that's essentially nothing. Then on a complex model — multiple layer types, more depth, running for hundreds of epochs — Keras wasn't just ahead. Keras was **2 to 3x faster** than Rocket-Lib.

A C++ engine losing badly to a Python framework on its own hardware. That's the crisis point — and the forcing function for understanding what was actually wrong.

The answer turned out to be two things, and neither of them was the math.

This article will also name what the codebase still gets wrong. A technical reader will find the inconsistencies in the repo within ten minutes, so they're better stated upfront.

---

## Build Flags: What's Actually Driving Performance

Before anything else, one thing needs to be said clearly: the benchmark numbers in this article aren't from `-O3` alone.

Rocket-Lib compiles with `-O3`, `-march=native`, and **`-ffast-math`**. The fast-math flag tells the compiler it can break strict IEEE 754 rules — reordering operations, assuming no NaNs or infinities, replacing divisions with reciprocal multiplications. On a neural engine doing billions of FMAs per training run, this contributes a meaningful share of the total speedup.

```cmake
# core/CMakeLists.txt
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -ffast-math -march=native")
endif()
```

If you're benchmarking against Rocket-Lib using only `-O3`, the comparison isn't apples-to-apples. Fast-math is a legitimate tool for neural compute — it's the same trade-off BLAS libraries make — but it should be visible, not buried in a build file.

---

## Why Threads Alone Did Nothing

Before fixing anything, the key question was why adding threads barely moved the needle.

The first version spawned fresh OS threads at the start of every forward pass and joined them at the end. The problem is that thread creation is a **kernel-space syscall**. On Linux with pthreads, `pthread_create` costs anywhere from 10 to 80 microseconds depending on scheduler load. Spawning 16 threads means paying that cost 16 times before a single multiply-accumulate happens.

For small to medium layers, that spawn overhead **dominates** compute entirely. You're not parallelizing the work — you're parallelizing the waiting.

But that still doesn't explain why complex models were slower than Keras. Larger matrices mean more work per thread, which should amortize the spawn cost. The real culprit was deeper — hiding inside the kernel itself.

---

## Optimization 1: Stop Thrashing Your Cache

### What the Naive Loop Does to Memory

The first version of Rocket-Lib's matrix multiply was textbook:

```cpp
// The naive i-j-k loop — what the engine started with
for (int i = 0; i < input.rows; ++i) {
    for (int j = 0; j < weights.cols; ++j) {
        double sum = biases.data[j];
        for (int k = 0; k < weights.rows; ++k) {
            sum += input.data[i * input.cols + k]
                 * weights.data[k * weights.cols + j];
        }
        output.data[i * weights.cols + j] = sum;
    }
}
```

Three nested loops. Clean, readable, completely correct. And for a modern CPU, nearly optimal at wasting memory bandwidth.

Inside the innermost loop, the access pattern for the weights is:

```
weights.data[k * weights.cols + j]
```

As `k` increments, this jumps through memory **column by column** in a row-major layout. Each new `k` skips an entire row width. If `weights.cols` is 256, that's a **2 KB jump per iteration**. The hardware prefetcher can't predict it. Every access to `weights` is potentially a cold cache miss.

On the i7-12650H, a cache hit costs roughly 4 clock cycles. A cache miss costs 200+. That's a 50× latency penalty, paid on the most-accessed operand, millions of times per layer. This is exactly why Keras was winning on complex models: its backend (backed by oneDNN) uses cache-aware kernels. The Python overhead was irrelevant. Kernel quality was the whole game.

### The Fix: i-k-j Loop Reorder

The solution is to swap the `j` and `k` loops. The full kernel, as it lives inline inside `DenseLayer::forward`, looks like this:

```cpp
// layer.cpp — the actual i-k-j kernel inside DenseLayer::forward
for (int i = start; i < end; ++i) {
    // Step 1: initialize output row with biases (sequential write)
    for (int j = 0; j < weights.cols; ++j) {
        output.data[i * weights.cols + j] = biases.data[j];
    }
    // Step 2: accumulate via i-k-j traversal
    for (int k = 0; k < weights.rows; ++k) {
        double in_val = input.data[i * input.cols + k]; // hoisted to register
        const double* weight_row = &weights.data[k * weights.cols];
        double* out_row = &output.data[i * weights.cols];
        for (int j = 0; j < weights.cols; ++j) {
            out_row[j] += in_val * weight_row[j]; // linear hot-path
        }
    }
}
```

The math is **identical**. The same multiplications, the same additions, the same numerical result. Only the traversal order and the bias initialization placement changed — and that changes everything.

**`weight_row` is now a sequential scan.** The inner `j` loop walks linearly through `weight_row[j]`. Cache lines load in order. The hardware prefetcher can see ahead and speculatively fetch the next line. Cache misses on weights effectively disappear.

**`in_val` becomes a register.** In the original i-j-k loop, `input.data[i * input.cols + k]` was re-loaded every iteration of the outer `k` loop. Now it's hoisted above the j-loop entirely and lives in a CPU register for all `weights.cols` iterations. Zero memory traffic for the hottest value.

**`out_row[j]` is a sequential write.** The output accumulation is a linear walk. Write-combining works. The prefetcher covers it.

**The bias init is separated.** Initializing the output row with biases before the k-loop means the accumulation `+= in_val * weight_row[j]` is always a pure add — no conditional, no branch in the hot path.

Three sequential memory streams, one register scalar. The compiler, seeing linear non-aliased streams under `-ffast-math`, can auto-vectorize with AVX2 — four `double` FMAs per instruction on the i7-12650H. Hardware SIMD acceleration without writing a single intrinsic.

### The Deeper Point

The algorithm didn't change. The traversal order changed.

Computation is cheap. Memory latency is the bottleneck. A cache-aware O(n³) GEMM beats a cache-oblivious one every time, regardless of how many threads you throw at the latter. When the complex multi-layer models were losing to Keras, they weren't losing because of bad math — they were losing because every layer was torching the cache on entry.

Fix the kernel first. Then parallelize. Parallelizing a slow kernel gives you more cores doing slow things simultaneously.

---

## Optimization 2: Stop Paying the Thread-Spawn Tax

With a cache-efficient kernel in place, parallelism can finally do something useful. But the naive threading model still needs to go.

### The Persistent Singleton ThreadPool

Instead of spawning threads per forward pass, Rocket-Lib creates them **once at startup** via a Meyers Singleton and keeps them alive for the entire program lifetime:

```cpp
// threadpool.h — full declaration
class ThreadPool {
public:
    static ThreadPool& getInstance() {
        static ThreadPool instance(std::thread::hardware_concurrency());
        return instance;
    }

    // Variadic template — accepts any callable, returns a typed future
    template<class F, class... Args>
    auto enqueue(F&& f, Args&&... args)
        -> std::future<typename std::result_of<F(Args...)>::type>;

private:
    ThreadPool(size_t threads);           // called once with hardware_concurrency()
    std::vector<std::thread> workers;
    std::queue<std::function<void()>> tasks;  // packaged_task erased to std::function
    std::mutex queue_mutex;
    std::condition_variable condition;
    bool stop;
};
```

The worker loop uses `for(;;)` — threads block on the condition variable and consume zero CPU while idle:

```cpp
// ThreadPool constructor — worker loop
workers.emplace_back([this] {
    for (;;) {
        std::function<void()> task;
        {
            std::unique_lock<std::mutex> lock(this->queue_mutex);
            this->condition.wait(lock, [this] {
                return this->stop || !this->tasks.empty();
            });
            if (this->stop && this->tasks.empty()) return;
            task = std::move(this->tasks.front());
            this->tasks.pop();
        }
        task(); // future signalled when the packaged_task inside completes
    }
});
```

`enqueue` wraps the callable in a `std::packaged_task`, type-erases it to `std::function<void()>`, pushes it onto the queue, then calls `condition.notify_one()` to wake exactly one waiting worker. The destructor sets `stop = true` and calls `condition.notify_all()` to drain all threads cleanly on shutdown.

The thread creation syscall now happens **once per program lifetime**, not once per forward pass. The dispatch cost per layer invocation drops from tens of microseconds per thread to a single lock-notify cycle — roughly a few microseconds for the entire enqueue operation.

### ROCKET_THRESHOLD: Knowing When Not to Parallelize

A ThreadPool alone isn't enough. You still need to decide *whether* to parallelize a given layer at all. Dispatching a 4×4 matrix to 16 threads spends more time on synchronization than it saves in parallel compute.

Rocket-Lib makes this decision explicitly with a FLOP-count heuristic. The forward pass threshold is **100,000 FLOPs**:

```cpp
// layer.cpp — DenseLayer::forward dispatch
int num_threads = std::thread::hardware_concurrency();
int total_work = input.rows * weights.cols * weights.rows;

if (total_work > 100000 && num_threads > 1) {
    std::vector<std::future<void>> futures;
    // Ceiling division — handles input.rows < num_threads safely
    int chunk_size = (input.rows + num_threads - 1) / num_threads;
    for (int t = 0; t < num_threads; ++t) {
        int start = t * chunk_size;
        int end = std::min(start + chunk_size, input.rows);
        if (start >= end) break; // no empty partitions
        futures.push_back(ThreadPool::getInstance().enqueue(
            [this, &input, start, end]() {
                for (int i = start; i < end; ++i) {
                    for (int j = 0; j < weights.cols; ++j)
                        output.data[i * weights.cols + j] = biases.data[j];
                    for (int k = 0; k < weights.rows; ++k) {
                        double in_val = input.data[i * input.cols + k];
                        const double* weight_row = &weights.data[k * weights.cols];
                        double* out_row = &output.data[i * weights.cols];
                        for (int j = 0; j < weights.cols; ++j)
                            out_row[j] += in_val * weight_row[j];
                    }
                }
            }
        ));
    }
    for (auto& f : futures) f.wait();
} else {
    // Single-threaded — no synchronization overhead
    for (int i = 0; i < input.rows; ++i) { /* same kernel */ }
}
```

Three details worth noting. First, the lambda captures `[this, &input, start, end]` — `this` gives the lambda access to `weights`, `biases`, and `output` as class members; `start` and `end` are captured by value so each thread has its own partition range. Second, ceiling division (`rows + threads - 1) / threads`) guarantees no partition is empty when `input.rows < num_threads`. Third, `f.wait()` is used over `f.get()` because the futures are `void` — `wait()` is the idiomatic choice.

### The Backward Pass Has Two Separate Strategies

The backward pass parallelizes two independent computations — weight gradients and input gradients — and each one has a different partition dimension and threshold.

**Weight gradients (grad_weights)** are partitioned by `input.cols` (the `k` dimension), with a lower threshold of **50,000 FLOPs**:

```cpp
// layer.cpp — DenseLayer::backward, grad_weights phase
int total_work_w = input.cols * input.rows * grad_output.cols;
if (total_work_w > 50000 && num_threads > 1) {
    int chunk_w = (input.cols + num_threads - 1) / num_threads;
    for (int t = 0; t < num_threads; ++t) {
        int start = t * chunk_w;
        int end = std::min(start + chunk_w, input.cols);
        if (start >= end) break;
        futures_w.push_back(ThreadPool::getInstance().enqueue(
            [this, &input, &grad_output, start, end]() {
                for (int k = start; k < end; ++k) {
                    for (int i = 0; i < input.rows; ++i) {
                        double in_val = input.data[i * input.cols + k];
                        for (int j = 0; j < grad_output.cols; ++j) {
                            grad_weights.data[k * grad_weights.cols + j] +=
                                in_val * grad_output.data[i * grad_output.cols + j];
                        }
                    }
                }
            }
        ));
    }
    for (auto& f : futures_w) f.wait();
}
```

Partitioning by `k` (input columns) means each thread owns a disjoint set of rows in `grad_weights` and never writes to the same memory as another thread. No locking needed, no false sharing.

**Input gradients (grad_input)** are partitioned by `grad_output.rows`, with a higher threshold of **100,000 FLOPs**:

```cpp
// layer.cpp — DenseLayer::backward, grad_input phase
int total_work_in = grad_output.rows * weights.rows * weights.cols;
if (total_work_in > 100000 && num_threads > 1) {
    int chunk_in = (grad_output.rows + num_threads - 1) / num_threads;
    for (int t = 0; t < num_threads; ++t) {
        int start = t * chunk_in;
        int end = std::min(start + chunk_in, grad_output.rows);
        if (start >= end) break;
        futures_in.push_back(ThreadPool::getInstance().enqueue(
            [this, &grad_output, start, end]() {
                for (int i = start; i < end; ++i) {
                    for (int j = 0; j < weights.rows; ++j) {
                        double sum = 0.0;
                        const double* grad_row = &grad_output.data[i * grad_output.cols];
                        const double* w_row    = &weights.data[j * weights.cols];
                        for (int k = 0; k < weights.cols; ++k)
                            sum += grad_row[k] * w_row[k];
                        grad_input.data[i * weights.rows + j] = sum;
                    }
                }
            }
        ));
    }
    for (auto& f : futures_in) f.wait();
}
```

The lower threshold on grad_weights (50k vs 100k) reflects that gradient accumulation involves more dependent writes and is more sensitive to synchronization overhead — parallelism pays off at a smaller work size during the backward pass.

### The Hybrid P/E-Core Architecture

The i7-12650H has 6 Performance cores and 4 Efficiency cores. The ThreadPool is sized to `hardware_concurrency()` — which returns 16 on this machine (6 P-cores × 2 HT + 4 E-cores). Rocket-Lib doesn't pin threads to specific core types; Linux's scheduler routes compute-heavy work to P-cores naturally. Explicit pinning via `pthread_setaffinity_np` would be a fine future optimization. The current approach captures most of the benefit because the workload is homogeneous.

---

## The DAG Execution Model

Rocket-Lib supports directed acyclic graph execution for ResNet-style architectures. The graph is compiled via Kahn's topological sort. Execution has an explicit fast path for single-predecessor nodes:

```cpp
// model.cpp — forward pass inside train()
// Maps declared once per epoch, reused across all batches within it
std::unordered_map<Layer*, Tensor> layer_outputs;
std::unordered_map<Layer*, Tensor> layer_grads;

for (/* each batch */) {
    // Forward
    for (Layer* layer : topological_order) {
        const auto& prevs = prev_layers_map[layer];
        if (prevs.size() == 1) {
            // Single predecessor: no temporary allocation
            layer_outputs[layer] = layer->forward(layer_outputs[prevs[0]]);
        } else {
            // Multi-predecessor (ResNet merge): sum inputs then forward
            Tensor combined_input = layer_outputs[prevs[0]];
            for (size_t p = 1; p < prevs.size(); ++p) {
                combined_input += layer_outputs[prevs[p]];
            }
            layer_outputs[layer] = layer->forward(combined_input);
        }
    }
    // layer_grads.clear() before backward — map reused within the epoch, not reallocated
}
```

Two things worth calling out. First, the maps are **declared once per epoch and reused across all batches within it** — eliminating the per-batch allocation churn that an earlier version had. The maps are still reallocated at the start of each epoch; moving them outside the epoch loop entirely and calling `.clear()` there is the remaining fix. Second, the single-predecessor path still performs a **deep copy** — the `Tensor` move constructor was deleted during a refactor, so `layer_outputs[layer] = ...` triggers the copy constructor rather than a move. Restoring the move constructor is the highest-priority code task; until then, every layer transition allocates and copies.

---

## What the Optimizer Actually Does

Adam is implemented from first principles — first-moment and second-moment estimates, bias correction, the standard update rule. But there is one addition: a **gradient clamp of ±10.0** before the moment update.

```cpp
// optimizer.cpp — Adam::update, gradient soft-clip (C++14 compatible)
double g = std::max(std::min(grad.data[i], 10.0), -10.0); // no std::clamp in C++14
m_t.data[i] = beta1 * m_t.data[i] + (1.0 - beta1) * g;
v_t.data[i] = beta2 * v_t.data[i] + (1.0 - beta2) * g * g;
```

Note `std::clamp` is C++17. Since Rocket-Lib targets C++14, the equivalent `std::max(std::min(...))` is used. The clamp itself is standard practice in production training pipelines — it prevents runaway gradients from destabilizing early training. It's not in the textbook derivation of Adam, but calling the implementation "first principles" without mentioning it would be misleading.

---

## Numerical Stability: Forward Pass Only

Rocket-Lib uses the Log-Sum-Exp trick in `BCEWithLogits` to prevent overflow:

```
log(1 + eˣ) = max(x, 0) + log(1 + e^−|x|)
```

This is applied in the **forward pass**:

```cpp
// loss.cpp — BCEWithLogits::forward
sum += std::max(x, 0.0) - x * t + std::log(1.0 + std::exp(-std::abs(x)));
```

The **backward pass** uses a standard sigmoid:

```cpp
// loss.cpp — BCEWithLogits::backward
double p = 1.0 / (1.0 + std::exp(-logits.data[i]));
grad.data[i] = (p - t) / size;
```

The `std::exp(-logits.data[i])` call is unprotected. For extreme positive logits, this underflows to zero harmlessly. For extreme negative logits, it overflows. In practice the benchmark inputs never hit this range, but it's a known inconsistency: the forward pass is stable by construction, the backward pass is not.

---

## A Note on Reproducibility

The `Tensor` base constructor uses a static seeded generator (`std::mt19937` with seed 42). However, `DenseLayer` — the primary layer used in all benchmark models — re-initializes weights using `std::random_device`:

```cpp
// layer.cpp — DenseLayer constructor
std::random_device rd;
std::mt19937 gen(rd());
std::uniform_real_distribution<> dis(-limit, limit);
for (int i = 0; i < input_dim * output_dim; ++i) {
    weights.data[i] = dis(gen);
}
```

`std::random_device` is non-deterministic. Two runs with identical hyperparameters will produce slightly different starting weights. The `DropoutLayer` reads from `ROCKET_SEED` and defaults to seed 42 if the variable is unset, so dropout masks are consistent out of the box. The benchmark scripts set `ROCKET_SEED=42` explicitly.

In practice, the final accuracy numbers are stable across runs — the loss landscape is smooth enough that initialization variance doesn't flip the outcome. But if you need to reproduce a specific loss curve, patch `DenseLayer` to use a seeded `std::mt19937` instead of `std::random_device`.

---

## The Benchmarks

All measurements on Ubuntu 22.04, Intel Core i7-12650H, compiled with `-O3 -march=native -ffast-math`. Keras benchmarks use TensorFlow CPU backend with default flags. Both training runs are wall-clock timed with `time.time()` and printed at the end of each script. The full benchmark suite runs via `sh tests.sh`; the primary parity comparison runs via `sh run_comparison.sh`.

The default epoch count in `compare_keras.py` is **500 epochs** — that's what the stress test numbers below reflect.

### Sequential Dense Forward Pass

| Implementation    | Time      |
|-------------------|-----------|
| Keras (TF CPU)    | 3.1289 s  |
| Rocket-Lib        | 0.2789 s  |
| **Speedup**       | **11.22×** |

Cache-aware GEMM eliminates the miss penalty. The persistent ThreadPool eliminates spawn overhead. The ceiling-division partitioning ensures no threads are wasted on empty partitions. The threshold heuristic keeps small layers on the single-threaded path.

### ResNet DAG Forward Pass

| Implementation    | Time      |
|-------------------|-----------|
| Keras (TF CPU)    | 3.3581 s  |
| Rocket-Lib        | 0.7873 s  |
| **Speedup**       | **4.27×** |

The smaller gain relative to the dense case is expected. Branch-merge points force tensor materialization, and without the move constructor active, even the single-predecessor path pays a deep copy cost at every layer transition.

### Comprehensive Multi-Layer Model

| Implementation    | Time      |
|-------------------|-----------|
| Keras (TF CPU)    | 9.7155 s  |
| Rocket-Lib        | 2.6683 s  |
| **Speedup**       | **3.64×** |

This is the model class where Keras was winning before the optimizations — same architecture, same hardware, the workload that had Keras running 2–3x faster. Now Rocket-Lib is on the right side of that gap.

### 500-Epoch Training Stress Test

| Implementation    | Time      |
|-------------------|-----------|
| Keras (TF CPU)    | 122.73 s  |
| Rocket-Lib        | 49.34 s   |
| **Speedup**       | **~2.5×** |

This is the default `compare_keras.py` run — 500 epochs, 10,000 samples, with Dropout and L2 regularization, timed end-to-end. The advantage holds across the full forward-backward-update cycle.

### Convergence

Speed only matters if accuracy holds:

| Metric      | Rocket-Lib | Keras   |
|-------------|------------|---------|
| BCE Loss    | 0.1125     | 0.1753  |
| Accuracy    | 97.25%     | 96.55%  |

Rocket-Lib converges to a better loss on this task. Given the non-deterministic DenseLayer initialization, treat these as representative rather than exactly reproducible. The accuracy advantage is consistent across runs.

---

## What the Code Still Gets Wrong

In the spirit of this being an engineering diary rather than a marketing document, here is the current known debt:

**`begin_step()` double-count in Adam.** `optimizer->begin_step()` — which increments Adam's `step_count` — is called twice per batch in the current `model.cpp`: once before the backward loop and once after it. Since `step_count` feeds directly into the bias correction denominators (`1 - β₁ᵗ` and `1 - β₂ᵗ`), doubling the increment causes the correction to converge twice as fast as the standard Adam derivation intends. One of the two calls needs to be removed.

**Epoch-level map reallocation.** `layer_outputs` and `layer_grads` are declared inside the epoch loop, so they are fully destroyed and reconstructed at the start of every epoch. For a 500-epoch run this creates unnecessary heap pressure. The fix is to move the declarations outside the epoch loop and call `.clear()` inside it instead.

**Move constructor is deleted.** The `Tensor` move constructor was removed during a refactor. All tensor transfers currently perform deep copies. The most visible consequence is in `ActivationLayer`, which had to adopt an explicit `memcpy` through a temporary to avoid double-free issues that arise without the move constructor:

```cpp
// layer.cpp — ActivationLayer::forward, consequence of missing move constructor
Tensor result = activation_fn->forward(input); // temporary allocation
std::memcpy(output.data, result.data,
            output.rows * output.cols * sizeof(double)); // explicit copy
```

Restoring the move constructor would eliminate the temporary allocation, the memcpy, and the deep copies in the DAG single-predecessor path in a single fix.

**Double initialization in `DenseLayer`.** The `Tensor` base constructor sweeps every element during allocation. The `DenseLayer` constructor then sweeps the weight matrix again for Xavier initialization. That's a 100% redundant sweep at startup — consolidating them into a single pass is a straightforward fix.

**Backward pass stability.** The LogSumExp trick in `BCEWithLogits` covers the forward pass only. The backward pass uses a raw `std::exp(-logits.data[i])` that can overflow for extreme negative logits.

**Non-deterministic weight initialization.** `DenseLayer` uses `std::random_device` rather than a seeded generator. Two training runs produce different starting weights. Patching this to use a seeded `std::mt19937` controlled by `ROCKET_SEED` would make results fully reproducible.

These are real. The benchmarks are also real. Engineering is the gap between the two.

---

## The Actual Takeaway

The arc of this project:

**0.8s/epoch** single-threaded → **0.70–0.75s/epoch** with naive threads → Keras **2–3x ahead** on complex models → two focused architectural fixes → Rocket-Lib **3–11x ahead** across the board.

Naive threads moved the needle by almost nothing. The reason wasn't threading — it was that the underlying kernel was cache-oblivious, and parallelizing a slow kernel gives you more cores doing slow things simultaneously. Keras's backend uses cache-aware kernels descended from decades of BLAS research. Matching that kernel quality, not the thread count, was what actually closed the gap.

**Fix 1: Loop reordering.** Swapping j and k transforms random column-stride access into sequential row-stride access. Bias initialization is separated from accumulation, keeping the hot inner loop pure. The compiler, seeing sequential non-aliased streams under `-ffast-math`, generates AVX2 automatically. No intrinsics, no assembly — just a different traversal order that respects the memory hierarchy.

**Fix 2: A persistent singleton ThreadPool with per-pass thresholds.** Thread creation happens once at startup via `hardware_concurrency()`. Per-layer dispatch costs a single lock-notify cycle. Ceiling-division partitioning handles any batch size safely. The backward pass uses two independent parallelization strategies with separately calibrated thresholds — grad_weights at 50k FLOPs partitioned by input column, grad_input at 100k FLOPs partitioned by row — because gradient accumulation has different memory access characteristics than the forward sweep.

Both fixes are architectural, not mathematical. The math was always correct. The problem was how the math moved through memory and how compute was scheduled around the hardware.

Fix the architecture. The math takes care of itself.

---

*Rocket-Lib is available at github.com/shrihari-s/Rocket-Lib. Built in C++14, tested on Ubuntu 22.04 with an Intel Core i7-12650H. Build flags: `-O3 -march=native -ffast-math`. Run `sh tests.sh` to reproduce all benchmarks.*

---

**Tags:** `C++` `Performance` `Multithreading` `Neural Networks` `Systems Programming` `Cache Optimization` `Computer Architecture`
