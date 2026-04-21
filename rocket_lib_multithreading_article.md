# From One Core to Many: How I Made My Neural Engine 11x Faster Without Touching the Math

### Threads alone did almost nothing. Here's what actually worked — and what the code still gets wrong.

---

*By Shri Hari S. | github.com/shrihari-s/Rocket-Lib*

---

## The Embarrassing Middle Chapter

Most performance articles show you the final scoreboard and reverse-engineer a clean story from it. This one won't do that.

I built Rocket-Lib — a custom C++14 neural engine — from scratch. Row-Major tensor storage, Adam optimizer implemented manually, BCEWithLogits loss with a numerical stability trick, the whole thing. The pitch was straightforward: hand-written C++ should be faster than Python-wrapped TensorFlow on the same laptop CPU.

On simple models, the single-threaded version was holding its own at around **0.8 seconds per epoch**. Not dominant, but credible. So I did what any engineer does next: I added threads.

The result? **0.70 to 0.75 seconds per epoch.**

A rounding error. On a 16-thread CPU, that's essentially nothing. Then the moment I tested on a complex model — multiple layer types, more depth, 400 epochs — Keras wasn't just ahead. Keras was **2 to 3x faster** than Rocket-Lib.

A C++ engine losing badly to a Python framework on its own hardware. That's the crisis point. That's what forced me to actually understand what was wrong.

The answer turned out to be two things, and neither of them was the math.

I'll also be honest about what the codebase still gets wrong. There are real inconsistencies between the clean story and the actual source — a technical reader will find them in the repo within ten minutes, so I'd rather name them upfront.

---

## Build Flags: What's Actually Driving Performance

Before getting into architecture, one thing needs to be said clearly: the benchmark numbers in this article aren't from `-O3` alone.

Rocket-Lib's `CMakeLists.txt` compiles with both `-O3` and **`-ffast-math`**. The fast-math flag tells the compiler it can break strict IEEE 754 floating-point rules — reordering operations, assuming no NaNs or infinities, replacing divisions with reciprocal multiplications. On a neural engine doing billions of FMAs per training run, this can contribute a meaningful share of the final speedup.

```cmake
# CMakeLists.txt
set(CMAKE_CXX_STANDARD 14)
target_compile_options(rocket PRIVATE -O3 -march=native -ffast-math)
```

If you're benchmarking against this and only using `-O3`, the comparison isn't apples-to-apples. Fast-math is a legitimate tool for neural compute — it's the same trade-off that BLAS libraries make — but it should be visible, not buried in a build file.

---

## Why Threads Alone Did Nothing

Before fixing the architecture, I had to understand why naive threads barely moved the needle.

The first version spawned fresh OS threads at the start of every forward pass and joined them at the end. The problem is that thread creation is a **kernel-space syscall**. On Linux with pthreads, `pthread_create` costs anywhere from 10 to 80 microseconds depending on scheduler load. Spawning 16 threads means paying that cost 16 times before a single multiply-accumulate happens.

For small to medium layers, the spawn overhead **dominates** compute entirely. You're not parallelizing the work — you're parallelizing the waiting.

But that still doesn't explain why complex models were slower than Keras. Larger matrices mean more work per thread, which should amortize the spawn cost. The real culprit was hiding inside the math kernel itself.

---

## Optimization 1: Stop Thrashing Your Cache

### What the Naive Loop Does to Memory

The first version of Rocket-Lib's matrix multiply was textbook:

```cpp
// The naive i-j-k loop order
for (int i = start; i < end; ++i) {
    for (int j = 0; j < weights_cols; ++j) {
        double sum = 0.0;
        for (int k = 0; k < weights.rows; ++k) {
            sum += input.data[i * input.cols + k]
                 * weights.data[k * weights_cols + j];
        }
        output.data[i * weights_cols + j] = sum;
    }
}
```

Three nested loops. Clean, readable, completely correct. And for a modern CPU, nearly optimal at wasting memory bandwidth.

Inside the innermost loop, the access pattern for weights is:

```
weights.data[k * weights_cols + j]
```

As `k` increments, this jumps through memory **column by column** in a row-major layout. Each new `k` skips an entire row width. If `weights_cols` is 256, that's a **2 KB jump per iteration**. The hardware prefetcher can't predict it. Every access to `weights` is potentially a cold cache miss.

On the i7-12650H, a cache hit costs roughly 4 clock cycles. A cache miss costs 200+. That's a 50x latency penalty, paid on the most-accessed operand, millions of times per layer. This is why Keras was winning on complex models: its backend uses cache-aware kernels. The Python overhead was irrelevant. Kernel quality was the whole game.

### The Fix: i-k-j Loop Reorder

The solution is to swap the `j` and `k` loops. The GEMM logic lives as an inline lambda inside `DenseLayer::forward`:

```cpp
// layer.cpp — inline inside DenseLayer::forward
// The i-k-j cache-aware kernel
for (int i = start; i < end; ++i) {
    for (int k = 0; k < weights.rows; ++k) {
        double in_val = input.data[i * input.cols + k]; // hoisted to register
        const double* weight_row = &weights.data[k * weights_cols];
        double* out_row = &output.data[i * weights_cols];
        for (int j = 0; j < weights_cols; ++j) {
            out_row[j] += in_val * weight_row[j]; // linear hot-path
        }
    }
}
```

The math is **identical**. The same multiplications, the same additions, the same numerical result. Only the traversal order changed.

**`weight_row` is now a sequential scan.** The inner `j` loop walks linearly through `weight_row[j]`. Cache lines load sequentially. The hardware prefetcher can see ahead. Cache misses on weights effectively disappear.

**`in_val` becomes a register.** In the original loop it was re-fetched every `k`. Now it's hoisted out of the j-loop. It lives in a CPU register for the entire inner loop. Zero memory traffic for the hottest value.

**`out_row[j]` is a sequential write.** The output accumulation is a linear walk too. Write-combining works. The prefetcher covers it.

Three sequential memory streams and one register scalar. The compiler, seeing linear non-aliased streams, can auto-vectorize with AVX2 — four `double` FMAs per instruction on the i7-12650H. You get hardware SIMD without writing a single intrinsic.

### The Deeper Point

The algorithm didn't change. The traversal order changed.

Computation is cheap. Memory latency is the bottleneck. A cache-aware O(n³) GEMM beats a cache-oblivious one every time, regardless of how many threads you throw at the latter. When the complex multi-layer models were losing to Keras, they weren't losing because of bad math — they were losing because every layer was torching the cache on entry.

Fix the kernel first. Then parallelize. Parallelizing a slow kernel gives you more cores doing slow things simultaneously.

---

## Optimization 2: Stop Paying the Thread-Spawn Tax

With a cache-efficient kernel in place, parallelism can finally do something useful. But the naive threading model still needs to go.

### The Persistent Singleton ThreadPool

Instead of spawning threads per forward pass, Rocket-Lib creates them **once at startup** and keeps them alive forever:

```cpp
// threadpool.h — Singleton declaration
class ThreadPool {
public:
    static ThreadPool& getInstance(int num_threads = 0);
    std::future<void> enqueue(std::function<void()> task);
private:
    ThreadPool(int num_threads);
    std::vector<std::thread> workers;
    std::queue<std::packaged_task<void()>> tasks;
    std::mutex queue_mutex;
    std::condition_variable condition;
    bool stop;
};
```

Workers run a tight condition-variable loop, consuming zero CPU while idle:

```cpp
// Worker loop — threads sleep until work arrives
workers.emplace_back([this] {
    while (true) {
        std::packaged_task<void()> task;
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            condition.wait(lock, [this] {
                return stop || !tasks.empty();
            });
            if (stop && tasks.empty()) return;
            task = std::move(tasks.front());
            tasks.pop();
        }
        task(); // completion signalled via the returned future
    }
});
```

Synchronization at the call site uses the `std::future<void>` returned by `enqueue`. The caller collects all futures and calls `.get()` on each — no separate `wait()` method, no atomic counter needed.

The kernel syscall for thread creation now happens **once per program lifetime**, not once per layer invocation. The dispatch cost per forward pass drops from tens of microseconds per thread to a few microseconds for the entire enqueue operation.

### ROCKET_THRESHOLD: When Not to Parallelize

A ThreadPool still isn't enough on its own. You need to decide *whether* to parallelize a given layer at all. Dispatching a tiny matrix to 16 threads wastes more time on synchronization than it saves in parallel compute.

Rocket-Lib makes this decision explicitly using a FLOP-count heuristic — and crucially, the threshold is **tuned separately per pass**:

```cpp
// layer.cpp — DenseLayer::forward dispatch
int total_work = input.rows * weights.cols * weights.rows;

if (total_work > 100000 && num_threads > 1) {
    std::vector<std::future<void>> futures;
    int rows_per_thread = input.rows / num_threads;
    for (int t = 0; t < num_threads; ++t) {
        int start = t * rows_per_thread;
        int end = (t == num_threads - 1) ? input.rows : start + rows_per_thread;
        futures.push_back(pool.enqueue([=, &input, &weights, &output]() {
            for (int i = start; i < end; ++i) {
                for (int k = 0; k < weights.rows; ++k) {
                    double in_val = input.data[i * input.cols + k];
                    const double* weight_row = &weights.data[k * weights_cols];
                    double* out_row = &output.data[i * weights_cols];
                    for (int j = 0; j < weights_cols; ++j) {
                        out_row[j] += in_val * weight_row[j];
                    }
                }
            }
        }));
    }
    for (auto& f : futures) f.get();
} else {
    // single-threaded — avoid sync overhead entirely
    for (int i = 0; i < input.rows; ++i) { /* same kernel */ }
}
```

The forward pass threshold is **100,000 FLOPs**. The backward pass uses a lower threshold of **50,000 FLOPs** — gradient accumulation involves more dependent memory writes and synchronization sensitivity, so parallelism pays off at a smaller work size. The threshold isn't a magic number; it's a break-even point calibrated per phase on this specific hardware.

### The Hybrid P/E-Core Architecture

The i7-12650H has 6 Performance cores and 4 Efficiency cores. P-cores run faster, support full AVX2, and have larger per-core caches. E-cores are tuned for throughput at lower power.

Rocket-Lib's ThreadPool doesn't pin threads to specific core types — Linux's scheduler tends to route compute-heavy work to P-cores naturally. Explicit pinning via `pthread_setaffinity_np` is a possible future optimization. The current approach captures most of the benefit because the workload is homogeneous and the scheduler makes sensible decisions.

---

## The DAG Execution Model

Rocket-Lib supports directed acyclic graph execution for ResNet-style architectures. The graph traversal uses topological degree forwarding:

```cpp
// model.cpp — DAG execution pass
if (prevs.size() == 1) {
    // Single predecessor — use its output directly
    layer_outputs[layer] = layer->forward(layer_outputs[prevs[0]]);
} else {
    // Multiple predecessors — combine inputs (ResNet merge)
    Tensor combined_input = layer_outputs[prevs[0]];
    for (size_t p = 1; p < prevs.size(); ++p) {
        combined_input = combined_input + layer_outputs[prevs[p]];
    }
    layer_outputs[layer] = layer->forward(combined_input);
}
```

One honest note here: with the move constructor currently absent from `tensor.cpp`, the single-predecessor path performs a **deep copy** rather than a pointer forward. The merge path is unaffected — it always materializes a new tensor. Restoring the move constructor is a pending code task; for now, single-predecessor layers pay the copy cost.

The state maps (`layer_outputs`) are reused across batches within an epoch, which avoids per-batch allocation churn. They are, however, reallocated at the start of each epoch — a source of unnecessary heap pressure on long training runs that a future version should address.

---

## What the Optimizer Actually Does

The article would be incomplete without being honest about the Adam implementation.

Yes, Adam is implemented from first principles — first-moment and second-moment estimates, bias correction, the standard update rule. But there is one addition: a **gradient clamp of ±10.0** applied before the update step.

```cpp
// optimizer.cpp — gradient soft-clip
double grad = std::clamp(raw_grad, -10.0, 10.0);
```

This is a standard stabilization technique in practice, common in production training pipelines. It prevents runaway gradients from destabilizing early training. It's not in the textbook derivation of Adam, but calling the implementation "from first principles" while omitting it would be misleading. The clamp is there, it helps, and you should know about it if you're building on this.

---

## Numerical Stability: Forward Pass Only

The article previously boasted that Rocket-Lib uses the Log-Sum-Exp trick throughout `BCEWithLogits` to prevent overflow. This needs to be scoped correctly.

The trick — $\log(1 + e^x) = \max(x, 0) + \log(1 + e^{-|x|})$ — is applied in the **forward pass**. The backward pass uses a standard sigmoid formula that doesn't carry the same protection. For the inputs seen in the benchmark tasks, this hasn't caused issues. But it's a known inconsistency: the forward pass is stable by construction, the backward pass is not.

---

## A Note on Reproducibility

The convergence numbers at the end of this article are real. But reproducing them exactly requires understanding how initialization works.

The `Tensor` class uses a seeded generator (`seed 42`) for weight initialization. However, `DenseLayer` — the primary layer type used in all benchmark models — initializes weights using `std::random_device`, which is non-deterministic. Two training runs with identical hyperparameters will produce slightly different weight initializations.

The `DropoutLayer`, if used, reads from a `ROCKET_SEED` environment variable. If that variable isn't set, dropout masks will differ between runs.

In practice the final accuracy numbers are stable across runs — the loss landscape is smooth enough that initialization variance doesn't change the outcome. But if you're trying to reproduce a specific loss curve exactly, you'll need to patch `DenseLayer` to use a seeded `std::mt19937`.

---

## The Benchmarks

All measurements taken on Ubuntu 22.04, Intel Core i7-12650H, compiled with `-O3 -march=native -ffast-math`. Keras benchmarks use TensorFlow CPU backend with default compilation flags.

### Sequential Dense Forward Pass

| Implementation    | Time      |
|-------------------|-----------|
| Keras (TF CPU)    | 3.1289 s  |
| Rocket-Lib        | 0.2789 s  |
| **Speedup**       | **11.22×** |

Cache-aware GEMM eliminates the miss penalty. The persistent ThreadPool eliminates spawn overhead. The threshold heuristic keeps small layers on the fast single-threaded path.

### ResNet DAG Forward Pass

| Implementation    | Time      |
|-------------------|-----------|
| Keras (TF CPU)    | 3.3581 s  |
| Rocket-Lib        | 0.7873 s  |
| **Speedup**       | **4.27×** |

The smaller gain relative to the dense case is expected. More branch-merge points mean more forced tensor allocations, and without move semantics currently active, single-predecessor transitions also pay a copy cost.

### Comprehensive Multi-Layer Model

| Implementation    | Time      |
|-------------------|-----------|
| Keras (TF CPU)    | 9.7155 s  |
| Rocket-Lib        | 2.6683 s  |
| **Speedup**       | **3.64×** |

This is the model class where Keras was winning before the optimizations. Same test. Same hardware. Same 400-epoch workload that had Keras running 2–3x faster. Now Rocket-Lib is on the right side of that gap.

### 500-Epoch Training Stress Test

| Implementation    | Time      |
|-------------------|-----------|
| Keras (TF CPU)    | 122.73 s  |
| Rocket-Lib        | 49.34 s   |
| **Speedup**       | **~2.5×** |

Sustained advantage across a full training run including backpropagation and Adam optimizer updates.

### Convergence

Speed only matters if accuracy holds:

| Metric      | Rocket-Lib | Keras   |
|-------------|------------|---------|
| BCE Loss    | 0.1125     | 0.1753  |
| Accuracy    | 97.25%     | 96.55%  |

Rocket-Lib converges to a better loss on this task. Given the non-deterministic initialization, treat these as representative rather than exact reproducible numbers. The accuracy advantage is consistent across runs.

---

## What the Code Still Gets Wrong

In the spirit of this article being a real engineering diary rather than a marketing document, here is the current known debt:

**Move constructor is deleted.** The `Tensor` move constructor was removed during a refactor. All tensor transfers currently perform deep copies. This is the highest-priority fix — restoring it would improve the DAG path and clean up the memory narrative.

**Double initialization in `DenseLayer`.** The constructor performs two full sweeps of every weight matrix: once in the `Tensor` constructor and once in the Xavier initialization pass. That's a 100% wasted allocation on startup that should be consolidated.

**Backward pass stability.** The LogSumExp trick in BCEWithLogits covers the forward pass only. The backward pass uses a naive sigmoid that can overflow on extreme inputs.

**Epoch-level map reallocation.** The DAG state maps are reallocated at the start of every epoch. For 500-epoch runs this creates significant heap churn that a persistent pre-allocated structure would eliminate.

These are real. The benchmarks are also real. Engineering is the gap between the two.

---

## The Actual Takeaway

The arc of this project:

**0.8s/epoch** single-threaded → **0.7–0.75s/epoch** with naive threads → Keras **2–3x ahead** on complex models → two focused architectural fixes → Rocket-Lib **3–11x ahead** across the board.

Naive threads moved the needle by almost nothing. The reason wasn't threading — it was that the underlying kernel was cache-oblivious, and parallelizing a slow kernel just gives you more cores doing slow things simultaneously. Keras's backend uses cache-aware kernels descended from decades of BLAS research. Matching that quality, not the thread count, was what actually closed the gap.

The two fixes that mattered:

**1. Loop reordering** made memory access sequential and let `-ffast-math` plus the compiler's auto-vectorizer generate AVX2 code. No intrinsics, no assembly, just a different traversal order that respects the memory hierarchy.

**2. A persistent thread pool with per-pass work thresholds** eliminated the spawn tax and avoided false parallelism on small layers. Forward threshold: 100,000 FLOPs. Backward threshold: 50,000 FLOPs. Threads that are already alive are cheap. Threads that are born and die per layer invocation are not.

Both are architectural decisions, not mathematical ones. The math was always correct. The problem was how the math moved through memory and how compute was scheduled around the hardware.

Fix the architecture. The math takes care of itself.

---

*Rocket-Lib is available at github.com/shrihari-s/Rocket-Lib. Built in C++14, tested on Ubuntu 22.04 with an Intel Core i7-12650H. All benchmark code lives in the `/testing` directory. Build flags: `-O3 -march=native -ffast-math`.*

---

**Tags:** `C++` `Performance` `Multithreading` `Neural Networks` `Systems Programming` `Cache Optimization` `Computer Architecture`
