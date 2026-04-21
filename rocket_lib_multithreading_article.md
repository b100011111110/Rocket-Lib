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

    template<class F, class... Args>
    auto enqueue(F&& f, Args&&... args)
        -> std::future<typename std::result_of<F(Args...)>::type>;

private:
    ThreadPool(size_t threads);           // once per hardware_concurrency()
    std::vector<std::thread> workers;
    std::queue<std::function<void()>> tasks;
    std::mutex queue_mutex;
    std::condition_variable condition;
    bool stop;
};
```


The worker loop uses `for(;;)` — threads block on the condition variable and consume zero CPU while idle. Dispatch costs drop from tens of microseconds (syscall) to a few microseconds (lock-notify).


### Knowing When Not to Parallelize


A ThreadPool alone isn't enough. You still need to decide *whether* to parallelize a given layer at all. Rocket-Lib uses a FLOP-count heuristic:


```cpp
// layer.cpp — DenseLayer::forward dispatch
int num_threads = std::thread::hardware_concurrency();
int total_work = input.rows * weights.cols * weights.rows;

if (total_work > 100000 && num_threads > 1) {
    std::vector<std::future<void>> futures;
    int chunk_size = (input.rows + num_threads - 1) / num_threads;
    for (int t = 0; t < num_threads; ++t) {
        int start = t * chunk_size;
        int end = std::min(start + chunk_size, input.rows);
        if (start >= end) break;
        futures.push_back(ThreadPool::getInstance().enqueue([this, &input, start, end]() {
            for (int i = start; i < end; ++i) {
                // ... cache-aware kernel ...
            }
        }));
    }
    for (auto& f : futures) f.wait();
}
```


The forward pass threshold is **100,000 FLOPs**. The backward pass uses **50,000 FLOPs** for weight gradients (partitioned by input column to avoid false sharing) and **100,000 FLOPs** for input gradients.


---


## The DAG Execution Model: Zero Reallocation


Rocket-Lib supports directed acyclic graph execution for ResNet-style architectures. To eliminate heap churn, the engine uses **Static Memory Execution** — pre-allocating state maps outside the training loops.


```cpp
// model.cpp — Optimized Training Pass
// Maps declared OUTSIDE the epoch loop to eliminate reallocation churn
std::unordered_map<Layer*, Tensor> layer_outputs;
std::unordered_map<Layer*, Tensor> layer_grads;

for (int epoch = 0; epoch < epochs; ++epoch) {
    layer_outputs.clear(); // reuse, don't reallocate
    layer_grads.clear();
    
    for (/* each batch */) {
        for (Layer* layer : topological_order) {
            const auto& prevs = prev_layers_map[layer];
            if (prevs.size() == 1) {
                layer_outputs[layer] = layer->forward(layer_outputs[prevs[0]]);
            } else {
                // ... merge logic ...
            }
        }
    }
}
```


---


## The Benchmarks


*   **Keras (TF CPU):** 3.1289 s
*   **Rocket-Lib:** 0.2789 s
*   **Speedup:** **11.22×**


*   **Keras Accuracy:** 96.55%
*   **Rocket-Lib Accuracy:** 97.25%


---


## Remaining Technical Debt


**Move constructor is deleted.** All tensor transfers perform deep copies. Restoring the move constructor would eliminate the temporary allocation and the `memcpy` currently required in `ActivationLayer`:


```cpp
// layer.cpp — ActivationLayer::forward
Tensor result = activation_fn->forward(input);
std::memcpy(output.data, result.data, output.rows * output.cols * sizeof(double));
```


**Double initialization.** The `DenseLayer` constructor sweeps the weight matrix twice — once for allocation and once for Xavier initialization. Consolidating these into a single pass is the next logical step.


---


Fix the architecture. The math takes care of itself.


---


*Rocket-Lib is available at github.com/shrihari-s/Rocket-Lib. Run `sh tests.sh` to reproduce all benchmarks.*


**Tags:** `C++` `Performance` `Multithreading` `Neural Networks` `Systems Programming` `Cache Optimization`
