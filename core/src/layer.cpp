#include "layer.h"
#include "activation.h"
#include "optimizer.h"
#include "threadpool.h"
#include <algorithm>
#include <cmath>
#include <random>
#include <cstring>
#include <thread>
#include <future>

// InputLayer
const Tensor &InputLayer::forward(const Tensor &input) {
  ensure_output_dims(input.rows, input.cols);
  for (int i = 0; i < input.rows * input.cols; ++i) {
    output.data[i] = input.data[i];
  }
  return output;
}

const Tensor &InputLayer::backward(const Tensor &input,
                                   const Tensor &grad_output) {
  ensure_grad_input_dims(grad_output.rows, grad_output.cols);
  for (int i = 0; i < grad_output.rows * grad_output.cols; ++i) {
    grad_input.data[i] = grad_output.data[i];
  }
  return grad_input;
}

DenseLayer::DenseLayer(int input_dim, int output_dim) {
  weights = Tensor(input_dim, output_dim);
  biases = Tensor(1, output_dim);
  grad_weights = Tensor(input_dim, output_dim);
  grad_biases = Tensor(1, output_dim);

  for (int i = 0; i < output_dim; ++i) {
    biases.data[i] = 0.0;
    grad_biases.data[i] = 0.0;
  }

  // Glorot/Xavier Uniform Initialization
  double limit = std::sqrt(6.0 / (input_dim + output_dim));
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(-limit, limit);

  for (int i = 0; i < input_dim * output_dim; ++i) {
    weights.data[i] = dis(gen);
    grad_weights.data[i] = 0.0;
  }
}

const Tensor &DenseLayer::forward(const Tensor &input) {
  ensure_output_dims(input.rows, weights.cols);
  
  int num_threads = std::thread::hardware_concurrency();
  int total_work = input.rows * weights.cols * weights.rows;
  
  // Only parallelize for significant workloads
  if (total_work > 100000 && num_threads > 1) {
    std::vector<std::future<void>> futures;
    int chunk_size = (input.rows + num_threads - 1) / num_threads;

    for (int t = 0; t < num_threads; ++t) {
      int start = t * chunk_size;
      int end = std::min(start + chunk_size, input.rows);
      if (start >= end) break;

      futures.push_back(ThreadPool::getInstance().enqueue([this, &input, start, end]() {
        for (int i = start; i < end; ++i) {
          for (int j = 0; j < weights.cols; ++j) {
            output.data[i * weights.cols + j] = biases.data[j];
          }
          for (int k = 0; k < weights.rows; ++k) {
            double in_val = input.data[i * input.cols + k];
            const double* weight_row = &weights.data[k * weights.cols];
            double* out_row = &output.data[i * weights.cols];
            for (int j = 0; j < weights.cols; ++j) {
              out_row[j] += in_val * weight_row[j];
            }
          }
        }
      }));
    }
    for (auto &f : futures) f.wait();
  } else {
    // Single-threaded optimized version
    for (int i = 0; i < input.rows; ++i) {
      for (int j = 0; j < weights.cols; ++j) {
        output.data[i * weights.cols + j] = biases.data[j];
      }
      for (int k = 0; k < weights.rows; ++k) {
        double in_val = input.data[i * input.cols + k];
        const double* weight_row = &weights.data[k * weights.cols];
        double* out_row = &output.data[i * weights.cols];
        for (int j = 0; j < weights.cols; ++j) {
          out_row[j] += in_val * weight_row[j];
        }
      }
    }
  }
  return output;
}

const Tensor &DenseLayer::backward(const Tensor &input,
                                   const Tensor &grad_output) {
  ensure_grad_input_dims(input.rows, input.cols);
  int num_threads = std::thread::hardware_concurrency();

  for (int i = 0; i < grad_weights.rows * grad_weights.cols; ++i)
    grad_weights.data[i] = 0.0;
  for (int i = 0; i < grad_biases.rows * grad_biases.cols; ++i)
    grad_biases.data[i] = 0.0;

  // Parallelize weight gradient calculation
  int total_work_w = input.cols * input.rows * grad_output.cols;
  if (total_work_w > 50000 && num_threads > 1) {
    std::vector<std::future<void>> futures_w;
    int chunk_w = (input.cols + num_threads - 1) / num_threads;

    for (int t = 0; t < num_threads; ++t) {
      int start = t * chunk_w;
      int end = std::min(start + chunk_w, input.cols);
      if (start >= end) break;

      futures_w.push_back(ThreadPool::getInstance().enqueue([this, &input, &grad_output, start, end]() {
        for (int k = start; k < end; ++k) {
          for (int i = 0; i < input.rows; ++i) {
            double in_val = input.data[i * input.cols + k];
            for (int j = 0; j < grad_output.cols; ++j) {
              grad_weights.data[k * grad_weights.cols + j] +=
                  in_val * grad_output.data[i * grad_output.cols + j];
            }
          }
        }
      }));
    }
    for (auto &f : futures_w) f.wait();
  } else {
    for (int k = 0; k < input.cols; ++k) {
      for (int i = 0; i < input.rows; ++i) {
        double in_val = input.data[i * input.cols + k];
        for (int j = 0; j < grad_output.cols; ++j) {
          grad_weights.data[k * grad_weights.cols + j] += in_val * grad_output.data[i * grad_output.cols + j];
        }
      }
    }
  }

  for (int i = 0; i < grad_output.rows; ++i) {
    for (int j = 0; j < grad_output.cols; ++j) {
      grad_biases.data[j] += grad_output.data[i * grad_output.cols + j];
    }
  }

  // grad_input = grad_output * weights^T
  int total_work_in = grad_output.rows * weights.rows * weights.cols;
  if (total_work_in > 100000 && num_threads > 1) {
    std::vector<std::future<void>> futures_in;
    int chunk_in = (grad_output.rows + num_threads - 1) / num_threads;

    for (int t = 0; t < num_threads; ++t) {
      int start = t * chunk_in;
      int end = std::min(start + chunk_in, grad_output.rows);
      if (start >= end) break;

      futures_in.push_back(ThreadPool::getInstance().enqueue([this, &grad_output, start, end]() {
        for (int i = start; i < end; ++i) {
          for (int j = 0; j < weights.rows; ++j) {
            double sum = 0.0;
            const double* grad_out_row = &grad_output.data[i * grad_output.cols];
            const double* weight_row = &weights.data[j * weights.cols];
            for (int k = 0; k < weights.cols; ++k) {
              sum += grad_out_row[k] * weight_row[k];
            }
            grad_input.data[i * weights.rows + j] = sum;
          }
        }
      }));
    }
    for (auto &f : futures_in) f.wait();
  } else {
    for (int i = 0; i < grad_output.rows; ++i) {
      for (int j = 0; j < weights.rows; ++j) {
        double sum = 0.0;
        for (int k = 0; k < weights.cols; ++k) {
          sum += grad_output.data[i * grad_output.cols + k] * weights.data[j * weights.cols + k];
        }
        grad_input.data[i * weights.rows + j] = sum;
      }
    }
  }

  return grad_input;
}

void DenseLayer::update(Optimizer *opt) {
  if (opt) {
    opt->update(weights, grad_weights);
    opt->update(biases, grad_biases);
  }
}

DropoutLayer::DropoutLayer(double rate)
    : rate(std::max(0.0, std::min(1.0, rate))), is_training(true) {}

void DropoutLayer::set_training(bool mode) { is_training = mode; }

const Tensor &DropoutLayer::forward(const Tensor &input) {
  ensure_output_dims(input.rows, input.cols);
  if (!is_training || rate >= 1.0) {
    for (int i = 0; i < input.rows * input.cols; ++i)
      output.data[i] = input.data[i];
    return output;
  }

  if (mask.rows != input.rows || mask.cols != input.cols) {
    mask = Tensor(input.rows, input.cols);
  }

  double scale = 1.0 / (1.0 - rate);

  static std::mt19937 gen([]() {
    const char *seed_str = std::getenv("ROCKET_SEED");
    return seed_str ? std::stoul(seed_str) : 42;
  }());
  std::uniform_real_distribution<> dis(0.0, 1.0);

  for (int i = 0; i < input.rows * input.cols; ++i) {
    double rand_val = dis(gen);
    if (rand_val > rate) {
      mask.data[i] = 1.0;
      output.data[i] = input.data[i] * scale;
    } else {
      mask.data[i] = 0.0;
      output.data[i] = 0.0;
    }
  }
  return output;
}

const Tensor &DropoutLayer::backward(const Tensor &input,
                                     const Tensor &grad_output) {
  ensure_grad_input_dims(grad_output.rows, grad_output.cols);
  if (!is_training || rate >= 1.0) {
    for (int i = 0; i < grad_output.rows * grad_output.cols; ++i) {
      grad_input.data[i] = grad_output.data[i];
    }
    return grad_input;
  }

  double scale = 1.0 / (1.0 - rate);
  for (int i = 0; i < grad_output.rows * grad_output.cols; ++i) {
    grad_input.data[i] = grad_output.data[i] * mask.data[i] * scale;
  }
  return grad_input;
}

RegularizationLayer::RegularizationLayer(double lambda, int type)
    : lambda(lambda), type(type) {}

const Tensor &RegularizationLayer::forward(const Tensor &input) {
  ensure_output_dims(input.rows, input.cols);
  for (int i = 0; i < input.rows * input.cols; ++i) {
    output.data[i] = input.data[i];
  }
  return output;
}

const Tensor &RegularizationLayer::backward(const Tensor &input,
                                            const Tensor &grad_output) {
  ensure_grad_input_dims(grad_output.rows, grad_output.cols);
  for (int i = 0; i < grad_output.rows * grad_output.cols; ++i) {
    double penalty = 0.0;
    if (type == 1) {
      penalty =
          lambda *
          ((input.data[i] > 0) ? 1.0 : ((input.data[i] < 0) ? -1.0 : 0.0));
    } else if (type == 2) {
      penalty = 2.0 * lambda * input.data[i] / grad_output.rows;
    }
    grad_input.data[i] = grad_output.data[i] + penalty;
  }
  return grad_input;
}

ActivationLayer::ActivationLayer(Activation *fn) : activation_fn(fn) {}

ActivationLayer::~ActivationLayer() {
  // Pybind11 manages the Python-created Activation object memory
}

const Tensor &ActivationLayer::forward(const Tensor &input) {
  ensure_output_dims(input.rows, input.cols);
  Tensor result = activation_fn->forward(input);
  std::memcpy(output.data, result.data, output.rows * output.cols * sizeof(double));
  return output;
}

const Tensor &ActivationLayer::backward(const Tensor &input,
                                        const Tensor &grad_output) {
  ensure_grad_input_dims(input.rows, input.cols);
  Tensor result = activation_fn->backward(input, grad_output);
  std::memcpy(grad_input.data, result.data, grad_input.rows * grad_input.cols * sizeof(double));
  return grad_input;
}
