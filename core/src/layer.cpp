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

DenseLayer::DenseLayer(int input_dim, int output_dim)
    : weights(input_dim, output_dim), biases(1, output_dim),
      grad_weights(input_dim, output_dim), grad_biases(1, output_dim) {
  weights.init_params();
}

const Tensor &DenseLayer::forward(const Tensor &input) {
  ensure_output_dims(input.rows, weights.cols);
  output = input * weights;
  for (int i = 0; i < output.rows; ++i) {
    for (int j = 0; j < output.cols; ++j) {
      output.data[i * output.cols + j] += biases.data[j];
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
            scalar in_val = input.data[i * input.cols + k];
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
        scalar in_val = input.data[i * input.cols + k];
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
            scalar sum = 0.0;
            const scalar* grad_out_row = &grad_output.data[i * grad_output.cols];
            const scalar* weight_row = &weights.data[j * weights.cols];
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
        scalar sum = 0.0;
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

void DenseLayer::save(std::ostream& os) const {
  weights.save(os);
  biases.save(os);
}

void DenseLayer::load(std::istream& is) {
  weights.load(is);
  biases.load(is);
}

DropoutLayer::DropoutLayer(scalar rate) : rate(rate), is_training(true) {}


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

  scalar scale = 1.0f / (1.0f - rate);

  static std::mt19937 gen([]() {
    const char *seed_str = std::getenv("ROCKET_SEED");
    return seed_str ? std::stoul(seed_str) : 42;
  }());
  std::uniform_real_distribution<scalar> dis(0.0f, 1.0f);

  for (int i = 0; i < input.rows * input.cols; ++i) {
    scalar rand_val = dis(gen);
    if (rand_val > rate) {
      mask.data[i] = 1.0f;
      output.data[i] = input.data[i] * scale;
    } else {
      mask.data[i] = 0.0f;
      output.data[i] = 0.0f;
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

  scalar scale = 1.0f / (1.0f - rate);
  for (int i = 0; i < grad_output.rows * grad_output.cols; ++i) {
    grad_input.data[i] = grad_output.data[i] * mask.data[i] * scale;
  }
  return grad_input;
}

RegularizationLayer::RegularizationLayer(scalar lambda, int type)
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
    scalar penalty = 0.0f;
    if (type == 1) {
      penalty =
          lambda *
          ((input.data[i] > 0.0f) ? 1.0f : ((input.data[i] < 0.0f) ? -1.0f : 0.0f));
    } else if (type == 2) {
      penalty = 2.0f * lambda * input.data[i] / grad_output.rows;
    }
    grad_input.data[i] = grad_output.data[i] + penalty;
  }
  return grad_input;
}

ActivationLayer::ActivationLayer(Activation *fn) : activation_fn(fn) {}

ActivationLayer::~ActivationLayer() {}

const Tensor &ActivationLayer::forward(const Tensor &input) {
  output = activation_fn->forward(input);
  return output;
}

const Tensor &ActivationLayer::backward(const Tensor &input,
                                        const Tensor &grad_output) {
  grad_input = activation_fn->backward(input, grad_output);
  return grad_input;
}

RNNLayer::RNNLayer(int input_dim, int hidden_dim, int seq_len, bool return_sequences)
    : input_dim(input_dim), hidden_dim(hidden_dim), seq_len(seq_len), return_sequences(return_sequences),
      weights_ih(input_dim, hidden_dim), weights_hh(hidden_dim, hidden_dim), biases(1, hidden_dim),
      grad_weights_ih(input_dim, hidden_dim), grad_weights_hh(hidden_dim, hidden_dim), grad_biases(1, hidden_dim) {
  weights_ih.init_params();
  weights_hh.init_params();
}

const Tensor &RNNLayer::forward(const Tensor &input) {
  int batch_size = input.rows / seq_len;
  if (h_states.rows != batch_size * seq_len || h_states.cols != hidden_dim) {
    h_states = Tensor(batch_size * seq_len, hidden_dim);
  }
  ensure_output_dims(return_sequences ? batch_size * seq_len : batch_size,
                     hidden_dim);

  // Precompute W_ih * X for all time steps
  Tensor x_ih = input * weights_ih;

  Tensor h_prev_batch(batch_size, hidden_dim);
  for(int i=0; i<batch_size*hidden_dim; ++i) h_prev_batch.data[i] = 0;

  for (int s = 0; s < seq_len; ++s) {
    Tensor gates_hh = h_prev_batch * weights_hh;

    ThreadPool::getInstance().parallel_for(0, batch_size, [this, &x_ih, &gates_hh, &h_prev_batch, s, batch_size](int b) {
      int row_idx = b * seq_len + s;
      int x_ih_base = row_idx * hidden_dim;
      int hh_base = b * hidden_dim;

      for (int h = 0; h < hidden_dim; ++h) {
        scalar val = biases.data[h] + x_ih.data[x_ih_base + h] + gates_hh.data[hh_base + h];
        scalar act = std::tanh(val);
        h_states.data[row_idx * hidden_dim + h] = act;
        h_prev_batch.data[b * hidden_dim + h] = act;

        if (return_sequences) {
          output.data[row_idx * hidden_dim + h] = act;
        } else if (s == seq_len - 1) {
          output.data[b * hidden_dim + h] = act;
        }
      }
    });
  }

  return output;
}

const Tensor &RNNLayer::backward(const Tensor &input, const Tensor &grad_output) {
  int batch_size = input.rows / seq_len;
  ensure_grad_input_dims(batch_size * seq_len, input_dim);
  
  std::memset(grad_weights_ih.data, 0, grad_weights_ih.rows * grad_weights_ih.cols * sizeof(scalar));
  std::memset(grad_weights_hh.data, 0, grad_weights_hh.rows * grad_weights_hh.cols * sizeof(scalar));
  std::memset(grad_biases.data, 0, grad_biases.rows * grad_biases.cols * sizeof(scalar));
  std::memset(grad_input.data, 0, grad_input.rows * grad_input.cols * sizeof(scalar));

  int num_threads = std::thread::hardware_concurrency();
  std::vector<std::future<void>> futures;
  int chunk_size = (batch_size + num_threads - 1) / num_threads;
  
  Tensor d_tanh_all(batch_size * seq_len, hidden_dim);

  for (int t = 0; t < num_threads; ++t) {
    int start_b = t * chunk_size;
    int end_b = std::min(start_b + chunk_size, batch_size);
    if (start_b >= end_b) break;

    futures.push_back(ThreadPool::getInstance().enqueue([this, &grad_output, start_b, end_b, &d_tanh_all]() {
      for (int b = start_b; b < end_b; ++b) {
        std::vector<scalar> dh_next(hidden_dim, 0.0f);
        for (int s = seq_len - 1; s >= 0; --s) {
          int row_idx = b * seq_len + s;
          for (int h = 0; h < hidden_dim; ++h) {
            scalar dh = dh_next[h];
            if (return_sequences) dh += grad_output.data[row_idx * hidden_dim + h];
            else if (s == seq_len - 1) dh += grad_output.data[b * hidden_dim + h];
            
            scalar h_val = h_states.data[row_idx * hidden_dim + h];
            scalar dtanh = dh * (1.0f - h_val * h_val);
            d_tanh_all.data[row_idx * hidden_dim + h] = dtanh;
          }
          for (int prev_h = 0; prev_h < hidden_dim; ++prev_h) {
            scalar dnext = 0.0f;
            for (int h = 0; h < hidden_dim; ++h) {
              dnext += d_tanh_all.data[row_idx * hidden_dim + h] * weights_hh.data[prev_h * hidden_dim + h];
            }
            dh_next[prev_h] = dnext;
          }
        }
      }
    }));
  }
  for (auto &f : futures) f.wait();

  ThreadPool::getInstance().parallel_for(0, input_dim, [this, &input, &d_tanh_all, batch_size](int i) {
    scalar *grad_row = &grad_weights_ih.data[i * hidden_dim];
    for (int r = 0; r < batch_size * seq_len; ++r) {
      scalar in_val = input.data[r * input_dim + i];
      if (in_val == 0) continue;
      const scalar *d_tanh_row = &d_tanh_all.data[r * hidden_dim];
      for (int h = 0; h < hidden_dim; ++h) {
        grad_row[h] += in_val * d_tanh_row[h];
      }
    }
  });

  ThreadPool::getInstance().parallel_for(0, hidden_dim, [this, &d_tanh_all, batch_size](int prev_h_idx) {
    scalar *grad_row = &grad_weights_hh.data[prev_h_idx * hidden_dim];
    for (int b = 0; b < batch_size; ++b) {
      int base = b * seq_len;
      for (int s = 1; s < seq_len; ++s) {
        scalar h_prev_val = h_states.data[(base + s - 1) * hidden_dim + prev_h_idx];
        if (h_prev_val == 0) continue;
        const scalar *d_tanh_row = &d_tanh_all.data[(base + s) * hidden_dim];
        for (int h = 0; h < hidden_dim; ++h) {
          grad_row[h] += h_prev_val * d_tanh_row[h];
        }
      }
    }
  });

  ThreadPool::getInstance().parallel_for(0, batch_size * seq_len, [this, &d_tanh_all](int r) {
    for (int i = 0; i < input_dim; ++i) {
      scalar sum = 0;
      for (int h = 0; h < hidden_dim; ++h) {
        sum += d_tanh_all.data[r * hidden_dim + h] * weights_ih.data[i * hidden_dim + h];
      }
      grad_input.data[r * input_dim + i] = sum;
    }
  });

  for (int r = 0; r < batch_size * seq_len; ++r) {
    for (int h = 0; h < hidden_dim; ++h) {
      grad_biases.data[h] += (scalar)d_tanh_all.data[r * hidden_dim + h];
    }
  }

  return grad_input;
}

void RNNLayer::update(Optimizer *opt) {
  if (opt) {
    opt->update(weights_ih, grad_weights_ih);
    opt->update(weights_hh, grad_weights_hh);
    opt->update(biases, grad_biases);
  }
}

void RNNLayer::save(std::ostream& os) const {
  weights_ih.save(os);
  weights_hh.save(os);
  biases.save(os);
}

void RNNLayer::load(std::istream& is) {
  weights_ih.load(is);
  weights_hh.load(is);
  biases.load(is);
}

LSTMLayer::LSTMLayer(int input_dim, int hidden_dim, int seq_len, bool return_sequences)
    : input_dim(input_dim), hidden_dim(hidden_dim), seq_len(seq_len), return_sequences(return_sequences),
      weights_ih(input_dim, 4 * hidden_dim), 
      weights_hh(hidden_dim, 4 * hidden_dim), biases(1, 4 * hidden_dim),
      grad_weights_ih(input_dim, 4 * hidden_dim),
      grad_weights_hh(hidden_dim, 4 * hidden_dim),
      grad_biases(1, 4 * hidden_dim) {
  // Glorot uniform per gate, matching Keras defaults
  scalar limit_ih = std::sqrt(6.0f / (input_dim + hidden_dim));
  scalar limit_hh = std::sqrt(6.0f / (hidden_dim + hidden_dim));
  static std::mt19937 gen(42);
  std::uniform_real_distribution<scalar> dis_ih(-limit_ih, limit_ih);
  std::uniform_real_distribution<scalar> dis_hh(-limit_hh, limit_hh);
  for (int i = 0; i < input_dim * 4 * hidden_dim; ++i)
    weights_ih.data[i] = dis_ih(gen);
  for (int i = 0; i < hidden_dim * 4 * hidden_dim; ++i)
    weights_hh.data[i] = dis_hh(gen);
  // Forget gate bias = 1.0 (Keras unit_forget_bias default, gate order: i,f,o,g)
  for (int h = hidden_dim; h < 2 * hidden_dim; ++h)
    biases.data[h] = 1.0f;
}
static inline scalar lstm_sigmoid(scalar x) {
  return 1.0f / (1.0f + std::exp(-x));
}

const Tensor &LSTMLayer::forward(const Tensor &input) {
  int batch_size = input.rows / seq_len;
  if (h_states.rows != batch_size * seq_len || h_states.cols != hidden_dim) {
    h_states = Tensor(batch_size * seq_len, hidden_dim);
    c_states = Tensor(batch_size * seq_len, hidden_dim);
    gates = Tensor(batch_size * seq_len, 4 * hidden_dim);
  }
  
  if (workspace_gates_hh.rows != batch_size || workspace_gates_hh.cols != 4 * hidden_dim) {
      workspace_gates_hh = Tensor(batch_size, 4 * hidden_dim);
      workspace_h_prev = Tensor(batch_size, hidden_dim);
      workspace_c_prev = Tensor(batch_size, hidden_dim);
  }
  
  std::memset(workspace_h_prev.data, 0, batch_size * hidden_dim * sizeof(scalar));
  std::memset(workspace_c_prev.data, 0, batch_size * hidden_dim * sizeof(scalar));

  ensure_output_dims(return_sequences ? batch_size * seq_len : batch_size, hidden_dim);

  Tensor x_ih = input * weights_ih;

  for (int s = 0; s < seq_len; ++s) {
        std::memset(workspace_gates_hh.data, 0, batch_size * 4 * hidden_dim * sizeof(scalar));

        ThreadPool::getInstance().parallel_for(0, batch_size, [this](int b) {
            int h_dim = this->hidden_dim;
            scalar* g_row = &workspace_gates_hh.data[b * 4 * h_dim];
            for (int i = 0; i < h_dim; ++i) {
                scalar h_prev_val = workspace_h_prev.data[b * h_dim + i];
                if (h_prev_val == 0) continue;
                const scalar* w_row = &weights_hh.data[i * 4 * h_dim];
                for (int j = 0; j < 4 * h_dim; ++j) {
                    g_row[j] += h_prev_val * w_row[j];
                }
            }
        });

    ThreadPool::getInstance().parallel_for(0, batch_size, [this, &x_ih, s, batch_size](int b) {
      int row_idx = b * seq_len + s;
      int gate_base = row_idx * 4 * hidden_dim;
      int x_ih_base = row_idx * 4 * hidden_dim;
      int hh_base = b * 4 * hidden_dim;

      for (int h = 0; h < 4 * hidden_dim; ++h) {
        scalar val = biases.data[h] + x_ih.data[x_ih_base + h] + workspace_gates_hh.data[hh_base + h];
        if (h < 3 * hidden_dim) {
          gates.data[gate_base + h] = lstm_sigmoid(val);
        } else {
          gates.data[gate_base + h] = std::tanh(val);
        }
      }

      for (int h = 0; h < hidden_dim; ++h) {
        scalar i_gate = gates.data[gate_base + h];
        scalar f_gate = gates.data[gate_base + hidden_dim + h];
        scalar o_gate = gates.data[gate_base + 2 * hidden_dim + h];
        scalar g_gate = gates.data[gate_base + 3 * hidden_dim + h];

        scalar c = f_gate * workspace_c_prev.data[b * hidden_dim + h] + i_gate * g_gate;
        c_states.data[row_idx * hidden_dim + h] = c;
        workspace_c_prev.data[b * hidden_dim + h] = c;

        scalar tanh_c = std::tanh(c);
        scalar h_out = o_gate * tanh_c;
        
        h_states.data[row_idx * hidden_dim + h] = h_out;
        workspace_h_prev.data[b * hidden_dim + h] = h_out;

        if (return_sequences) {
          output.data[row_idx * hidden_dim + h] = h_out;
        } else if (s == seq_len - 1) {
          output.data[b * hidden_dim + h] = h_out;
        }
      }
    });
  }

  return output;
}
const Tensor &LSTMLayer::backward(const Tensor &input, const Tensor &grad_output) {
  int batch_size = input.rows / seq_len;
  ensure_grad_input_dims(batch_size * seq_len, input_dim);
  
  std::memset(grad_weights_ih.data, 0, grad_weights_ih.rows * grad_weights_ih.cols * sizeof(scalar));
  std::memset(grad_weights_hh.data, 0, grad_weights_hh.rows * grad_weights_hh.cols * sizeof(scalar));
  std::memset(grad_biases.data, 0, grad_biases.rows * grad_biases.cols * sizeof(scalar));
  std::memset(grad_input.data, 0, grad_input.rows * grad_input.cols * sizeof(scalar));

  Tensor d_gates_all(batch_size * seq_len, 4 * hidden_dim);
  std::memset(d_gates_all.data, 0, d_gates_all.rows * d_gates_all.cols * sizeof(scalar));

  int num_threads = std::thread::hardware_concurrency();
  std::vector<std::future<void>> futures;
  int chunk_size = (batch_size + num_threads - 1) / num_threads;

  for (int t = 0; t < num_threads; ++t) {
    int start_b = t * chunk_size;
    int end_b = std::min(start_b + chunk_size, batch_size);
    if (start_b >= end_b) break;

    futures.push_back(ThreadPool::getInstance().enqueue([this, &grad_output, start_b, end_b, &d_gates_all]() {
      for (int b = start_b; b < end_b; ++b) {
        std::vector<scalar> dh_next(hidden_dim, 0.0f);
        std::vector<scalar> dc_next(hidden_dim, 0.0f);
        for (int s = seq_len - 1; s >= 0; --s) {
          int row_idx = b * seq_len + s;
          
          for (int h = 0; h < hidden_dim; ++h) {
            scalar dh = dh_next[h];
            if (return_sequences) dh += grad_output.data[row_idx * hidden_dim + h];
            else if (s == seq_len - 1) dh += grad_output.data[b * hidden_dim + h];
            
            scalar c = c_states.data[row_idx * hidden_dim + h];
            scalar prev_c_val = (s > 0) ? c_states.data[(row_idx - 1) * hidden_dim + h] : 0.0f;
            
            scalar i = gates.data[row_idx * 4 * hidden_dim + h];
            scalar f = gates.data[row_idx * 4 * hidden_dim + hidden_dim + h];
            scalar o = gates.data[row_idx * 4 * hidden_dim + 2 * hidden_dim + h];
            scalar g = gates.data[row_idx * 4 * hidden_dim + 3 * hidden_dim + h];
            
            scalar tanh_c = std::tanh(c);
            scalar do_gate = dh * tanh_c;
            scalar dc = dh * o * (1.0f - tanh_c * tanh_c) + dc_next[h];
            
            scalar df_gate = dc * prev_c_val;
            scalar di_gate = dc * g;
            scalar dg_gate = dc * i;
            
            dc_next[h] = dc * f;
            
            d_gates_all.data[row_idx * 4 * hidden_dim + h] = di_gate * i * (1.0f - i);
            d_gates_all.data[row_idx * 4 * hidden_dim + hidden_dim + h] = df_gate * f * (1.0f - f);
            d_gates_all.data[row_idx * 4 * hidden_dim + 2 * hidden_dim + h] = do_gate * o * (1.0f - o);
            d_gates_all.data[row_idx * 4 * hidden_dim + 3 * hidden_dim + h] = dg_gate * (1.0f - g * g);
          }
          
          for (int prev_h = 0; prev_h < hidden_dim; ++prev_h) {
            scalar dnext = 0.0f;
            for (int h = 0; h < 4 * hidden_dim; ++h) {
              dnext += d_gates_all.data[row_idx * 4 * hidden_dim + h] * weights_hh.data[prev_h * 4 * hidden_dim + h];
            }
            dh_next[prev_h] = dnext;
          }
        }
      }
    }));
  }
  for (auto &f : futures) f.wait();

  ThreadPool::getInstance().parallel_for(0, input_dim, [this, &input, &d_gates_all, batch_size](int i) {
    scalar *grad_row = &grad_weights_ih.data[i * 4 * hidden_dim];
    for (int r = 0; r < batch_size * seq_len; ++r) {
      scalar in_val = input.data[r * input_dim + i];
      if (in_val == 0) continue;
      const scalar *d_gates_row = &d_gates_all.data[r * 4 * hidden_dim];
      for (int g = 0; g < 4 * hidden_dim; ++g) {
        grad_row[g] += in_val * d_gates_row[g];
      }
    }
  });

  ThreadPool::getInstance().parallel_for(0, hidden_dim, [this, &d_gates_all, batch_size](int prev_h_idx) {
    scalar *grad_row = &grad_weights_hh.data[prev_h_idx * 4 * hidden_dim];
    for (int b = 0; b < batch_size; ++b) {
      int base = b * seq_len;
      for (int s = 1; s < seq_len; ++s) {
        scalar h_prev_val = h_states.data[(base + s - 1) * hidden_dim + prev_h_idx];
        if (h_prev_val == 0) continue;
        const scalar *d_gates_row = &d_gates_all.data[(base + s) * 4 * hidden_dim];
        for (int g = 0; g < 4 * hidden_dim; ++g) {
          grad_row[g] += h_prev_val * d_gates_row[g];
        }
      }
    }
  });

  ThreadPool::getInstance().parallel_for(0, batch_size * seq_len, [this, &d_gates_all](int r) {
    for (int i = 0; i < input_dim; ++i) {
      scalar sum = 0;
      for (int g = 0; g < 4 * hidden_dim; ++g) {
        sum += d_gates_all.data[r * 4 * hidden_dim + g] * weights_ih.data[i * 4 * hidden_dim + g];
      }
      grad_input.data[r * input_dim + i] = sum;
    }
  });

  // grad_biases
  ThreadPool::getInstance().parallel_for(0, 4 * hidden_dim, [this, &d_gates_all, batch_size](int g) {
    scalar sum = 0;
    for (int r = 0; r < batch_size * seq_len; ++r) {
      sum += d_gates_all.data[r * 4 * hidden_dim + g];
    }
    grad_biases.data[g] = sum;
  });

  return grad_input;
}

void LSTMLayer::update(Optimizer *opt) {
  if (opt) {
    opt->update(weights_ih, grad_weights_ih);
    opt->update(weights_hh, grad_weights_hh);
    opt->update(biases, grad_biases);
  }
}

void LSTMLayer::save(std::ostream& os) const {
  weights_ih.save(os);
  weights_hh.save(os);
  biases.save(os);
}

void LSTMLayer::load(std::istream& is) {
  weights_ih.load(is);
  weights_hh.load(is);
  biases.load(is);
}
