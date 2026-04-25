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
  weights_hh.load(is);
  biases.load(is);
}

// --- LayerNormLayer ---
LayerNormLayer::LayerNormLayer(int feature_dim, scalar epsilon)
    : feature_dim(feature_dim), epsilon(epsilon), gamma(1, feature_dim),
      beta(1, feature_dim), grad_gamma(1, feature_dim), grad_beta(1, feature_dim) {
  for (int i = 0; i < feature_dim; ++i) {
    gamma.data[i] = 1.0f;
    beta.data[i] = 0.0f;
  }
}

const Tensor &LayerNormLayer::forward(const Tensor &input) {
  ensure_output_dims(input.rows, input.cols);
  if (x_centered.rows != input.rows || x_centered.cols != input.cols) {
    x_centered = Tensor(input.rows, input.cols);
    variance = Tensor(input.rows, 1);
    stddev = Tensor(input.rows, 1);
    x_norm = Tensor(input.rows, input.cols);
  }

  ThreadPool::getInstance().parallel_for(0, input.rows, [this, &input](int i) {
    scalar mean = 0.0f;
    for (int j = 0; j < feature_dim; ++j) {
      mean += input.data[i * feature_dim + j];
    }
    mean /= feature_dim;

    scalar var = 0.0f;
    for (int j = 0; j < feature_dim; ++j) {
      scalar diff = input.data[i * feature_dim + j] - mean;
      x_centered.data[i * feature_dim + j] = diff;
      var += diff * diff;
    }
    var /= feature_dim;
    variance.data[i] = var;
    
    scalar std_val = std::sqrt(var + epsilon);
    stddev.data[i] = std_val;

    for (int j = 0; j < feature_dim; ++j) {
      x_norm.data[i * feature_dim + j] = x_centered.data[i * feature_dim + j] / std_val;
      output.data[i * feature_dim + j] = gamma.data[j] * x_norm.data[i * feature_dim + j] + beta.data[j];
    }
  });

  return output;
}

const Tensor &LayerNormLayer::backward(const Tensor &input, const Tensor &grad_output) {
  ensure_grad_input_dims(input.rows, input.cols);
  
  std::memset(grad_gamma.data, 0, grad_gamma.cols * sizeof(scalar));
  std::memset(grad_beta.data, 0, grad_beta.cols * sizeof(scalar));
  
  for (int i = 0; i < input.rows; ++i) {
    for (int j = 0; j < feature_dim; ++j) {
      grad_gamma.data[j] += grad_output.data[i * feature_dim + j] * x_norm.data[i * feature_dim + j];
      grad_beta.data[j] += grad_output.data[i * feature_dim + j];
    }
  }

  ThreadPool::getInstance().parallel_for(0, input.rows, [this, &grad_output](int i) {
    scalar sum_dx_norm = 0;
    scalar sum_dx_norm_x_centered = 0;
    for (int j = 0; j < feature_dim; ++j) {
      scalar dx_norm = grad_output.data[i * feature_dim + j] * gamma.data[j];
      sum_dx_norm += dx_norm;
      sum_dx_norm_x_centered += dx_norm * x_centered.data[i * feature_dim + j];
    }

    scalar m = feature_dim;
    scalar std_val = stddev.data[i];
    scalar var_eps = variance.data[i] + epsilon;

    for (int j = 0; j < feature_dim; ++j) {
      scalar dx_norm = grad_output.data[i * feature_dim + j] * gamma.data[j];
      scalar dx = (m * dx_norm - sum_dx_norm - x_centered.data[i * feature_dim + j] * sum_dx_norm_x_centered / var_eps) / (m * std_val);
      grad_input.data[i * feature_dim + j] = dx;
    }
  });

  return grad_input;
}

void LayerNormLayer::update(Optimizer *opt) {
  if (opt) {
    opt->update(gamma, grad_gamma);
    opt->update(beta, grad_beta);
  }
}

void LayerNormLayer::save(std::ostream& os) const { gamma.save(os); beta.save(os); }
void LayerNormLayer::load(std::istream& is) { gamma.load(is); beta.load(is); }

// --- SelfAttentionLayer ---
SelfAttentionLayer::SelfAttentionLayer(int embed_dim, int seq_len)
    : embed_dim(embed_dim), seq_len(seq_len),
      W_q(embed_dim, embed_dim), W_k(embed_dim, embed_dim), W_v(embed_dim, embed_dim),
      b_q(1, embed_dim), b_k(1, embed_dim), b_v(1, embed_dim),
      grad_W_q(embed_dim, embed_dim), grad_W_k(embed_dim, embed_dim), grad_W_v(embed_dim, embed_dim),
      grad_b_q(1, embed_dim), grad_b_k(1, embed_dim), grad_b_v(1, embed_dim) {
  W_q.init_params(); W_k.init_params(); W_v.init_params();
}

const Tensor &SelfAttentionLayer::forward(const Tensor &input) {
  int batch_size = input.rows / seq_len;
  ensure_output_dims(input.rows, embed_dim);

  Q = input * W_q; K = input * W_k; V = input * W_v;
  
  for(int i=0; i<input.rows; ++i) {
    for(int j=0; j<embed_dim; ++j) {
      Q.data[i*embed_dim+j] += b_q.data[j];
      K.data[i*embed_dim+j] += b_k.data[j];
      V.data[i*embed_dim+j] += b_v.data[j];
    }
  }

  if (scores.rows != batch_size * seq_len || scores.cols != seq_len) {
    scores = Tensor(batch_size * seq_len, seq_len);
    attention_weights = Tensor(batch_size * seq_len, seq_len);
  }
  
  scalar scale = 1.0f / std::sqrt((scalar)embed_dim);

  ThreadPool::getInstance().parallel_for(0, batch_size, [this, scale](int b) {
    for (int i = 0; i < seq_len; ++i) {
      scalar max_score = -1e9f;
      for (int j = 0; j < seq_len; ++j) {
        scalar dot = 0.0f;
        for (int d = 0; d < embed_dim; ++d) {
          dot += Q.data[(b * seq_len + i) * embed_dim + d] * K.data[(b * seq_len + j) * embed_dim + d];
        }
        scalar s = dot * scale;
        scores.data[(b * seq_len + i) * seq_len + j] = s;
        if (s > max_score) max_score = s;
      }
      
      scalar sum_exp = 0.0f;
      for (int j = 0; j < seq_len; ++j) {
        scalar e = std::exp(scores.data[(b * seq_len + i) * seq_len + j] - max_score);
        attention_weights.data[(b * seq_len + i) * seq_len + j] = e;
        sum_exp += e;
      }
      for (int j = 0; j < seq_len; ++j) {
        attention_weights.data[(b * seq_len + i) * seq_len + j] /= sum_exp;
      }
    }
  });

  ThreadPool::getInstance().parallel_for(0, batch_size, [this](int b) {
    for (int i = 0; i < seq_len; ++i) {
      for (int d = 0; d < embed_dim; ++d) {
        scalar sum = 0.0f;
        for (int j = 0; j < seq_len; ++j) {
          sum += attention_weights.data[(b * seq_len + i) * seq_len + j] * V.data[(b * seq_len + j) * embed_dim + d];
        }
        output.data[(b * seq_len + i) * embed_dim + d] = sum;
      }
    }
  });

  return output;
}

const Tensor &SelfAttentionLayer::backward(const Tensor &input, const Tensor &grad_output) {
  int batch_size = input.rows / seq_len;
  ensure_grad_input_dims(input.rows, input.cols);

  Tensor dQ(input.rows, embed_dim);
  Tensor dK(input.rows, embed_dim);
  Tensor dV(input.rows, embed_dim);
  
  std::memset(dQ.data, 0, dQ.rows * dQ.cols * sizeof(scalar));
  std::memset(dK.data, 0, dK.rows * dK.cols * sizeof(scalar));
  std::memset(dV.data, 0, dV.rows * dV.cols * sizeof(scalar));
  
  scalar scale = 1.0f / std::sqrt((scalar)embed_dim);

  ThreadPool::getInstance().parallel_for(0, batch_size, [this, &grad_output, &dQ, &dK, &dV, scale](int b) {
    for (int i = 0; i < seq_len; ++i) {
      for (int j = 0; j < seq_len; ++j) {
        scalar w = attention_weights.data[(b * seq_len + i) * seq_len + j];
        for (int d = 0; d < embed_dim; ++d) {
          dV.data[(b * seq_len + j) * embed_dim + d] += w * grad_output.data[(b * seq_len + i) * embed_dim + d];
        }
      }
    }

    // Precompute per-query scalar: sum_k a_ik * (dout_i · v_k) — O(n²) not O(n³)
    std::vector<scalar> scalar_i(seq_len, 0.0f);
    for (int i = 0; i < seq_len; ++i) {
      for (int k = 0; k < seq_len; ++k) {
        scalar wk = attention_weights.data[(b * seq_len + i) * seq_len + k];
        scalar dot = 0.0f;
        for (int d = 0; d < embed_dim; ++d)
          dot += grad_output.data[(b * seq_len + i) * embed_dim + d] * V.data[(b * seq_len + k) * embed_dim + d];
        scalar_i[i] += wk * dot;
      }
    }

    for (int i = 0; i < seq_len; ++i) {
      for (int j = 0; j < seq_len; ++j) {
        scalar w = attention_weights.data[(b * seq_len + i) * seq_len + j];
        scalar dscore = 0.0f;
        for (int d = 0; d < embed_dim; ++d)
          dscore += grad_output.data[(b * seq_len + i) * embed_dim + d] * V.data[(b * seq_len + j) * embed_dim + d];

        scalar ds = w * (dscore - scalar_i[i]) * scale;
        for (int d = 0; d < embed_dim; ++d) {
          dQ.data[(b * seq_len + i) * embed_dim + d] += ds * K.data[(b * seq_len + j) * embed_dim + d];
          dK.data[(b * seq_len + j) * embed_dim + d] += ds * Q.data[(b * seq_len + i) * embed_dim + d];
        }
      }
    }
  });

  std::memset(grad_W_q.data, 0, grad_W_q.rows * grad_W_q.cols * sizeof(scalar));
  std::memset(grad_W_k.data, 0, grad_W_k.rows * grad_W_k.cols * sizeof(scalar));
  std::memset(grad_W_v.data, 0, grad_W_v.rows * grad_W_v.cols * sizeof(scalar));
  std::memset(grad_b_q.data, 0, grad_b_q.rows * grad_b_q.cols * sizeof(scalar));
  std::memset(grad_b_k.data, 0, grad_b_k.rows * grad_b_k.cols * sizeof(scalar));
  std::memset(grad_b_v.data, 0, grad_b_v.rows * grad_b_v.cols * sizeof(scalar));

  for(int i=0; i<input.rows; ++i) {
    for(int d=0; d<embed_dim; ++d) {
      grad_b_q.data[d] += dQ.data[i*embed_dim+d];
      grad_b_k.data[d] += dK.data[i*embed_dim+d];
      grad_b_v.data[d] += dV.data[i*embed_dim+d];
    }
  }

  for(int d1=0; d1<embed_dim; ++d1) {
    for(int d2=0; d2<embed_dim; ++d2) {
      scalar sq = 0, sk = 0, sv = 0;
      for(int i=0; i<input.rows; ++i) {
        scalar in_val = input.data[i*embed_dim + d1];
        sq += in_val * dQ.data[i*embed_dim + d2];
        sk += in_val * dK.data[i*embed_dim + d2];
        sv += in_val * dV.data[i*embed_dim + d2];
      }
      grad_W_q.data[d1*embed_dim+d2] = sq;
      grad_W_k.data[d1*embed_dim+d2] = sk;
      grad_W_v.data[d1*embed_dim+d2] = sv;
    }
  }

  for(int i=0; i<input.rows; ++i) {
    for(int d=0; d<embed_dim; ++d) {
      scalar sum = 0;
      for(int d2=0; d2<embed_dim; ++d2) {
        sum += dQ.data[i*embed_dim+d2] * W_q.data[d*embed_dim+d2] +
               dK.data[i*embed_dim+d2] * W_k.data[d*embed_dim+d2] +
               dV.data[i*embed_dim+d2] * W_v.data[d*embed_dim+d2];
      }
      grad_input.data[i*embed_dim+d] = sum;
    }
  }

  return grad_input;
}

void SelfAttentionLayer::update(Optimizer *opt) {
  if (opt) {
    opt->update(W_q, grad_W_q); opt->update(b_q, grad_b_q);
    opt->update(W_k, grad_W_k); opt->update(b_k, grad_b_k);
    opt->update(W_v, grad_W_v); opt->update(b_v, grad_b_v);
  }
}
void SelfAttentionLayer::save(std::ostream& os) const { W_q.save(os); W_k.save(os); W_v.save(os); b_q.save(os); b_k.save(os); b_v.save(os); }
void SelfAttentionLayer::load(std::istream& is) { W_q.load(is); W_k.load(is); W_v.load(is); b_q.load(is); b_k.load(is); b_v.load(is); }

// --- TransformerEncoderLayer ---
static ReLU relu_fn_inst;

TransformerEncoderLayer::TransformerEncoderLayer(int embed_dim, int seq_len, int ff_dim)
    : embed_dim(embed_dim), seq_len(seq_len), ff_dim(ff_dim == -1 ? 4 * embed_dim : ff_dim),
      attention(embed_dim, seq_len), norm1(embed_dim),
      ff1(embed_dim, this->ff_dim), relu(&relu_fn_inst),
      ff2(this->ff_dim, embed_dim), norm2(embed_dim) {
}

const Tensor &TransformerEncoderLayer::forward(const Tensor &input) {
  ensure_output_dims(input.rows, input.cols);

  // Sublayer 1: Self-Attention + Residual + LayerNorm
  att_output = attention.forward(input);
  Tensor add1 = input + att_output;
  norm1_output = norm1.forward(add1);

  // Sublayer 2: FFN + Residual + LayerNorm
  ff1_output = ff1.forward(norm1_output);
  relu_output = relu.forward(ff1_output);
  ff2_output = ff2.forward(relu_output);
  
  Tensor add2 = norm1_output + ff2_output;
  output = norm2.forward(add2);

  return output;
}

const Tensor &TransformerEncoderLayer::backward(const Tensor &input, const Tensor &grad_output) {
  ensure_grad_input_dims(input.rows, input.cols);

  Tensor grad_add2 = norm2.backward(norm1_output + ff2_output, grad_output);
  
  Tensor grad_ff2_out = grad_add2; // branch 2
  Tensor grad_norm1_out_1 = grad_add2; // branch 1

  Tensor grad_relu_out = ff2.backward(relu_output, grad_ff2_out);
  Tensor grad_ff1_out = relu.backward(ff1_output, grad_relu_out);
  Tensor grad_norm1_in = ff1.backward(norm1_output, grad_ff1_out);

  Tensor grad_norm1_out = grad_norm1_out_1 + grad_norm1_in;

  Tensor grad_add1 = norm1.backward(input + att_output, grad_norm1_out);

  Tensor grad_att_out = grad_add1;
  Tensor grad_att_in = attention.backward(input, grad_att_out);

  grad_input = grad_add1 + grad_att_in;

  return grad_input;
}

void TransformerEncoderLayer::set_training(bool training) {
  attention.set_training(training);
  norm1.set_training(training);
  ff1.set_training(training);
  relu.set_training(training);
  ff2.set_training(training);
  norm2.set_training(training);
}

void TransformerEncoderLayer::update(Optimizer *opt) {
  attention.update(opt);
  norm1.update(opt);
  ff1.update(opt);
  ff2.update(opt);
  norm2.update(opt);
}

void TransformerEncoderLayer::save(std::ostream& os) const { attention.save(os); norm1.save(os); ff1.save(os); ff2.save(os); norm2.save(os); }
void TransformerEncoderLayer::load(std::istream& is) { attention.load(is); norm1.load(is); ff1.load(is); ff2.load(is); norm2.load(is); }

// --- MaskedSelfAttentionLayer ---
MaskedSelfAttentionLayer::MaskedSelfAttentionLayer(int embed_dim, int seq_len)
    : embed_dim(embed_dim), seq_len(seq_len),
      W_q(embed_dim, embed_dim), W_k(embed_dim, embed_dim), W_v(embed_dim, embed_dim),
      b_q(1, embed_dim), b_k(1, embed_dim), b_v(1, embed_dim),
      grad_W_q(embed_dim, embed_dim), grad_W_k(embed_dim, embed_dim), grad_W_v(embed_dim, embed_dim),
      grad_b_q(1, embed_dim), grad_b_k(1, embed_dim), grad_b_v(1, embed_dim) {
  W_q.init_params(); W_k.init_params(); W_v.init_params();
}

const Tensor &MaskedSelfAttentionLayer::forward(const Tensor &input) {
  int batch_size = input.rows / seq_len;
  ensure_output_dims(input.rows, embed_dim);

  Q = input * W_q; K = input * W_k; V = input * W_v;
  
  for(int i=0; i<input.rows; ++i) {
    for(int j=0; j<embed_dim; ++j) {
      Q.data[i*embed_dim+j] += b_q.data[j];
      K.data[i*embed_dim+j] += b_k.data[j];
      V.data[i*embed_dim+j] += b_v.data[j];
    }
  }

  if (scores.rows != batch_size * seq_len || scores.cols != seq_len) {
    scores = Tensor(batch_size * seq_len, seq_len);
    attention_weights = Tensor(batch_size * seq_len, seq_len);
  }
  
  scalar scale = 1.0f / std::sqrt((scalar)embed_dim);

  ThreadPool::getInstance().parallel_for(0, batch_size, [this, scale](int b) {
    for (int i = 0; i < seq_len; ++i) {
      scalar max_score = -1e9f;
      for (int j = 0; j < seq_len; ++j) {
        if (j > i) {
          scores.data[(b * seq_len + i) * seq_len + j] = -1e9f; // causal mask
          continue;
        }
        scalar dot = 0.0f;
        for (int d = 0; d < embed_dim; ++d) {
          dot += Q.data[(b * seq_len + i) * embed_dim + d] * K.data[(b * seq_len + j) * embed_dim + d];
        }
        scalar s = dot * scale;
        scores.data[(b * seq_len + i) * seq_len + j] = s;
        if (s > max_score) max_score = s;
      }
      
      scalar sum_exp = 0.0f;
      for (int j = 0; j <= i; ++j) {
        scalar e = std::exp(scores.data[(b * seq_len + i) * seq_len + j] - max_score);
        attention_weights.data[(b * seq_len + i) * seq_len + j] = e;
        sum_exp += e;
      }
      for (int j = i + 1; j < seq_len; ++j) {
        attention_weights.data[(b * seq_len + i) * seq_len + j] = 0.0f;
      }
      for (int j = 0; j <= i; ++j) {
        attention_weights.data[(b * seq_len + i) * seq_len + j] /= sum_exp;
      }
    }
  });

  ThreadPool::getInstance().parallel_for(0, batch_size, [this](int b) {
    for (int i = 0; i < seq_len; ++i) {
      for (int d = 0; d < embed_dim; ++d) {
        scalar sum = 0.0f;
        for (int j = 0; j <= i; ++j) {
          sum += attention_weights.data[(b * seq_len + i) * seq_len + j] * V.data[(b * seq_len + j) * embed_dim + d];
        }
        output.data[(b * seq_len + i) * embed_dim + d] = sum;
      }
    }
  });

  return output;
}

const Tensor &MaskedSelfAttentionLayer::backward(const Tensor &input, const Tensor &grad_output) {
  int batch_size = input.rows / seq_len;
  ensure_grad_input_dims(input.rows, input.cols);

  Tensor dQ(input.rows, embed_dim);
  Tensor dK(input.rows, embed_dim);
  Tensor dV(input.rows, embed_dim);
  
  std::memset(dQ.data, 0, dQ.rows * dQ.cols * sizeof(scalar));
  std::memset(dK.data, 0, dK.rows * dK.cols * sizeof(scalar));
  std::memset(dV.data, 0, dV.rows * dV.cols * sizeof(scalar));
  
  scalar scale = 1.0f / std::sqrt((scalar)embed_dim);

  ThreadPool::getInstance().parallel_for(0, batch_size, [this, &grad_output, &dQ, &dK, &dV, scale](int b) {
    for (int i = 0; i < seq_len; ++i) {
      for (int j = 0; j <= i; ++j) {
        scalar w = attention_weights.data[(b * seq_len + i) * seq_len + j];
        for (int d = 0; d < embed_dim; ++d) {
          dV.data[(b * seq_len + j) * embed_dim + d] += w * grad_output.data[(b * seq_len + i) * embed_dim + d];
        }
      }
    }

    // Precompute per-query scalar: sum_k a_ik * (dout_i · v_k) — O(n²) not O(n³)
    std::vector<scalar> scalar_i(seq_len, 0.0f);
    for (int i = 0; i < seq_len; ++i) {
      for (int k = 0; k <= i; ++k) {
        scalar wk = attention_weights.data[(b * seq_len + i) * seq_len + k];
        scalar dot = 0.0f;
        for (int d = 0; d < embed_dim; ++d)
          dot += grad_output.data[(b * seq_len + i) * embed_dim + d] * V.data[(b * seq_len + k) * embed_dim + d];
        scalar_i[i] += wk * dot;
      }
    }

    for (int i = 0; i < seq_len; ++i) {
      for (int j = 0; j <= i; ++j) {
        scalar w = attention_weights.data[(b * seq_len + i) * seq_len + j];
        scalar dscore = 0.0f;
        for (int d = 0; d < embed_dim; ++d)
          dscore += grad_output.data[(b * seq_len + i) * embed_dim + d] * V.data[(b * seq_len + j) * embed_dim + d];

        scalar ds = w * (dscore - scalar_i[i]) * scale;
        for (int d = 0; d < embed_dim; ++d) {
          dQ.data[(b * seq_len + i) * embed_dim + d] += ds * K.data[(b * seq_len + j) * embed_dim + d];
          dK.data[(b * seq_len + j) * embed_dim + d] += ds * Q.data[(b * seq_len + i) * embed_dim + d];
        }
      }
    }
  });

  std::memset(grad_W_q.data, 0, grad_W_q.rows * grad_W_q.cols * sizeof(scalar));
  std::memset(grad_W_k.data, 0, grad_W_k.rows * grad_W_k.cols * sizeof(scalar));
  std::memset(grad_W_v.data, 0, grad_W_v.rows * grad_W_v.cols * sizeof(scalar));
  std::memset(grad_b_q.data, 0, grad_b_q.rows * grad_b_q.cols * sizeof(scalar));
  std::memset(grad_b_k.data, 0, grad_b_k.rows * grad_b_k.cols * sizeof(scalar));
  std::memset(grad_b_v.data, 0, grad_b_v.rows * grad_b_v.cols * sizeof(scalar));

  for(int i=0; i<input.rows; ++i) {
    for(int d=0; d<embed_dim; ++d) {
      grad_b_q.data[d] += dQ.data[i*embed_dim+d];
      grad_b_k.data[d] += dK.data[i*embed_dim+d];
      grad_b_v.data[d] += dV.data[i*embed_dim+d];
    }
  }

  for(int d1=0; d1<embed_dim; ++d1) {
    for(int d2=0; d2<embed_dim; ++d2) {
      scalar sq = 0, sk = 0, sv = 0;
      for(int i=0; i<input.rows; ++i) {
        scalar in_val = input.data[i*embed_dim + d1];
        sq += in_val * dQ.data[i*embed_dim + d2];
        sk += in_val * dK.data[i*embed_dim + d2];
        sv += in_val * dV.data[i*embed_dim + d2];
      }
      grad_W_q.data[d1*embed_dim+d2] = sq;
      grad_W_k.data[d1*embed_dim+d2] = sk;
      grad_W_v.data[d1*embed_dim+d2] = sv;
    }
  }

  for(int i=0; i<input.rows; ++i) {
    for(int d=0; d<embed_dim; ++d) {
      scalar sum = 0;
      for(int d2=0; d2<embed_dim; ++d2) {
        sum += dQ.data[i*embed_dim+d2] * W_q.data[d*embed_dim+d2] +
               dK.data[i*embed_dim+d2] * W_k.data[d*embed_dim+d2] +
               dV.data[i*embed_dim+d2] * W_v.data[d*embed_dim+d2];
      }
      grad_input.data[i*embed_dim+d] = sum;
    }
  }

  return grad_input;
}

void MaskedSelfAttentionLayer::update(Optimizer *opt) {
  if (opt) {
    opt->update(W_q, grad_W_q); opt->update(b_q, grad_b_q);
    opt->update(W_k, grad_W_k); opt->update(b_k, grad_b_k);
    opt->update(W_v, grad_W_v); opt->update(b_v, grad_b_v);
  }
}
void MaskedSelfAttentionLayer::save(std::ostream& os) const { W_q.save(os); W_k.save(os); W_v.save(os); b_q.save(os); b_k.save(os); b_v.save(os); }
void MaskedSelfAttentionLayer::load(std::istream& is) { W_q.load(is); W_k.load(is); W_v.load(is); b_q.load(is); b_k.load(is); b_v.load(is); }


// --- TransformerDecoderLayer ---
extern class ReLU relu_fn;
static ReLU relu_fn_dec_inst;

TransformerDecoderLayer::TransformerDecoderLayer(int embed_dim, int seq_len, int ff_dim)
    : embed_dim(embed_dim), seq_len(seq_len), ff_dim(ff_dim == -1 ? 4 * embed_dim : ff_dim),
      attention(embed_dim, seq_len), norm1(embed_dim),
      ff1(embed_dim, this->ff_dim), relu(&relu_fn_dec_inst),
      ff2(this->ff_dim, embed_dim), norm2(embed_dim) {
}

const Tensor &TransformerDecoderLayer::forward(const Tensor &input) {
  ensure_output_dims(input.rows, input.cols);

  att_output = attention.forward(input);
  Tensor add1 = input + att_output;
  norm1_output = norm1.forward(add1);

  ff1_output = ff1.forward(norm1_output);
  relu_output = relu.forward(ff1_output);
  ff2_output = ff2.forward(relu_output);
  
  Tensor add2 = norm1_output + ff2_output;
  output = norm2.forward(add2);

  return output;
}

const Tensor &TransformerDecoderLayer::backward(const Tensor &input, const Tensor &grad_output) {
  ensure_grad_input_dims(input.rows, input.cols);

  Tensor grad_add2 = norm2.backward(norm1_output + ff2_output, grad_output);
  
  Tensor grad_ff2_out = grad_add2;
  Tensor grad_norm1_out_1 = grad_add2;

  Tensor grad_relu_out = ff2.backward(relu_output, grad_ff2_out);
  Tensor grad_ff1_out = relu.backward(ff1_output, grad_relu_out);
  Tensor grad_norm1_in = ff1.backward(norm1_output, grad_ff1_out);

  Tensor grad_norm1_out = grad_norm1_out_1 + grad_norm1_in;

  Tensor grad_add1 = norm1.backward(input + att_output, grad_norm1_out);

  Tensor grad_att_out = grad_add1;
  Tensor grad_att_in = attention.backward(input, grad_att_out);

  grad_input = grad_add1 + grad_att_in;

  return grad_input;
}

void TransformerDecoderLayer::set_training(bool training) {
  attention.set_training(training);
  norm1.set_training(training);
  ff1.set_training(training);
  relu.set_training(training);
  ff2.set_training(training);
  norm2.set_training(training);
}

void TransformerDecoderLayer::update(Optimizer *opt) {
  attention.update(opt);
  norm1.update(opt);
  ff1.update(opt);
  ff2.update(opt);
  norm2.update(opt);
}

void TransformerDecoderLayer::save(std::ostream& os) const { attention.save(os); norm1.save(os); ff1.save(os); ff2.save(os); norm2.save(os); }
void TransformerDecoderLayer::load(std::istream& is) { attention.load(is); norm1.load(is); ff1.load(is); ff2.load(is); norm2.load(is); }


// --- MaskedMultiHeadAttentionLayer ---
MaskedMultiHeadAttentionLayer::MaskedMultiHeadAttentionLayer(int embed_dim, int seq_len, int num_heads)
    : embed_dim(embed_dim), seq_len(seq_len), num_heads(num_heads), head_dim(embed_dim / num_heads),
      W_q(embed_dim, embed_dim), W_k(embed_dim, embed_dim), W_v(embed_dim, embed_dim),
      b_q(1, embed_dim), b_k(1, embed_dim), b_v(1, embed_dim),
      W_o(embed_dim, embed_dim), b_o(1, embed_dim),
      grad_W_q(embed_dim, embed_dim), grad_W_k(embed_dim, embed_dim), grad_W_v(embed_dim, embed_dim),
      grad_b_q(1, embed_dim), grad_b_k(1, embed_dim), grad_b_v(1, embed_dim),
      grad_W_o(embed_dim, embed_dim), grad_b_o(1, embed_dim) {
  
  if (embed_dim % num_heads != 0) {
    throw std::invalid_argument("embed_dim must be divisible by num_heads");
  }
  
  W_q.init_params(); W_k.init_params(); W_v.init_params(); W_o.init_params();
}

const Tensor &MaskedMultiHeadAttentionLayer::forward(const Tensor &input) {
  int batch_size = input.rows / seq_len;
  ensure_output_dims(input.rows, embed_dim);
  
  if (concat_out.rows != input.rows || concat_out.cols != embed_dim) {
      concat_out = Tensor(input.rows, embed_dim);
  }

  Q = input * W_q; K = input * W_k; V = input * W_v;
  
  for(int i=0; i<input.rows; ++i) {
    for(int j=0; j<embed_dim; ++j) {
      Q.data[i*embed_dim+j] += b_q.data[j];
      K.data[i*embed_dim+j] += b_k.data[j];
      V.data[i*embed_dim+j] += b_v.data[j];
    }
  }

  if (scores.rows != batch_size * num_heads * seq_len || scores.cols != seq_len) {
    scores = Tensor(batch_size * num_heads * seq_len, seq_len);
    attention_weights = Tensor(batch_size * num_heads * seq_len, seq_len);
  }
  
  scalar scale = 1.0f / std::sqrt((scalar)head_dim);

  ThreadPool::getInstance().parallel_for(0, batch_size * num_heads, [this, scale](int task_idx) {
    int b = task_idx / num_heads;
    int h = task_idx % num_heads;
    int head_offset = h * head_dim;
    
    for (int i = 0; i < seq_len; ++i) {
      scalar max_score = -1e9f;
      for (int j = 0; j < seq_len; ++j) {
        if (j > i) {
          scores.data[(task_idx * seq_len + i) * seq_len + j] = -1e9f; // causal mask
          continue;
        }
        scalar dot = 0.0f;
        for (int d = 0; d < head_dim; ++d) {
          dot += Q.data[(b * seq_len + i) * embed_dim + head_offset + d] * K.data[(b * seq_len + j) * embed_dim + head_offset + d];
        }
        scalar s = dot * scale;
        scores.data[(task_idx * seq_len + i) * seq_len + j] = s;
        if (s > max_score) max_score = s;
      }
      
      scalar sum_exp = 0.0f;
      for (int j = 0; j <= i; ++j) {
        scalar e = std::exp(scores.data[(task_idx * seq_len + i) * seq_len + j] - max_score);
        attention_weights.data[(task_idx * seq_len + i) * seq_len + j] = e;
        sum_exp += e;
      }
      for (int j = i + 1; j < seq_len; ++j) {
        attention_weights.data[(task_idx * seq_len + i) * seq_len + j] = 0.0f;
      }
      for (int j = 0; j <= i; ++j) {
        attention_weights.data[(task_idx * seq_len + i) * seq_len + j] /= sum_exp;
      }
    }
  });

  ThreadPool::getInstance().parallel_for(0, batch_size * num_heads, [this](int task_idx) {
    int b = task_idx / num_heads;
    int h = task_idx % num_heads;
    int head_offset = h * head_dim;
    
    for (int i = 0; i < seq_len; ++i) {
      for (int d = 0; d < head_dim; ++d) {
        scalar sum = 0.0f;
        for (int j = 0; j <= i; ++j) {
          sum += attention_weights.data[(task_idx * seq_len + i) * seq_len + j] * V.data[(b * seq_len + j) * embed_dim + head_offset + d];
        }
        concat_out.data[(b * seq_len + i) * embed_dim + head_offset + d] = sum;
      }
    }
  });

  // Final linear projection
  output = concat_out * W_o;
  for(int i=0; i<output.rows; ++i) {
    for(int j=0; j<embed_dim; ++j) {
      output.data[i*embed_dim+j] += b_o.data[j];
    }
  }

  return output;
}

const Tensor &MaskedMultiHeadAttentionLayer::backward(const Tensor &input, const Tensor &grad_output) {
  int batch_size = input.rows / seq_len;
  ensure_grad_input_dims(input.rows, input.cols);

  // Backprop through final projection
  std::memset(grad_b_o.data, 0, embed_dim * sizeof(scalar));
  for(int i=0; i<input.rows; ++i) {
      for(int j=0; j<embed_dim; ++j) grad_b_o.data[j] += grad_output.data[i*embed_dim+j];
  }

  std::memset(grad_W_o.data, 0, embed_dim * embed_dim * sizeof(scalar));
  for(int d1=0; d1<embed_dim; ++d1) {
    for(int d2=0; d2<embed_dim; ++d2) {
      scalar sum = 0;
      for(int i=0; i<input.rows; ++i) {
        sum += concat_out.data[i*embed_dim + d1] * grad_output.data[i*embed_dim + d2];
      }
      grad_W_o.data[d1*embed_dim+d2] = sum;
    }
  }

  Tensor d_concat(input.rows, embed_dim);
  for(int i=0; i<input.rows; ++i) {
    for(int d1=0; d1<embed_dim; ++d1) {
      scalar sum = 0;
      for(int d2=0; d2<embed_dim; ++d2) {
        sum += grad_output.data[i*embed_dim+d2] * W_o.data[d1*embed_dim+d2];
      }
      d_concat.data[i*embed_dim+d1] = sum;
    }
  }

  // Backprop through heads
  Tensor dQ(input.rows, embed_dim);
  Tensor dK(input.rows, embed_dim);
  Tensor dV(input.rows, embed_dim);
  
  std::memset(dQ.data, 0, dQ.rows * dQ.cols * sizeof(scalar));
  std::memset(dK.data, 0, dK.rows * dK.cols * sizeof(scalar));
  std::memset(dV.data, 0, dV.rows * dV.cols * sizeof(scalar));
  
  scalar scale = 1.0f / std::sqrt((scalar)head_dim);

  ThreadPool::getInstance().parallel_for(0, batch_size * num_heads, [this, &d_concat, &dQ, &dK, &dV, scale](int task_idx) {
    int b = task_idx / num_heads;
    int h = task_idx % num_heads;
    int head_offset = h * head_dim;

    for (int i = 0; i < seq_len; ++i) {
      for (int j = 0; j <= i; ++j) {
        scalar w = attention_weights.data[(task_idx * seq_len + i) * seq_len + j];
        for (int d = 0; d < head_dim; ++d) {
          dV.data[(b * seq_len + j) * embed_dim + head_offset + d] += w * d_concat.data[(b * seq_len + i) * embed_dim + head_offset + d];
        }
      }
    }

    // Precompute per-query scalar: sum_k a_ik * (dout_i · v_k) — O(n²) not O(n³)
    std::vector<scalar> scalar_i(seq_len, 0.0f);
    for (int i = 0; i < seq_len; ++i) {
      for (int k = 0; k <= i; ++k) {
        scalar wk = attention_weights.data[(task_idx * seq_len + i) * seq_len + k];
        scalar dot = 0.0f;
        for (int d = 0; d < head_dim; ++d)
          dot += d_concat.data[(b * seq_len + i) * embed_dim + head_offset + d] * V.data[(b * seq_len + k) * embed_dim + head_offset + d];
        scalar_i[i] += wk * dot;
      }
    }

    for (int i = 0; i < seq_len; ++i) {
      for (int j = 0; j <= i; ++j) {
        scalar w = attention_weights.data[(task_idx * seq_len + i) * seq_len + j];
        scalar dscore = 0.0f;
        for (int d = 0; d < head_dim; ++d)
          dscore += d_concat.data[(b * seq_len + i) * embed_dim + head_offset + d] * V.data[(b * seq_len + j) * embed_dim + head_offset + d];

        scalar ds = w * (dscore - scalar_i[i]) * scale;
        for (int d = 0; d < head_dim; ++d) {
          dQ.data[(b * seq_len + i) * embed_dim + head_offset + d] += ds * K.data[(b * seq_len + j) * embed_dim + head_offset + d];
          dK.data[(b * seq_len + j) * embed_dim + head_offset + d] += ds * Q.data[(b * seq_len + i) * embed_dim + head_offset + d];
        }
      }
    }
  });

  std::memset(grad_W_q.data, 0, grad_W_q.rows * grad_W_q.cols * sizeof(scalar));
  std::memset(grad_W_k.data, 0, grad_W_k.rows * grad_W_k.cols * sizeof(scalar));
  std::memset(grad_W_v.data, 0, grad_W_v.rows * grad_W_v.cols * sizeof(scalar));
  std::memset(grad_b_q.data, 0, grad_b_q.rows * grad_b_q.cols * sizeof(scalar));
  std::memset(grad_b_k.data, 0, grad_b_k.rows * grad_b_k.cols * sizeof(scalar));
  std::memset(grad_b_v.data, 0, grad_b_v.rows * grad_b_v.cols * sizeof(scalar));

  for(int i=0; i<input.rows; ++i) {
    for(int d=0; d<embed_dim; ++d) {
      grad_b_q.data[d] += dQ.data[i*embed_dim+d];
      grad_b_k.data[d] += dK.data[i*embed_dim+d];
      grad_b_v.data[d] += dV.data[i*embed_dim+d];
    }
  }

  for(int d1=0; d1<embed_dim; ++d1) {
    for(int d2=0; d2<embed_dim; ++d2) {
      scalar sq = 0, sk = 0, sv = 0;
      for(int i=0; i<input.rows; ++i) {
        scalar in_val = input.data[i*embed_dim + d1];
        sq += in_val * dQ.data[i*embed_dim + d2];
        sk += in_val * dK.data[i*embed_dim + d2];
        sv += in_val * dV.data[i*embed_dim + d2];
      }
      grad_W_q.data[d1*embed_dim+d2] = sq;
      grad_W_k.data[d1*embed_dim+d2] = sk;
      grad_W_v.data[d1*embed_dim+d2] = sv;
    }
  }

  for(int i=0; i<input.rows; ++i) {
    for(int d=0; d<embed_dim; ++d) {
      scalar sum = 0;
      for(int d2=0; d2<embed_dim; ++d2) {
        sum += dQ.data[i*embed_dim+d2] * W_q.data[d*embed_dim+d2] +
               dK.data[i*embed_dim+d2] * W_k.data[d*embed_dim+d2] +
               dV.data[i*embed_dim+d2] * W_v.data[d*embed_dim+d2];
      }
      grad_input.data[i*embed_dim+d] = sum;
    }
  }

  return grad_input;
}

void MaskedMultiHeadAttentionLayer::update(Optimizer *opt) {
  if (opt) {
    opt->update(W_q, grad_W_q); opt->update(b_q, grad_b_q);
    opt->update(W_k, grad_W_k); opt->update(b_k, grad_b_k);
    opt->update(W_v, grad_W_v); opt->update(b_v, grad_b_v);
    opt->update(W_o, grad_W_o); opt->update(b_o, grad_b_o);
  }
}
void MaskedMultiHeadAttentionLayer::save(std::ostream& os) const { W_q.save(os); W_k.save(os); W_v.save(os); W_o.save(os); b_q.save(os); b_k.save(os); b_v.save(os); b_o.save(os); }
void MaskedMultiHeadAttentionLayer::load(std::istream& is) { W_q.load(is); W_k.load(is); W_v.load(is); W_o.load(is); b_q.load(is); b_k.load(is); b_v.load(is); b_o.load(is); }


// --- TransformerMHDecoderLayer ---
static ReLU relu_fn_mhdec_inst;

TransformerMHDecoderLayer::TransformerMHDecoderLayer(int embed_dim, int seq_len, int ff_dim, int num_heads)
    : embed_dim(embed_dim), seq_len(seq_len), ff_dim(ff_dim == -1 ? 4 * embed_dim : ff_dim), num_heads(num_heads),
      attention(embed_dim, seq_len, num_heads), norm1(embed_dim),
      ff1(embed_dim, this->ff_dim), relu(&relu_fn_mhdec_inst),
      ff2(this->ff_dim, embed_dim), norm2(embed_dim) {
}

const Tensor &TransformerMHDecoderLayer::forward(const Tensor &input) {
  ensure_output_dims(input.rows, input.cols);

  att_output = attention.forward(input);
  Tensor add1 = input + att_output;
  norm1_output = norm1.forward(add1);

  ff1_output = ff1.forward(norm1_output);
  relu_output = relu.forward(ff1_output);
  ff2_output = ff2.forward(relu_output);
  
  Tensor add2 = norm1_output + ff2_output;
  output = norm2.forward(add2);

  return output;
}

const Tensor &TransformerMHDecoderLayer::backward(const Tensor &input, const Tensor &grad_output) {
  ensure_grad_input_dims(input.rows, input.cols);

  Tensor grad_add2 = norm2.backward(norm1_output + ff2_output, grad_output);
  
  Tensor grad_ff2_out = grad_add2;
  Tensor grad_norm1_out_1 = grad_add2;

  Tensor grad_relu_out = ff2.backward(relu_output, grad_ff2_out);
  Tensor grad_ff1_out = relu.backward(ff1_output, grad_relu_out);
  Tensor grad_norm1_in = ff1.backward(norm1_output, grad_ff1_out);

  Tensor grad_norm1_out = grad_norm1_out_1 + grad_norm1_in;

  Tensor grad_add1 = norm1.backward(input + att_output, grad_norm1_out);

  Tensor grad_att_out = grad_add1;
  Tensor grad_att_in = attention.backward(input, grad_att_out);

  grad_input = grad_add1 + grad_att_in;

  return grad_input;
}

void TransformerMHDecoderLayer::set_training(bool training) {
  attention.set_training(training);
  norm1.set_training(training);
  ff1.set_training(training);
  relu.set_training(training);
  ff2.set_training(training);
  norm2.set_training(training);
}

void TransformerMHDecoderLayer::update(Optimizer *opt) {
  attention.update(opt);
  norm1.update(opt);
  ff1.update(opt);
  ff2.update(opt);
  norm2.update(opt);
}

void TransformerMHDecoderLayer::save(std::ostream& os) const { attention.save(os); norm1.save(os); ff1.save(os); ff2.save(os); norm2.save(os); }
void TransformerMHDecoderLayer::load(std::istream& is) { attention.load(is); norm1.load(is); ff1.load(is); ff2.load(is); norm2.load(is); }

// --- MultiHeadAttentionLayer ---
MultiHeadAttentionLayer::MultiHeadAttentionLayer(int embed_dim, int seq_len, int num_heads)
    : embed_dim(embed_dim), seq_len(seq_len), num_heads(num_heads), head_dim(embed_dim / num_heads),
      W_q(embed_dim, embed_dim), W_k(embed_dim, embed_dim), W_v(embed_dim, embed_dim),
      b_q(1, embed_dim), b_k(1, embed_dim), b_v(1, embed_dim),
      W_o(embed_dim, embed_dim), b_o(1, embed_dim),
      grad_W_q(embed_dim, embed_dim), grad_W_k(embed_dim, embed_dim), grad_W_v(embed_dim, embed_dim),
      grad_b_q(1, embed_dim), grad_b_k(1, embed_dim), grad_b_v(1, embed_dim),
      grad_W_o(embed_dim, embed_dim), grad_b_o(1, embed_dim) {
  
  if (embed_dim % num_heads != 0) {
    throw std::invalid_argument("embed_dim must be divisible by num_heads");
  }
  
  W_q.init_params(); W_k.init_params(); W_v.init_params(); W_o.init_params();
}

const Tensor &MultiHeadAttentionLayer::forward(const Tensor &input) {
  int batch_size = input.rows / seq_len;
  ensure_output_dims(input.rows, embed_dim);
  
  if (concat_out.rows != input.rows || concat_out.cols != embed_dim) {
      concat_out = Tensor(input.rows, embed_dim);
  }

  Q = input * W_q; K = input * W_k; V = input * W_v;
  
  for(int i=0; i<input.rows; ++i) {
    for(int j=0; j<embed_dim; ++j) {
      Q.data[i*embed_dim+j] += b_q.data[j];
      K.data[i*embed_dim+j] += b_k.data[j];
      V.data[i*embed_dim+j] += b_v.data[j];
    }
  }

  if (scores.rows != batch_size * num_heads * seq_len || scores.cols != seq_len) {
    scores = Tensor(batch_size * num_heads * seq_len, seq_len);
    attention_weights = Tensor(batch_size * num_heads * seq_len, seq_len);
  }
  
  scalar scale = 1.0f / std::sqrt((scalar)head_dim);

  ThreadPool::getInstance().parallel_for(0, batch_size * num_heads, [this, scale](int task_idx) {
    int b = task_idx / num_heads;
    int h = task_idx % num_heads;
    int head_offset = h * head_dim;
    
    for (int i = 0; i < seq_len; ++i) {
      scalar max_score = -1e9f;
      for (int j = 0; j < seq_len; ++j) {
        scalar dot = 0.0f;
        for (int d = 0; d < head_dim; ++d) {
          dot += Q.data[(b * seq_len + i) * embed_dim + head_offset + d] * K.data[(b * seq_len + j) * embed_dim + head_offset + d];
        }
        scalar s = dot * scale;
        scores.data[(task_idx * seq_len + i) * seq_len + j] = s;
        if (s > max_score) max_score = s;
      }
      
      scalar sum_exp = 0.0f;
      for (int j = 0; j < seq_len; ++j) {
        scalar e = std::exp(scores.data[(task_idx * seq_len + i) * seq_len + j] - max_score);
        attention_weights.data[(task_idx * seq_len + i) * seq_len + j] = e;
        sum_exp += e;
      }
      for (int j = 0; j < seq_len; ++j) {
        attention_weights.data[(task_idx * seq_len + i) * seq_len + j] /= sum_exp;
      }
    }
  });

  ThreadPool::getInstance().parallel_for(0, batch_size * num_heads, [this](int task_idx) {
    int b = task_idx / num_heads;
    int h = task_idx % num_heads;
    int head_offset = h * head_dim;
    
    for (int i = 0; i < seq_len; ++i) {
      for (int d = 0; d < head_dim; ++d) {
        scalar sum = 0.0f;
        for (int j = 0; j < seq_len; ++j) {
          sum += attention_weights.data[(task_idx * seq_len + i) * seq_len + j] * V.data[(b * seq_len + j) * embed_dim + head_offset + d];
        }
        concat_out.data[(b * seq_len + i) * embed_dim + head_offset + d] = sum;
      }
    }
  });

  // Final linear projection
  output = concat_out * W_o;
  for(int i=0; i<output.rows; ++i) {
    for(int j=0; j<embed_dim; ++j) {
      output.data[i*embed_dim+j] += b_o.data[j];
    }
  }

  return output;
}

const Tensor &MultiHeadAttentionLayer::backward(const Tensor &input, const Tensor &grad_output) {
  int batch_size = input.rows / seq_len;
  ensure_grad_input_dims(input.rows, input.cols);

  // Backprop through final projection
  std::memset(grad_b_o.data, 0, embed_dim * sizeof(scalar));
  for(int i=0; i<input.rows; ++i) {
      for(int j=0; j<embed_dim; ++j) grad_b_o.data[j] += grad_output.data[i*embed_dim+j];
  }

  std::memset(grad_W_o.data, 0, embed_dim * embed_dim * sizeof(scalar));
  for(int d1=0; d1<embed_dim; ++d1) {
    for(int d2=0; d2<embed_dim; ++d2) {
      scalar sum = 0;
      for(int i=0; i<input.rows; ++i) {
        sum += concat_out.data[i*embed_dim + d1] * grad_output.data[i*embed_dim + d2];
      }
      grad_W_o.data[d1*embed_dim+d2] = sum;
    }
  }

  Tensor d_concat(input.rows, embed_dim);
  for(int i=0; i<input.rows; ++i) {
    for(int d1=0; d1<embed_dim; ++d1) {
      scalar sum = 0;
      for(int d2=0; d2<embed_dim; ++d2) {
        sum += grad_output.data[i*embed_dim+d2] * W_o.data[d1*embed_dim+d2];
      }
      d_concat.data[i*embed_dim+d1] = sum;
    }
  }

  // Backprop through heads
  Tensor dQ(input.rows, embed_dim);
  Tensor dK(input.rows, embed_dim);
  Tensor dV(input.rows, embed_dim);
  
  std::memset(dQ.data, 0, dQ.rows * dQ.cols * sizeof(scalar));
  std::memset(dK.data, 0, dK.rows * dK.cols * sizeof(scalar));
  std::memset(dV.data, 0, dV.rows * dV.cols * sizeof(scalar));
  
  scalar scale = 1.0f / std::sqrt((scalar)head_dim);

  ThreadPool::getInstance().parallel_for(0, batch_size * num_heads, [this, &d_concat, &dQ, &dK, &dV, scale](int task_idx) {
    int b = task_idx / num_heads;
    int h = task_idx % num_heads;
    int head_offset = h * head_dim;

    for (int i = 0; i < seq_len; ++i) {
      for (int j = 0; j < seq_len; ++j) {
        scalar w = attention_weights.data[(task_idx * seq_len + i) * seq_len + j];
        for (int d = 0; d < head_dim; ++d) {
          dV.data[(b * seq_len + j) * embed_dim + head_offset + d] += w * d_concat.data[(b * seq_len + i) * embed_dim + head_offset + d];
        }
      }
    }

    // Precompute per-query scalar: sum_k a_ik * (dout_i · v_k) — O(n²) not O(n³)
    std::vector<scalar> scalar_i(seq_len, 0.0f);
    for (int i = 0; i < seq_len; ++i) {
      for (int k = 0; k < seq_len; ++k) {
        scalar wk = attention_weights.data[(task_idx * seq_len + i) * seq_len + k];
        scalar dot = 0.0f;
        for (int d = 0; d < head_dim; ++d)
          dot += d_concat.data[(b * seq_len + i) * embed_dim + head_offset + d] * V.data[(b * seq_len + k) * embed_dim + head_offset + d];
        scalar_i[i] += wk * dot;
      }
    }

    for (int i = 0; i < seq_len; ++i) {
      for (int j = 0; j < seq_len; ++j) {
        scalar w = attention_weights.data[(task_idx * seq_len + i) * seq_len + j];
        scalar dscore = 0.0f;
        for (int d = 0; d < head_dim; ++d)
          dscore += d_concat.data[(b * seq_len + i) * embed_dim + head_offset + d] * V.data[(b * seq_len + j) * embed_dim + head_offset + d];

        scalar ds = w * (dscore - scalar_i[i]) * scale;
        for (int d = 0; d < head_dim; ++d) {
          dQ.data[(b * seq_len + i) * embed_dim + head_offset + d] += ds * K.data[(b * seq_len + j) * embed_dim + head_offset + d];
          dK.data[(b * seq_len + j) * embed_dim + head_offset + d] += ds * Q.data[(b * seq_len + i) * embed_dim + head_offset + d];
        }
      }
    }
  });

  std::memset(grad_W_q.data, 0, grad_W_q.rows * grad_W_q.cols * sizeof(scalar));
  std::memset(grad_W_k.data, 0, grad_W_k.rows * grad_W_k.cols * sizeof(scalar));
  std::memset(grad_W_v.data, 0, grad_W_v.rows * grad_W_v.cols * sizeof(scalar));
  std::memset(grad_b_q.data, 0, grad_b_q.rows * grad_b_q.cols * sizeof(scalar));
  std::memset(grad_b_k.data, 0, grad_b_k.rows * grad_b_k.cols * sizeof(scalar));
  std::memset(grad_b_v.data, 0, grad_b_v.rows * grad_b_v.cols * sizeof(scalar));

  for(int i=0; i<input.rows; ++i) {
    for(int d=0; d<embed_dim; ++d) {
      grad_b_q.data[d] += dQ.data[i*embed_dim+d];
      grad_b_k.data[d] += dK.data[i*embed_dim+d];
      grad_b_v.data[d] += dV.data[i*embed_dim+d];
    }
  }

  for(int d1=0; d1<embed_dim; ++d1) {
    for(int d2=0; d2<embed_dim; ++d2) {
      scalar sq = 0, sk = 0, sv = 0;
      for(int i=0; i<input.rows; ++i) {
        scalar in_val = input.data[i*embed_dim + d1];
        sq += in_val * dQ.data[i*embed_dim + d2];
        sk += in_val * dK.data[i*embed_dim + d2];
        sv += in_val * dV.data[i*embed_dim + d2];
      }
      grad_W_q.data[d1*embed_dim+d2] = sq;
      grad_W_k.data[d1*embed_dim+d2] = sk;
      grad_W_v.data[d1*embed_dim+d2] = sv;
    }
  }

  for(int i=0; i<input.rows; ++i) {
    for(int d=0; d<embed_dim; ++d) {
      scalar sum = 0;
      for(int d2=0; d2<embed_dim; ++d2) {
        sum += dQ.data[i*embed_dim+d2] * W_q.data[d*embed_dim+d2] +
               dK.data[i*embed_dim+d2] * W_k.data[d*embed_dim+d2] +
               dV.data[i*embed_dim+d2] * W_v.data[d*embed_dim+d2];
      }
      grad_input.data[i*embed_dim+d] = sum;
    }
  }

  return grad_input;
}

void MultiHeadAttentionLayer::update(Optimizer *opt) {
  if (opt) {
    opt->update(W_q, grad_W_q); opt->update(b_q, grad_b_q);
    opt->update(W_k, grad_W_k); opt->update(b_k, grad_b_k);
    opt->update(W_v, grad_W_v); opt->update(b_v, grad_b_v);
    opt->update(W_o, grad_W_o); opt->update(b_o, grad_b_o);
  }
}
void MultiHeadAttentionLayer::save(std::ostream& os) const { W_q.save(os); W_k.save(os); W_v.save(os); W_o.save(os); b_q.save(os); b_k.save(os); b_v.save(os); b_o.save(os); }
void MultiHeadAttentionLayer::load(std::istream& is) { W_q.load(is); W_k.load(is); W_v.load(is); W_o.load(is); b_q.load(is); b_k.load(is); b_v.load(is); b_o.load(is); }


// --- TransformerMHEncoderLayer ---
static ReLU relu_fn_mhenc_inst;

TransformerMHEncoderLayer::TransformerMHEncoderLayer(int embed_dim, int seq_len, int ff_dim, int num_heads, scalar dropout_rate)
    : embed_dim(embed_dim), seq_len(seq_len), ff_dim(ff_dim == -1 ? 4 * embed_dim : ff_dim), num_heads(num_heads),
      attention(embed_dim, seq_len, num_heads), norm1(embed_dim), drop1(dropout_rate),
      ff1(embed_dim, this->ff_dim), relu(&relu_fn_mhenc_inst),
      ff2(this->ff_dim, embed_dim), drop2(dropout_rate), norm2(embed_dim) {
}

const Tensor &TransformerMHEncoderLayer::forward(const Tensor &input) {
  ensure_output_dims(input.rows, input.cols);

  att_output = attention.forward(input);
  drop1_output = drop1.forward(att_output);
  Tensor add1 = input + drop1_output;
  norm1_output = norm1.forward(add1);

  ff1_output = ff1.forward(norm1_output);
  relu_output = relu.forward(ff1_output);
  ff2_output = ff2.forward(relu_output);
  drop2_output = drop2.forward(ff2_output);
  
  Tensor add2 = norm1_output + drop2_output;
  output = norm2.forward(add2);

  return output;
}

const Tensor &TransformerMHEncoderLayer::backward(const Tensor &input, const Tensor &grad_output) {
  ensure_grad_input_dims(input.rows, input.cols);

  Tensor grad_add2 = norm2.backward(norm1_output + drop2_output, grad_output);
  
  Tensor grad_drop2_out = grad_add2;
  Tensor grad_norm1_out_1 = grad_add2;

  Tensor grad_ff2_out = drop2.backward(ff2_output, grad_drop2_out);
  Tensor grad_relu_out = ff2.backward(relu_output, grad_ff2_out);
  Tensor grad_ff1_out = relu.backward(ff1_output, grad_relu_out);
  Tensor grad_norm1_in = ff1.backward(norm1_output, grad_ff1_out);

  Tensor grad_norm1_out = grad_norm1_out_1 + grad_norm1_in;

  Tensor grad_add1 = norm1.backward(input + drop1_output, grad_norm1_out);

  Tensor grad_drop1_out = grad_add1;
  Tensor grad_att_out = drop1.backward(att_output, grad_drop1_out);
  Tensor grad_att_in = attention.backward(input, grad_att_out);

  grad_input = grad_add1 + grad_att_in;

  return grad_input;
}

void TransformerMHEncoderLayer::set_training(bool training) {
  attention.set_training(training);
  norm1.set_training(training);
  drop1.set_training(training);
  ff1.set_training(training);
  relu.set_training(training);
  ff2.set_training(training);
  drop2.set_training(training);
  norm2.set_training(training);
}

void TransformerMHEncoderLayer::update(Optimizer *opt) {
  attention.update(opt);
  norm1.update(opt);
  ff1.update(opt);
  ff2.update(opt);
  norm2.update(opt);
}

void TransformerMHEncoderLayer::save(std::ostream& os) const { attention.save(os); norm1.save(os); ff1.save(os); ff2.save(os); norm2.save(os); }
void TransformerMHEncoderLayer::load(std::istream& is) { attention.load(is); norm1.load(is); ff1.load(is); ff2.load(is); norm2.load(is); }

// --- GlobalAveragePooling1DLayer ---
const Tensor &GlobalAveragePooling1DLayer::forward(const Tensor &input) {
  int batch_size = input.rows / seq_len;
  ensure_output_dims(batch_size, embed_dim);

  for (int b = 0; b < batch_size; ++b) {
    for (int d = 0; d < embed_dim; ++d) {
      scalar sum = 0.0f;
      for (int i = 0; i < seq_len; ++i) {
        sum += input.data[(b * seq_len + i) * embed_dim + d];
      }
      output.data[b * embed_dim + d] = sum / seq_len;
    }
  }
  return output;
}

const Tensor &GlobalAveragePooling1DLayer::backward(const Tensor &input, const Tensor &grad_output) {
  int batch_size = input.rows / seq_len;
  ensure_grad_input_dims(input.rows, embed_dim);

  scalar scale = 1.0f / seq_len;
  for (int b = 0; b < batch_size; ++b) {
    for (int d = 0; d < embed_dim; ++d) {
      scalar go = grad_output.data[b * embed_dim + d] * scale;
      for (int i = 0; i < seq_len; ++i) {
        grad_input.data[(b * seq_len + i) * embed_dim + d] = go;
      }
    }
  }
  return grad_input;
}
