#ifndef LAYER_H
#define LAYER_H

#include "tensor.h"
#include <string>
#include <unordered_map>

class Optimizer;

class Layer {
public:
  Tensor output;
  Tensor grad_input;

  void ensure_output_dims(int r, int c) {
    if (output.rows != r || output.cols != c) {
      output = Tensor(r, c);
    }
  }

  void ensure_grad_input_dims(int r, int c) {
    if (grad_input.rows != r || grad_input.cols != c) {
      grad_input = Tensor(r, c);
    }
  }

public:
  virtual ~Layer() = default;

  virtual const Tensor &forward(const Tensor &input) = 0;
  virtual const Tensor &backward(const Tensor &input,
                                 const Tensor &grad_output) = 0;
  virtual void set_training(bool training) {}
  virtual void update(Optimizer *opt) {}
  virtual std::string get_name() const = 0;
  virtual int get_params_count() const { return 0; }
  virtual std::unordered_map<std::string, std::string> get_details() const {
    return {};
  }
  virtual void save(std::ostream& os) const {}
  virtual void load(std::istream& is) {}
};

class InputLayer : public Layer {
public:
  const Tensor &forward(const Tensor &input) override;
  const Tensor &backward(const Tensor &input,
                         const Tensor &grad_output) override;
  std::string get_name() const override { return "InputLayer"; }
};

class DenseLayer : public Layer {
public:
  Tensor weights;
  Tensor biases;
  Tensor grad_weights;
  Tensor grad_biases;

  DenseLayer(int input_dim, int output_dim);
  const Tensor &forward(const Tensor &input) override;
  const Tensor &backward(const Tensor &input,
                         const Tensor &grad_output) override;
  void update(Optimizer *opt) override;
  std::string get_name() const override { return "DenseLayer"; }
  int get_params_count() const override {
    return (weights.rows * weights.cols) + (biases.rows * biases.cols);
  }
  std::unordered_map<std::string, std::string> get_details() const override {
    return {{"input_dim", std::to_string(weights.rows)},
            {"output_dim", std::to_string(weights.cols)}};
  }
  void save(std::ostream& os) const override;
  void load(std::istream& is) override;
};

class DropoutLayer : public Layer {
public:
  scalar rate;
  bool is_training;
  Tensor mask;

  DropoutLayer(scalar rate = 0.5);
  const Tensor &forward(const Tensor &input) override;
  const Tensor &backward(const Tensor &input,
                         const Tensor &grad_output) override;
  std::string get_name() const override { return "DropoutLayer"; }
  void set_training(bool training) override { is_training = training; }
  std::unordered_map<std::string, std::string> get_details() const override {
    return {{"rate", std::to_string(rate)}};
  }
};

class RegularizationLayer : public Layer {
public:
  scalar lambda;
  int type; // 1 for L1, 2 for L2

  RegularizationLayer(scalar lambda = 0.01, int type = 2);
  const Tensor &forward(const Tensor &input) override;
  const Tensor &backward(const Tensor &input,
                         const Tensor &grad_output) override;
  std::string get_name() const override {
    return (type == 1 ? "L1" : "L2") + std::string("RegularizationLayer");
  }
  std::unordered_map<std::string, std::string> get_details() const override {
    return {{"lambda", std::to_string(lambda)},
            {"type", (type == 1 ? "L1" : "L2")}};
  }
};

class Activation;

class ActivationLayer : public Layer {
public:
  Activation *activation_fn;
  ActivationLayer(Activation *fn);
  ~ActivationLayer();
  const Tensor &forward(const Tensor &input) override;
  const Tensor &backward(const Tensor &input,
                         const Tensor &grad_output) override;
  std::string get_name() const override { return "ActivationLayer"; }
};

class RNNLayer : public Layer {
public:
  int input_dim;
  int hidden_dim;
  int seq_len;
  bool return_sequences;

  Tensor weights_ih;
  Tensor weights_hh;
  Tensor biases;

  Tensor grad_weights_ih;
  Tensor grad_weights_hh;
  Tensor grad_biases;

  Tensor h_states;

  RNNLayer(int input_dim, int hidden_dim, int seq_len, bool return_sequences = false);
  const Tensor &forward(const Tensor &input) override;
  const Tensor &backward(const Tensor &input,
                         const Tensor &grad_output) override;
  void update(Optimizer *opt) override;
  std::string get_name() const override { return "RNNLayer"; }
  int get_params_count() const override {
    return (weights_ih.rows * weights_ih.cols) + (weights_hh.rows * weights_hh.cols) + (biases.rows * biases.cols);
  }
  std::unordered_map<std::string, std::string> get_details() const override {
    return {{"input_dim", std::to_string(input_dim)},
            {"hidden_dim", std::to_string(hidden_dim)},
            {"seq_len", std::to_string(seq_len)},
            {"return_sequences", return_sequences ? "True" : "False"}};
  }
  void save(std::ostream& os) const override;
  void load(std::istream& is) override;
};

class LSTMLayer : public Layer {
public:
  int input_dim;
  int hidden_dim;
  int seq_len;
  bool return_sequences;

  Tensor weights_ih;
  Tensor weights_hh;
  Tensor biases;

  Tensor grad_weights_ih;
  Tensor grad_weights_hh;
  Tensor grad_biases;

  Tensor h_states;
  Tensor c_states;
  Tensor gates;

  // Workspace for optimization
  Tensor workspace_gates_hh;
  Tensor workspace_h_prev;
  Tensor workspace_c_prev; // Pre-activations + activations

  LSTMLayer(int input_dim, int hidden_dim, int seq_len, bool return_sequences = false);
  const Tensor &forward(const Tensor &input) override;
  const Tensor &backward(const Tensor &input,
                         const Tensor &grad_output) override;
  void update(Optimizer *opt) override;
  std::string get_name() const override { return "LSTMLayer"; }
  int get_params_count() const override {
    return (weights_ih.rows * weights_ih.cols) + (weights_hh.rows * weights_hh.cols) + (biases.rows * biases.cols);
  }
  std::unordered_map<std::string, std::string> get_details() const override {
    return {{"input_dim", std::to_string(input_dim)},
            {"hidden_dim", std::to_string(hidden_dim)},
            {"seq_len", std::to_string(seq_len)},
            {"return_sequences", return_sequences ? "True" : "False"}};
  }
  void save(std::ostream& os) const override;
  void load(std::istream& is) override;
};

class LayerNormLayer : public Layer {
public:
  int feature_dim;
  scalar epsilon;
  Tensor gamma;
  Tensor beta;
  Tensor grad_gamma;
  Tensor grad_beta;

  // Cache for backward pass
  Tensor x_centered;
  Tensor variance;
  Tensor stddev;
  Tensor x_norm;

  LayerNormLayer(int feature_dim, scalar epsilon = 1e-5);
  const Tensor &forward(const Tensor &input) override;
  const Tensor &backward(const Tensor &input,
                         const Tensor &grad_output) override;
  void update(Optimizer *opt) override;
  std::string get_name() const override { return "LayerNormLayer"; }
  int get_params_count() const override {
    return (gamma.rows * gamma.cols) + (beta.rows * beta.cols);
  }
  std::unordered_map<std::string, std::string> get_details() const override {
    return {{"feature_dim", std::to_string(feature_dim)}};
  }
  void save(std::ostream& os) const override;
  void load(std::istream& is) override;
};

class SelfAttentionLayer : public Layer {
public:
  int embed_dim;
  int seq_len;
  
  Tensor W_q, W_k, W_v;
  Tensor b_q, b_k, b_v;
  
  Tensor grad_W_q, grad_W_k, grad_W_v;
  Tensor grad_b_q, grad_b_k, grad_b_v;

  // Caches for backward
  Tensor Q, K, V;
  Tensor scores; // Q * K^T
  Tensor attention_weights; // softmax(scores)

  SelfAttentionLayer(int embed_dim, int seq_len);
  const Tensor &forward(const Tensor &input) override;
  const Tensor &backward(const Tensor &input,
                         const Tensor &grad_output) override;
  void update(Optimizer *opt) override;
  std::string get_name() const override { return "SelfAttentionLayer"; }
  int get_params_count() const override {
    return 3 * ((embed_dim * embed_dim) + embed_dim);
  }
  std::unordered_map<std::string, std::string> get_details() const override {
    return {{"embed_dim", std::to_string(embed_dim)},
            {"seq_len", std::to_string(seq_len)}};
  }
  void save(std::ostream& os) const override;
  void load(std::istream& is) override;
};

class TransformerEncoderLayer : public Layer {
public:
  int embed_dim;
  int seq_len;
  int ff_dim;

  SelfAttentionLayer attention;
  LayerNormLayer norm1;
  DenseLayer ff1;
  ActivationLayer relu;
  DenseLayer ff2;
  LayerNormLayer norm2;

  // Cache
  Tensor att_output;
  Tensor norm1_output;
  Tensor ff1_output;
  Tensor relu_output;
  Tensor ff2_output;

  TransformerEncoderLayer(int embed_dim, int seq_len, int ff_dim = -1);
  const Tensor &forward(const Tensor &input) override;
  const Tensor &backward(const Tensor &input,
                         const Tensor &grad_output) override;
  void set_training(bool training) override;
  void update(Optimizer *opt) override;
  std::string get_name() const override { return "TransformerEncoderLayer"; }
  int get_params_count() const override {
    return attention.get_params_count() + norm1.get_params_count() +
           ff1.get_params_count() + ff2.get_params_count() + norm2.get_params_count();
  }
  std::unordered_map<std::string, std::string> get_details() const override {
    return {{"embed_dim", std::to_string(embed_dim)},
            {"seq_len", std::to_string(seq_len)},
            {"ff_dim", std::to_string(ff_dim)}};
  }
  void save(std::ostream& os) const override;
  void load(std::istream& is) override;
};

class MaskedSelfAttentionLayer : public Layer {
public:
  int embed_dim;
  int seq_len;
  
  Tensor W_q, W_k, W_v;
  Tensor b_q, b_k, b_v;
  
  Tensor grad_W_q, grad_W_k, grad_W_v;
  Tensor grad_b_q, grad_b_k, grad_b_v;

  // Caches for backward
  Tensor Q, K, V;
  Tensor scores; // Q * K^T
  Tensor attention_weights; // softmax(scores)

  MaskedSelfAttentionLayer(int embed_dim, int seq_len);
  const Tensor &forward(const Tensor &input) override;
  const Tensor &backward(const Tensor &input,
                         const Tensor &grad_output) override;
  void update(Optimizer *opt) override;
  std::string get_name() const override { return "MaskedSelfAttentionLayer"; }
  int get_params_count() const override {
    return 3 * ((embed_dim * embed_dim) + embed_dim);
  }
  std::unordered_map<std::string, std::string> get_details() const override {
    return {{"embed_dim", std::to_string(embed_dim)},
            {"seq_len", std::to_string(seq_len)}};
  }
  void save(std::ostream& os) const override;
  void load(std::istream& is) override;
};

class TransformerDecoderLayer : public Layer {
public:
  int embed_dim;
  int seq_len;
  int ff_dim;

  MaskedSelfAttentionLayer attention;
  LayerNormLayer norm1;
  DenseLayer ff1;
  ActivationLayer relu;
  DenseLayer ff2;
  LayerNormLayer norm2;

  // Cache
  Tensor att_output;
  Tensor norm1_output;
  Tensor ff1_output;
  Tensor relu_output;
  Tensor ff2_output;

  TransformerDecoderLayer(int embed_dim, int seq_len, int ff_dim = -1);
  const Tensor &forward(const Tensor &input) override;
  const Tensor &backward(const Tensor &input,
                         const Tensor &grad_output) override;
  void set_training(bool training) override;
  void update(Optimizer *opt) override;
  std::string get_name() const override { return "TransformerDecoderLayer"; }
  int get_params_count() const override {
    return attention.get_params_count() + norm1.get_params_count() +
           ff1.get_params_count() + ff2.get_params_count() + norm2.get_params_count();
  }
  std::unordered_map<std::string, std::string> get_details() const override {
    return {{"embed_dim", std::to_string(embed_dim)},
            {"seq_len", std::to_string(seq_len)},
            {"ff_dim", std::to_string(ff_dim)}};
  }
  void save(std::ostream& os) const override;
  void load(std::istream& is) override;
};

class MaskedMultiHeadAttentionLayer : public Layer {
public:
  int embed_dim;
  int seq_len;
  int num_heads;
  int head_dim;
  
  Tensor W_q, W_k, W_v;
  Tensor b_q, b_k, b_v;
  
  Tensor W_o;
  Tensor b_o;
  
  Tensor grad_W_q, grad_W_k, grad_W_v;
  Tensor grad_b_q, grad_b_k, grad_b_v;

  Tensor grad_W_o;
  Tensor grad_b_o;

  // Caches for backward
  Tensor Q, K, V;
  Tensor scores; // Q * K^T per head
  Tensor attention_weights; // softmax(scores) per head
  Tensor concat_out;

  MaskedMultiHeadAttentionLayer(int embed_dim, int seq_len, int num_heads);
  const Tensor &forward(const Tensor &input) override;
  const Tensor &backward(const Tensor &input,
                         const Tensor &grad_output) override;
  void update(Optimizer *opt) override;
  std::string get_name() const override { return "MaskedMultiHeadAttentionLayer"; }
  int get_params_count() const override {
    return 3 * ((embed_dim * embed_dim) + embed_dim) + (embed_dim * embed_dim) + embed_dim;
  }
  std::unordered_map<std::string, std::string> get_details() const override {
    return {{"embed_dim", std::to_string(embed_dim)},
            {"seq_len", std::to_string(seq_len)},
            {"num_heads", std::to_string(num_heads)}};
  }
  void save(std::ostream& os) const override;
  void load(std::istream& is) override;
};

class TransformerMHDecoderLayer : public Layer {
public:
  int embed_dim;
  int seq_len;
  int ff_dim;
  int num_heads;

  MaskedMultiHeadAttentionLayer attention;
  LayerNormLayer norm1;
  DenseLayer ff1;
  ActivationLayer relu;
  DenseLayer ff2;
  LayerNormLayer norm2;

  // Cache
  Tensor att_output;
  Tensor norm1_output;
  Tensor ff1_output;
  Tensor relu_output;
  Tensor ff2_output;

  TransformerMHDecoderLayer(int embed_dim, int seq_len, int ff_dim = -1, int num_heads = 1);
  const Tensor &forward(const Tensor &input) override;
  const Tensor &backward(const Tensor &input,
                         const Tensor &grad_output) override;
  void set_training(bool training) override;
  void update(Optimizer *opt) override;
  std::string get_name() const override { return "TransformerMHDecoderLayer"; }
  int get_params_count() const override {
    return attention.get_params_count() + norm1.get_params_count() +
           ff1.get_params_count() + ff2.get_params_count() + norm2.get_params_count();
  }
  std::unordered_map<std::string, std::string> get_details() const override {
    return {{"embed_dim", std::to_string(embed_dim)},
            {"seq_len", std::to_string(seq_len)},
            {"ff_dim", std::to_string(ff_dim)},
            {"num_heads", std::to_string(num_heads)}};
  }
  void save(std::ostream& os) const override;
  void load(std::istream& is) override;
};

class MultiHeadAttentionLayer : public Layer {
public:
  int embed_dim;
  int seq_len;
  int num_heads;
  int head_dim;
  
  Tensor W_q, W_k, W_v;
  Tensor b_q, b_k, b_v;
  
  Tensor W_o;
  Tensor b_o;
  
  Tensor grad_W_q, grad_W_k, grad_W_v;
  Tensor grad_b_q, grad_b_k, grad_b_v;

  Tensor grad_W_o;
  Tensor grad_b_o;

  // Caches for backward
  Tensor Q, K, V;
  Tensor scores; // Q * K^T per head
  Tensor attention_weights; // softmax(scores) per head
  Tensor concat_out;

  MultiHeadAttentionLayer(int embed_dim, int seq_len, int num_heads);
  const Tensor &forward(const Tensor &input) override;
  const Tensor &backward(const Tensor &input,
                         const Tensor &grad_output) override;
  void update(Optimizer *opt) override;
  std::string get_name() const override { return "MultiHeadAttentionLayer"; }
  int get_params_count() const override {
    return 3 * ((embed_dim * embed_dim) + embed_dim) + (embed_dim * embed_dim) + embed_dim;
  }
  std::unordered_map<std::string, std::string> get_details() const override {
    return {{"embed_dim", std::to_string(embed_dim)},
            {"seq_len", std::to_string(seq_len)},
            {"num_heads", std::to_string(num_heads)}};
  }
  void save(std::ostream& os) const override;
  void load(std::istream& is) override;
};

class TransformerMHEncoderLayer : public Layer {
public:
  int embed_dim;
  int seq_len;
  int ff_dim;
  int num_heads;

  MultiHeadAttentionLayer attention;
  LayerNormLayer norm1;
  DropoutLayer drop1;
  DenseLayer ff1;
  ActivationLayer relu;
  DenseLayer ff2;
  DropoutLayer drop2;
  LayerNormLayer norm2;

  // Cache
  Tensor att_output;
  Tensor drop1_output;
  Tensor norm1_output;
  Tensor ff1_output;
  Tensor relu_output;
  Tensor ff2_output;
  Tensor drop2_output;

  TransformerMHEncoderLayer(int embed_dim, int seq_len, int ff_dim = -1, int num_heads = 1, scalar dropout_rate = 0.1);
  const Tensor &forward(const Tensor &input) override;
  const Tensor &backward(const Tensor &input,
                         const Tensor &grad_output) override;
  void set_training(bool training) override;
  void update(Optimizer *opt) override;
  std::string get_name() const override { return "TransformerMHEncoderLayer"; }
  int get_params_count() const override {
    return attention.get_params_count() + norm1.get_params_count() + drop1.get_params_count() +
           ff1.get_params_count() + ff2.get_params_count() + drop2.get_params_count() + norm2.get_params_count();
  }
  std::unordered_map<std::string, std::string> get_details() const override {
    return {{"embed_dim", std::to_string(embed_dim)},
            {"seq_len", std::to_string(seq_len)},
            {"ff_dim", std::to_string(ff_dim)},
            {"num_heads", std::to_string(num_heads)}};
  }
  void save(std::ostream& os) const override;
  void load(std::istream& is) override;
};

class GlobalAveragePooling1DLayer : public Layer {
public:
  int seq_len;
  int embed_dim;

  GlobalAveragePooling1DLayer(int seq_len, int embed_dim) : seq_len(seq_len), embed_dim(embed_dim) {}
  
  const Tensor &forward(const Tensor &input) override;
  const Tensor &backward(const Tensor &input, const Tensor &grad_output) override;
  
  std::string get_name() const override { return "GlobalAveragePooling1DLayer"; }
  int get_params_count() const override { return 0; }
  std::unordered_map<std::string, std::string> get_details() const override {
    return {{"seq_len", std::to_string(seq_len)},
            {"embed_dim", std::to_string(embed_dim)}};
  }
};

#endif
