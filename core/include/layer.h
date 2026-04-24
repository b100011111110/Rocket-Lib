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

#endif
