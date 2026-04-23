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
  virtual void update(Optimizer *opt) {}
  virtual std::string get_name() const = 0;
  virtual int get_params_count() const { return 0; }
  virtual std::unordered_map<std::string, std::string> get_details() const { return {}; }
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
  double rate;
  Tensor mask;
  bool is_training;

  DropoutLayer(double rate = 0.5);
  void set_training(bool mode);
  const Tensor &forward(const Tensor &input) override;
  const Tensor &backward(const Tensor &input,
                         const Tensor &grad_output) override;
  std::string get_name() const override { return "DropoutLayer"; }
  std::unordered_map<std::string, std::string> get_details() const override {
    return {{"rate", std::to_string(rate)}};
  }
};

class RegularizationLayer : public Layer {
public:
  double lambda;
  int type; // 1 for L1, 2 for L2
  RegularizationLayer(double lambda, int type = 2);
  const Tensor &forward(const Tensor &input) override;
  const Tensor &backward(const Tensor &input,
                         const Tensor &grad_output) override;
  std::string get_name() const override { return "RegularizationLayer"; }
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

#endif
