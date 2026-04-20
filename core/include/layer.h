#ifndef LAYER_H
#define LAYER_H

#include "tensor.h"

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
};

class InputLayer : public Layer {
public:
  const Tensor &forward(const Tensor &input) override;
  const Tensor &backward(const Tensor &input,
                         const Tensor &grad_output) override;
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
};

class RegularizationLayer : public Layer {
public:
  double lambda;
  int type; // 1 for L1, 2 for L2
  RegularizationLayer(double lambda, int type = 2);
  const Tensor &forward(const Tensor &input) override;
  const Tensor &backward(const Tensor &input,
                         const Tensor &grad_output) override;
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
};

#endif
