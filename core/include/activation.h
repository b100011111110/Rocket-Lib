#ifndef ACTIVATION_H
#define ACTIVATION_H

#include "tensor.h"

class Activation {
public:
  virtual ~Activation() = default;
  virtual Tensor forward(const Tensor &input) = 0;
  virtual Tensor backward(const Tensor &input, const Tensor &grad_output) = 0;
};

class Linear : public Activation {
public:
  Tensor forward(const Tensor &input) override;
  Tensor backward(const Tensor &input, const Tensor &grad_output) override;
};

class ReLU : public Activation {
public:
  Tensor forward(const Tensor &input) override;
  Tensor backward(const Tensor &input, const Tensor &grad_output) override;
};

class LeakyReLU : public Activation {
public:
  double alpha;
  LeakyReLU(double alpha = 0.01) : alpha(alpha) {}
  Tensor forward(const Tensor &input) override;
  Tensor backward(const Tensor &input, const Tensor &grad_output) override;
};

class Tanh : public Activation {
public:
  Tensor forward(const Tensor &input) override;
  Tensor backward(const Tensor &input, const Tensor &grad_output) override;
};

class Sigmoid : public Activation {
public:
  Tensor forward(const Tensor &input) override;
  Tensor backward(const Tensor &input, const Tensor &grad_output) override;
};

class Softplus : public Activation {
public:
  Tensor forward(const Tensor &input) override;
  Tensor backward(const Tensor &input, const Tensor &grad_output) override;
};

class Softmax : public Activation {
public:
  Tensor forward(const Tensor &input) override;
  Tensor backward(const Tensor &input, const Tensor &grad_output) override;
};

#endif
