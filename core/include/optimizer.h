#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include "tensor.h"
#include <cmath>
#include <unordered_map>

class Optimizer {
public:
  virtual ~Optimizer() = default;
  virtual void begin_step() {}
  virtual void update(Tensor &param, const Tensor &grad) = 0;
};

class SGD : public Optimizer {
public:
  scalar learning_rate;

  SGD(scalar lr = 0.01);
  void update(Tensor &param, const Tensor &grad) override;
};

class Adam : public Optimizer {
public:
  scalar learning_rate, beta1, beta2, epsilon;
  int step_count;
  std::unordered_map<int, Tensor> m;
  std::unordered_map<int, Tensor> v;

  Adam(scalar lr = 0.001, scalar b1 = 0.9, scalar b2 = 0.999, scalar eps = 1e-8);
  void update(Tensor &param, const Tensor &grad) override;
  void begin_step() override;
};

class RMSprop : public Optimizer {
public:
  scalar learning_rate, rho, epsilon;
  std::unordered_map<int, Tensor> v;

  RMSprop(scalar lr = 0.001, scalar rho = 0.9, scalar eps = 1e-8);
  void update(Tensor &param, const Tensor &grad) override;
};

#endif
