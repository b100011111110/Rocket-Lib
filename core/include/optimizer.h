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
  double learning_rate;

  SGD(double lr = 0.01);
  void update(Tensor &param, const Tensor &grad) override;
};

class Adam : public Optimizer {
public:
  double learning_rate;
  double beta1;
  double beta2;
  double epsilon;
  int step_count;

  std::unordered_map<double *, Tensor> m;
  std::unordered_map<double *, Tensor> v;

  Adam(double lr = 0.001, double b1 = 0.9, double b2 = 0.999,
       double eps = 1e-8);
  void begin_step() override;
  void update(Tensor &param, const Tensor &grad) override;
};

class RMSprop : public Optimizer {
public:
  double learning_rate;
  double rho;
  double epsilon;

  std::unordered_map<double *, Tensor> v;

  RMSprop(double lr = 0.001, double rho = 0.9, double eps = 1e-8);
  void update(Tensor &param, const Tensor &grad) override;
};

#endif
