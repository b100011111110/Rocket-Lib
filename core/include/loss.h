#ifndef LOSS_H
#define LOSS_H

#include "tensor.h"

class Loss {
public:
  virtual ~Loss() = default;
  virtual double forward(const Tensor &y_pred, const Tensor &y_true) = 0;
  virtual Tensor backward(const Tensor &y_pred, const Tensor &y_true) = 0;
};

// forward is for error calculation
// backward is for back propogation

class MSE : public Loss {
public:
  double forward(const Tensor &y_pred, const Tensor &y_true) override;
  Tensor backward(const Tensor &y_pred, const Tensor &y_true) override;
};

class MAE : public Loss {
public:
  double forward(const Tensor &y_pred, const Tensor &y_true) override;
  Tensor backward(const Tensor &y_pred, const Tensor &y_true) override;
};

class Huber : public Loss {
private:
  double delta;

public:
  Huber(double delta = 1.0);
  double forward(const Tensor &y_pred, const Tensor &y_true) override;
  Tensor backward(const Tensor &y_pred, const Tensor &y_true) override;
};

class BCE : public Loss {
public:
  double forward(const Tensor &y_pred, const Tensor &y_true) override;
  Tensor backward(const Tensor &y_pred, const Tensor &y_true) override;
};

class CCE : public Loss {
public:
  double forward(const Tensor &y_pred, const Tensor &y_true) override;
  Tensor backward(const Tensor &y_pred, const Tensor &y_true) override;
};

#endif
