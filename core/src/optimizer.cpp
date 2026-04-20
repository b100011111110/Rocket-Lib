#include "optimizer.h"

SGD::SGD(double lr) : learning_rate(lr) {}

void SGD::update(Tensor &param, const Tensor &grad) {
  for (int i = 0; i < param.rows * param.cols; ++i) {
    param.data[i] -= learning_rate * grad.data[i];
  }
}

Adam::Adam(double lr, double b1, double b2, double eps)
    : learning_rate(lr), beta1(b1), beta2(b2), epsilon(eps), step_count(0) {}

void Adam::begin_step() { step_count++; }

void Adam::update(Tensor &param, const Tensor &grad) {
  if (m.find(param.data) == m.end()) {
    m[param.data] = Tensor(param.rows, param.cols);
    v[param.data] = Tensor(param.rows, param.cols);
    for (int i = 0; i < param.rows * param.cols; ++i) {
      m[param.data].data[i] = 0;
      v[param.data].data[i] = 0;
    }
  }

  int t = std::max(step_count, 1);
  double beta1_correction = 1.0 - std::pow(beta1, t);
  double beta2_correction = 1.0 - std::pow(beta2, t);

  Tensor &m_t = m[param.data];
  Tensor &v_t = v[param.data];

  for (int i = 0; i < param.rows * param.cols; ++i) {
    double g = grad.data[i];
    g = std::max(std::min(g, 10.0), -10.0); // Soft clip to prevent massive explosions

    m_t.data[i] = beta1 * m_t.data[i] + (1.0 - beta1) * g;
    v_t.data[i] =
        beta2 * v_t.data[i] + (1.0 - beta2) * g * g;

    double m_hat = m_t.data[i] / beta1_correction;
    double v_hat = v_t.data[i] / beta2_correction;
    param.data[i] -=
        learning_rate * m_hat / (std::sqrt(v_hat) + epsilon);
  }
}

RMSprop::RMSprop(double lr, double rho, double eps)
    : learning_rate(lr), rho(rho), epsilon(eps) {}

void RMSprop::update(Tensor &param, const Tensor &grad) {
  if (v.find(param.data) == v.end()) {
    v[param.data] = Tensor(param.rows, param.cols);
    for (int i = 0; i < param.rows * param.cols; ++i) {
      v[param.data].data[i] = 0;
    }
  }

  Tensor &v_t = v[param.data];

  for (int i = 0; i < param.rows * param.cols; ++i) {
    v_t.data[i] = rho * v_t.data[i] + (1.0 - rho) * grad.data[i] * grad.data[i];
    param.data[i] -=
        learning_rate * grad.data[i] / (std::sqrt(v_t.data[i]) + epsilon);
  }
}
