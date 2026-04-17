#include "loss.h"
#include <cmath>
#include <stdexcept>

double MSE::forward(const Tensor &y_pred, const Tensor &y_true) {
  if (y_pred.rows != y_true.rows || y_pred.cols != y_true.cols) {
    throw std::invalid_argument("MSE: Dimensions mismatch");
  }
  double sum = 0.0;
  int size = y_pred.rows * y_pred.cols;
  for (int i = 0; i < size; ++i) {
    double diff = y_pred.data[i] - y_true.data[i];
    sum += diff * diff;
  }
  return sum / size;
}

Tensor MSE::backward(const Tensor &y_pred, const Tensor &y_true) {
  if (y_pred.rows != y_true.rows || y_pred.cols != y_true.cols) {
    throw std::invalid_argument("MSE Backward: Dimensions mismatch");
  }
  int rows = y_pred.rows;
  int cols = y_pred.cols;
  int size = rows * cols;
  Tensor grad(rows, cols);
  for (int i = 0; i < size; ++i) {
    grad.data[i] = 2.0 * (y_pred.data[i] - y_true.data[i]) / size;
  }
  return grad;
}

double MAE::forward(const Tensor &y_pred, const Tensor &y_true) {
  if (y_pred.rows != y_true.rows || y_pred.cols != y_true.cols) {
    throw std::invalid_argument("MAE: Dimensions mismatch");
  }
  double sum = 0.0;
  int size = y_pred.rows * y_pred.cols;
  for (int i = 0; i < size; ++i) {
    sum += std::abs(y_pred.data[i] - y_true.data[i]);
  }
  return sum / size;
}

Tensor MAE::backward(const Tensor &y_pred, const Tensor &y_true) {
  if (y_pred.rows != y_true.rows || y_pred.cols != y_true.cols) {
    throw std::invalid_argument("MAE Backward: Dimensions mismatch");
  }
  int rows = y_pred.rows;
  int cols = y_pred.cols;
  int size = rows * cols;
  Tensor grad(rows, cols);
  for (int i = 0; i < size; ++i) {
    double diff = y_pred.data[i] - y_true.data[i];
    if (diff > 0)
      grad.data[i] = 1.0 / size;
    else if (diff < 0)
      grad.data[i] = -1.0 / size;
    else
      grad.data[i] = 0.0;
  }
  return grad;
}

Huber::Huber(double d) : delta(d) {}

double Huber::forward(const Tensor &y_pred, const Tensor &y_true) {
  if (y_pred.rows != y_true.rows || y_pred.cols != y_true.cols) {
    throw std::invalid_argument("Huber: Dimensions mismatch");
  }
  double sum = 0.0;
  int size = y_pred.rows * y_pred.cols;
  for (int i = 0; i < size; ++i) {
    double diff = std::abs(y_pred.data[i] - y_true.data[i]);
    if (diff <= delta) {
      sum += 0.5 * diff * diff;
    } else {
      sum += delta * (diff - 0.5 * delta);
    }
  }
  return sum / size;
}

Tensor Huber::backward(const Tensor &y_pred, const Tensor &y_true) {
  if (y_pred.rows != y_true.rows || y_pred.cols != y_true.cols) {
    throw std::invalid_argument("Huber Backward: Dimensions mismatch");
  }
  int rows = y_pred.rows;
  int cols = y_pred.cols;
  int size = rows * cols;
  Tensor grad(rows, cols);
  for (int i = 0; i < size; ++i) {
    double diff = y_pred.data[i] - y_true.data[i];
    if (std::abs(diff) <= delta) {
      grad.data[i] = diff / size;
    } else {
      grad.data[i] = (diff > 0 ? delta : -delta) / size;
    }
  }
  return grad;
}

double BCE::forward(const Tensor &y_pred, const Tensor &y_true) {
  if (y_pred.rows != y_true.rows || y_pred.cols != y_true.cols) {
    throw std::invalid_argument("BCE: Dimensions mismatch");
  }
  double sum = 0.0;
  int size = y_pred.rows * y_pred.cols;
  const double epsilon = 1e-15;
  for (int i = 0; i < size; ++i) {
    double p = std::max(epsilon, std::min(1.0 - epsilon, y_pred.data[i]));
    double t = y_true.data[i];
    sum -= (t * std::log(p) + (1.0 - t) * std::log(1.0 - p));
  }
  return sum / size;
}

Tensor BCE::backward(const Tensor &y_pred, const Tensor &y_true) {
  if (y_pred.rows != y_true.rows || y_pred.cols != y_true.cols) {
    throw std::invalid_argument("BCE Backward: Dimensions mismatch");
  }
  int rows = y_pred.rows;
  int cols = y_pred.cols;
  int size = rows * cols;
  Tensor grad(rows, cols);
  const double epsilon = 1e-15;
  for (int i = 0; i < size; ++i) {
    double p = std::max(epsilon, std::min(1.0 - epsilon, y_pred.data[i]));
    double t = y_true.data[i];
    grad.data[i] = (-(t / p) + (1.0 - t) / (1.0 - p)) / size;
  }
  return grad;
}

double CCE::forward(const Tensor &y_pred, const Tensor &y_true) {
  if (y_pred.rows != y_true.rows || y_pred.cols != y_true.cols) {
    throw std::invalid_argument("CCE: Dimensions mismatch");
  }
  double sum = 0.0;
  int size = y_pred.rows * y_pred.cols;
  const double epsilon = 1e-15;
  for (int i = 0; i < size; ++i) {
    double p = std::max(epsilon, std::min(1.0 - epsilon, y_pred.data[i]));
    sum -= (y_true.data[i] * std::log(p));
  }
  return sum / size;
}

Tensor CCE::backward(const Tensor &y_pred, const Tensor &y_true) {
  if (y_pred.rows != y_true.rows || y_pred.cols != y_true.cols) {
    throw std::invalid_argument("CCE Backward: Dimensions mismatch");
  }
  int rows = y_pred.rows;
  int cols = y_pred.cols;
  int size = rows * cols;
  Tensor grad(rows, cols);
  const double epsilon = 1e-15;
  for (int i = 0; i < size; ++i) {
    double p = std::max(epsilon, std::min(1.0 - epsilon, y_pred.data[i]));
    grad.data[i] = (-y_true.data[i] / p) / size;
  }
  return grad;
}
