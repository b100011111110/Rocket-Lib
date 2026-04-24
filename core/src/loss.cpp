#include "loss.h"
#include <cmath>
#include <stdexcept>

scalar MSE::forward(const Tensor &y_pred, const Tensor &y_true) {
  if (y_pred.rows != y_true.rows || y_pred.cols != y_true.cols) {
    throw std::invalid_argument("MSE: Dimensions mismatch");
  }
  scalar sum = 0.0f;
  int size = y_pred.rows * y_pred.cols;
  for (int i = 0; i < size; ++i) {
    scalar diff = y_pred.data[i] - y_true.data[i];
    sum += diff * diff;
  }
  return sum / size;
}

Tensor MSE::backward(const Tensor &y_pred, const Tensor &y_true) {
  if (y_pred.rows != y_true.rows || y_pred.cols != y_true.cols) {
    throw std::invalid_argument("MSE Backward: Dimensions mismatch");
  }
  int size = y_pred.rows * y_pred.cols;
  Tensor grad(y_pred.rows, y_pred.cols);
  for (int i = 0; i < size; ++i) {
    grad.data[i] = 2.0f * (y_pred.data[i] - y_true.data[i]) / size;
  }
  return grad;
}

scalar MAE::forward(const Tensor &y_pred, const Tensor &y_true) {
  if (y_pred.rows != y_true.rows || y_pred.cols != y_true.cols) {
    throw std::invalid_argument("MAE: Dimensions mismatch");
  }
  scalar sum = 0.0f;
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
  int size = y_pred.rows * y_pred.cols;
  Tensor grad(y_pred.rows, y_pred.cols);
  for (int i = 0; i < size; ++i) {
    scalar diff = y_pred.data[i] - y_true.data[i];
    if (diff > 0)
      grad.data[i] = 1.0f / size;
    else if (diff < 0)
      grad.data[i] = -1.0f / size;
    else
      grad.data[i] = 0.0f;
  }
  return grad;
}

Huber::Huber(scalar d) : delta(d) {}

scalar Huber::forward(const Tensor &y_pred, const Tensor &y_true) {
  if (y_pred.rows != y_true.rows || y_pred.cols != y_true.cols) {
    throw std::invalid_argument("Huber: Dimensions mismatch");
  }
  scalar sum = 0.0f;
  int size = y_pred.rows * y_pred.cols;
  for (int i = 0; i < size; ++i) {
    scalar diff = std::abs(y_pred.data[i] - y_true.data[i]);
    if (diff <= delta) {
      sum += 0.5f * diff * diff;
    } else {
      sum += delta * (diff - 0.5f * delta);
    }
  }
  return sum / size;
}

Tensor Huber::backward(const Tensor &y_pred, const Tensor &y_true) {
  if (y_pred.rows != y_true.rows || y_pred.cols != y_true.cols) {
    throw std::invalid_argument("Huber Backward: Dimensions mismatch");
  }
  int size = y_pred.rows * y_pred.cols;
  Tensor grad(y_pred.rows, y_pred.cols);
  for (int i = 0; i < size; ++i) {
    scalar diff = y_pred.data[i] - y_true.data[i];
    if (std::abs(diff) <= delta) {
      grad.data[i] = diff / size;
    } else {
      grad.data[i] = (diff > 0 ? delta : -delta) / size;
    }
  }
  return grad;
}

scalar BCE::forward(const Tensor &y_pred, const Tensor &y_true) {
  if (y_pred.rows != y_true.rows || y_pred.cols != y_true.cols) {
    throw std::invalid_argument("BCE: Dimensions mismatch");
  }
  scalar sum = 0.0f;
  int size = y_pred.rows * y_pred.cols;
  const scalar epsilon = 1e-7f;
  for (int i = 0; i < size; ++i) {
    scalar p = std::max(epsilon, std::min(1.0f - epsilon, y_pred.data[i]));
    scalar t = y_true.data[i];
    sum -= (t * std::log(p) + (1.0f - t) * std::log(1.0f - p));
  }
  return sum / size;
}

Tensor BCE::backward(const Tensor &y_pred, const Tensor &y_true) {
  if (y_pred.rows != y_true.rows || y_pred.cols != y_true.cols) {
    throw std::invalid_argument("BCE Backward: Dimensions mismatch");
  }
  int size = y_pred.rows * y_pred.cols;
  Tensor grad(y_pred.rows, y_pred.cols);
  const scalar epsilon = 1e-7f;
  for (int i = 0; i < size; ++i) {
    scalar p = std::max(epsilon, std::min(1.0f - epsilon, y_pred.data[i]));
    scalar t = y_true.data[i];
    grad.data[i] = (p - t) / (p * (1.0f - p) * size);
  }
  return grad;
}

scalar BCEWithLogits::forward(const Tensor &logits, const Tensor &y_true) {
  if (logits.rows != y_true.rows || logits.cols != y_true.cols) {
    throw std::invalid_argument("BCEWithLogits: Dimensions mismatch");
  }
  scalar sum = 0.0f;
  int size = logits.rows * logits.cols;
  for (int i = 0; i < size; ++i) {
    scalar x = logits.data[i];
    scalar t = y_true.data[i];
    sum += std::max(x, 0.0f) - x * t + std::log(1.0f + std::exp(-std::abs(x)));
  }
  return sum / size;
}

Tensor BCEWithLogits::backward(const Tensor &logits, const Tensor &y_true) {
  if (logits.rows != y_true.rows || logits.cols != y_true.cols) {
    throw std::invalid_argument("BCEWithLogits: Dimensions mismatch");
  }
  int size = logits.rows * logits.cols;
  Tensor grad(logits.rows, logits.cols);
  for (int i = 0; i < size; ++i) {
    scalar p = 1.0f / (1.0f + std::exp(-logits.data[i]));
    scalar t = y_true.data[i];
    grad.data[i] = (p - t) / size;
  }
  return grad;
}

scalar CCE::forward(const Tensor &y_pred, const Tensor &y_true) {
  if (y_pred.rows != y_true.rows || y_pred.cols != y_true.cols) {
    throw std::invalid_argument("CCE: Dimensions mismatch");
  }
  scalar sum = 0.0f;
  int size = y_pred.rows * y_pred.cols;
  const scalar epsilon = 1e-15f;
  for (int i = 0; i < size; ++i) {
    scalar p = std::max(epsilon, std::min(1.0f - epsilon, y_pred.data[i]));
    sum -= (y_true.data[i] * std::log(p));
  }
  return sum / size;
}

Tensor CCE::backward(const Tensor &y_pred, const Tensor &y_true) {
  if (y_pred.rows != y_true.rows || y_pred.cols != y_true.cols) {
    throw std::invalid_argument("CCE Backward: Dimensions mismatch");
  }
  int size = y_pred.rows * y_pred.cols;
  Tensor grad(y_pred.rows, y_pred.cols);
  const scalar epsilon = 1e-15f;
  for (int i = 0; i < size; ++i) {
    scalar p = std::max(epsilon, std::min(1.0f - epsilon, y_pred.data[i]));
    grad.data[i] = (-y_true.data[i] / p) / size;
  }
  return grad;
}
