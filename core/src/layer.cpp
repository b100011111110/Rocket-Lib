#include "layer.h"
#include <algorithm>
#include <cmath>
#include <random>

// InputLayer
const Tensor &InputLayer::forward(const Tensor &input) {
  ensure_output_dims(input.rows, input.cols);
  for (int i = 0; i < input.rows * input.cols; ++i) {
    output.data[i] = input.data[i];
  }
  return output;
}

const Tensor &InputLayer::backward(const Tensor &input,
                                   const Tensor &grad_output) {
  ensure_grad_input_dims(grad_output.rows, grad_output.cols);
  for (int i = 0; i < grad_output.rows * grad_output.cols; ++i) {
    grad_input.data[i] = grad_output.data[i];
  }
  return grad_input;
}

DenseLayer::DenseLayer(int input_dim, int output_dim) {
  weights = Tensor(input_dim, output_dim);
  biases = Tensor(1, output_dim);
  grad_weights = Tensor(input_dim, output_dim);
  grad_biases = Tensor(1, output_dim);

  for (int i = 0; i < output_dim; ++i) {
    biases.data[i] = 0.0;
    grad_biases.data[i] = 0.0;
  }
  for (int i = 0; i < input_dim * output_dim; ++i) {
    grad_weights.data[i] = 0.0;
  }
}

const Tensor &DenseLayer::forward(const Tensor &input) {
  ensure_output_dims(input.rows, weights.cols);
  for (int i = 0; i < input.rows; ++i) {
    for (int j = 0; j < weights.cols; ++j) {
      double sum = biases.data[j];
      for (int k = 0; k < weights.rows; ++k) {
        sum +=
            input.data[i * input.cols + k] * weights.data[k * weights.cols + j];
      }
      output.data[i * weights.cols + j] = sum;
    }
  }
  return output;
}

const Tensor &DenseLayer::backward(const Tensor &input,
                                   const Tensor &grad_output) {
  ensure_grad_input_dims(input.rows, input.cols);

  for (int i = 0; i < grad_weights.rows * grad_weights.cols; ++i)
    grad_weights.data[i] = 0.0;
  for (int i = 0; i < grad_biases.rows * grad_biases.cols; ++i)
    grad_biases.data[i] = 0.0;

  for (int i = 0; i < input.rows; ++i) {
    for (int k = 0; k < input.cols; ++k) {
      double in_val = input.data[i * input.cols + k];
      for (int j = 0; j < grad_output.cols; ++j) {
        grad_weights.data[k * grad_weights.cols + j] +=
            in_val * grad_output.data[i * grad_output.cols + j];
      }
    }
  }

  for (int i = 0; i < grad_output.rows; ++i) {
    for (int j = 0; j < grad_output.cols; ++j) {
      grad_biases.data[j] += grad_output.data[i * grad_output.cols + j];
    }
  }

  for (int i = 0; i < grad_output.rows; ++i) {
    for (int j = 0; j < weights.rows; ++j) {
      double sum = 0.0;
      for (int k = 0; k < weights.cols; ++k) {
        sum += grad_output.data[i * grad_output.cols + k] *
               weights.data[j * weights.cols + k];
      }
      grad_input.data[i * weights.rows + j] = sum;
    }
  }

  return grad_input;
}

DropoutLayer::DropoutLayer(double rate)
    : rate(std::max(0.0, std::min(1.0, rate))), is_training(true) {}

void DropoutLayer::set_training(bool mode) { is_training = mode; }

const Tensor &DropoutLayer::forward(const Tensor &input) {
  ensure_output_dims(input.rows, input.cols);
  if (!is_training || rate >= 1.0) {
    for (int i = 0; i < input.rows * input.cols; ++i)
      output.data[i] = input.data[i];
    return output;
  }

  if (mask.rows != input.rows || mask.cols != input.cols) {
    mask = Tensor(input.rows, input.cols);
  }

  double scale = 1.0 / (1.0 - rate);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(0.0, 1.0);

  for (int i = 0; i < input.rows * input.cols; ++i) {
    if (dis(gen) > rate) {
      mask.data[i] = 1.0;
      output.data[i] = input.data[i] * scale;
    } else {
      mask.data[i] = 0.0;
      output.data[i] = 0.0;
    }
  }
  return output;
}

const Tensor &DropoutLayer::backward(const Tensor &input,
                                     const Tensor &grad_output) {
  ensure_grad_input_dims(grad_output.rows, grad_output.cols);
  if (!is_training || rate >= 1.0) {
    for (int i = 0; i < grad_output.rows * grad_output.cols; ++i) {
      grad_input.data[i] = grad_output.data[i];
    }
    return grad_input;
  }

  double scale = 1.0 / (1.0 - rate);
  for (int i = 0; i < grad_output.rows * grad_output.cols; ++i) {
    grad_input.data[i] = grad_output.data[i] * mask.data[i] * scale;
  }
  return grad_input;
}

RegularizationLayer::RegularizationLayer(double lambda, int type)
    : lambda(lambda), type(type) {}

const Tensor &RegularizationLayer::forward(const Tensor &input) {
  ensure_output_dims(input.rows, input.cols);
  for (int i = 0; i < input.rows * input.cols; ++i) {
    output.data[i] = input.data[i];
  }
  return output;
}

const Tensor &RegularizationLayer::backward(const Tensor &input,
                                            const Tensor &grad_output) {
  ensure_grad_input_dims(grad_output.rows, grad_output.cols);
  for (int i = 0; i < grad_output.rows * grad_output.cols; ++i) {
    double penalty = 0.0;
    if (type == 1) {
      penalty =
          lambda *
          ((input.data[i] > 0) ? 1.0 : ((input.data[i] < 0) ? -1.0 : 0.0));
    } else if (type == 2) {
      penalty = lambda * input.data[i];
    }
    grad_input.data[i] = grad_output.data[i] + penalty;
  }
  return grad_input;
}
