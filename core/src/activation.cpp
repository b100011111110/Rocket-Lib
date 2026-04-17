#include <algorithm>
#include <cmath>

#include "activation.h"
#include "tensor.h"

Tensor ReLU::forward(const Tensor &input) {
  Tensor result(input.rows, input.cols);
  for (int i = 0; i < input.rows * input.cols; ++i) {
    result.data[i] = std::max(0.0, input.data[i]);
  }
  return result;
}

Tensor ReLU::backward(const Tensor &input, const Tensor &grad_output) {
  Tensor grad_input(input.rows, input.cols);
  for (int i = 0; i < input.rows * input.cols; ++i) {
    grad_input.data[i] = (input.data[i] > 0) ? grad_output.data[i] : 0.0;
  }
  return grad_input;
}

Tensor LeakyReLU::forward(const Tensor &input) {
  Tensor result(input.rows, input.cols);
  for (int i = 0; i < input.rows * input.cols; ++i) {
    result.data[i] =
        (input.data[i] > 0) ? input.data[i] : alpha * input.data[i];
  }
  return result;
}

Tensor LeakyReLU::backward(const Tensor &input, const Tensor &grad_output) {
  Tensor grad_input(input.rows, input.cols);
  for (int i = 0; i < input.rows * input.cols; ++i) {
    grad_input.data[i] =
        (input.data[i] > 0) ? grad_output.data[i] : alpha * grad_output.data[i];
  }
  return grad_input;
}

Tensor Tanh::forward(const Tensor &input) {
  Tensor result(input.rows, input.cols);
  for (int i = 0; i < input.rows * input.cols; ++i) {
    result.data[i] = std::tanh(input.data[i]);
  }
  return result;
}

Tensor Tanh::backward(const Tensor &input, const Tensor &grad_output) {
  Tensor grad_input(input.rows, input.cols);
  for (int i = 0; i < input.rows * input.cols; ++i) {
    double t = std::tanh(input.data[i]);
    grad_input.data[i] = grad_output.data[i] * (1.0 - t * t);
  }
  return grad_input;
}

Tensor Sigmoid::forward(const Tensor &input) {
  Tensor result(input.rows, input.cols);
  for (int i = 0; i < input.rows * input.cols; ++i) {
    result.data[i] = 1.0 / (1.0 + std::exp(-input.data[i]));
  }
  return result;
}

Tensor Sigmoid::backward(const Tensor &input, const Tensor &grad_output) {
  Tensor grad_input(input.rows, input.cols);
  for (int i = 0; i < input.rows * input.cols; ++i) {
    double s = 1.0 / (1.0 + std::exp(-input.data[i]));
    grad_input.data[i] = grad_output.data[i] * s * (1.0 - s);
  }
  return grad_input;
}

// Softplus
Tensor Softplus::forward(const Tensor &input) {
  Tensor result(input.rows, input.cols);
  for (int i = 0; i < input.rows * input.cols; ++i) {
    result.data[i] = std::log(1.0 + std::exp(input.data[i]));
  }
  return result;
}

Tensor Softplus::backward(const Tensor &input, const Tensor &grad_output) {
  Tensor grad_input(input.rows, input.cols);
  for (int i = 0; i < input.rows * input.cols; ++i) {
    double s = 1.0 / (1.0 + std::exp(-input.data[i]));
    grad_input.data[i] = grad_output.data[i] * s;
  }
  return grad_input;
}

// Linear
Tensor Linear::forward(const Tensor &input) {
  Tensor result(input.rows, input.cols);
  for (int i = 0; i < input.rows * input.cols; ++i) {
    result.data[i] = input.data[i];
  }
  return result;
}

Tensor Linear::backward(const Tensor &input, const Tensor &grad_output) {
  Tensor grad_input(input.rows, input.cols);
  for (int i = 0; i < input.rows * input.cols; ++i) {
    grad_input.data[i] = grad_output.data[i];
  }
  return grad_input;
}

// Softmax
Tensor Softmax::forward(const Tensor &input) {
  Tensor result(input.rows, input.cols);
  for (int i = 0; i < input.rows; ++i) {
    double max_val = input.data[i * input.cols];
    for (int j = 1; j < input.cols; ++j) {
      if (input.data[i * input.cols + j] > max_val) {
        max_val = input.data[i * input.cols + j];
      }
    }

    double sum = 0.0;
    for (int j = 0; j < input.cols; ++j) {
      double e = std::exp(input.data[i * input.cols + j] - max_val);
      result.data[i * input.cols + j] = e;
      sum += e;
    }

    for (int j = 0; j < input.cols; ++j) {
      result.data[i * input.cols + j] /= sum;
    }
  }
  return result;
}

Tensor Softmax::backward(const Tensor &input, const Tensor &grad_output) {
  Tensor result = this->forward(input);
  Tensor grad_input(input.rows, input.cols);

  for (int i = 0; i < input.rows; ++i) {
    double dot_product = 0.0;
    for (int k = 0; k < input.cols; ++k) {
      dot_product += grad_output.data[i * input.cols + k] *
                     result.data[i * input.cols + k];
    }

    for (int j = 0; j < input.cols; ++j) {
      grad_input.data[i * input.cols + j] =
          result.data[i * input.cols + j] *
          (grad_output.data[i * input.cols + j] - dot_product);
    }
  }
  return grad_input;
}
