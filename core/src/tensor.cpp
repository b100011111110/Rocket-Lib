#include <cmath>
#include <cstring>
#include <random>
#include <stdexcept>
#include <utility>

#include "tensor.h"

Tensor::Tensor() : rows(0), cols(0), data(nullptr), owns_memory(false) {}

Tensor::Tensor(int r, int c) : rows(r), cols(c), owns_memory(true) {
  if (r <= 0 || c <= 0) {
    throw std::invalid_argument("Tensor dimensions must be positive");
  }
  data = new double[r * c];

  // Xavier/Glorot Initialization
  // For Xavier Initialization, variance is 2.0 / (fan_in + fan_out)
  double limit = std::sqrt(6.0 / (rows + cols)); // Uniform Xavier

  static std::mt19937 gen(42);
  std::uniform_real_distribution<> dis(-limit, limit);

  for (int i = 0; i < r * c; ++i) {
    data[i] = dis(gen);
  }
}

Tensor::Tensor(int r, int c, double *weights)
    : rows(r), cols(c), owns_memory(false) {
  if (r <= 0 || c <= 0) {
    throw std::invalid_argument("Tensor dimensions must be positive");
  }
  if (!weights) {
    throw std::invalid_argument("Weights pointer cannot be null");
  }
  data = weights;
}

Tensor::Tensor(const Tensor &other)
    : rows(other.rows), cols(other.cols), owns_memory(true) {
  if (rows > 0 && cols > 0) {
    data = new double[rows * cols];
    std::memcpy(data, other.data, rows * cols * sizeof(double));
  } else {
    data = nullptr;
  }
}

Tensor::~Tensor() {
  if (owns_memory && data != nullptr) {
    delete[] data;
  }
}

Tensor &Tensor::operator=(Tensor other) {
  std::swap(rows, other.rows);
  std::swap(cols, other.cols);
  std::swap(data, other.data);
  std::swap(owns_memory, other.owns_memory);
  return *this;
}

Tensor Tensor::operator+(const Tensor &other) const {
  if (rows != other.rows || cols != other.cols) {
    throw std::invalid_argument("Matrix dimensions must match for addition");
  }
  Tensor result(rows, cols);
  for (int i = 0; i < rows * cols; ++i) {
    result.data[i] = data[i] + other.data[i];
  }
  return result;
}

Tensor &Tensor::operator+=(const Tensor &other) {
  if (rows != other.rows || cols != other.cols) {
    throw std::invalid_argument("Matrix dimensions must match for addition");
  }
  for (int i = 0; i < rows * cols; ++i) {
    data[i] += other.data[i];
  }
  return *this;
}

Tensor Tensor::operator*(const Tensor &other) const {
  if (cols != other.rows) {
    throw std::invalid_argument(
        "Dimensions mismatch for matrix multiplication");
  }
  Tensor result(rows, other.cols);
  for (int i = 0; i < rows * other.cols; ++i) {
    result.data[i] = 0.0;
  }

  for (int i = 0; i < rows; ++i) {
    for (int k = 0; k < cols; ++k) {
      double temp = data[i * cols + k];
      for (int j = 0; j < other.cols; ++j) {
        result.data[i * other.cols + j] +=
            temp * other.data[k * other.cols + j];
      }
    }
  }
  return result;
}

Tensor &Tensor::operator*=(const Tensor &other) {
  *this = *this * other;
  return *this;
}

Tensor Tensor::operator-() const {
  Tensor result(rows, cols);
  for (int i = 0; i < rows * cols; ++i) {
    result.data[i] = -data[i];
  }
  return result;
}

void Tensor::print() const {
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      std::cout << data[i * cols + j] << " ";
    }
    std::cout << "\n";
  }
}
