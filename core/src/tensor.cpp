#include <cmath>
#include <cstring>
#include <random>
#include <stdexcept>
#include <utility>
#include <thread>
#include <vector>
#include <future>

#include "tensor.h"
#include "threadpool.h"

static int next_tensor_id = 0;
static std::mutex id_mutex;

Tensor::Tensor() : rows(0), cols(0), data(nullptr), owns_memory(false) {
    std::lock_guard<std::mutex> lock(id_mutex);
    id = next_tensor_id++;
}

Tensor::Tensor(int r, int c) : rows(r), cols(c), owns_memory(true) {
  {
    std::lock_guard<std::mutex> lock(id_mutex);
    id = next_tensor_id++;
  }
  if (r <= 0 || c <= 0) {
    data = nullptr;
    return;
  }
  data = new scalar[r * c];
  std::memset(data, 0, r * c * sizeof(scalar));
}

void Tensor::init_params() {
  if (!data || rows <= 0 || cols <= 0) return;
  
  scalar limit = std::sqrt(6.0f / (rows + cols));
  static std::mt19937 gen(42);
  std::uniform_real_distribution<scalar> dis(-limit, limit);

  for (int i = 0; i < rows * cols; ++i) {
    data[i] = dis(gen);
  }
}

Tensor::Tensor(int r, int c, scalar *weights)
    : rows(r), cols(c), owns_memory(false) {
  {
    std::lock_guard<std::mutex> lock(id_mutex);
    id = next_tensor_id++;
  }
  if (r <= 0 || c <= 0) {
    throw std::invalid_argument("Tensor dimensions must be positive");
  }
  if (!weights) {
    throw std::invalid_argument("Weights pointer cannot be null");
  }
  data = weights;
}

Tensor::Tensor(const Tensor &other) : rows(other.rows), cols(other.cols), owns_memory(true) {
  {
    std::lock_guard<std::mutex> lock(id_mutex);
    id = next_tensor_id++;
  }
  if (other.data) {
    data = new scalar[rows * cols];
    std::memcpy(data, other.data, rows * cols * sizeof(scalar));
  } else {
    data = nullptr;
  }
}

Tensor::Tensor(Tensor &&other) noexcept : rows(other.rows), cols(other.cols), data(other.data), owns_memory(other.owns_memory), id(other.id) {
  other.rows = 0;
  other.cols = 0;
  other.data = nullptr;
  other.owns_memory = false;
  other.id = -1;
}

Tensor::~Tensor() {
  if (owns_memory && data != nullptr) {
    delete[] data;
    data = nullptr;
  }
}

Tensor &Tensor::operator=(const Tensor &other) {
  if (this != &other) {
    if (owns_memory && data) delete[] data;
    rows = other.rows;
    cols = other.cols;
    owns_memory = true;
    {
      std::lock_guard<std::mutex> lock(id_mutex);
      id = next_tensor_id++;
    }
    if (other.data) {
      data = new scalar[rows * cols];
      std::memcpy(data, other.data, rows * cols * sizeof(scalar));
    } else {
      data = nullptr;
    }
  }
  return *this;
}

Tensor &Tensor::operator=(Tensor &&other) noexcept {
  if (this != &other) {
    if (owns_memory && data) delete[] data;
    rows = other.rows;
    cols = other.cols;
    data = other.data;
    owns_memory = other.owns_memory;
    id = other.id;
    other.rows = 0;
    other.cols = 0;
    other.data = nullptr;
    other.owns_memory = false;
    other.id = -1;
  }
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

  int num_threads = std::thread::hardware_concurrency();
  int total_elements = rows * other.cols;
  
  if (total_elements > 10000 && num_threads > 1) {
    std::vector<std::future<void>> futures;
    int chunk = (rows + num_threads - 1) / num_threads;

    for (int t = 0; t < num_threads; ++t) {
      int start = t * chunk;
      int end = std::min(start + chunk, rows);
      if (start >= end) break;

      futures.push_back(ThreadPool::getInstance().enqueue([this, &other, &result, start, end]() {
        for (int i = start; i < end; ++i) {
          for (int k = 0; k < cols; ++k) {
            scalar temp = data[i * cols + k];
            if (temp == 0) continue;
            for (int j = 0; j < other.cols; ++j) {
              result.data[i * other.cols + j] +=
                  temp * other.data[k * other.cols + j];
            }
          }
        }
      }));
    }
    for (auto &f : futures) f.wait();
  } else {
    for (int i = 0; i < rows; ++i) {
      for (int k = 0; k < cols; ++k) {
        scalar temp = data[i * cols + k];
        if (temp == 0) continue;
        for (int j = 0; j < other.cols; ++j) {
          result.data[i * other.cols + j] +=
              temp * other.data[k * other.cols + j];
        }
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

void Tensor::save(std::ostream& os) const {
  os.write(reinterpret_cast<const char*>(&rows), sizeof(rows));
  os.write(reinterpret_cast<const char*>(&cols), sizeof(cols));
  if (rows > 0 && cols > 0 && data != nullptr) {
    os.write(reinterpret_cast<const char*>(data), rows * cols * sizeof(scalar));
  }
}

void Tensor::load(std::istream& is) {
  int r, c;
  is.read(reinterpret_cast<char*>(&r), sizeof(r));
  is.read(reinterpret_cast<char*>(&c), sizeof(c));
  
  if (r != rows || c != cols) {
      // If dimensions don't match and we own memory, reallocate
      if (owns_memory) {
          delete[] data;
          rows = r;
          cols = c;
          data = new scalar[rows * cols];
      } else {
          throw std::runtime_error("Tensor dimension mismatch during load on non-owning tensor");
      }
  }
  if (rows > 0 && cols > 0 && data != nullptr) {
    is.read(reinterpret_cast<char*>(data), rows * cols * sizeof(scalar));
  }
}
