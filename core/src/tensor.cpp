#include <cmath>
#include <chrono>
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
  static std::mt19937 gen([]() {
    const char *seed_str = std::getenv("ROCKET_SEED");
    if (seed_str) {
      return std::mt19937(std::stoul(seed_str));
    }
    return std::mt19937(std::chrono::steady_clock::now().time_since_epoch().count());
  }());
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
  std::memset(result.data, 0, rows * other.cols * sizeof(scalar));

  const int BLOCK_SIZE = 64;

  #pragma omp parallel for collapse(2) schedule(dynamic)
  for (int ih = 0; ih < rows; ih += BLOCK_SIZE) {
    for (int jh = 0; jh < other.cols; jh += BLOCK_SIZE) {
      int i_max = std::min(ih + BLOCK_SIZE, rows);
      int j_max = std::min(jh + BLOCK_SIZE, other.cols);
      
      for (int kh = 0; kh < cols; kh += BLOCK_SIZE) {
        int k_max = std::min(kh + BLOCK_SIZE, cols);
        
        for (int i = ih; i < i_max; ++i) {
          int base_res = i * other.cols;
          int base_this = i * cols;
          for (int k = kh; k < k_max; ++k) {
            scalar temp = data[base_this + k];
            if (temp == 0.0f) continue;
            
            int base_other = k * other.cols;
            for (int j = jh; j < j_max; ++j) {
              result.data[base_res + j] += temp * other.data[base_other + j];
            }
          }
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
