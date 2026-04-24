#ifndef TENSOR_H
#define TENSOR_H

#include <iostream>
#include <stdexcept>
#include <vector>

typedef float scalar;

class Tensor {
public:
  int rows, cols;
  scalar *data;
  bool owns_memory;
  int id;

  Tensor();
  Tensor(int r, int c);
  Tensor(int r, int c, scalar *weights);
  Tensor(const Tensor &other);
  Tensor(Tensor &&other) noexcept;
  Tensor &operator=(const Tensor &other);
  Tensor &operator=(Tensor &&other) noexcept;
  ~Tensor();

  void init_params();

  Tensor operator+(const Tensor &other) const;
  Tensor &operator+=(const Tensor &other);
  Tensor operator*(const Tensor &other) const;
  Tensor &operator*=(const Tensor &other);
  Tensor operator-() const; // Unary minus

  void print() const;
  void save(std::ostream& os) const;
  void load(std::istream& is);
};

#endif
