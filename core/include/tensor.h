#ifndef TENSOR_H
#define TENSOR_H

#include <iostream>
#include <stdexcept>

class Tensor {
public:
  int rows;
  int cols;
  double *data;
  bool owns_memory;

  Tensor();
  Tensor(int r, int c);
  Tensor(int r, int c, double *weights);

  Tensor(const Tensor &other);

  ~Tensor();
  Tensor &operator=(Tensor other);

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
