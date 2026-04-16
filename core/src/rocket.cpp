#include "tensor.h"
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

PYBIND11_MODULE(rocket, m) {
  m.doc() = "Rocket Core Module providing Tensor functionality";

  py::class_<Tensor>(m, "Tensor")
      .def(py::init<int, int>(), py::arg("rows"), py::arg("cols"))
      .def("print", &Tensor::print)
      .def(py::self + py::self)
      .def(py::self += py::self)
      .def(py::self * py::self)
      .def(py::self *= py::self)
      .def(-py::self)
      .def_readwrite("rows", &Tensor::rows)
      .def_readwrite("cols", &Tensor::cols)
      .def_readwrite("owns_memory", &Tensor::owns_memory);
}
