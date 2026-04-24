#include "activation.h"
#include "layer.h"
#include "loss.h"
#include "model.h"
#include "optimizer.h"
#include "tensor.h"
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

// Trampoline class for Layer since it has pure virtual methods
class PyLayer : public Layer {
public:
  using Layer::Layer; // Inherit constructors
  const Tensor &forward(const Tensor &input) override {
    PYBIND11_OVERRIDE_PURE(
        const Tensor &, /* Return type */
        Layer,          /* Parent class */
        forward,        /* Name of function in C++ (must match Python name) */
        input           /* Argument(s) */
    );
  }
  const Tensor &backward(const Tensor &input,
                         const Tensor &grad_output) override {
    PYBIND11_OVERRIDE_PURE(const Tensor &, Layer, backward, input, grad_output);
  }
  std::string get_name() const override {
    PYBIND11_OVERRIDE_PURE(std::string, Layer, get_name);
  }
  int get_params_count() const override {
    PYBIND11_OVERRIDE(int, Layer, get_params_count);
  }
  using DetailsMap = std::unordered_map<std::string, std::string>;
  DetailsMap get_details() const override {
    PYBIND11_OVERRIDE(DetailsMap, Layer, get_details);
  }
};

// Trampoline class for Loss
class PyLoss : public Loss {
public:
  using Loss::Loss;
  scalar forward(const Tensor &y_pred, const Tensor &y_true) override {
    PYBIND11_OVERRIDE_PURE(scalar, Loss, forward, y_pred, y_true);
  }
  Tensor backward(const Tensor &y_pred, const Tensor &y_true) override {
    PYBIND11_OVERRIDE_PURE(Tensor, Loss, backward, y_pred, y_true);
  }
};

// Trampoline class for Optimizer
class PyOptimizer : public Optimizer {
public:
  using Optimizer::Optimizer;
  void update(Tensor &param, const Tensor &grad) override {
    PYBIND11_OVERRIDE_PURE(void, Optimizer, update, param, grad);
  }
};

// Trampoline class for Activation
class PyActivation : public Activation {
public:
  using Activation::Activation;
  Tensor forward(const Tensor &input) override {
    PYBIND11_OVERRIDE_PURE(Tensor, Activation, forward, input);
  }
  Tensor backward(const Tensor &input, const Tensor &grad_output) override {
    PYBIND11_OVERRIDE_PURE(Tensor, Activation, backward, input, grad_output);
  }
};

PYBIND11_MODULE(rocket, m) {
  m.doc() = "Rocket Core Module providing Tensor and ML functionality";

  py::class_<Tensor>(m, "Tensor")
      .def(py::init<int, int>(), py::arg("rows"), py::arg("cols"))
      .def(py::init([](py::array_t<scalar> b) {
            py::buffer_info info = b.request();
            if (info.ndim != 2) throw std::runtime_error("Incompatible buffer dimension!");
            Tensor t(info.shape[0], info.shape[1]);
            std::memcpy(t.data, info.ptr, info.shape[0] * info.shape[1] * sizeof(scalar));
            return t;
        }))
      .def("print", &Tensor::print)
      .def(py::self + py::self)
      .def(py::self += py::self)
      .def(py::self * py::self)
      .def(py::self *= py::self)
      .def(-py::self)
      .def_readwrite("rows", &Tensor::rows)
      .def_readwrite("cols", &Tensor::cols)
      .def_readwrite("owns_memory", &Tensor::owns_memory)
      .def("set_val",
           [](Tensor &t, int i, int j, scalar val) {
             if (i >= 0 && i < t.rows && j >= 0 && j < t.cols) {
               t.data[i * t.cols + j] = val;
             }
           })
      .def("get_val", [](const Tensor &t, int i, int j) {
        if (i >= 0 && i < t.rows && j >= 0 && j < t.cols) {
          return t.data[i * t.cols + j];
        }
        throw std::out_of_range("Tensor indices out of range");
      });

  // Bind Layers
  py::class_<Layer, PyLayer>(m, "Layer").def(py::init<>());

  py::class_<InputLayer, Layer>(m, "InputLayer").def(py::init<>());

  py::class_<DenseLayer, Layer>(m, "DenseLayer")
      .def(py::init<int, int>(), py::arg("input_dim"), py::arg("output_dim"))
      .def_readwrite("weights", &DenseLayer::weights)
      .def_readwrite("biases", &DenseLayer::biases)
      .def_readwrite("grad_weights", &DenseLayer::grad_weights)
      .def_readwrite("grad_biases", &DenseLayer::grad_biases)
      .def_readwrite("grad_input", &DenseLayer::grad_input);

  py::class_<DropoutLayer, Layer>(m, "DropoutLayer")
      .def(py::init<scalar>(), py::arg("rate") = 0.5)
      .def("set_training", &DropoutLayer::set_training, py::arg("mode"));

  py::class_<RegularizationLayer, Layer>(m, "RegularizationLayer")
      .def(py::init<scalar, int>(), py::arg("lambda"), py::arg("type") = 2);

  py::class_<ActivationLayer, Layer>(m, "ActivationLayer")
      .def(py::init<Activation *>(), py::arg("fn"), py::keep_alive<1, 2>())
      .def_readwrite("fn", &ActivationLayer::activation_fn);

  py::class_<RNNLayer, Layer>(m, "RNNLayer")
      .def(py::init<int, int, int, bool>(), py::arg("input_dim"), py::arg("hidden_dim"), py::arg("seq_len"), py::arg("return_sequences") = false)
      .def_readwrite("weights_ih", &RNNLayer::weights_ih)
      .def_readwrite("weights_hh", &RNNLayer::weights_hh)
      .def_readwrite("biases", &RNNLayer::biases);

  py::class_<LSTMLayer, Layer>(m, "LSTMLayer")
      .def(py::init<int, int, int, bool>(), py::arg("input_dim"), py::arg("hidden_dim"), py::arg("seq_len"), py::arg("return_sequences") = false)
      .def_readwrite("weights_ih", &LSTMLayer::weights_ih)
      .def_readwrite("weights_hh", &LSTMLayer::weights_hh)
      .def_readwrite("biases", &LSTMLayer::biases);

  // Bind Activations
  py::class_<Activation, PyActivation>(m, "Activation").def(py::init<>());

  py::class_<ReLU, Activation>(m, "ReLU").def(py::init<>());

  py::class_<Sigmoid, Activation>(m, "Sigmoid").def(py::init<>());

  py::class_<Tanh, Activation>(m, "Tanh").def(py::init<>());

  py::class_<Linear, Activation>(m, "Linear").def(py::init<>());

  // Bind Losses
  py::class_<Loss, PyLoss>(m, "Loss").def(py::init<>());

  py::class_<MSE, Loss>(m, "MSE").def(py::init<>());

  py::class_<BCE, Loss>(m, "BCE").def(py::init<>());

  py::class_<BCEWithLogits, Loss>(m, "BCEWithLogits").def(py::init<>());

  // Bind Optimizers
  py::class_<Optimizer, PyOptimizer>(m, "Optimizer").def(py::init<>());

  py::class_<SGD, Optimizer>(m, "SGD").def(py::init<scalar>(),
                                           py::arg("lr") = 0.01);

  py::class_<Adam, Optimizer>(m, "Adam").def(
      py::init<scalar, scalar, scalar, scalar>(), py::arg("lr") = 0.001,
      py::arg("b1") = 0.9, py::arg("b2") = 0.999, py::arg("eps") = 1e-8);

  // Bind Model
  py::class_<Model>(m, "Model")
      .def(py::init<>())
      .def("add", &Model::add, py::arg("layer"), py::arg("prev_layers"),
           py::keep_alive<1, 2>())
      .def("compile", &Model::compile, py::arg("loss"), py::arg("opt"),
           py::keep_alive<1, 2>(), py::keep_alive<1, 3>())
      .def("setInputOutputLayers", &Model::setInputOutputLayers,
           py::arg("inputs"), py::arg("outputs"))
      .def("train", &Model::train, py::arg("xtrain"), py::arg("ytrain"),
           py::arg("xtest"), py::arg("ytest"), py::arg("epochs"),
           py::arg("batch_size") = 1)
      .def("predict", &Model::predict, py::arg("x"))
      .def("test", &Model::test, py::arg("x"), py::arg("y"), py::arg("metric"))
      .def("summary", &Model::summary)
      .def("details", &Model::details)
      .def("weights", &Model::weights)
      .def("save", &Model::save, py::arg("path"))
      .def("load", &Model::load, py::arg("path"));
}
