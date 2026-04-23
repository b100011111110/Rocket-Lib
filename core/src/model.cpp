#include "model.h"
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <queue>
#include <random>

Model::Model() : loss_fn(nullptr), optimizer(nullptr) {}

Model::~Model() {}

void Model::add(Layer *layer, const std::vector<Layer *> prev_layers) {
  if (prev_layers_map.find(layer) == prev_layers_map.end()) {
    prev_layers_map[layer] = prev_layers;
  }
  for (Layer *prev : prev_layers) {
    next_layers_map[prev].push_back(layer);
  }
}

void Model::setInputOutputLayers(const std::vector<Layer *> &inputs,
                                 const std::vector<Layer *> &outputs) {
  this->inputs = inputs;
  this->outputs = outputs;

  // Topological sort using Kahn's algorithm
  std::unordered_map<Layer *, int> in_degree;

  for (auto &pair : prev_layers_map) {
    in_degree[pair.first] = pair.second.size();
  }
  for (Layer *input_layer : inputs) {
    in_degree[input_layer] = 0;
  }

  std::queue<Layer *> q;
  for (auto &pair : in_degree) {
    if (pair.second == 0) {
      q.push(pair.first);
    }
  }

  topological_order.clear();
  while (!q.empty()) {
    Layer *curr = q.front();
    q.pop();
    topological_order.push_back(curr);

    for (Layer *next : next_layers_map[curr]) {
      in_degree[next]--;
      if (in_degree[next] == 0) {
        q.push(next);
      }
    }
  }
}

void Model::compile(Loss *loss, Optimizer *opt) {
  this->loss_fn = loss;
  this->optimizer = opt;
}

std::vector<Tensor> Model::predict(const std::vector<Tensor> &x) {
  std::unordered_map<Layer *, Tensor> layer_outputs;

  for (size_t i = 0; i < inputs.size() && i < x.size(); ++i) {
    layer_outputs[inputs[i]] = inputs[i]->forward(x[i]);
  }

  for (Layer *layer : topological_order) {
    if (std::find(inputs.begin(), inputs.end(), layer) != inputs.end())
      continue;

    // Sum inputs from previous layers
    const auto &prevs = prev_layers_map[layer];
    if (prevs.empty())
      continue;

    Tensor combined_input = layer_outputs[prevs[0]];
    for (size_t i = 1; i < prevs.size(); ++i) {
      combined_input += layer_outputs[prevs[i]];
    }

    layer_outputs[layer] = layer->forward(combined_input);
  }

  std::vector<Tensor> result;
  for (Layer *out_layer : outputs) {
    result.push_back(layer_outputs[out_layer]);
  }
  return result;
}

void Model::train(const std::vector<Tensor> &xtrain,
                  const std::vector<Tensor> &ytrain,
                  const std::vector<Tensor> &xtest,
                  const std::vector<Tensor> &ytest, int epochs,
                  int batch_size) {

  if (xtrain.empty() || ytrain.empty()) {
    throw std::invalid_argument("Training data cannot be empty");
  }
  if (xtrain.size() != ytrain.size()) {
    throw std::invalid_argument("xtrain and ytrain must have the same size");
  }
  if (batch_size <= 0) {
    throw std::invalid_argument("batch_size must be positive");
  }

  std::vector<int> indices(xtrain.size());
  for (size_t i = 0; i < indices.size(); ++i) {
    indices[i] = i;
  }

  const char *seed_str = std::getenv("ROCKET_SEED");
  unsigned int seed = seed_str ? std::stoul(seed_str) : 42;
  std::mt19937 gen(seed);

  for (int epoch = 0; epoch < epochs; ++epoch) {
    auto epoch_start = std::chrono::high_resolution_clock::now();
    double total_loss = 0.0;

    // Shuffle indices at the start of each epoch if enabled
    if (std::getenv("ROCKET_SHUFFLE") == nullptr ||
        std::string(std::getenv("ROCKET_SHUFFLE")) != "0") {
      std::shuffle(indices.begin(), indices.end(), gen);
    }

    // Pre-allocate maps to avoid per-batch allocation
    std::unordered_map<Layer *, Tensor> layer_outputs;
    std::unordered_map<Layer *, Tensor> layer_grads;

    for (size_t batch_start = 0; batch_start < indices.size();
         batch_start += batch_size) {
      int current_batch_size =
          std::min(batch_size, static_cast<int>(indices.size() - batch_start));
      Tensor x_batch(current_batch_size, xtrain[0].cols);
      Tensor y_batch(current_batch_size, ytrain[0].cols);

      for (int batch_row = 0; batch_row < current_batch_size; ++batch_row) {
        int sample_idx = indices[batch_start + batch_row];
        const Tensor &x_sample = xtrain[sample_idx];
        const Tensor &y_sample = ytrain[sample_idx];

        for (int col = 0; col < x_sample.cols; ++col) {
          x_batch.data[batch_row * x_batch.cols + col] = x_sample.data[col];
        }
        for (int col = 0; col < y_sample.cols; ++col) {
          y_batch.data[batch_row * y_batch.cols + col] = y_sample.data[col];
        }
      }

      std::vector<Tensor> x_single = {x_batch};

      // Forward pass - REUSING MAP
      layer_outputs[inputs[0]] = inputs[0]->forward(x_single[0]);

      for (Layer *layer : topological_order) {
        if (std::find(inputs.begin(), inputs.end(), layer) != inputs.end())
          continue;

        const auto &prevs = prev_layers_map[layer];
        if (prevs.size() == 1) {
          // Zero-copy path for single-input layers
          layer_outputs[layer] = layer->forward(layer_outputs[prevs[0]]);
        } else {
          // Summation path for multi-input (ResNet) layers
          Tensor combined_input = layer_outputs[prevs[0]];
          for (size_t p = 1; p < prevs.size(); ++p) {
            combined_input += layer_outputs[prevs[p]];
          }
          layer_outputs[layer] = layer->forward(combined_input);
        }
      }

      // Calculate Loss and Backward pass - REUSING MAP
      layer_grads.clear();

      Tensor pred = layer_outputs[outputs[0]];
      Tensor target = y_batch;

      total_loss += loss_fn->forward(pred, target) * current_batch_size;
      Tensor grad_out = loss_fn->backward(pred, target);

      layer_grads[outputs[0]] = grad_out;
      if (optimizer) {
        optimizer->begin_step();
      }

      for (auto it = topological_order.rbegin(); it != topological_order.rend();
           ++it) {
        Layer *layer = *it;

        Tensor current_grad;
        if (std::find(outputs.begin(), outputs.end(), layer) != outputs.end()) {
          current_grad = layer_grads[layer];
        } else {
          const auto &nexts = next_layers_map[layer];
          if (!nexts.empty()) {
            current_grad = layer_grads[nexts[0]];
            for (size_t n = 1; n < nexts.size(); ++n) {
              current_grad += layer_grads[nexts[n]];
            }
          }
        }

        if (std::find(inputs.begin(), inputs.end(), layer) != inputs.end()) {
          layer_grads[layer] = layer->backward(x_single[0], current_grad);
        } else {
          const auto &prevs = prev_layers_map[layer];
          if (prevs.size() == 1) {
            // Zero-copy path
            layer_grads[layer] =
                layer->backward(layer_outputs[prevs[0]], current_grad);
          } else {
            Tensor combined_input = layer_outputs[prevs[0]];
            for (size_t p = 1; p < prevs.size(); ++p) {
              combined_input += layer_outputs[prevs[p]];
            }
            layer_grads[layer] = layer->backward(combined_input, current_grad);
          }
        }

        const auto &prevs = prev_layers_map[layer];
        for (Layer *prev : prevs) {
          if (layer_grads.find(prev) == layer_grads.end()) {
            layer_grads[prev] = layer_grads[layer];
          } else {
            layer_grads[prev] += layer_grads[layer];
          }
        }

        layer->update(optimizer);
      }
      if (optimizer) {
        optimizer->begin_step();
      }
    }
    auto epoch_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> epoch_duration = epoch_end - epoch_start;

    std::cout << "Epoch " << epoch + 1 << "/" << epochs
              << " - Loss: " << (total_loss / xtrain.size())
              << " - Time: " << epoch_duration.count() << "s" << std::endl;
  }
}

void Model::test(const std::vector<Tensor> &x, const std::vector<Tensor> &y,
                 const std::string &metric) {
  // Evaluation is primarily handled through the Python API
}

void Model::summary() const {
  std::cout << "\nModel Summary" << std::endl;
  std::cout << std::string(70, '-') << std::endl;
  std::cout << std::left << std::setw(25) << "Layer (type)" << std::setw(25)
            << "Output Shape" << std::setw(15) << "Param #" << std::endl;
  std::cout << std::string(70, '=') << std::endl;

  int total_params = 0;
  for (Layer *layer : topological_order) {
    std::string name = layer->get_name();
    std::string shape = "[batch, " + std::to_string(layer->output.cols) + "]";

    if (name == "DenseLayer") {
      shape = "[batch, " +
              std::to_string(static_cast<DenseLayer *>(layer)->weights.cols) +
              "]";
    } else if (layer->output.cols == 0) {
      shape = "[batch, ?]";
    }

    int params = layer->get_params_count();
    total_params += params;

    std::cout << std::left << std::setw(25) << name << std::setw(25) << shape
              << std::setw(15) << params << std::endl;
  }

  std::cout << std::string(70, '=') << std::endl;
  std::cout << "Total params: " << total_params << std::endl;
  std::cout << std::string(70, '-') << std::endl << std::endl;
}

void Model::details() const {
  std::cout << "\nModel Details (DAG Structure)" << std::endl;
  std::cout << std::string(90, '-') << std::endl;
  std::cout << std::left << std::setw(20) << "Layer" 
            << std::setw(25) << "Predecessors" 
            << std::setw(25) << "Successors" 
            << std::setw(20) << "Configuration" << std::endl;
  std::cout << std::string(90, '=') << std::endl;

  for (Layer *layer : topological_order) {
    std::string name = layer->get_name();
    
    // Get Predecessors
    std::string preds = "";
    auto it_p = prev_layers_map.find(layer);
    if (it_p != prev_layers_map.end()) {
      for (Layer* p : it_p->second) {
        preds += p->get_name() + ",";
      }
      if (!preds.empty()) preds.pop_back();
    }
    if (preds == "") preds = "Input";

    // Get Successors
    std::string succs = "";
    auto it_n = next_layers_map.find(layer);
    if (it_n != next_layers_map.end()) {
      for (Layer* n : it_n->second) {
        succs += n->get_name() + ",";
      }
      if (!succs.empty()) succs.pop_back();
    }
    if (succs == "") succs = "Output";

    // Get Configuration Details
    std::string config = "";
    auto details_map = layer->get_details();
    for (auto const& pair : details_map) {
      config += pair.first + ":" + pair.second + " ";
    }

    std::cout << std::left << std::setw(20) << name 
              << std::setw(25) << (preds.length() > 23 ? preds.substr(0, 20) + "..." : preds)
              << std::setw(25) << (succs.length() > 23 ? succs.substr(0, 20) + "..." : succs)
              << std::setw(20) << config << std::endl;
  }
  std::cout << std::string(90, '-') << std::endl << std::endl;
}

void Model::weights() const {
  std::cout << "\n--- Model Weights Registry ---" << std::endl;
  for (Layer *layer : topological_order) {
    if (layer->get_name() == "DenseLayer") {
      DenseLayer* dl = static_cast<DenseLayer*>(layer);
      std::cout << "\n[ " << dl->get_name() << " ]" << std::endl;
      std::cout << "Weights Shape: (" << dl->weights.rows << ", " << dl->weights.cols << ")" << std::endl;
      dl->weights.print();
      std::cout << "Biases Shape: (1, " << dl->biases.cols << ")" << std::endl;
      dl->biases.print();
    }
  }
  std::cout << "\n------------------------------" << std::endl;
}

void Model::save(const std::string& path) const {
    std::ofstream os(path, std::ios::binary);
    if (!os) {
        throw std::runtime_error("Could not open file for saving: " + path);
    }
    
    // 1. Write Header/Version
    uint32_t magic = 0x524F434B; // "ROCK"
    os.write(reinterpret_cast<const char*>(&magic), sizeof(magic));
    
    // 2. Write number of layers
    uint32_t num_layers = static_cast<uint32_t>(topological_order.size());
    os.write(reinterpret_cast<const char*>(&num_layers), sizeof(num_layers));
    
    // 3. Save each layer
    for (Layer* layer : topological_order) {
        layer->save(os);
    }
    
    os.close();
}

void Model::load(const std::string& path) {
    std::ifstream is(path, std::ios::binary);
    if (!is) {
        throw std::runtime_error("Could not open file for loading: " + path);
    }
    
    // 1. Check Header
    uint32_t magic;
    is.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    if (magic != 0x524F434B) {
        throw std::runtime_error("Invalid model file format (Magic mismatch)");
    }
    
    // 2. Check number of layers
    uint32_t num_layers;
    is.read(reinterpret_cast<char*>(&num_layers), sizeof(num_layers));
    if (num_layers != topological_order.size()) {
        throw std::runtime_error("Model architecture mismatch: number of layers doesn't match");
    }
    
    // 3. Load each layer
    for (Layer* layer : topological_order) {
        layer->load(is);
    }
    
    is.close();
}
