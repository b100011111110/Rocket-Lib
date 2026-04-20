#include "model.h"
#include <algorithm>
#include <iostream>
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
}

void Model::compile(Loss *loss, Optimizer *opt) {
  this->loss_fn = loss;
  this->optimizer = opt;

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
    double total_loss = 0.0;

    // Shuffle indices at the start of each epoch if enabled
    if (std::getenv("ROCKET_SHUFFLE") == nullptr ||
        std::string(std::getenv("ROCKET_SHUFFLE")) != "0") {
      std::shuffle(indices.begin(), indices.end(), gen);
    }

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

      // Forward pass
      std::unordered_map<Layer *, Tensor> layer_outputs;
      layer_outputs[inputs[0]] = inputs[0]->forward(x_single[0]);

      for (Layer *layer : topological_order) {
        if (std::find(inputs.begin(), inputs.end(), layer) != inputs.end())
          continue;

        const auto &prevs = prev_layers_map[layer];
        Tensor combined_input = layer_outputs[prevs[0]];
        for (size_t p = 1; p < prevs.size(); ++p) {
          combined_input += layer_outputs[prevs[p]];
        }
        layer_outputs[layer] = layer->forward(combined_input);
      }

      // Calculate Loss and Backward pass
      std::unordered_map<Layer *, Tensor> layer_grads;

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

        Tensor layer_input;
        if (std::find(inputs.begin(), inputs.end(), layer) != inputs.end()) {
          layer_input = x_single[0];
        } else {
          const auto &prevs = prev_layers_map[layer];
          layer_input = layer_outputs[prevs[0]];
          for (size_t p = 1; p < prevs.size(); ++p) {
            layer_input += layer_outputs[prevs[p]];
          }
        }

        Tensor grad_in = layer->backward(layer_input, current_grad);

        const auto &prevs = prev_layers_map[layer];
        for (Layer *prev : prevs) {
          if (layer_grads.find(prev) == layer_grads.end()) {
            layer_grads[prev] = grad_in;
          } else {
            layer_grads[prev] += grad_in;
          }
        }

        layer->update(optimizer);
      }
    }
    std::cout << "Epoch " << epoch + 1 << "/" << epochs
              << " - Loss: " << (total_loss / xtrain.size()) << std::endl;
  }
}

void Model::test(const std::vector<Tensor> &x, const std::vector<Tensor> &y,
                 const std::string &metric) {
  // Simple mock implementation for metrics
  std::cout << "Testing model on " << x.size()
            << " samples with metric: " << metric << std::endl;
  int correct = 0;
  for (size_t i = 0; i < x.size(); ++i) {
    std::vector<Tensor> pred_vec = predict({x[i]});
    Tensor pred = pred_vec[0];
  }

  if (metric == "accuracy") {
    std::cout << "Accuracy metric calculation placeholder." << std::endl;
  } else if (metric == "precision") {
    std::cout << "Precision metric calculation placeholder." << std::endl;
  } else if (metric == "recall") {
    std::cout << "Recall metric calculation placeholder." << std::endl;
  } else if (metric == "f1") {
    std::cout << "F1 score calculation placeholder." << std::endl;
  } else if (metric == "matrix") {
    std::cout << "Confusion matrix placeholder." << std::endl;
  }
}
