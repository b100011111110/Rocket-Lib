#ifndef MODEL_H
#define MODEL_H

#include "layer.h"
#include "loss.h"
#include "optimizer.h"
#include <vector>
#include <unordered_map>
#include <string>

class Model {
public:
    std::unordered_map<Layer*, std::vector<Layer*>> prev_layers_map;
    std::unordered_map<Layer*, std::vector<Layer*>> next_layers_map;
    
    std::vector<Layer*> inputs;
    std::vector<Layer*> outputs;
    std::vector<Layer*> topological_order;

    Loss* loss_fn;
    Optimizer* optimizer;

    Model();
    ~Model();

    void add(Layer* layer, const std::vector<Layer*> prev_layers);
    void setInputOutputLayers(const std::vector<Layer*>& inputs, const std::vector<Layer*>& outputs);
    void compile(Loss* loss, Optimizer* opt);
    
    std::vector<Tensor> predict(const std::vector<Tensor>& x);
    void train(const std::vector<Tensor>& xtrain, const std::vector<Tensor>& ytrain, 
               const std::vector<Tensor>& xtest, const std::vector<Tensor>& ytest, 
               int epochs, int batch_size = 1);
    void test(const std::vector<Tensor>& x, const std::vector<Tensor>& y, const std::string& metric);
};

#endif
