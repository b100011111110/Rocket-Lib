import os
import sys
import time
import numpy as np

# Ensure we can find the built rocket module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "build")))
import rocket

def main():
    print("Running Comprehensive Rocket-Lib Showcase...")
    
    # 1. Initialize Model
    model = rocket.Model()

    # 2. Layer Definitions
    layers = {
        "input": rocket.InputLayer(),
        "dense1": rocket.DenseLayer(20, 128),
        "relu1": rocket.ActivationLayer(rocket.ReLU()),
        "reg1": rocket.RegularizationLayer(0.001), # L2 Regularization
        "drop1": rocket.DropoutLayer(0.2),                 # 20% Dropout
        "dense2": rocket.DenseLayer(128, 64),
        "relu2": rocket.ActivationLayer(rocket.ReLU()),
        "drop2": rocket.DropoutLayer(0.1),                 # 10% Dropout
        "dense_out": rocket.DenseLayer(64, 1)
    }

    # 3. Graph Assembly (The DAG)
    model.add(layers["input"], [])
    model.add(layers["dense1"], [layers["input"]])
    model.add(layers["relu1"], [layers["dense1"]])
    model.add(layers["reg1"], [layers["relu1"]])
    model.add(layers["drop1"], [layers["reg1"]])
    model.add(layers["dense2"], [layers["drop1"]])
    model.add(layers["relu2"], [layers["dense2"]])
    model.add(layers["drop2"], [layers["relu2"]])
    model.add(layers["dense_out"], [layers["drop2"]])

    # 4. Model Configuration
    model.setInputOutputLayers([layers["input"]], [layers["dense_out"]])
    model.compile(rocket.BCEWithLogits(), rocket.Adam(lr=0.001))
    model.summary()
    # 5. Data Preparation
    X = np.random.randn(2000, 20).astype(np.float32)
    y = (np.sum(X[:, :10], axis=1) > 0).astype(np.float32).reshape(-1, 1)

    def to_rocket(arr):
        tensors = []
        for row in arr:
            t = rocket.Tensor(1, row.shape[0])
            for i, v in enumerate(row): t.set_val(0, i, float(v))
            tensors.append(t)
        return tensors

    x_train = to_rocket(X)
    y_train = to_rocket(y)

    # 6. Train Rocket
    print("\n[1/2] Training Rocket Comprehensive Model (50 epochs)...")
    os.environ["ROCKET_SEED"] = "42"
    start_rocket = time.time()
    model.train(x_train, y_train, x_train, y_train, epochs=50, batch_size=32)
    time_rocket = time.time() - start_rocket

    # 7. Benchmark PyTorch
    time_pytorch = None
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
        print("\n[2/2] Training PyTorch Equivalent (50 epochs)...")
        
        class PyTorchComprehensive(nn.Module):
            def __init__(self):
                super().__init__()
                self.dense1 = nn.Linear(20, 128)
                self.relu1 = nn.ReLU()
                self.drop1 = nn.Dropout(0.2)
                self.dense2 = nn.Linear(128, 64)
                self.relu2 = nn.ReLU()
                self.drop2 = nn.Dropout(0.1)
                self.dense_out = nn.Linear(64, 1)

            def forward(self, x):
                h1_relu = self.relu1(self.dense1(x))
                x = self.drop1(h1_relu)
                h2_relu = self.relu2(self.dense2(x))
                x = self.drop2(h2_relu)
                out = self.dense_out(x)
                return out, h1_relu

        pt_model = PyTorchComprehensive()
        optimizer = optim.Adam(pt_model.parameters(), lr=0.001)
        criterion = nn.BCEWithLogitsLoss()
        
        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.float32)
        dataset = torch.utils.data.TensorDataset(X_t, y_t)
        loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
        
        start_pytorch = time.time()
        pt_model.train()
        for epoch in range(50):
            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                out, h1_relu = pt_model(batch_X)
                loss = criterion(out, batch_y)
                # Keras L2 Activity Regularization: lambda * sum(x^2)
                reg_penalty = 0.001 * torch.mean(torch.sum(h1_relu ** 2, dim=1))
                total_loss = loss + reg_penalty
                total_loss.backward()
                optimizer.step()
        time_pytorch = time.time() - start_pytorch
    except ImportError:
        print("\nPyTorch not found. Skipping benchmark.")

    # 8. Results Comparison
    print("\n" + "="*35)
    print(" COMPREHENSIVE PERFORMANCE & ACCURACY")
    print("="*35)
    print(f"Rocket Time: {time_rocket:.4f}s")
    
    # Evaluate Rocket Accuracy
    r_preds = []
    for x in x_train:
        out = model.predict([x])[0]
        r_preds.append(out.get_val(0, 0))
    r_probs = 1.0 / (1.0 + np.exp(-np.array(r_preds)))
    r_acc = np.mean((r_probs > 0.5) == y.flatten())
    print(f"Rocket Accuracy: {r_acc*100:.2f}%")

    if time_pytorch:
        print(f"PyTorch Time:  {time_pytorch:.4f}s")
        pt_model.eval()
        with torch.no_grad():
            pt_preds, _ = pt_model(X_t)
            pt_probs = torch.sigmoid(pt_preds.flatten()).numpy()
        pt_acc = np.mean((pt_probs > 0.5) == y.flatten())
        print(f"PyTorch Accuracy: {pt_acc*100:.2f}%")
        print("-" * 35)
        print(f"Rocket is {time_pytorch/time_rocket:.2f}x faster!")
    print("="*35)

if __name__ == "__main__":
    main()
