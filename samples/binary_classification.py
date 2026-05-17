import os
import sys
import time
import numpy as np

# Ensure 100% stable determinism
os.environ["ROCKET_SEED"] = "42"
np.random.seed(42)

# Ensure we can find the built rocket module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "build")))
import rocket

def main():
    # 1. Create Synthetic Data
    print("Generating training data...")
    X = np.random.randn(1000, 20).astype(np.float32)
    # Simple rule: if sum of first 5 features > 0, class 1, else 0
    y = (np.sum(X[:, :5], axis=1) > 0).astype(np.float32).reshape(-1, 1)

    # 2. Build the Rocket Model
    model = rocket.Model()

    # Define layers
    input_layer = rocket.InputLayer()
    dense1 = rocket.DenseLayer(20, 64)
    relu1 = rocket.ActivationLayer(rocket.ReLU())
    dense_out = rocket.DenseLayer(64, 1)

    # Connect the graph
    model.add(input_layer, [])
    model.add(dense1, [input_layer])
    model.add(relu1, [dense1])
    model.add(dense_out, [relu1])

    # Finalize model boundary
    model.setInputOutputLayers([input_layer], [dense_out])

    # 3. Compile with Adam optimizer
    optimizer = rocket.Adam(lr=0.01)
    model.compile(rocket.BCEWithLogits(), optimizer)
    model.summary()


    # 4. Convert numpy to Rocket Tensors
    def to_rocket(arr):
        tensors = []
        for row in arr:
            t = rocket.Tensor(1, row.shape[0])
            for i, val in enumerate(row):
                t.set_val(0, i, float(val))
            tensors.append(t)
        return tensors

    x_train = to_rocket(X)
    y_train = to_rocket(y)

    # 5. Train Rocket
    print("\n[1/2] Training Rocket-Lib (50 epochs)...")
    start_rocket = time.time()
    model.train(x_train, y_train, x_train, y_train, epochs=50, batch_size=32)
    time_rocket = time.time() - start_rocket

    # 6. Benchmark PyTorch (if available)
    time_pytorch = None
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
        print("\n[2/2] Training PyTorch Equivalent (50 epochs)...")
        
        # Build PyTorch model
        class PyTorchModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(20, 64),
                    nn.ReLU(),
                    nn.Linear(64, 1)
                )
            def forward(self, x):
                return self.net(x)

        pt_model = PyTorchModel()
        optimizer = optim.Adam(pt_model.parameters(), lr=0.01)
        criterion = nn.BCEWithLogitsLoss()
        
        # Prepare data loader
        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.float32)
        dataset = torch.utils.data.TensorDataset(X_t, y_t)
        loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
        
        start_pytorch = time.time()
        pt_model.train()
        for epoch in range(50):
            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                out = pt_model(batch_X)
                loss = criterion(out, batch_y)
                loss.backward()
                optimizer.step()
        time_pytorch = time.time() - start_pytorch
    except ImportError:
        print("\nPyTorch not found. Skipping PyTorch benchmark.")

    # 7. Results Comparison
    print("\n" + "="*35)
    print(" PERFORMANCE & ACCURACY COMPARISON")
    print("="*35)
    print(f"Rocket-Lib Time: {time_rocket:.4f}s")
    
    # Evaluate Rocket Accuracy
    r_preds = []
    for x in x_train:
        out = model.predict([x])[0]
        r_preds.append(out.get_val(0, 0))
    r_probs = 1.0 / (1.0 + np.exp(-np.array(r_preds)))
    r_acc = np.mean((r_probs > 0.5) == y.flatten())
    print(f"Rocket Accuracy: {r_acc*100:.2f}%")

    if time_pytorch:
        print(f"PyTorch Time:    {time_pytorch:.4f}s")
        pt_model.eval()
        with torch.no_grad():
            pt_preds = pt_model(X_t).flatten()
            pt_probs = torch.sigmoid(pt_preds).numpy()
        pt_acc = np.mean((pt_probs > 0.5) == y.flatten())
        print(f"PyTorch Accuracy: {pt_acc*100:.2f}%")
        print("-" * 35)
        print(f"Rocket is {time_pytorch/time_rocket:.2f}x faster!")
    print("="*35)

if __name__ == "__main__":
    main()
