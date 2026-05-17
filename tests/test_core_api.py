import os
import sys
import time
import numpy as np

# Ensure we can find the built rocket module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "build")))
try:
    import rocket
except ImportError as e:
    print(f"Failed to import rocket module: {e}")
    sys.exit(1)

# Ensure determinism
os.environ["ROCKET_SEED"] = "42"
np.random.seed(42)


def test_tensor():
    print("\n" + "="*40)
    print(" 1. Testing Rocket Tensor Functionalities")
    print("="*40)

    print("\nCreating t1 (2x3)...")
    t1 = rocket.Tensor(2, 3)
    t1.print()

    print("\nCreating t2 (3x2)...")
    t2 = rocket.Tensor(3, 2)
    t2.print()

    print("\nMatrix Multiplication (t3 = t1 * t2)...")
    t3 = t1 * t2
    t3.print()
    
    print("\nMatrix Addition (t4 = t3 + t3)...")
    t4 = t3 + t3
    t4.print()

    print("\nUnary Negation (t5 = -t4)...")
    t5 = -t4
    t5.print()
    
    print("\n✅ Tensor tests completed successfully!")


def test_features():
    print("\n" + "="*40)
    print(" 2. Rocket-Lib API Feature Verification")
    print("="*40)
    
    print("\nConstructing Model...")
    model = rocket.Model()
    
    in_layer = rocket.InputLayer()
    d1 = rocket.DenseLayer(10, 32)
    a1 = rocket.ActivationLayer(rocket.ReLU())
    d2 = rocket.DenseLayer(32, 1)
    
    model.add(in_layer, [])
    model.add(d1, [in_layer])
    model.add(a1, [d1])
    model.add(d2, [a1])
    
    model.setInputOutputLayers([in_layer], [d2])
    model.compile(rocket.MSE(), rocket.SGD(0.01))
    
    print("\nTesting Summary & Details...")
    model.summary()
    model.details()
    
    print("\nTesting Weights Access...")
    model.weights()
    
    print("\nTesting Serialization (Save/Load)...")
    save_path = "feature_test.rocket"
    
    test_input = rocket.Tensor(1, 10)
    for i in range(10): 
        test_input.set_val(0, i, 1.0)
    
    baseline_pred = model.predict([test_input])[0].get_val(0, 0)
    print(f"Baseline Prediction: {baseline_pred:.6f}")
    
    print(f"Saving model to {save_path}...")
    model.save(save_path)
    
    print("Creating fresh model and loading weights...")
    new_model = rocket.Model()
    
    ni = rocket.InputLayer()
    nd1 = rocket.DenseLayer(10, 32)
    na1 = rocket.ActivationLayer(rocket.ReLU())
    nd2 = rocket.DenseLayer(32, 1)
    
    new_model.add(ni, [])
    new_model.add(nd1, [ni])
    new_model.add(na1, [nd1])
    new_model.add(nd2, [na1])
    
    new_model.setInputOutputLayers([ni], [nd2])
    new_model.load(save_path)
    
    loaded_pred = new_model.predict([test_input])[0].get_val(0, 0)
    print(f"Loaded Prediction:   {loaded_pred:.6f}")
    
    if abs(baseline_pred - loaded_pred) < 1e-7:
        print("✅ SUCCESS: Save/Load parity verified.")
    else:
        print("❌ FAILURE: Save/Load parity mismatch.")
        sys.exit(1)
        
    if os.path.exists(save_path):
        os.remove(save_path)
    
    print("\n✅ All features verified successfully!")


def test_binary_classification():
    print("\n" + "="*40)
    print(" 3. Running Binary Classification Benchmark")
    print("="*40)
    
    print("Generating training data...")
    X = np.random.randn(25000, 50).astype(np.float32)
    y = (np.sum(X[:, :5], axis=1) > 0).astype(np.float32).reshape(-1, 1)

    model = rocket.Model()
    input_layer = rocket.InputLayer()
    dense1 = rocket.DenseLayer(50, 64)
    relu1 = rocket.ActivationLayer(rocket.ReLU())
    dense_out = rocket.DenseLayer(64, 1)

    model.add(input_layer, [])
    model.add(dense1, [input_layer])
    model.add(relu1, [dense1])
    model.add(dense_out, [relu1])

    model.setInputOutputLayers([input_layer], [dense_out])
    optimizer = rocket.Adam(lr=0.01)
    model.compile(rocket.BCEWithLogits(), optimizer)
    model.summary()

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

    print("\n[1/2] Training Rocket-Lib (50 epochs)...")
    start_rocket = time.time()
    model.train(x_train, y_train, x_train, y_train, epochs=50, batch_size=32)
    time_rocket = time.time() - start_rocket

    time_pytorch = None
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
        print("\n[2/2] Training PyTorch Equivalent (50 epochs)...")
        
        class PyTorchModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(50, 64),
                    nn.ReLU(),
                    nn.Linear(64, 1)
                )
            def forward(self, x):
                return self.net(x)

        pt_model = PyTorchModel()
        optimizer = optim.Adam(pt_model.parameters(), lr=0.01)
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
                out = pt_model(batch_X)
                loss = criterion(out, batch_y)
                loss.backward()
                optimizer.step()
        time_pytorch = time.time() - start_pytorch
    except ImportError:
        print("\nPyTorch not found. Skipping PyTorch benchmark.")

    print("\n" + "="*35)
    print(" PERFORMANCE & ACCURACY COMPARISON")
    print("="*35)
    print(f"Rocket-Lib Time: {time_rocket:.4f}s")
    
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


def test_comprehensive_model():
    print("\n" + "="*40)
    print(" 4. Running Comprehensive Model Benchmark")
    print("="*40)
    
    model = rocket.Model()
    layers = {
        "input": rocket.InputLayer(),
        "dense1": rocket.DenseLayer(20, 128),
        "relu1": rocket.ActivationLayer(rocket.ReLU()),
        "reg1": rocket.RegularizationLayer(0.001), 
        "drop1": rocket.DropoutLayer(0.2),
        "dense2": rocket.DenseLayer(128, 64),
        "relu2": rocket.ActivationLayer(rocket.ReLU()),
        "drop2": rocket.DropoutLayer(0.1),
        "dense_out": rocket.DenseLayer(64, 1)
    }

    model.add(layers["input"], [])
    model.add(layers["dense1"], [layers["input"]])
    model.add(layers["relu1"], [layers["dense1"]])
    model.add(layers["reg1"], [layers["relu1"]])
    model.add(layers["drop1"], [layers["reg1"]])
    model.add(layers["dense2"], [layers["drop1"]])
    model.add(layers["relu2"], [layers["dense2"]])
    model.add(layers["drop2"], [layers["relu2"]])
    model.add(layers["dense_out"], [layers["drop2"]])

    model.setInputOutputLayers([layers["input"]], [layers["dense_out"]])
    model.compile(rocket.BCEWithLogits(), rocket.Adam(lr=0.001))
    model.summary()

    X = np.random.randn(2000, 20).astype(np.float32)
    y = (np.sum(X[:, :10], axis=1) > 0).astype(np.float32).reshape(-1, 1)

    def to_rocket(arr):
        tensors = []
        for row in arr:
            t = rocket.Tensor(1, row.shape[0])
            for i, v in enumerate(row): 
                t.set_val(0, i, float(v))
            tensors.append(t)
        return tensors

    x_train = to_rocket(X)
    y_train = to_rocket(y)

    print("\n[1/2] Training Rocket Comprehensive Model (50 epochs)...")
    start_rocket = time.time()
    model.train(x_train, y_train, x_train, y_train, epochs=50, batch_size=32)
    time_rocket = time.time() - start_rocket

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
                reg_penalty = 0.001 * torch.mean(torch.sum(h1_relu ** 2, dim=1))
                total_loss = loss + reg_penalty
                total_loss.backward()
                optimizer.step()
        time_pytorch = time.time() - start_pytorch
    except ImportError:
        print("\nPyTorch not found. Skipping benchmark.")

    print("\n" + "="*35)
    print(" COMPREHENSIVE PERFORMANCE & ACCURACY")
    print("="*35)
    print(f"Rocket Time: {time_rocket:.4f}s")
    
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


def test_resnet_dag():
    print("\n" + "="*40)
    print(" 5. Running ResNet-style DAG Architecture Benchmark")
    print("="*40)
    
    model = rocket.Model()
    input_layer = rocket.InputLayer()
    dense1 = rocket.DenseLayer(20, 64)
    relu1 = rocket.ActivationLayer(rocket.ReLU())
    dense_res = rocket.DenseLayer(64, 64)
    bypass = rocket.ActivationLayer(rocket.Linear())
    relu_sum = rocket.ActivationLayer(rocket.ReLU())
    dense_out = rocket.DenseLayer(64, 1)

    model.add(input_layer, [])
    model.add(dense1, [input_layer])
    model.add(relu1, [dense1])
    model.add(dense_res, [relu1])
    model.add(bypass, [relu1])
    model.add(relu_sum, [dense_res, bypass])
    model.add(dense_out, [relu_sum])

    model.setInputOutputLayers([input_layer], [dense_out])
    model.compile(rocket.BCEWithLogits(), rocket.Adam(lr=0.005))
    model.summary()

    X = np.random.randn(1000, 20).astype(np.float32)
    y = (np.sum(X**2, axis=1) > 20).astype(np.float32).reshape(-1, 1)

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

    print("\n[1/2] Training Rocket Residual Model (50 epochs)...")
    start_rocket = time.time()
    model.train(x_train, y_train, x_train, y_train, epochs=50, batch_size=32)
    time_rocket = time.time() - start_rocket

    time_pytorch = None
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
        print("\n[2/2] Training PyTorch Equivalent (50 epochs)...")
        
        class PyTorchResNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.dense1 = nn.Linear(20, 64)
                self.relu1 = nn.ReLU()
                self.dense_res = nn.Linear(64, 64)
                self.relu_sum = nn.ReLU()
                self.dense_out = nn.Linear(64, 1)

            def forward(self, x):
                x1 = self.relu1(self.dense1(x))
                path_a = self.dense_res(x1)
                path_b = x1
                x2 = self.relu_sum(path_a + path_b)
                out = self.dense_out(x2)
                return out

        pt_model = PyTorchResNet()
        optimizer = optim.Adam(pt_model.parameters(), lr=0.005)
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
                out = pt_model(batch_X)
                loss = criterion(out, batch_y)
                loss.backward()
                optimizer.step()
        time_pytorch = time.time() - start_pytorch
    except ImportError:
        print("\nPyTorch not found. Skipping benchmark.")

    print("\n" + "="*35)
    print(" RESNET PERFORMANCE & ACCURACY")
    print("="*35)
    print(f"Rocket Time: {time_rocket:.4f}s")
    
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
            pt_preds = pt_model(X_t).flatten()
            pt_probs = torch.sigmoid(pt_preds).numpy()
        pt_acc = np.mean((pt_probs > 0.5) == y.flatten())
        print(f"PyTorch Accuracy: {pt_acc*100:.2f}%")
        print("-" * 35)
        print(f"Rocket is {time_pytorch/time_rocket:.2f}x faster!")
    print("="*35)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Rocket-Lib Core API and DAG Model Tests")
    parser.add_argument(
        "--test", 
        type=str, 
        choices=["tensor", "features", "binary", "comprehensive", "resnet", "all"], 
        default="all",
        help="Specify which test to execute (default: all)"
    )
    args = parser.parse_args()

    if args.test == "all":
        test_tensor()
        test_features()
        test_binary_classification()
        test_comprehensive_model()
        test_resnet_dag()
    elif args.test == "tensor":
        test_tensor()
    elif args.test == "features":
        test_features()
    elif args.test == "binary":
        test_binary_classification()
    elif args.test == "comprehensive":
        test_comprehensive_model()
    elif args.test == "resnet":
        test_resnet_dag()
