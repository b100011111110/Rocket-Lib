import sys
import os
import time

# Ensure we can find the built rocket module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "build")))
import rocket

def test_features():
    print("=== Rocket-Lib Feature Verification ===")
    
    # 1. Build a DAG Model
    print("\n[1] Constructing Model...")
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
    
    # 2. Test Introspection Methods
    print("\n[2] Testing Summary & Details...")
    model.summary()
    model.details()
    
    print("\n[3] Testing Weights Access...")
    model.weights()
    
    # 3. Test Save & Load
    print("\n[4] Testing Serialization (Save/Load)...")
    save_path = "feature_test.rocket"
    
    # Get a baseline prediction
    test_input = rocket.Tensor(1, 10)
    for i in range(10): test_input.set_val(0, i, 1.0)
    
    baseline_pred = model.predict([test_input])[0].get_val(0, 0)
    print(f"Baseline Prediction: {baseline_pred:.6f}")
    
    print(f"Saving model to {save_path}...")
    model.save(save_path)
    
    # Create a fresh model with same architecture
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
        
    # Cleanup
    if os.path.exists(save_path):
        os.remove(save_path)
    
    print("\n=== All Features Verified Successfully ===")

if __name__ == "__main__":
    test_features()
