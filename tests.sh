#!/bin/bash
set -e

echo "===================================================="
echo "          ROCKET-LIB TEST & BENCHMARK SUITE          "
echo "===================================================="

# 1. Build Engine
echo "\n[1/6] Building Core Engine..."
mkdir -p build
cd build
cmake ../core > /dev/null
make -j$(nproc) > /dev/null
cd ..

# 2. Set Optimal Performance Environment
export ROCKET_SEED=42
export ROCKET_SHUFFLE=1

# 3. Run Standard Binary Classification Benchmark
echo "\n[2/6] Running Binary Classification Benchmark..."
./venv/bin/python3 samples/binary_classification.py

# 4. Run ResNet DAG Benchmark
echo "\n[3/6] Running ResNet DAG Benchmark..."
./venv/bin/python3 samples/resnet_dag.py

# 5. Run Comprehensive Model (Dropout/L2) Benchmark
echo "\n[4/6] Running Comprehensive Model Benchmark..."
./venv/bin/python3 samples/comprehensive_model.py

# 6. Run Full Parity Deep-Test (10k samples)
echo "\n[5/6] Running Full Parity & Metric Verification..."
./venv/bin/python3 testing/compare_keras.py

# 7. Run Feature Verification (Summary/Details/Weights/Save/Load)
echo "\n[6/6] Verifying API Features & Serialization..."
./venv/bin/python3 testing/feature_test.py

echo "\n===================================================="
echo "          ALL TESTS COMPLETED SUCCESSFULLY          "
echo "===================================================="
