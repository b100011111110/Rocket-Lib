#!/bin/bash
set -e

echo "===================================================="
echo "          ROCKET-LIB TEST & BENCHMARK SUITE          "
echo "===================================================="

# 1. Build Engine
echo "\n[1/5] Building Core Engine..."
mkdir -p build
cd build
cmake ../core > /dev/null
make -j$(nproc) > /dev/null
cd ..

# 2. Set Optimal Performance Environment
export ROCKET_SEED=42
export ROCKET_SHUFFLE=1

# 3. Run Standard Binary Classification Benchmark
echo "\n[2/5] Running Binary Classification Benchmark..."
./venv/bin/python3 samples/binary_classification.py

# 4. Run ResNet DAG Benchmark
echo "\n[3/5] Running ResNet DAG Benchmark..."
./venv/bin/python3 samples/resnet_dag.py

# 5. Run Comprehensive Model (Dropout/L2) Benchmark
echo "\n[4/5] Running Comprehensive Model Benchmark..."
./venv/bin/python3 samples/comprehensive_model.py

# 6. Run Full Parity Deep-Test (10k samples)
echo "\n[5/5] Running Full Parity & Metric Verification..."
./venv/bin/python3 testing/compare_keras.py

echo "\n===================================================="
echo "          ALL TESTS COMPLETED SUCCESSFULLY          "
echo "===================================================="
