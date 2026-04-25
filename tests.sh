#!/bin/bash
set -e

echo "===================================================="
echo "          ROCKET-LIB TEST & BENCHMARK SUITE         "
echo "===================================================="

echo "\n[1/7] Building Core Engine..."
mkdir -p build
cd build
cmake ../core > /dev/null
make -j$(nproc) > /dev/null
cd ..

export ROCKET_SEED=42
export ROCKET_SHUFFLE=1
export PYTHONPATH=$(pwd)/build

echo "\n[2/7] Running Binary Classification Benchmark..."
python3 samples/binary_classification.py

echo "\n[3/7] Running ResNet DAG Benchmark..."
python3 samples/resnet_dag.py

echo "\n[4/7] Running Comprehensive Model Benchmark..."
python3 samples/comprehensive_model.py

echo "\n[5/7] Running Full Parity & Metric Verification..."
python3 testing/compare_keras.py

echo "\n[6/7] Verifying API Features & Serialization..."
python3 testing/feature_test.py

echo "\n[7/7] Running RNN/LSTM Benchmarks & Correctness..."
python3 testing/test_rnn_lstm.py
python3 testing/verify_correctness.py

echo "\n----------------------------------------------------"
echo "   Stacked LSTM Spam Correctness (Acc/Prec/Rec/F1)  "
echo "----------------------------------------------------"
python3 samples/spam_benchmark.py

echo "\n[8/8] Running Transformer Encoder Keras Comparison..."
# Ensure dataset is present before running
mkdir -p samples/data
if [ ! -f samples/data/sms.tsv ]; then
    echo "Downloading SMS Spam Collection Dataset..."
    curl -sSL https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv -o samples/data/sms.tsv
fi
python3 testing/compare_transformer_keras.py

echo "\n===================================================="
echo "          ALL TESTS COMPLETED SUCCESSFULLY          "
echo "===================================================="
