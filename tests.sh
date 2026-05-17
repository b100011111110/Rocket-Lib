#!/bin/bash
set -e

echo "===================================================="
echo "          ROCKET-LIB TEST & BENCHMARK SUITE         "
echo "===================================================="

echo -e "\n[1/5] Building Core Engine..."
mkdir -p build
cd build
cmake ../core > /dev/null
make -j$(nproc) > /dev/null
cd ..

export ROCKET_SEED=42
export ROCKET_SHUFFLE=1
export PYTHONPATH=$(pwd)/build

# Make sure tests/data directory exists
mkdir -p tests/data

# Download datasets if not already present
if [ ! -f tests/data/sms.tsv ]; then
    echo "Downloading SMS Spam Collection Dataset..."
    curl -sSL https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv -o tests/data/sms.tsv
fi

if [ ! -f tests/data/emotion.csv ]; then
    echo "Downloading Emotion Dataset..."
    curl -sSL https://raw.githubusercontent.com/dair-ai/emotion_dataset/main/dist/emotion.csv -o tests/data/emotion.csv
fi

echo -e "\n[2/5] Running Core API & DAG Model Tests..."
python3 tests/test_core_api.py

echo -e "\n[3/5] Running Feed-Forward PyTorch Parity Tests..."
python3 tests/test_ff_parity.py

echo -e "\n[4/5] Running RNN & LSTM Parity & Spam Benchmarks..."
python3 tests/test_rnn_lstm_parity.py

echo -e "\n[5/5] Running Transformer Encoder & Decoder Parity Tests..."
python3 tests/test_transformer_parity.py

echo -e "\n===================================================="
echo "          ALL TESTS COMPLETED SUCCESSFULLY          "
echo "===================================================="
