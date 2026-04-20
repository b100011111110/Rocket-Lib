#!/bin/bash

# Ensure we're in the right directory
cd "$(dirname "$0")"

echo "Building Rocket-Lib..."
rm -rf build
mkdir -p build
cd build
cmake ../core
make -j4
cd ..

echo "Running Comparison Script..."
venv/bin/python testing/compare_keras.py
