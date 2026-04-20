cd build
cmake ../core
make -j4
cd ..

# Optimal deterministic configuration
export ROCKET_SEED=42
export ROCKET_REG_LAMBDA=0.001
export ROCKET_LR=0.005
export ROCKET_SHUFFLE=1
export ROCKET_DROPOUT=0.15

# Ensure keras compares symmetrically
sed -i 's/shuffle=False/shuffle=True/g' testing/compare_keras.py

sh run_comparison.sh
