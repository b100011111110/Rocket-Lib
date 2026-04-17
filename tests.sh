rm -rf build
mkdir build
cd build
cmake ../core
make
python3 ../testing/rocket_test.py