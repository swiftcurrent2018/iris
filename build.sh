rm -rf build
mkdir build
cd build
cmake .. -DCMAKE_INSTALL_PREFIX=$HOME/.local -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++ -DCMAKE_BUILD_TYPE=DEBUG -DUSE_FORTRAN=OFF -DUSE_PYTHON=ON
make -j install

