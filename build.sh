rm -rf build
mkdir build
cd build
cmake .. -DCMAKE_INSTALL_PREFIX=../install -DCMAKE_BUILD_TYPE=DEBUG -DUSE_FORTRAN=ON
make -j install
