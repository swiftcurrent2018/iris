rm -rf build
mkdir build
cd build
cmake -DCMAKE_INSTALL_PREFIX=../install ..
make -j install
