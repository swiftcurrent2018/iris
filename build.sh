rm -rf build
mkdir build
cd build
cmake -DCMAKE_INSTALL_PREFIX=../install -DCMAKE_BUILD_TYPE=DEBUG ..
make -j install
