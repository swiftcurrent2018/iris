rm -rf build
mkdir build
cd build
CC=gcc
CXX=g++
#CC=xlc
#CXX=xlC
cmake .. -DCMAKE_INSTALL_PREFIX=$HOME/.local -DCMAKE_C_COMPILER=$CC -DCMAKE_CXX_COMPILER=$CXX -DCMAKE_BUILD_TYPE=DEBUG -DUSE_FORTRAN=OFF -DUSE_PYTHON=ON
make -j install

#export CPATH=$CPATH:$HOME/.local/include
#export LIBRARY_PATH=$LIBRARY_PATH:$HOME/.local/lib64

