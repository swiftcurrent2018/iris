export INTELFPGAOCLSDKROOT=/auto/software/altera/17.1/hld
export AOCL_BOARD_PACKAGE_ROOT=$INTELFPGAOCLSDKROOT/board/nalla_pcie
source $INTELFPGAOCLSDKROOT/init_opencl.sh
export LD_LIBRARY_PATH=$HOME/install/lib:$HOME/install/opt/intel/opencl/exp-runtime-2.1/lib64:$INTELFPGAOCLSDKROOT/board/nalla_pcie/linux64/lib:$INTELFPGAOCLSDKROOT/host/linux64/lib
