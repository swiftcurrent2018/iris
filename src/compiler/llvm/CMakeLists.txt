include_directories($ENV{HOME}/install/include)
include_directories($ENV{HOME}/install/include/polly)

add_llvm_library( libLLVMBrisbane MODULE
  BrisbaneLLVM.cpp

  DEPENDS
  PLUGIN_TOOL
  opt
  )
