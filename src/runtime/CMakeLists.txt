option(BUILD_SHARED_LIBS "Build shared libs" ON)

include_directories(${BRISBANE_INC_DIR})
include_directories(${BRISBANE_BIN_DIR}/runtime)

configure_file(Config.h.in Config.h)

set(BRISBANE_RUNTIME_SOURCES
  CAPI.cpp
  Command.cpp
  Consistency.cpp
  Device.cpp
  DeviceCUDA.cpp
  DeviceHIP.cpp
  DeviceOpenCL.cpp
  DeviceOpenMP.cpp
  FilterTaskSplit.cpp
  History.cpp
  HubClient.cpp
  Kernel.cpp
  Loader.cpp
  LoaderCUDA.cpp
  LoaderHIP.cpp
  LoaderOpenCL.cpp
  LoaderOpenMP.cpp
  LoaderPolicy.cpp
  Message.cpp
  Mem.cpp
  MemRange.cpp
  Platform.cpp
  Policies.cpp
  Policy.cpp
  PolicyAll.cpp
  PolicyAny.cpp
  PolicyData.cpp
  PolicyDefault.cpp
  PolicyDevice.cpp
  PolicyProfile.cpp
  PolicyRandom.cpp
  Polyhedral.cpp
  Profiler.cpp
  ProfilerDOT.cpp
  ProfilerGoogleCharts.cpp
  Reduction.cpp
  Retainable.cpp
  Scheduler.cpp
  Task.cpp
  TaskQueue.cpp
  Thread.cpp
  Timer.cpp
  Utils.cpp
  Worker.cpp)

if(USE_FORTRAN)
  MESSAGE(STATUS "USE_FORTRAN")
  enable_language(Fortran)
  list(APPEND BRISBANE_RUNTIME_SOURCES FortranAPI.f90)
  install(DIRECTORY ${BRISBANE_BIN_DIR}/runtime/brisbane.mod DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/brisbane)
endif()

if(USE_PYTHON)
  MESSAGE(STATUS "USE_PYTHON")
  install(FILES brisbane.py DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/brisbane)
endif()

add_library(brisbane ${BRISBANE_RUNTIME_SOURCES})

#target_compile_options(brisbane PRIVATE -Wall)

install(TARGETS brisbane DESTINATION ${CMAKE_INSTALL_LIBDIR})

