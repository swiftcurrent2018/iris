option(BUILD_SHARED_LIBS "Build shared libs" ON)

MESSAGE("BRISBANE_BIN_DIR ${BRISBANE_BIN_DIR}")

include_directories(${BRISBANE_INC_DIR})
include_directories(${BRISBANE_BIN_DIR}/runtime)

option(USE_FORTRAN "Enable support for Fortran 90" OFF)
if(USE_FORTRAN)
  enable_language(Fortran)
endif()

configure_file(Config.h.in Config.h)

set(BRISBANE_RUNTIME_SOURCES
  CAPI.cpp
  Command.cpp
  Consistency.cpp
  Device.cpp
  FilterTaskSplit.cpp
  History.cpp
  HubClient.cpp
  Kernel.cpp
  Message.cpp
  Mem.cpp
  Object.cpp
  Platform.cpp
  Policies.cpp
  Policy.cpp
  PolicyData.cpp
  PolicyDefault.cpp
  PolicyEager.cpp
  PolicyProfile.cpp
  PolicyRandom.cpp
  PolicySpecific.cpp
  Polyhedral.cpp
  Reduction.cpp
  Scheduler.cpp
  Task.cpp
  TaskQueue.cpp
  Thread.cpp
  Timer.cpp
  Utils.cpp
  Worker.cpp)

if(USE_FORTRAN)
  list(APPEND BRISBANE_RUNTIME_SOURCES FortranAPI.f90)
endif()

add_library(brisbane ${BRISBANE_RUNTIME_SOURCES})

#target_compile_options(brisbane PRIVATE -Wall)

install(TARGETS brisbane DESTINATION ${CMAKE_INSTALL_LIBDIR})

if(USE_FORTRAN)
  install(DIRECTORY ${PROJECT_BINARY_DIR}/src/brisbane.mod DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/brisbane)
endif()
