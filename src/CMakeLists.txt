option(BUILD_SHARED_LIBS "Build shared libs" ON)

include_directories(${PROJECT_SOURCE_DIR}/include)

option(USE_FORTRAN "Enable support for Fortran 90" OFF)
if (USE_FORTRAN)
  enable_language(Fortran)
endif()

configure_file(Config.h.in Config.h)

include_directories(${PROJECT_BINARY_DIR}/src)

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

SET_TARGET_PROPERTIES(brisbane PROPERTIES OUTPUT_NAME brisbane)

set(BRISBANE_HUB_SOURCES
  Hub.cpp
  HubClient.cpp
  Message.cpp)

add_executable(brisbane-hub ${BRISBANE_HUB_SOURCES})

#target_compile_options(brisbane PRIVATE -Wall)

install(TARGETS brisbane brisbane-hub
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

if(USE_FORTRAN)
  install(DIRECTORY ${PROJECT_BINARY_DIR}/src/brisbane.mod DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/brisbane)
endif()
