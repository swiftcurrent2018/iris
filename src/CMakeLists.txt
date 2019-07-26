include_directories(${PROJECT_SOURCE_DIR}/include)

option(USE_HIP  "Use AMD HIP API" OFF)

configure_file (Config.h.in Config.h)

include_directories(${PROJECT_BINARY_DIR}/src)

set(libsrc
    CAPI.cpp
    Command.cpp
    Dependency.cpp
    Device.cpp
    History.cpp
    Kernel.cpp
    Mem.cpp
    Object.cpp
    Platform.cpp
    Policies.cpp
    Policy.cpp
    PolicyData.cpp
    PolicyDefault.cpp
    PolicyHistory.cpp
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
    WorkloadManager.cpp
    )

add_library(brisbane STATIC ${libsrc})

#target_compile_options(brisbane PRIVATE -Wall)

install(TARGETS brisbane DESTINATION lib)
