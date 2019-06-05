include_directories(${PROJECT_SOURCE_DIR}/include)

set(libsrc
    CAPI.cpp
    Command.cpp
    Device.cpp
    Executor.cpp
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
    Scheduler.cpp
    Task.cpp
    Thread.cpp
    Timer.cpp
    )

add_library(brisbane STATIC ${libsrc})

install(TARGETS brisbane DESTINATION lib)
