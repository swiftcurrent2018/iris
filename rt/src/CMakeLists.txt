include_directories(${PROJECT_SOURCE_DIR}/include)

set(libsrc
    CAPI.cpp
    Command.cpp
    Device.cpp
    Executor.cpp
    Kernel.cpp
    Mem.cpp
    Object.cpp
    Platform.cpp
    Task.cpp
    )

add_library(brisbane STATIC ${libsrc})

install(TARGETS brisbane DESTINATION lib)
