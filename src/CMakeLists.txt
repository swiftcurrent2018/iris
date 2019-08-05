include_directories(${PROJECT_SOURCE_DIR}/include)

option(USE_FORTRAN "Enable support for Fortran 90" OFF)
if (USE_FORTRAN)
    enable_language(Fortran)
endif()

configure_file (Config.h.in Config.h)

include_directories(${PROJECT_BINARY_DIR}/src)

set(libsrc
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
    Worker.cpp
    )

if(USE_FORTRAN)
    set(libsrc ${libsrc} FortranAPI.f90)
endif()

add_library(brisbane STATIC SHARED ${libsrc})

set(hubsrc
    Hub.cpp
    HubClient.cpp
    Message.cpp
    )

add_executable(brisbane-hub ${hubsrc})

#target_compile_options(brisbane PRIVATE -Wall)

install(TARGETS brisbane DESTINATION lib)
install(TARGETS brisbane-hub DESTINATION bin)

if(USE_FORTRAN)
    install(DIRECTORY ${PROJECT_BINARY_DIR}/src/brisbane.mod DESTINATION include/brisbane)
endif()
