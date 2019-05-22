include_directories(${PROJECT_SOURCE_DIR}/include)

set(libsrc
    CAPI.cpp
    Platform.cpp
    )

add_library(brisbane STATIC ${libsrc})

install(TARGETS brisbane DESTINATION lib)
