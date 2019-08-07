include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_SOURCE_DIR}/src)
include_directories(${PROJECT_BINARY_DIR}/src)

set(BRISBANE_HUB_SOURCES
  brisbane-hub.cpp
  ../Hub.cpp
  ../HubClient.cpp
  ../Message.cpp)

add_executable(brisbane-hub ${BRISBANE_HUB_SOURCES})

install(TARGETS brisbane-hub DESTINATION ${CMAKE_INSTALL_BINDIR})
