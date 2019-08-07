include_directories(${BRISBANE_INC_DIR})
include_directories(${BRISBANE_SRC_DIR}/runtime)
include_directories(${BRISBANE_BIN_DIR}/runtime)

set(BRISBANE_HUB_SOURCES
  brisbane-hub.cpp
  ${BRISBANE_SRC_DIR}/runtime/Hub.cpp
  ${BRISBANE_SRC_DIR}/runtime/HubClient.cpp
  ${BRISBANE_SRC_DIR}/runtime/Message.cpp)

add_executable(brisbane-hub ${BRISBANE_HUB_SOURCES})

install(TARGETS brisbane-hub DESTINATION ${CMAKE_INSTALL_BINDIR})

