cmake_minimum_required(VERSION 3.1)

project(BRISBANE)
include(GNUInstallDirs)

set(BRISBANE_VERSION_MAJOR 1)
set(BRISBANE_VERSION_MINOR 0)
set(BRISBANE_VERSION_PATCH 0)

set(BRISBANE_SRC_DIR ${PROJECT_SOURCE_DIR}/src)
set(BRISBANE_INC_DIR ${PROJECT_SOURCE_DIR}/include)
set(BRISBANE_BIN_DIR ${PROJECT_BINARY_DIR}/src)

option(USE_FORTRAN  "Enable support for Fortran 90"     OFF)
option(USE_PYTHON   "Enable support for Python"         ON )

add_subdirectory(include)
add_subdirectory(src)

