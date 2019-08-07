cmake_minimum_required(VERSION 2.8)

project(BRISBANE)
include(GNUInstallDirs)

set(BRISBANE_VERSION_MAJOR 1)
set(BRISBANE_VERSION_MINOR 0)
set(BRISBANE_VERSION_PATCH 0)

add_subdirectory(include)
add_subdirectory(src)

