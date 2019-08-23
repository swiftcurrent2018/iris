install(FILES
  brisbane.h brisbane_errno.h brisbane_llvm.h brisbane_openmp.h brisbane_poly.h brisbane_poly_types.h brisbane_runtime.h
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/brisbane)

if(USE_PYTHON)
  MESSAGE(STATUS "USE_PYTHON")
  install(FILES brisbane.py DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/brisbane)
endif()

