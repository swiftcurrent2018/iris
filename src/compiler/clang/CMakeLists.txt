set(LLVM_LINK_COMPONENTS
  Support
  )

add_clang_executable(brisbane-clang
  BrisbaneClang.cpp
  )

target_link_libraries(brisbane-clang
  PRIVATE
  clangAST
  clangASTMatchers
  clangBasic
  clangFrontend
  clangTooling
  clangToolingRefactoring
  )

install(TARGETS brisbane-clang DESTINATION ${CMAKE_INSTALL_BINDIR})
