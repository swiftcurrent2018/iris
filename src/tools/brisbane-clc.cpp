#include <stdio.h>
#include <stdlib.h>

int usage(char** argv) {
  printf("Usage: %s [ kernel.cl ]\n", argv[0]);
  return 1;
}

int compile(char* file) {
  char cmd[256];
  char cd[256];
  sprintf(cd, "cd brisbane &&");
  sprintf(cmd, "mkdir -p brisbane");
  if (system(cmd) == -1) perror(cmd);
  sprintf(cmd, "cp %s brisbane/", file);
  if (system(cmd) == -1) perror(cmd);
  sprintf(cmd, "%s brisbane-clang %s -- 2>/dev/null", cd, file);
  if (system(cmd) == -1) perror(cmd);
  sprintf(cmd, "%s clang -S -emit-llvm %s.llvm.c -Xclang -disable-O0-optnone", cd, file);
  if (system(cmd) == -1) perror(cmd);
  sprintf(cmd, "%s opt -S -polly-canonicalize %s.llvm.ll -o %s.llvm.preopt.ll", cd, file, file);
  if (system(cmd) == -1) perror(cmd);
  sprintf(cmd, "%s opt -load libLLVMBrisbane.so -basicaa -brisbane -analyze %s.llvm.preopt.ll -polly-process-unprofitable -polly-use-llvm-names", cd, file);
  if (system(cmd) == -1) perror(cmd);
  sprintf(cmd, "%s c++ -fPIC -shared -rdynamic -o libbrisbane_poly.so kernel-poly.c", cd);
  if (system(cmd) == -1) perror(cmd);
  return 1;
}

int main(int argc, char** argv) {
  if (argc == 1) return usage(argv);
  return compile(argv[1]);
}
