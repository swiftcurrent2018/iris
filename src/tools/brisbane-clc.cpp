#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

int usage(char** argv) {
  printf("Usage: %s [-k] kernel.cl\n", argv[0]);
  return 1;
}

int compile(char* file, int keep) {
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
  sprintf(cmd, "%s c++ -fPIC -shared -rdynamic -o ../libbrisbane-poly.so %s.poly.c", cd, file);
  if (system(cmd) == -1) perror(cmd);
  if (!keep) {
    sprintf(cmd, "rm -rf brisbane");
    if (system(cmd) == -1) perror(cmd);
  }
  return 1;
}

int main(int argc, char** argv) {
  int keep = 0;
  int c;

  while ((c = getopt(argc, argv, "k")) != -1) {
    switch (c) {
      case 'k': keep = 1; break;
      default : return usage(argv);
    }
  }
  if (optind == argc) return usage(argv);
  return compile(argv[optind], keep);
}

