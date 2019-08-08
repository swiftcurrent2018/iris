#include <brisbane/brisbane.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char** argv) {
  brisbane_init(&argc, &argv, true);
  brisbane_finalize();
  return 0;
}
