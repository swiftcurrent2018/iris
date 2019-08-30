#include <brisbane/brisbane.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char** argv) {
  brisbane_init(&argc, &argv, true);

  char vendor[64];
  char name[64];
  int type;
  int nplatforms = 0;
  int ndevs = 0;
  brisbane_platform_count(&nplatforms);
  for (int i = 0; i < nplatforms; i++) {
    size_t size;
    brisbane_platform_info(i, brisbane_name, name, &size);
    printf("platform[%d] name[%s]\n", i, name);
  }

  brisbane_device_count(&ndevs);
  for (int i = 0; i < ndevs; i++) {
    size_t size;
    brisbane_device_info(i, brisbane_vendor, vendor, &size);
    brisbane_device_info(i, brisbane_name, name, &size);
    brisbane_device_info(i, brisbane_type, &type, &size);
    printf("dev[%d] vendor[%s] name[%s] type[0x%x]\n", i, vendor, name, type);
  }
  brisbane_finalize();

  return 0;
}
