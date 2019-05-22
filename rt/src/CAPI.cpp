#include <brisbane/brisbane.h>
#include "Debug.h"
#include "Platform.h"

using namespace brisbane::rt;

int brisbane_init(int* argc, char*** argv) {
    return Platform::GetPlatform()->Init(argc, argv);
}

int brisbane_finalize() {
    return Platform::Finalize();
}

