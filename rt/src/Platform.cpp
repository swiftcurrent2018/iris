#include "Platform.h"
#include "Debug.h"
#include <unistd.h>

namespace brisbane {
namespace rt {

char debug_prefix_[256];

Platform::Platform() {
    init_ = false;
}

Platform::~Platform() {
    _check();
    if (!init_) return;
}

int Platform::Init(int* argc, char*** argv) {
    if (init_) return BRISBANE_ERR;
    gethostname(debug_prefix_, 256);
    _check();
    return BRISBANE_OK;
}

Platform* Platform::singleton_ = NULL;

Platform* Platform::GetPlatform() {
    if (singleton_ == NULL) singleton_ = new Platform();
    return singleton_;
}

int Platform::Finalize() {
    _check();
    if (singleton_ == NULL) return BRISBANE_ERR;
    if (singleton_) delete singleton_;
    singleton_ = NULL;
    return BRISBANE_OK;
}

} /* namespace rt */
} /* namespace brisbane */
