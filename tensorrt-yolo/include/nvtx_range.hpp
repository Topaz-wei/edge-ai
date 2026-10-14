#pragma once

#include <nvtx3/nvToolsExt.h>

// NVTX ranges describe CPU submission/wait scopes, not GPU execution duration.
// RAII keeps the per-thread range stack balanced on exceptions as well.
class NvtxRange {
public:
    explicit NvtxRange(const char* name) noexcept {
        nvtxRangePushA(name);
    }

    ~NvtxRange() {
        nvtxRangePop();
    }

    NvtxRange(const NvtxRange&) = delete;
    NvtxRange& operator=(const NvtxRange&) = delete;
};
