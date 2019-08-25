#ifndef HIP_INCLUDE_HIP_HCC_DETAIL_HIP_RUNTIME_API_H
#define HIP_INCLUDE_HIP_HCC_DETAIL_HIP_RUNTIME_API_H

#ifdef __cplusplus
extern "C" {
#endif

typedef enum hipError_t {
  hipSuccess = 0,
} hipError_t;

typedef int hipDevice_t;
typedef void* hipDeviceptr_t;

typedef struct ihipCtx_t* hipCtx_t;
typedef struct ihipModule_t* hipModule_t;
typedef struct ihipModuleSymbol_t* hipFunction_t;
typedef struct ihipStream_t* hipStream_t;

hipError_t hipInit(unsigned int flags);
hipError_t hipDriverGetVersion(int* driverVersion);
hipError_t hipGetDeviceCount(int* count);
hipError_t hipDeviceGet(hipDevice_t* device, int ordinal);
hipError_t hipDeviceGetName(char* name, int len, hipDevice_t device);
hipError_t hipCtxCreate(hipCtx_t* ctx, unsigned int flags, hipDevice_t device);
hipError_t hipModuleLoad(hipModule_t* module, const char* fname);
hipError_t hipModuleGetFunction(hipFunction_t* function, hipModule_t module, const char* kname);
hipError_t hipMalloc(void** ptr, size_t size);
hipError_t hipFree(void* ptr);
hipError_t hipMemcpyHtoD(hipDeviceptr_t dst, void* src, size_t sizeBytes);
hipError_t hipMemcpyDtoH(void* dst, hipDeviceptr_t src, size_t sizeBytes);
hipError_t hipModuleLaunchKernel(hipFunction_t f, unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ, unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ, unsigned int sharedMemBytes, hipStream_t stream, void** kernelParams, void** extra);
hipError_t hipDeviceSynchronize(void);

#ifdef __cplusplus
} /* end of extern "C" */
#endif

#endif /* HIP_INCLUDE_HIP_HCC_DETAIL_HIP_RUNTIME_API_H */
