/***************************************************************************
 *cr
 *cr            (C) Copyright 2007 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ***************************************************************************/

#ifdef SPEC_NO_INLINE
#define INLINE 
#else
#ifdef SPEC_NO_STATIC_INLINE
#define INLINE inline
#else
#define INLINE static inline
#endif
#endif

#define PI   3.1415926535897932384626433832795029f
#define PIx2 6.2831853071795864769252867665590058f

#define MIN(X,Y) ((X) < (Y) ? (X) : (Y))
#define K_ELEMS_PER_GRID 2048

struct kValues {
  float Kx;
  float Ky;
  float Kz;
  float PhiMag;
};

INLINE
void 
ComputePhiMagCPU(int numK, 
                 float* phiR, float* phiI, float* phiMag) {
  brisbane_mem mem_phiR;
  brisbane_mem mem_phiI;
  brisbane_mem mem_phiMag;
  brisbane_mem_create(numK * sizeof(float), &mem_phiR);
  brisbane_mem_create(numK * sizeof(float), &mem_phiI);
  brisbane_mem_create(numK * sizeof(float), &mem_phiMag);

  brisbane_kernel kernel_ComputePhiMag;
  brisbane_kernel_create("ComputePhiMag", &kernel_ComputePhiMag);
  brisbane_kernel_setarg(kernel_ComputePhiMag, 0, sizeof(int), &numK);
  brisbane_kernel_setmem(kernel_ComputePhiMag, 1, mem_phiR, brisbane_rd);
  brisbane_kernel_setmem(kernel_ComputePhiMag, 2, mem_phiI, brisbane_rd);
  brisbane_kernel_setmem(kernel_ComputePhiMag, 3, mem_phiMag, brisbane_wr);

  brisbane_task task0;
  brisbane_task_create(&task0);
  brisbane_task_h2d(task0, mem_phiR, 0, numK * sizeof(float), phiR);
  brisbane_task_h2d(task0, mem_phiI, 0, numK * sizeof(float), phiI);
  size_t kernel_ComputePhiMag_off[1] = { 0 };
  size_t kernel_ComputePhiMag_idx[1] = { numK };
  brisbane_task_kernel(task0, kernel_ComputePhiMag, 1, kernel_ComputePhiMag_off, kernel_ComputePhiMag_idx);
  brisbane_task_d2h(task0, mem_phiMag, 0, numK * sizeof(float), phiMag);
  brisbane_task_submit(task0, brisbane_device_gpu, NULL, true);

/*
  int indexK = 0;
#pragma omp target map(from:phiMag[0:numK]) map(to:phiR[0:numK],phiI[0:numK])
#pragma omp teams distribute parallel for simd
  for (indexK = 0; indexK < numK; indexK++) {
    float real = phiR[indexK];
    float imag = phiI[indexK];
    phiMag[indexK] = real*real + imag*imag;
  }
*/
}

INLINE
void
ComputeQCPU(int numK, int numX,
            struct kValues *kVals,
            float* x, float* y, float* z,
            float *Qr, float *Qi) {
  brisbane_mem mem_kVals;
  brisbane_mem mem_x;
  brisbane_mem mem_y;
  brisbane_mem mem_z;
  brisbane_mem mem_Qr;
  brisbane_mem mem_Qi;
  brisbane_mem_create(numK * sizeof(struct kValues), &mem_kVals);
  brisbane_mem_create(numX * sizeof(float), &mem_x);
  brisbane_mem_create(numX * sizeof(float), &mem_y);
  brisbane_mem_create(numX * sizeof(float), &mem_z);
  brisbane_mem_create(numX * sizeof(float), &mem_Qr);
  brisbane_mem_create(numX * sizeof(float), &mem_Qi);

  brisbane_kernel kernel_ComputeQ;
  brisbane_kernel_create("ComputeQ", &kernel_ComputeQ);
  brisbane_kernel_setarg(kernel_ComputeQ, 0, sizeof(int), &numK);
  brisbane_kernel_setarg(kernel_ComputeQ, 1, sizeof(int), &numX);
  brisbane_kernel_setmem(kernel_ComputeQ, 2, mem_kVals, brisbane_rd);
  brisbane_kernel_setmem(kernel_ComputeQ, 3, mem_x, brisbane_rd);
  brisbane_kernel_setmem(kernel_ComputeQ, 4, mem_y, brisbane_rd);
  brisbane_kernel_setmem(kernel_ComputeQ, 5, mem_z, brisbane_rd);
  brisbane_kernel_setmem(kernel_ComputeQ, 6, mem_Qr, brisbane_wr);
  brisbane_kernel_setmem(kernel_ComputeQ, 7, mem_Qi, brisbane_wr);

  brisbane_task task0;
  brisbane_task_create(&task0);
  brisbane_task_h2d(task0, mem_kVals, 0, numK * sizeof(struct kValues), kVals);
  brisbane_task_h2d(task0, mem_x, 0, numX * sizeof(float), x);
  brisbane_task_h2d(task0, mem_y, 0, numX * sizeof(float), y);
  brisbane_task_h2d(task0, mem_z, 0, numX * sizeof(float), z);
  brisbane_task_h2d(task0, mem_Qr, 0, numX * sizeof(float), Qr);
  brisbane_task_h2d(task0, mem_Qi, 0, numX * sizeof(float), Qi);
  size_t kernel_ComputeQ_off[1] = { 0 };
  size_t kernel_ComputeQ_idx[1] = { numX };
  brisbane_task_kernel(task0, kernel_ComputeQ, 1, kernel_ComputeQ_off, kernel_ComputeQ_idx);
  brisbane_task_d2h(task0, mem_Qr, 0, numX * sizeof(float), Qr);
  brisbane_task_d2h(task0, mem_Qi, 0, numX * sizeof(float), Qi);
  brisbane_task_submit(task0, brisbane_device_gpu, NULL, true);

  /*
  float expArg;
  float cosArg;
  float sinArg;

  int indexK, indexX;
#pragma omp target map(to:kVals[0:numK], x[0:numX], y[0:numX], z[0:numX]), \
                     map(tofrom:Qr[0:numX], Qi[0:numX])
  {
#pragma omp teams distribute parallel for private(expArg, cosArg, sinArg)
    for (indexX = 0; indexX < numX; indexX++) {

      float QrSum = 0.0;
      float QiSum = 0.0;
      
#pragma omp simd private(expArg, cosArg, sinArg) reduction(+:QrSum, QiSum)
      for (indexK = 0; indexK < numK; indexK++) {
        expArg = PIx2 * (kVals[indexK].Kx * x[indexX] +
        kVals[indexK].Ky * y[indexX] +
        kVals[indexK].Kz * z[indexX]);

        cosArg = cosf(expArg);
        sinArg = sinf(expArg);

        float phi = kVals[indexK].PhiMag;
        QrSum += phi * cosArg;
        QiSum += phi * sinArg;
      }

      Qr[indexX] += QrSum;
      Qi[indexX] += QiSum;
    }
  }
  */
}

void createDataStructsCPU(int numK, int numX, float** phiMag,
	 float** Qr, float** Qi)
{
  size_t alignment = SPEC_ALIGNMENT_SIZE;
  *phiMag = (float* ) memalign(alignment, numK * sizeof(float));
  *Qr = (float*) memalign(alignment, numX * sizeof (float));
  memset((void *)*Qr, 0, numX * sizeof(float));
  *Qi = (float*) memalign(alignment, numX * sizeof (float));
  memset((void *)*Qi, 0, numX * sizeof(float));
}
