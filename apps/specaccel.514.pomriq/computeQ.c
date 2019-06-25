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
  int indexK = 0;
#pragma omp target map(from:phiMag[0:numK]) map(to:phiR[0:numK],phiI[0:numK])
#pragma omp teams distribute parallel for simd
  for (indexK = 0; indexK < numK; indexK++) {
    float real = phiR[indexK];
    float imag = phiI[indexK];
    phiMag[indexK] = real*real + imag*imag;
  }
}

INLINE
void
ComputeQCPU(int numK, int numX,
            struct kValues *kVals,
            float* x, float* y, float* z,
            float *Qr, float *Qi) {
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
