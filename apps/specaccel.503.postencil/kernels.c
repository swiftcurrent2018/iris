/***************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ***************************************************************************/

#include "common.h"
#include "kernels.h"

void cpu_stencil(float c0,float c1, float *A0,float *Anext,const int nx, const int ny, const int nz)
{
  int i, j, k;  
  int size=nx*ny*nz;
#pragma omp target map(alloc:A0[0:size], Anext[0:size])
#ifdef SPEC_USE_INNER_SIMD
#pragma omp teams distribute parallel for collapse(2)
#else
#pragma omp teams distribute parallel for simd collapse(3)
#endif
	for(k=1;k<nz-1;k++)
	{
		for(j=1;j<ny-1;j++)
		{
#ifdef SPEC_USE_INNER_SIMD
#pragma omp simd
#endif
			for(i=1;i<nx-1;i++)
			{
				Anext[Index3D (nx, ny, i, j, k)] = 
				(A0[Index3D (nx, ny, i, j, k + 1)] +
				A0[Index3D (nx, ny, i, j, k - 1)] +
				A0[Index3D (nx, ny, i, j + 1, k)] +
				A0[Index3D (nx, ny, i, j - 1, k)] +
				A0[Index3D (nx, ny, i + 1, j, k)] +
				A0[Index3D (nx, ny, i - 1, j, k)])*c1
				- A0[Index3D (nx, ny, i, j, k)]*c0;
			}
		}
  }

}


