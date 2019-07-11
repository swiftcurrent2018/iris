//-------------------------------------------------------------------------//
//                                                                         //
//  This benchmark is a serial C version of the NPB SP code. This C        //
//  version is developed by the Center for Manycore Programming at Seoul   //
//  National University and derived from the serial Fortran versions in    //
//  "NPB3.3-SER" developed by NAS.                                         //
//                                                                         //
//  Permission to use, copy, distribute and modify this software for any   //
//  purpose with or without fee is hereby granted. This software is        //
//  provided "as is" without express or implied warranty.                  //
//                                                                         //
//  Information on NPB 3.3, including the technical report, the original   //
//  specifications, source code, results and information on how to submit  //
//  new results, is available at:                                          //
//                                                                         //
//           http://www.nas.nasa.gov/Software/NPB/                         //
//                                                                         //
//  Send comments or suggestions for this C version to cmp@aces.snu.ac.kr  //
//                                                                         //
//          Center for Manycore Programming                                //
//          School of Computer Science and Engineering                     //
//          Seoul National University                                      //
//          Seoul 151-744, Korea                                           //
//                                                                         //
//          E-mail:  cmp@aces.snu.ac.kr                                    //
//                                                                         //
//-------------------------------------------------------------------------//

//-------------------------------------------------------------------------//
// Authors: Sangmin Seo, Jungwon Kim, Jun Lee, Jeongho Nah, Gangwon Jo,    //
//          and Jaejin Lee                                                 //
//-------------------------------------------------------------------------//

#include "header-brisbane.h"

//---------------------------------------------------------------------
// addition of update to the vector u
//---------------------------------------------------------------------
void add()
{
  int i, j, k, m;

  size_t kernel_add_0_off[3] = { 1, 1, 1 };
  size_t kernel_add_0_idx[3] = { nx2, ny2, nz2 };
  brisbane_kernel kernel_add_0;
  brisbane_kernel_create("add_0", &kernel_add_0);
  brisbane_kernel_setmem(kernel_add_0, 0, mem_u, brisbane_rw);
  brisbane_kernel_setmem(kernel_add_0, 1, mem_rhs, brisbane_rd);

  brisbane_task task0;
  brisbane_task_create(&task0);
  brisbane_task_kernel(task0, kernel_add_0, 3, kernel_add_0_off, kernel_add_0_idx);
  brisbane_task_submit(task0, brisbane_gpu, NULL, true);

#if 0
#ifdef SPEC_USE_INNER_SIMD
    #pragma omp target teams distribute parallel for private(i,j,k,m) collapse(2) 
#else
    #pragma omp target teams distribute parallel for simd collapse(4)
#endif
    for (k = 1; k <= nz2; k++) {
      for (j = 1; j <= ny2; j++) {
#ifdef SPEC_USE_INNER_SIMD
      #pragma omp simd
#endif
        for (i = 1; i <= nx2; i++) {
        for (m = 0; m < 5; m++) {
          u[m][k][j][i] = u[m][k][j][i] + rhs[m][k][j][i];
        }
      }
    }
  }
#endif
}
