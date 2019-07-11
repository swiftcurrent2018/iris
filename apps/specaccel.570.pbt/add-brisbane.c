//-------------------------------------------------------------------------//
//                                                                         //
//  This benchmark is a serial C version of the NPB BT code. This C        //
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
//#include "timers.h"

//---------------------------------------------------------------------
// addition of update to the vector u
//---------------------------------------------------------------------
void add()
{
  int i, j, k, m;
  int gp22, gp12, gp02;

  gp22 = grid_points[2]-2;
  gp12 = grid_points[1]-2;
  gp02 = grid_points[0]-2;

  size_t kernel_add_0_off[3] = { 1, 1, 1 };
  size_t kernel_add_0_idx[3] = { gp02, gp12, gp22 };
  brisbane_kernel kernel_add_0;
  brisbane_kernel_create("add_0", &kernel_add_0);
  brisbane_kernel_setmem(kernel_add_0, 0, mem_u, brisbane_rw);
  brisbane_kernel_setmem(kernel_add_0, 1, mem_rhs, brisbane_r);

  brisbane_task task0;
  brisbane_task_create(&task0);
  brisbane_task_h2d_full(task0, mem_u, u);
  brisbane_task_h2d_full(task0, mem_rhs, rhs);
  brisbane_task_kernel(task0, kernel_add_0, 3, kernel_add_0_off, kernel_add_0_idx);
  brisbane_task_d2h_full(task0, mem_u, u);
  brisbane_task_submit(task0, brisbane_cpu, NULL, true);
#if 0
#pragma omp target //present(rhs,u)
#ifdef SPEC_USE_INNER_SIMD
  #pragma omp teams distribute parallel for private(i,j,k) collapse(2)
#else
  #pragma omp teams distribute parallel for simd collapse(3)
#endif
  for (k = 1; k <= gp22; k++) {
    for (j = 1; j <= gp12; j++) {
#ifdef SPEC_USE_INNER_SIMD
#pragma omp simd    
#endif
      for (i = 1; i <= gp02; i++) {
          u[k][j][i][0] = u[k][j][i][0] + rhs[k][j][i][0];
          u[k][j][i][1] = u[k][j][i][1] + rhs[k][j][i][1];
          u[k][j][i][2] = u[k][j][i][2] + rhs[k][j][i][2];
          u[k][j][i][3] = u[k][j][i][3] + rhs[k][j][i][3];
          u[k][j][i][4] = u[k][j][i][4] + rhs[k][j][i][4];
      }
    }
  }
#endif
}
