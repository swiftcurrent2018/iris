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
// block-diagonal matrix-vector multiplication                  
//---------------------------------------------------------------------
void txinvr()
{
  int i, j, k;
  double t1, t2, t3, ac, ru1, uu, vv, ww, r1, r2, r3, r4, r5, ac2inv;

  size_t kernel_txinvr_0_off[3] = { 1, 1, 1 };
  size_t kernel_txinvr_0_idx[3] = { nx2, ny2, nz2 };
  brisbane_kernel kernel_txinvr_0;
  brisbane_kernel_create("txinvr_0", &kernel_txinvr_0);
  brisbane_kernel_setmem(kernel_txinvr_0, 0, mem_rho_i, brisbane_rd);
  brisbane_kernel_setmem(kernel_txinvr_0, 1, mem_us, brisbane_rd);
  brisbane_kernel_setmem(kernel_txinvr_0, 2, mem_vs, brisbane_rd);
  brisbane_kernel_setmem(kernel_txinvr_0, 3, mem_ws, brisbane_rd);
  brisbane_kernel_setmem(kernel_txinvr_0, 4, mem_rhs, brisbane_rw);
  brisbane_kernel_setmem(kernel_txinvr_0, 5, mem_speed, brisbane_rd);
  brisbane_kernel_setmem(kernel_txinvr_0, 6, mem_qs, brisbane_rd);
  brisbane_kernel_setarg(kernel_txinvr_0, 7, sizeof(double), &c2);
  brisbane_kernel_setarg(kernel_txinvr_0, 8, sizeof(double), &bt);

  brisbane_task task0;
  brisbane_task_create(&task0);
  brisbane_task_h2d_full(task0, mem_rhs, rhs);
  brisbane_task_kernel(task0, kernel_txinvr_0, 3, kernel_txinvr_0_off, kernel_txinvr_0_idx);
  brisbane_task_submit(task0, brisbane_gpu, NULL, true);
#if 0
//  if (timeron) timer_start(t_txinvr);
#pragma omp target //present(rho_i,us,vs,ws,rhs,speed,qs)
{
#ifdef SPEC_USE_INNER_SIMD
  #pragma omp teams distribute parallel for collapse(2) private(t1, t2, t3, ac, ru1, uu, vv, ww, r1, r2, r3, r4, r5, ac2inv)
#else
  #pragma omp teams distribute parallel for simd collapse(3) private(t1, t2, t3, ac, ru1, uu, vv, ww, r1, r2, r3, r4, r5, ac2inv)
#endif
  for (k = 1; k <= nz2; k++) {
    for (j = 1; j <= ny2; j++) {
#ifdef SPEC_USE_INNER_SIMD
      #pragma omp simd
#endif
      for (i = 1; i <= nx2; i++) {
        ru1 = rho_i[k][j][i];
        uu = us[k][j][i];
        vv = vs[k][j][i];
        ww = ws[k][j][i];
        ac = speed[k][j][i];
        ac2inv = ac*ac;

        r1 = rhs[0][k][j][i];
        r2 = rhs[1][k][j][i];
        r3 = rhs[2][k][j][i];
        r4 = rhs[3][k][j][i];
        r5 = rhs[4][k][j][i];

        t1 = c2 / ac2inv * ( qs[k][j][i]*r1 - uu*r2  - vv*r3 - ww*r4 + r5 );
        t2 = bt * ru1 * ( uu * r1 - r2 );
        t3 = ( bt * ru1 * ac ) * t1;

        rhs[0][k][j][i] = r1 - t1;
        rhs[1][k][j][i] = - ru1 * ( ww*r1 - r4 );
        rhs[2][k][j][i] =   ru1 * ( vv*r1 - r3 );
        rhs[3][k][j][i] = - t2 + t3;
        rhs[4][k][j][i] =   t2 + t3;
      }
    }
  }
}/*end kernels*/
#endif
//  if (timeron) timer_stop(t_txinvr);
}

