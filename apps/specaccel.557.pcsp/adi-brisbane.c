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

void adi()
{
  compute_rhs();
  
  txinvr();

  x_solve();

  y_solve();

  z_solve();
  
  add();
}


void ninvr()
{
  int i, j, k;
  double r1, r2, r3, r4, r5, t1, t2;

  size_t kernel_ninvr_0_off[3] = { 1, 1, 1 };
  size_t kernel_ninvr_0_idx[3] = { nx2, ny2, nz2 };
  brisbane_kernel kernel_ninvr_0;
  brisbane_kernel_create("ninvr_0", &kernel_ninvr_0);
  brisbane_kernel_setmem(kernel_ninvr_0, 0, mem_rhs, brisbane_rw);
  brisbane_kernel_setarg(kernel_ninvr_0, 1, sizeof(double), &bt);

  brisbane_task task0;
  brisbane_task_create(&task0);
  brisbane_task_kernel(task0, kernel_ninvr_0, 3, kernel_ninvr_0_off, kernel_ninvr_0_idx);
  brisbane_task_submit(task0, brisbane_gpu, NULL, true);
#if 0
#pragma omp target //present(rhs)
#ifdef SPEC_USE_INNER_SIMD
  #pragma omp teams distribute parallel for collapse(2) private(i,j,k,r1,r2,r3,r4,r5,t1,t2) 
#else
  #pragma omp teams distribute parallel for simd collapse(3) private(r1,r2,r3,r4,r5,t1,t2) 
#endif
  for (k = 1; k <= nz2; k++) {
    for (j = 1; j <= ny2; j++) {
#ifdef SPEC_USE_INNER_SIMD
      #pragma omp simd private(r1,r2,r3,r4,r5,t1,t2)
#endif
      for (i = 1; i <= nx2; i++) {
        r1 = rhs[0][k][j][i];
        r2 = rhs[1][k][j][i];
        r3 = rhs[2][k][j][i];
        r4 = rhs[3][k][j][i];
        r5 = rhs[4][k][j][i];

        t1 = bt * r3;
        t2 = 0.5 * ( r4 + r5 );

        rhs[0][k][j][i] = -r2;
        rhs[1][k][j][i] =  r1;
        rhs[2][k][j][i] = bt * ( r4 - r5 );
        rhs[3][k][j][i] = -t1 + t2;
        rhs[4][k][j][i] =  t1 + t2;
      }
    }
  }
#endif

}

void pinvr()
{
  int i, j, k;
  double r1, r2, r3, r4, r5, t1, t2;

  size_t kernel_pinvr_0_off[3] = { 1, 1, 1 };
  size_t kernel_pinvr_0_idx[3] = { nx2, ny2, nz2 };
  brisbane_kernel kernel_pinvr_0;
  brisbane_kernel_create("pinvr_0", &kernel_pinvr_0);
  brisbane_kernel_setmem(kernel_pinvr_0, 0, mem_rhs, brisbane_rw);
  brisbane_kernel_setarg(kernel_pinvr_0, 1, sizeof(double), &bt);

  brisbane_task task0;
  brisbane_task_create(&task0);
  brisbane_task_kernel(task0, kernel_pinvr_0, 3, kernel_pinvr_0_off, kernel_pinvr_0_idx);
  brisbane_task_submit(task0, brisbane_gpu, NULL, true);
#if 0
#pragma omp target //present(rhs)
#ifdef SPEC_USE_INNER_SIMD
  #pragma omp teams distribute parallel for private(i,j,k,r1,r2,r3,r4,r5,t1,t2) collapse(2) 
#else
  #pragma omp teams distribute parallel for simd private(r1,r2,r3,r4,r5,t1,t2) collapse(3)
#endif
  for (k = 1; k <= nz2; k++) {
    for (j = 1; j <= ny2; j++) {
#ifdef SPEC_USE_INNER_SIMD
      #pragma omp simd private(r1,r2,r3,r4,r5,t1,t2)
#endif
      for (i = 1; i <= nx2; i++) {
        r1 = rhs[0][k][j][i];
        r2 = rhs[1][k][j][i];
        r3 = rhs[2][k][j][i];
        r4 = rhs[3][k][j][i];
        r5 = rhs[4][k][j][i];

        t1 = bt * r1;
        t2 = 0.5 * ( r4 + r5 );

        rhs[0][k][j][i] =  bt * ( r4 - r5 );
        rhs[1][k][j][i] = -r3;
        rhs[2][k][j][i] =  r2;
        rhs[3][k][j][i] = -t1 + t2;
        rhs[4][k][j][i] =  t1 + t2;
      }
    }
  }
#endif

}

void tzetar()
{
  int i, j, k;
  double t1, t2, t3, ac, xvel, yvel, zvel, r1, r2, r3, r4, r5;
  double btuz, ac2u, uzik1;

  size_t kernel_tzetar_0_off[3] = { 1, 1, 1 };
  size_t kernel_tzetar_0_idx[3] = { nx2, ny2, nz2 };
  brisbane_kernel kernel_tzetar_0;
  brisbane_kernel_create("tzetar_0", &kernel_tzetar_0);
  brisbane_kernel_setmem(kernel_tzetar_0, 0, mem_us, brisbane_rd);
  brisbane_kernel_setmem(kernel_tzetar_0, 1, mem_vs, brisbane_rd);
  brisbane_kernel_setmem(kernel_tzetar_0, 2, mem_ws, brisbane_rd);
  brisbane_kernel_setmem(kernel_tzetar_0, 3, mem_qs, brisbane_rd);
  brisbane_kernel_setmem(kernel_tzetar_0, 4, mem_u, brisbane_rd);
  brisbane_kernel_setmem(kernel_tzetar_0, 5, mem_speed, brisbane_rd);
  brisbane_kernel_setmem(kernel_tzetar_0, 6, mem_rhs, brisbane_rw);
  brisbane_kernel_setarg(kernel_tzetar_0, 7, sizeof(double), &bt);
  brisbane_kernel_setarg(kernel_tzetar_0, 8, sizeof(double), &c2iv);

  brisbane_task task0;
  brisbane_task_create(&task0);
  brisbane_task_kernel(task0, kernel_tzetar_0, 3, kernel_tzetar_0_off, kernel_tzetar_0_idx);
  brisbane_task_submit(task0, brisbane_gpu, NULL, true);

#if 0
#pragma omp target //present(us,vs,ws,qs,u,speed,rhs)
#ifdef SPEC_USE_INNER_SIMD
  #pragma omp teams distribute parallel for collapse(2) private(i,j,k,t1,t2,t3,ac,xvel,yvel,zvel,r1,r2,r3,r4,r5,btuz,ac2u,uzik1)
#else
  #pragma omp teams distribute parallel for simd collapse(3) private(t1,t2,t3,ac,xvel,yvel,zvel,r1,r2,r3,r4,r5,btuz,ac2u,uzik1)
#endif
  for (k = 1; k <= nz2; k++) {
    for (j = 1; j <= ny2; j++) {
#ifdef SPEC_USE_INNER_SIMD
      #pragma omp simd private(t1,t2,t3,ac,xvel,yvel,zvel,r1,r2,r3,r4,r5,btuz,ac2u,uzik1)
#endif
      for (i = 1; i <= nx2; i++) {
        xvel = us[k][j][i];
        yvel = vs[k][j][i];
        zvel = ws[k][j][i];
        ac   = speed[k][j][i];

        ac2u = ac*ac;

        r1 = rhs[0][k][j][i];
        r2 = rhs[1][k][j][i];
        r3 = rhs[2][k][j][i];
        r4 = rhs[3][k][j][i];
        r5 = rhs[4][k][j][i];     

        uzik1 = u[0][k][j][i];
        btuz  = bt * uzik1;

        t1 = btuz/ac * (r4 + r5);
        t2 = r3 + t1;
        t3 = btuz * (r4 - r5);

        rhs[0][k][j][i] = t2;
        rhs[1][k][j][i] = -uzik1*r2 + xvel*t2;
        rhs[2][k][j][i] =  uzik1*r1 + yvel*t2;
        rhs[3][k][j][i] =  zvel*t2  + t3;
        rhs[4][k][j][i] =  uzik1*(-xvel*r2 + yvel*r1) + 
                           qs[k][j][i]*t2 + c2iv*ac2u*t1 + zvel*t3;
      }
    }
  }
#endif

}

void x_solve()
{
  int i, j, k, i1, i2, m;
  int gp01,gp02,gp03,gp04;
  double ru1, fac1, fac2;
  double lhsX[5][nz2+1][IMAXP+1][IMAXP+1];
  double lhspX[5][nz2+1][IMAXP+1][IMAXP+1];
  double lhsmX[5][nz2+1][IMAXP+1][IMAXP+1];
  double rhonX[nz2+1][IMAXP+1][PROBLEM_SIZE];
  double rhsX[5][nz2+1][IMAXP+1][JMAXP+1];

  int ni=nx2+1;
  gp01=grid_points[0]-1;
  gp02=grid_points[0]-2;
  gp03=grid_points[0]-3;
  gp04=grid_points[0]-4;

  brisbane_mem mem_lhsX;
  brisbane_mem mem_lhspX;
  brisbane_mem mem_lhsmX;
  brisbane_mem mem_rhonX;
  brisbane_mem mem_rhsX;
  brisbane_mem_create(5 * (nz2 + 1) * (IMAXP + 1) * (IMAXP + 1) * sizeof(double), &mem_lhsX);
  brisbane_mem_create(5 * (nz2 + 1) * (IMAXP + 1) * (IMAXP + 1) * sizeof(double), &mem_lhspX);
  brisbane_mem_create(5 * (nz2 + 1) * (IMAXP + 1) * (IMAXP + 1) * sizeof(double), &mem_lhsmX);
  brisbane_mem_create((nz2 + 1) * (IMAXP + 1) * (PROBLEM_SIZE) * sizeof(double), &mem_rhonX);
  brisbane_mem_create(5 * (nz2 + 1) * (IMAXP + 1) * (JMAXP + 1) * sizeof(double), &mem_rhsX);

  #pragma omp target data map(alloc:lhsX[:][:][:][:],lhspX[:][:][:][:], lhsmX[:][:][:][:],rhonX[:][:][:],rhsX[:][:][:][:]) //present(rho_i,us,speed,rhs)
  {
    size_t kernel_x_solve_0_off[3] = { 0, 0, 0 };
    size_t kernel_x_solve_0_idx[3] = { IMAXP + 1, JMAXP + 1, nz2 + 1 };
    brisbane_kernel kernel_x_solve_0;
    brisbane_kernel_create("x_solve_0", &kernel_x_solve_0);
    brisbane_kernel_setmem(kernel_x_solve_0, 0, mem_rhsX, brisbane_wr);
    brisbane_kernel_setmem(kernel_x_solve_0, 1, mem_rhs, brisbane_rd);

    brisbane_task task0;
    brisbane_task_create(&task0);
    brisbane_task_kernel(task0, kernel_x_solve_0, 3, kernel_x_solve_0_off, kernel_x_solve_0_idx);
    brisbane_task_submit(task0, brisbane_gpu, NULL, true);

#if 0
#ifdef SPEC_USE_INNER_SIMD
    #pragma omp target teams distribute parallel for collapse(2) private(i,j,k)
#else
    #pragma omp target teams distribute parallel for simd collapse(3)
#endif
    for (k = 0; k <= nz2; k++) {
    for (j = 0; j <= JMAXP; j++) {
#ifdef SPEC_USE_INNER_SIMD
      #pragma omp simd
#endif
      for (i = 0; i <= IMAXP; i++) {
	  rhsX[0][k][i][j] = rhs[0][k][j][i];
	  rhsX[1][k][i][j] = rhs[1][k][j][i];
	  rhsX[2][k][i][j] = rhs[2][k][j][i];
	  rhsX[3][k][i][j] = rhs[3][k][j][i];
	  rhsX[4][k][i][j] = rhs[4][k][j][i];
      }
    }
  }
#endif

    size_t kernel_x_solve_1_off[2] = { 1, 1 };
    size_t kernel_x_solve_1_idx[2] = { ny2, nz2 };
    brisbane_kernel kernel_x_solve_1;
    brisbane_kernel_create("x_solve_1", &kernel_x_solve_1);
    brisbane_kernel_setmem(kernel_x_solve_1, 0, mem_lhsX, brisbane_wr);
    brisbane_kernel_setmem(kernel_x_solve_1, 1, mem_lhspX, brisbane_wr);
    brisbane_kernel_setmem(kernel_x_solve_1, 2, mem_lhsmX, brisbane_wr);
    brisbane_kernel_setarg(kernel_x_solve_1, 3, sizeof(int), &ni);

    brisbane_task task1;
    brisbane_task_create(&task1);
    brisbane_task_kernel(task1, kernel_x_solve_1, 2, kernel_x_solve_1_off, kernel_x_solve_1_idx);
    brisbane_task_submit(task1, brisbane_gpu, NULL, true);
#if 0
    #pragma omp target teams distribute parallel for private(k,j,m) collapse(2)
    for (k = 1; k <= nz2; k++) {
      for (j = 1; j <= ny2; j++) {
      for (m = 0; m < 5; m++) {
          lhsX[m][k][0][j] = 0.0;
          lhspX[m][k][0][j] = 0.0;
          lhsmX[m][k][0][j] = 0.0;
          lhsX[m][k][ni][j] = 0.0;
          lhspX[m][k][ni][j] = 0.0;
          lhsmX[m][k][ni][j] = 0.0;
      }
      lhsX[2][k][0][j] = 1.0;
      lhspX[2][k][0][j] = 1.0;
      lhsmX[2][k][0][j] = 1.0;
      lhsX[2][k][ni][j] = 1.0;
      lhspX[2][k][ni][j] = 1.0;
      lhsmX[2][k][ni][j] = 1.0;
    }
  }
#endif

    //---------------------------------------------------------------------
    // Computes the left hand side for the three x-factors  
    //---------------------------------------------------------------------
  
    //---------------------------------------------------------------------
    // first fill the lhs for the u-eigenvalue                   
    //---------------------------------------------------------------------
  size_t kernel_x_solve_2_off[2] = { 1, 1 };
  size_t kernel_x_solve_2_idx[2] = { ny2, nz2 };
  brisbane_kernel kernel_x_solve_2;
  brisbane_kernel_create("x_solve_2", &kernel_x_solve_2);
  brisbane_kernel_setmem(kernel_x_solve_2, 0, mem_rho_i, brisbane_rd);
  brisbane_kernel_setmem(kernel_x_solve_2, 1, mem_rhonX, brisbane_rw);
  brisbane_kernel_setmem(kernel_x_solve_2, 2, mem_lhsX, brisbane_rw);
  brisbane_kernel_setmem(kernel_x_solve_2, 3, mem_us, brisbane_rd);
  brisbane_kernel_setarg(kernel_x_solve_2, 4, sizeof(int), &gp01);
  brisbane_kernel_setarg(kernel_x_solve_2, 5, sizeof(double), &dx1);
  brisbane_kernel_setarg(kernel_x_solve_2, 6, sizeof(double), &dx2);
  brisbane_kernel_setarg(kernel_x_solve_2, 7, sizeof(double), &dx5);
  brisbane_kernel_setarg(kernel_x_solve_2, 8, sizeof(double), &dxmax);
  brisbane_kernel_setarg(kernel_x_solve_2, 9, sizeof(double), &c1c5);
  brisbane_kernel_setarg(kernel_x_solve_2, 10, sizeof(double), &c3c4);
  brisbane_kernel_setarg(kernel_x_solve_2, 11, sizeof(double), &dttx1);
  brisbane_kernel_setarg(kernel_x_solve_2, 12, sizeof(double), &dttx2);
  brisbane_kernel_setarg(kernel_x_solve_2, 13, sizeof(double), &c2dttx1);
  brisbane_kernel_setarg(kernel_x_solve_2, 14, sizeof(double), &con43);

  brisbane_task task2;
  brisbane_task_create(&task2);
  brisbane_task_kernel(task2, kernel_x_solve_2, 2, kernel_x_solve_2_off, kernel_x_solve_2_idx);
  brisbane_task_submit(task2, brisbane_gpu, NULL, true);
#if 0
  #pragma omp target teams distribute parallel for collapse(2) private(i,j,k,ru1)
  for (k = 1; k <= nz2; k++) {
    for (j = 1; j <= ny2; j++) {
      #pragma omp simd private(ru1)
      for (i = 0; i <= gp01; i++) {
        ru1 = c3c4*rho_i[k][j][i];
        rhonX[k][j][i] = max(max(dx2+con43*ru1,dx5+c1c5*ru1), max(dxmax+ru1,dx1));
      }
    #pragma omp simd
    for (i = 1; i <= nx2; i++) {
        lhsX[0][k][i][j] =  0.0;
        lhsX[1][k][i][j] = -dttx2 * us[k][j][i-1] - dttx1 * rhonX[k][j][i-1];
        lhsX[2][k][i][j] =  1.0 + c2dttx1 * rhonX[k][j][i];
        lhsX[3][k][i][j] =  dttx2 * us[k][j][i+1] - dttx1 * rhonX[k][j][i+1];
        lhsX[4][k][i][j] =  0.0;
      }
    }
  }
#endif

    //---------------------------------------------------------------------
    // add fourth order dissipation                             
    //---------------------------------------------------------------------
  i = 1;

  size_t kernel_x_solve_3_off[2] = { 1, 1 };
  size_t kernel_x_solve_3_idx[2] = { ny2, nz2 };
  brisbane_kernel kernel_x_solve_3;
  brisbane_kernel_create("x_solve_3", &kernel_x_solve_3);
  brisbane_kernel_setmem(kernel_x_solve_3, 0, mem_lhsX, brisbane_rw);
  brisbane_kernel_setarg(kernel_x_solve_3, 1, sizeof(int), &i);
  brisbane_kernel_setarg(kernel_x_solve_3, 2, sizeof(double), &comz1);
  brisbane_kernel_setarg(kernel_x_solve_3, 3, sizeof(double), &comz4);
  brisbane_kernel_setarg(kernel_x_solve_3, 4, sizeof(double), &comz5);
  brisbane_kernel_setarg(kernel_x_solve_3, 5, sizeof(double), &comz6);

  brisbane_task task3;
  brisbane_task_create(&task3);
  brisbane_task_kernel(task3, kernel_x_solve_3, 2, kernel_x_solve_3_off, kernel_x_solve_3_idx);
  brisbane_task_submit(task3, brisbane_gpu, NULL, true);
#if 0
#ifdef SPEC_USE_INNER_SIMD
  #pragma omp target teams distribute parallel for private(k,j) 
#else
  #pragma omp target teams distribute parallel for simd collapse(2)
#endif
  for (k = 1; k <= nz2; k++) {
#ifdef SPEC_USE_INNER_SIMD
    #pragma omp simd
#endif
    for (j = 1; j <= ny2; j++) {
      lhsX[2][k][i][j] = lhsX[2][k][i][j] + comz5;
      lhsX[3][k][i][j] = lhsX[3][k][i][j] - comz4;
      lhsX[4][k][i][j] = lhsX[4][k][i][j] + comz1;

      lhsX[1][k][i+1][j] = lhsX[1][k][i+1][j] - comz4;
      lhsX[2][k][i+1][j] = lhsX[2][k][i+1][j] + comz6;
      lhsX[3][k][i+1][j] = lhsX[3][k][i+1][j] - comz4;
      lhsX[4][k][i+1][j] = lhsX[4][k][i+1][j] + comz1;
    }
  }
#endif

  size_t kernel_x_solve_4_off[3] = { 3, 1, 1 };
  size_t kernel_x_solve_4_idx[3] = { gp04 - 2, ny2, nz2 };
  brisbane_kernel kernel_x_solve_4;
  brisbane_kernel_create("x_solve_4", &kernel_x_solve_4);
  brisbane_kernel_setmem(kernel_x_solve_4, 0, mem_lhsX, brisbane_rw);
  brisbane_kernel_setarg(kernel_x_solve_4, 1, sizeof(double), &comz1);
  brisbane_kernel_setarg(kernel_x_solve_4, 2, sizeof(double), &comz4);
  brisbane_kernel_setarg(kernel_x_solve_4, 3, sizeof(double), &comz6);

  brisbane_task task4;
  brisbane_task_create(&task4);
  brisbane_task_kernel(task4, kernel_x_solve_4, 3, kernel_x_solve_4_off, kernel_x_solve_4_idx);
  brisbane_task_submit(task4, brisbane_gpu, NULL, true);
#if 0
#ifdef SPEC_USE_INNER_SIMD
  #pragma omp target teams distribute parallel for private(i,j,k) collapse(2)
#else
  #pragma omp target teams distribute parallel for simd collapse(3)
#endif
  for (k = 1; k <= nz2; k++) {
    for (j = 1; j <= ny2; j++) {
#ifdef SPEC_USE_INNER_SIMD
    #pragma omp simd
#endif
    for (i = 3; i <= gp04; i++) {
        lhsX[0][k][i][j] = lhsX[0][k][i][j] + comz1;
        lhsX[1][k][i][j] = lhsX[1][k][i][j] - comz4;
        lhsX[2][k][i][j] = lhsX[2][k][i][j] + comz6;
        lhsX[3][k][i][j] = lhsX[3][k][i][j] - comz4;
        lhsX[4][k][i][j] = lhsX[4][k][i][j] + comz1;
      }
    }
  }
#endif

  i = gp03;

  size_t kernel_x_solve_5_off[2] = { 1, 1 };
  size_t kernel_x_solve_5_idx[2] = { ny2, nz2 };
  brisbane_kernel kernel_x_solve_5;
  brisbane_kernel_create("x_solve_5", &kernel_x_solve_5);
  brisbane_kernel_setmem(kernel_x_solve_5, 0, mem_lhsX, brisbane_rw);
  brisbane_kernel_setarg(kernel_x_solve_5, 1, sizeof(int), &i);
  brisbane_kernel_setarg(kernel_x_solve_5, 2, sizeof(double), &comz1);
  brisbane_kernel_setarg(kernel_x_solve_5, 3, sizeof(double), &comz4);
  brisbane_kernel_setarg(kernel_x_solve_5, 4, sizeof(double), &comz5);
  brisbane_kernel_setarg(kernel_x_solve_5, 5, sizeof(double), &comz6);

  brisbane_task task5;
  brisbane_task_create(&task5);
  brisbane_task_kernel(task5, kernel_x_solve_5, 2, kernel_x_solve_5_off, kernel_x_solve_5_idx);
  brisbane_task_submit(task5, brisbane_gpu, NULL, true);
#if 0
#ifdef SPEC_USE_INNER_SIMD
  #pragma omp target teams distribute parallel for private(j,k)
#else
  #pragma omp target teams distribute parallel for simd collapse(2)
#endif
  for (k = 1; k <= nz2; k++) {
#ifdef SPEC_USE_INNER_SIMD
    #pragma omp simd
#endif
    for (j = 1; j <= ny2; j++) {
      lhsX[0][k][i][j] = lhsX[0][k][i][j] + comz1;
      lhsX[1][k][i][j] = lhsX[1][k][i][j] - comz4;
      lhsX[2][k][i][j] = lhsX[2][k][i][j] + comz6;
      lhsX[3][k][i][j] = lhsX[3][k][i][j] - comz4;

      lhsX[0][k][i+1][j] = lhsX[0][k][i+1][j] + comz1;
      lhsX[1][k][i+1][j] = lhsX[1][k][i+1][j] - comz4;
      lhsX[2][k][i+1][j] = lhsX[2][k][i+1][j] + comz5;
    }
  }
#endif

    //---------------------------------------------------------------------
    // subsequently, fill the other factors (u+c), (u-c) by adding to 
    // the first  
    //---------------------------------------------------------------------
  size_t kernel_x_solve_6_off[3] = { 1, 1, 1 };
  size_t kernel_x_solve_6_idx[3] = { nx2, ny2, nz2 };
  brisbane_kernel kernel_x_solve_6;
  brisbane_kernel_create("x_solve_6", &kernel_x_solve_6);
  brisbane_kernel_setmem(kernel_x_solve_6, 0, mem_lhspX, brisbane_wr);
  brisbane_kernel_setmem(kernel_x_solve_6, 1, mem_lhsmX, brisbane_wr);
  brisbane_kernel_setmem(kernel_x_solve_6, 2, mem_lhsX, brisbane_rd);
  brisbane_kernel_setmem(kernel_x_solve_6, 3, mem_speed, brisbane_rd);
  brisbane_kernel_setarg(kernel_x_solve_6, 4, sizeof(double), &dttx2);

  brisbane_task task6;
  brisbane_task_create(&task6);
  brisbane_task_kernel(task6, kernel_x_solve_6, 3, kernel_x_solve_6_off, kernel_x_solve_6_idx);
  brisbane_task_submit(task6, brisbane_gpu, NULL, true);
#if 0
#ifdef SPEC_USE_INNER_SIMD
  #pragma omp target teams distribute parallel for private(i,j,k) collapse(2) 
#else
  #pragma omp target teams distribute parallel for simd collapse(3)
#endif
  for (k = 1; k <= nz2; k++) {
    for (j = 1; j <= ny2; j++) {
#ifdef SPEC_USE_INNER_SIMD
    #pragma omp simd
#endif
    for (i = 1; i <= nx2; i++) {
        lhspX[0][k][i][j] = lhsX[0][k][i][j];
        lhspX[1][k][i][j] = lhsX[1][k][i][j] - dttx2 * speed[k][j][i-1];
        lhspX[2][k][i][j] = lhsX[2][k][i][j];
        lhspX[3][k][i][j] = lhsX[3][k][i][j] + dttx2 * speed[k][j][i+1];
        lhspX[4][k][i][j] = lhsX[4][k][i][j];
        lhsmX[0][k][i][j] = lhsX[0][k][i][j];
        lhsmX[1][k][i][j] = lhsX[1][k][i][j] + dttx2 * speed[k][j][i-1];
        lhsmX[2][k][i][j] = lhsX[2][k][i][j];
        lhsmX[3][k][i][j] = lhsX[3][k][i][j] - dttx2 * speed[k][j][i+1];
        lhsmX[4][k][i][j] = lhsX[4][k][i][j];
      }
    }
  }
#endif

    //---------------------------------------------------------------------
    // FORWARD ELIMINATION  
    //---------------------------------------------------------------------

    //---------------------------------------------------------------------
    // perform the Thomas algorithm; first, FORWARD ELIMINATION     
    //---------------------------------------------------------------------
  size_t kernel_x_solve_7_off[2] = { 1, 1 };
  size_t kernel_x_solve_7_idx[2] = { ny2, nz2 };
  brisbane_kernel kernel_x_solve_7;
  brisbane_kernel_create("x_solve_7", &kernel_x_solve_7);
  brisbane_kernel_setmem(kernel_x_solve_7, 0, mem_lhsX, brisbane_rw);
  brisbane_kernel_setmem(kernel_x_solve_7, 1, mem_rhsX, brisbane_rw);
  brisbane_kernel_setarg(kernel_x_solve_7, 2, sizeof(int), &gp03);

  brisbane_task task7;
  brisbane_task_create(&task7);
  brisbane_task_kernel(task7, kernel_x_solve_7, 2, kernel_x_solve_7_off, kernel_x_solve_7_idx);
  brisbane_task_submit(task7, brisbane_gpu, NULL, true);
#if 0
#ifdef SPEC_USE_INNER_SIMD
  #pragma omp target teams distribute parallel for private(i,j,k,m)
#else
  #pragma omp target teams distribute parallel for simd collapse(2) private(i,m,i1,i2,fac1)
#endif
  for (k = 1; k <= nz2; k++) {
#ifdef SPEC_USE_INNER_SIMD
  #pragma omp simd private(i1,i2,fac1)
#endif
    for (j = 1; j <= ny2; j++) {
    for (i = 0; i <= gp03; i++) {
      i1 = i + 1;
      i2 = i + 2;
        fac1 = 1.0/lhsX[2][k][i][j];
        lhsX[3][k][i][j] = fac1*lhsX[3][k][i][j];
        lhsX[4][k][i][j] = fac1*lhsX[4][k][i][j];
        for (m = 0; m < 3; m++) {
          rhsX[m][k][i][j] = fac1*rhsX[m][k][i][j];
        }
        lhsX[2][k][i1][j] = lhsX[2][k][i1][j] - lhsX[1][k][i1][j]*lhsX[3][k][i][j];
        lhsX[3][k][i1][j] = lhsX[3][k][i1][j] - lhsX[1][k][i1][j]*lhsX[4][k][i][j];
        for (m = 0; m < 3; m++) {
          rhsX[m][k][i1][j] = rhsX[m][k][i1][j] - lhsX[1][k][i1][j]*rhsX[m][k][i][j];
        }
        lhsX[1][k][i2][j] = lhsX[1][k][i2][j] - lhsX[0][k][i2][j]*lhsX[3][k][i][j];
        lhsX[2][k][i2][j] = lhsX[2][k][i2][j] - lhsX[0][k][i2][j]*lhsX[4][k][i][j];
        for (m = 0; m < 3; m++) {
          rhsX[m][k][i2][j] = rhsX[m][k][i2][j] - lhsX[0][k][i2][j]*rhsX[m][k][i][j];
        }
      }
    }
  }
#endif

    //---------------------------------------------------------------------
    // The last two rows in this grid block are a bit different, 
    // since they for (not have two more rows available for the
    // elimination of off-diagonal entries
    //---------------------------------------------------------------------
  i  = gp02;
  i1 = gp01;

  size_t kernel_x_solve_8_off[2] = { 1, 1 };
  size_t kernel_x_solve_8_idx[2] = { ny2, nz2 };
  brisbane_kernel kernel_x_solve_8;
  brisbane_kernel_create("x_solve_8", &kernel_x_solve_8);
  brisbane_kernel_setmem(kernel_x_solve_8, 0, mem_lhsX, brisbane_wr);
  brisbane_kernel_setmem(kernel_x_solve_8, 1, mem_rhsX, brisbane_wr);
  brisbane_kernel_setarg(kernel_x_solve_8, 2, sizeof(int), &i);
  brisbane_kernel_setarg(kernel_x_solve_8, 3, sizeof(int), &i1);

  brisbane_task task8;
  brisbane_task_create(&task8);
  brisbane_task_kernel(task8, kernel_x_solve_8, 2, kernel_x_solve_8_off, kernel_x_solve_8_idx);
  brisbane_task_submit(task8, brisbane_gpu, NULL, true);
#if 0
  #pragma omp target teams distribute parallel for private(j,k,m,fac1,fac2) collapse(2)
  for (k = 1; k <= nz2; k++) {
    for (j = 1; j <= ny2; j++) {
      fac1 = 1.0/lhsX[2][k][i][j];
      lhsX[3][k][i][j] = fac1*lhsX[3][k][i][j];
      lhsX[4][k][i][j] = fac1*lhsX[4][k][i][j];
      for (m = 0; m < 3; m++) {
        rhsX[m][k][i][j] = fac1*rhsX[m][k][i][j];
      }
      lhsX[2][k][i1][j] = lhsX[2][k][i1][j] - lhsX[1][k][i1][j]*lhsX[3][k][i][j];
      lhsX[3][k][i1][j] = lhsX[3][k][i1][j] - lhsX[1][k][i1][j]*lhsX[4][k][i][j];
      for (m = 0; m < 3; m++) {
        rhsX[m][k][i1][j] = rhsX[m][k][i1][j] - lhsX[1][k][i1][j]*rhsX[m][k][i][j];
      }

      //---------------------------------------------------------------------
      // scale the last row immediately 
      //---------------------------------------------------------------------
      fac2 = 1.0/lhsX[2][k][i1][j];
      for (m = 0; m < 3; m++) {
        rhsX[m][k][i1][j] = fac2*rhsX[m][k][i1][j];
      }
    }
  }
#endif

    //---------------------------------------------------------------------
    // for (the u+c and the u-c factors                 
    //---------------------------------------------------------------------
  size_t kernel_x_solve_9_off[2] = { 1, 1 };
  size_t kernel_x_solve_9_idx[2] = { ny2, nz2 };
  brisbane_kernel kernel_x_solve_9;
  brisbane_kernel_create("x_solve_9", &kernel_x_solve_9);
  brisbane_kernel_setmem(kernel_x_solve_9, 0, mem_lhspX, brisbane_wr);
  brisbane_kernel_setmem(kernel_x_solve_9, 1, mem_lhsmX, brisbane_wr);
  brisbane_kernel_setmem(kernel_x_solve_9, 2, mem_rhsX, brisbane_wr);
  brisbane_kernel_setarg(kernel_x_solve_9, 3, sizeof(int), &gp03);

  brisbane_task task9;
  brisbane_task_create(&task9);
  brisbane_task_kernel(task9, kernel_x_solve_9, 2, kernel_x_solve_9_off, kernel_x_solve_9_idx);
  brisbane_task_submit(task9, brisbane_gpu, NULL, true);
#if 0
#ifdef SPEC_USE_INNER_SIMD
  #pragma omp target teams distribute parallel for private(i,j,k,m) 
#else
  #pragma omp target teams distribute parallel for simd collapse(2) private(i,m,fac1,i1,i2)
#endif
  for (k = 1; k <= nz2; k++) {
#ifdef SPEC_USE_INNER_SIMD
  #pragma omp simd private(fac1,i1,i2)
#endif
    for (j = 1; j <= ny2; j++) {
    for (i = 0; i <= gp03; i++) {
      i1 = i + 1;
      i2 = i + 2;
        m = 3;
        fac1 = 1.0/lhspX[2][k][i][j];
        lhspX[3][k][i][j]    = fac1*lhspX[3][k][i][j];
        lhspX[4][k][i][j]    = fac1*lhspX[4][k][i][j];
        rhsX[m][k][i][j]  = fac1*rhsX[m][k][i][j];
        lhspX[2][k][i1][j]   = lhspX[2][k][i1][j] - lhspX[1][k][i1][j]*lhspX[3][k][i][j];
        lhspX[3][k][i1][j]   = lhspX[3][k][i1][j] - lhspX[1][k][i1][j]*lhspX[4][k][i][j];
        rhsX[m][k][i1][j] = rhsX[m][k][i1][j] - lhspX[1][k][i1][j]*rhsX[m][k][i][j];
        lhspX[1][k][i2][j]   = lhspX[1][k][i2][j] - lhspX[0][k][i2][j]*lhspX[3][k][i][j];
        lhspX[2][k][i2][j]   = lhspX[2][k][i2][j] - lhspX[0][k][i2][j]*lhspX[4][k][i][j];
        rhsX[m][k][i2][j] = rhsX[m][k][i2][j] - lhspX[0][k][i2][j]*rhsX[m][k][i][j];
 
        m = 4;
        fac1 = 1.0/lhsmX[2][k][i][j];
        lhsmX[3][k][i][j]    = fac1*lhsmX[3][k][i][j];
        lhsmX[4][k][i][j]    = fac1*lhsmX[4][k][i][j];
        rhsX[m][k][i][j]  = fac1*rhsX[m][k][i][j];
        lhsmX[2][k][i1][j]   = lhsmX[2][k][i1][j] - lhsmX[1][k][i1][j]*lhsmX[3][k][i][j];
        lhsmX[3][k][i1][j]   = lhsmX[3][k][i1][j] - lhsmX[1][k][i1][j]*lhsmX[4][k][i][j];
        rhsX[m][k][i1][j] = rhsX[m][k][i1][j] - lhsmX[1][k][i1][j]*rhsX[m][k][i][j];
        lhsmX[1][k][i2][j]   = lhsmX[1][k][i2][j] - lhsmX[0][k][i2][j]*lhsmX[3][k][i][j];
        lhsmX[2][k][i2][j]   = lhsmX[2][k][i2][j] - lhsmX[0][k][i2][j]*lhsmX[4][k][i][j];
        rhsX[m][k][i2][j] = rhsX[m][k][i2][j] - lhsmX[0][k][i2][j]*rhsX[m][k][i][j];
      }
    }
  }
#endif

    //---------------------------------------------------------------------
    // And again the last two rows separately
    //---------------------------------------------------------------------
  i  = gp02;
  i1 = gp01;

  size_t kernel_x_solve_10_off[2] = { 1, 1 };
  size_t kernel_x_solve_10_idx[2] = { ny2, nz2 };
  brisbane_kernel kernel_x_solve_10;
  brisbane_kernel_create("x_solve_10", &kernel_x_solve_10);
  brisbane_kernel_setmem(kernel_x_solve_10, 0, mem_lhspX, brisbane_wr);
  brisbane_kernel_setmem(kernel_x_solve_10, 1, mem_lhsmX, brisbane_wr);
  brisbane_kernel_setmem(kernel_x_solve_10, 2, mem_rhsX, brisbane_wr);
  brisbane_kernel_setarg(kernel_x_solve_10, 3, sizeof(int), &i);
  brisbane_kernel_setarg(kernel_x_solve_10, 4, sizeof(int), &i1);

  brisbane_task task10;
  brisbane_task_create(&task10);
  brisbane_task_kernel(task10, kernel_x_solve_10, 2, kernel_x_solve_10_off, kernel_x_solve_10_idx);
  brisbane_task_submit(task10, brisbane_gpu, NULL, true);
#if 0
#ifdef SPEC_USE_INNER_SIMD
  #pragma omp target teams distribute parallel for private(j,k,m,fac1)
#else
  #pragma omp target teams distribute parallel for simd collapse(2) private(m,fac1)
#endif
  for (k = 1; k <= nz2; k++) {
#ifdef SPEC_USE_INNER_SIMD
    #pragma omp simd
#endif
    for (j = 1; j <= ny2; j++) {

      m = 3;
      fac1 = 1.0/lhspX[2][k][i][j];
      lhspX[3][k][i][j]    = fac1*lhspX[3][k][i][j];
      lhspX[4][k][i][j]    = fac1*lhspX[4][k][i][j];
      rhsX[m][k][i][j]  = fac1*rhsX[m][k][i][j];
      lhspX[2][k][i1][j]   = lhspX[2][k][i1][j] - lhspX[1][k][i1][j]*lhspX[3][k][i][j];
      lhspX[3][k][i1][j]   = lhspX[3][k][i1][j] - lhspX[1][k][i1][j]*lhspX[4][k][i][j];
      rhsX[m][k][i1][j] = rhsX[m][k][i1][j] - lhspX[1][k][i1][j]*rhsX[m][k][i][j];

      m = 4;
      fac1 = 1.0/lhsmX[2][k][i][j];
      lhsmX[3][k][i][j]    = fac1*lhsmX[3][k][i][j];
      lhsmX[4][k][i][j]    = fac1*lhsmX[4][k][i][j];
      rhsX[m][k][i][j]  = fac1*rhsX[m][k][i][j];
      lhsmX[2][k][i1][j]   = lhsmX[2][k][i1][j] - lhsmX[1][k][i1][j]*lhsmX[3][k][i][j];
      lhsmX[3][k][i1][j]   = lhsmX[3][k][i1][j] - lhsmX[1][k][i1][j]*lhsmX[4][k][i][j];
      rhsX[m][k][i1][j] = rhsX[m][k][i1][j] - lhsmX[1][k][i1][j]*rhsX[m][k][i][j];

      //---------------------------------------------------------------------
      // Scale the last row immediately
      //---------------------------------------------------------------------
      rhsX[3][k][i1][j] = rhsX[3][k][i1][j]/lhspX[2][k][i1][j];
      rhsX[4][k][i1][j] = rhsX[4][k][i1][j]/lhsmX[2][k][i1][j];
    }
  }
#endif

    //---------------------------------------------------------------------
    // BACKSUBSTITUTION 
    //---------------------------------------------------------------------
  i  = gp02;
  i1 = gp01;

  size_t kernel_x_solve_11_off[2] = { 1, 1 };
  size_t kernel_x_solve_11_idx[2] = { ny2, nz2 };
  brisbane_kernel kernel_x_solve_11;
  brisbane_kernel_create("x_solve_11", &kernel_x_solve_11);
  brisbane_kernel_setmem(kernel_x_solve_11, 0, mem_rhsX, brisbane_rw);
  brisbane_kernel_setmem(kernel_x_solve_11, 1, mem_lhsX, brisbane_rd);
  brisbane_kernel_setmem(kernel_x_solve_11, 2, mem_lhspX, brisbane_rd);
  brisbane_kernel_setmem(kernel_x_solve_11, 3, mem_lhsmX, brisbane_rd);
  brisbane_kernel_setarg(kernel_x_solve_11, 4, sizeof(int), &i);
  brisbane_kernel_setarg(kernel_x_solve_11, 5, sizeof(int), &i1);

  brisbane_task task11;
  brisbane_task_create(&task11);
  brisbane_task_kernel(task11, kernel_x_solve_11, 2, kernel_x_solve_11_off, kernel_x_solve_11_idx);
  brisbane_task_submit(task11, brisbane_gpu, NULL, true);
#if 0
  #pragma omp target teams distribute parallel for private(j,k,m) collapse(2)
  for (k = 1; k <= nz2; k++) {
    for (j = 1; j <= ny2; j++) {
      for (m = 0; m < 3; m++) {
        rhsX[m][k][i][j] = rhsX[m][k][i][j] - lhsX[3][k][i][j]*rhsX[m][k][i1][j];
      }

      rhsX[3][k][i][j] = rhsX[3][k][i][j] - lhspX[3][k][i][j]*rhsX[3][k][i1][j];
      rhsX[4][k][i][j] = rhsX[4][k][i][j] - lhsmX[3][k][i][j]*rhsX[4][k][i1][j];
    }
  }
#endif

    //---------------------------------------------------------------------
    // The first three factors
    //---------------------------------------------------------------------
  size_t kernel_x_solve_12_off[2] = { 1, 1 };
  size_t kernel_x_solve_12_idx[2] = { ny2, nz2 };
  brisbane_kernel kernel_x_solve_12;
  brisbane_kernel_create("x_solve_12", &kernel_x_solve_12);
  brisbane_kernel_setmem(kernel_x_solve_12, 0, mem_rhsX, brisbane_rw);
  brisbane_kernel_setmem(kernel_x_solve_12, 1, mem_lhsX, brisbane_rd);
  brisbane_kernel_setmem(kernel_x_solve_12, 2, mem_lhspX, brisbane_rd);
  brisbane_kernel_setmem(kernel_x_solve_12, 3, mem_lhsmX, brisbane_rd);
  brisbane_kernel_setarg(kernel_x_solve_12, 4, sizeof(int), &gp03);

  brisbane_task task12;
  brisbane_task_create(&task12);
  brisbane_task_kernel(task12, kernel_x_solve_12, 2, kernel_x_solve_12_off, kernel_x_solve_12_idx);
  brisbane_task_submit(task12, brisbane_gpu, NULL, true);
#if 0
#ifdef SPEC_USE_INNER_SIMD
  #pragma omp target teams distribute parallel for private(i,j,k,m) 
#else
  #pragma omp target teams distribute parallel for simd collapse(2) private(i,m,i1,i2)
#endif
  for (k = 1; k <= nz2; k++) {
#ifdef SPEC_USE_INNER_SIMD
  #pragma omp simd private(i1,i2)
#endif
    for (j = 1; j <= ny2; j++) {
    for (i = gp03; i >= 0; i--) {
      i1 = i + 1;
      i2 = i + 2;
        for (m = 0; m < 3; m++) {
          rhsX[m][k][i][j] = rhsX[m][k][i][j] - 
                            lhsX[3][k][i][j]*rhsX[m][k][i1][j] -
                            lhsX[4][k][i][j]*rhsX[m][k][i2][j];
        }

        //-------------------------------------------------------------------
        // And the remaining two
        //-------------------------------------------------------------------
        rhsX[3][k][i][j] = rhsX[3][k][i][j] - 
                          lhspX[3][k][i][j]*rhsX[3][k][i1][j] -
                          lhspX[4][k][i][j]*rhsX[3][k][i2][j];
        rhsX[4][k][i][j] = rhsX[4][k][i][j] - 
                          lhsmX[3][k][i][j]*rhsX[4][k][i1][j] -
                          lhsmX[4][k][i][j]*rhsX[4][k][i2][j];
      }
    }
  }  
#endif

  size_t kernel_x_solve_13_off[3] = { 0, 0, 0 };
  size_t kernel_x_solve_13_idx[3] = { IMAXP + 1, JMAXP + 1, nz2 + 1 };
  brisbane_kernel kernel_x_solve_13;
  brisbane_kernel_create("x_solve_13", &kernel_x_solve_13);
  brisbane_kernel_setmem(kernel_x_solve_13, 0, mem_rhs, brisbane_wr);
  brisbane_kernel_setmem(kernel_x_solve_13, 1, mem_rhsX, brisbane_rd);

  brisbane_task task13;
  brisbane_task_create(&task13);
  brisbane_task_kernel(task13, kernel_x_solve_13, 3, kernel_x_solve_13_off, kernel_x_solve_13_idx);
  brisbane_task_submit(task13, brisbane_gpu, NULL, true);
#if 0
#ifdef SPEC_USE_INNER_SIMD
#pragma omp target teams distribute parallel for private(i,j,k) collapse(2)
#else
#pragma omp target teams distribute parallel for simd collapse(3)
#endif
    for (k = 0; k <= nz2; k++) {
      for (j = 0; j <= JMAXP; j++) {
#ifdef SPEC_USE_INNER_SIMD
        #pragma omp simd
#endif
        for (i = 0; i <= IMAXP; i++) {
	  rhs[0][k][j][i] = rhsX[0][k][i][j];
	  rhs[1][k][j][i] = rhsX[1][k][i][j];
	  rhs[2][k][j][i] = rhsX[2][k][i][j];
	  rhs[3][k][j][i] = rhsX[3][k][i][j];
	  rhs[4][k][j][i] = rhsX[4][k][i][j];
      }
  }
  }
#endif

  brisbane_mem_release(mem_lhsX);
  brisbane_mem_release(mem_lhspX);
  brisbane_mem_release(mem_lhsmX);
  brisbane_mem_release(mem_rhonX);
  brisbane_mem_release(mem_rhsX);
}/* end omp target data  */

  //---------------------------------------------------------------------
  // Do the block-diagonal inversion          
  //---------------------------------------------------------------------
  ninvr();
}

void y_solve()
{
  int i, j, k, j1, j2, m;
  int gp0, gp1, gp2;
  double ru1, fac1, fac2;
  double lhsY[5][nz2+1][IMAXP+1][IMAXP+1];
  double lhspY[5][nz2+1][IMAXP+1][IMAXP+1];
  double lhsmY[5][nz2+1][IMAXP+1][IMAXP+1];
  double rhoqY[nz2+1][IMAXP+1][PROBLEM_SIZE];

  int ni=ny2+1;
  gp0=grid_points[0];
  gp1=grid_points[1];
  gp2=grid_points[2];

  brisbane_mem mem_lhsY;
  brisbane_mem mem_lhspY;
  brisbane_mem mem_lhsmY;
  brisbane_mem mem_rhoqY;
  brisbane_mem_create(5 * (nz2 + 1) * (IMAXP + 1) * (IMAXP + 1) * sizeof(double), &mem_lhsY);
  brisbane_mem_create(5 * (nz2 + 1) * (IMAXP + 1) * (IMAXP + 1) * sizeof(double), &mem_lhspY);
  brisbane_mem_create(5 * (nz2 + 1) * (IMAXP + 1) * (IMAXP + 1) * sizeof(double), &mem_lhsmY);
  brisbane_mem_create((nz2 + 1) * (IMAXP + 1) * (PROBLEM_SIZE) * sizeof(double), &mem_rhoqY);

#pragma omp target data map(alloc:lhsY[:][:][:][:],lhspY[:][:][:][:],lhsmY[:][:][:][:],rhoqY[:][:][:]) //present(rho_i,vs,speed,rhs)
{
    size_t kernel_y_solve_0_off[2] = { 1, 1 };
    size_t kernel_y_solve_0_idx[2] = { nx2, nz2 };
    brisbane_kernel kernel_y_solve_0;
    brisbane_kernel_create("y_solve_0", &kernel_y_solve_0);
    brisbane_kernel_setmem(kernel_y_solve_0, 0, mem_lhsY, brisbane_wr);
    brisbane_kernel_setmem(kernel_y_solve_0, 1, mem_lhspY, brisbane_wr);
    brisbane_kernel_setmem(kernel_y_solve_0, 2, mem_lhsmY, brisbane_wr);
    brisbane_kernel_setarg(kernel_y_solve_0, 3, sizeof(int), &ni);

    brisbane_task task0;
    brisbane_task_create(&task0);
    brisbane_task_kernel(task0, kernel_y_solve_0, 2, kernel_y_solve_0_off, kernel_y_solve_0_idx);
    brisbane_task_submit(task0, brisbane_gpu, NULL, true);
#if 0
    #pragma omp target teams distribute parallel for private(i,k,m) collapse(2)
    for (k = 1; k <= nz2; k++) {
      for (i = 1; i <= nx2; i++) {
      for (m = 0; m < 5; m++) {
        lhsY[m][k][0][i] = 0.0;
        lhspY[m][k][0][i] = 0.0;
        lhsmY[m][k][0][i] = 0.0;
        lhsY[m][k][ni][i] = 0.0;
        lhspY[m][k][ni][i] = 0.0;
        lhsmY[m][k][ni][i] = 0.0;
      }
      lhsY[2][k][0][i] = 1.0;
      lhspY[2][k][0][i] = 1.0;
      lhsmY[2][k][0][i] = 1.0;
      lhsY[2][k][ni][i] = 1.0;
      lhspY[2][k][ni][i] = 1.0;
      lhsmY[2][k][ni][i] = 1.0;
    }
  }
#endif

    //---------------------------------------------------------------------
    // Computes the left hand side for the three y-factors   
    //---------------------------------------------------------------------

    //---------------------------------------------------------------------
    // first fill the lhs for the u-eigenvalue         
    //---------------------------------------------------------------------
    size_t kernel_y_solve_1_off[2] = { 1, 1 };
    size_t kernel_y_solve_1_idx[2] = { gp0 - 2, nz2 };
    brisbane_kernel kernel_y_solve_1;
    brisbane_kernel_create("y_solve_1", &kernel_y_solve_1);
    brisbane_kernel_setmem(kernel_y_solve_1, 0, mem_rho_i, brisbane_rd);
    brisbane_kernel_setmem(kernel_y_solve_1, 1, mem_rhoqY, brisbane_rw);
    brisbane_kernel_setmem(kernel_y_solve_1, 2, mem_lhsY, brisbane_wr);
    brisbane_kernel_setmem(kernel_y_solve_1, 3, mem_vs, brisbane_rd);
    brisbane_kernel_setarg(kernel_y_solve_1, 4, sizeof(int), &gp1);
    brisbane_kernel_setarg(kernel_y_solve_1, 5, sizeof(double), &dy1);
    brisbane_kernel_setarg(kernel_y_solve_1, 6, sizeof(double), &dy3);
    brisbane_kernel_setarg(kernel_y_solve_1, 7, sizeof(double), &dy5);
    brisbane_kernel_setarg(kernel_y_solve_1, 8, sizeof(double), &dymax);
    brisbane_kernel_setarg(kernel_y_solve_1, 9, sizeof(double), &c1c5);
    brisbane_kernel_setarg(kernel_y_solve_1, 10, sizeof(double), &c3c4);
    brisbane_kernel_setarg(kernel_y_solve_1, 11, sizeof(double), &dtty1);
    brisbane_kernel_setarg(kernel_y_solve_1, 12, sizeof(double), &dtty2);
    brisbane_kernel_setarg(kernel_y_solve_1, 13, sizeof(double), &c2dtty1);
    brisbane_kernel_setarg(kernel_y_solve_1, 14, sizeof(double), &con43);

    brisbane_task task1;
    brisbane_task_create(&task1);
    brisbane_task_kernel(task1, kernel_y_solve_1, 2, kernel_y_solve_1_off, kernel_y_solve_1_idx);
    brisbane_task_submit(task1, brisbane_gpu, NULL, true);
#if 0
  #pragma omp target teams distribute parallel for collapse(2) private(i,j,k,ru1)
  for (k = 1; k <= nz2; k++) {
    for (i = 1; i <= gp0-2; i++) {
      #pragma omp simd private(ru1)
      for (j = 0; j <= gp1-1; j++) {
        ru1 = c3c4*rho_i[k][j][i];
        rhoqY[k][j][i] = max(max(dy3+con43*ru1, dy5+c1c5*ru1), max(dymax+ru1, dy1));
      }
      #pragma omp simd
      for (j = 1; j <= gp1-2; j++) {
        lhsY[0][k][j][i] =  0.0;
        lhsY[1][k][j][i] = -dtty2 * vs[k][j-1][i] - dtty1 * rhoqY[k][j-1][i];
        lhsY[2][k][j][i] =  1.0 + c2dtty1 * rhoqY[k][j][i];
        lhsY[3][k][j][i] =  dtty2 * vs[k][j+1][i] - dtty1 * rhoqY[k][j+1][i];
        lhsY[4][k][j][i] =  0.0;
      }
    }
  }
#endif

    //---------------------------------------------------------------------
    // add fourth order dissipation                             
    //---------------------------------------------------------------------
  j = 1;

  size_t kernel_y_solve_2_off[2] = { 1, 1 };
  size_t kernel_y_solve_2_idx[2] = { gp0 - 2, gp2 - 2 };
  brisbane_kernel kernel_y_solve_2;
  brisbane_kernel_create("y_solve_2", &kernel_y_solve_2);
  brisbane_kernel_setmem(kernel_y_solve_2, 0, mem_lhsY, brisbane_rw);
  brisbane_kernel_setarg(kernel_y_solve_2, 1, sizeof(double), &comz1);
  brisbane_kernel_setarg(kernel_y_solve_2, 2, sizeof(double), &comz4);
  brisbane_kernel_setarg(kernel_y_solve_2, 3, sizeof(double), &comz5);
  brisbane_kernel_setarg(kernel_y_solve_2, 4, sizeof(double), &comz6);
  brisbane_kernel_setarg(kernel_y_solve_2, 5, sizeof(int), &j);

  brisbane_task task2;
  brisbane_task_create(&task2);
  brisbane_task_kernel(task2, kernel_y_solve_2, 2, kernel_y_solve_2_off, kernel_y_solve_2_idx);
  brisbane_task_submit(task2, brisbane_gpu, NULL, true);
#if 0
#ifdef SPEC_USE_INNER_SIMD
  #pragma omp target teams distribute parallel for private(i,k)
#else
  #pragma omp target teams distribute parallel for simd collapse(2)
#endif
  for (k = 1; k <= gp2-2; k++) {
#ifdef SPEC_USE_INNER_SIMD
    #pragma omp simd
#endif
    for (i = 1; i <= gp0-2; i++) {
      lhsY[2][k][j][i] = lhsY[2][k][j][i] + comz5;
      lhsY[3][k][j][i] = lhsY[3][k][j][i] - comz4;
      lhsY[4][k][j][i] = lhsY[4][k][j][i] + comz1;

      lhsY[1][k][j+1][i] = lhsY[1][k][j+1][i] - comz4;
      lhsY[2][k][j+1][i] = lhsY[2][k][j+1][i] + comz6;
      lhsY[3][k][j+1][i] = lhsY[3][k][j+1][i] - comz4;
      lhsY[4][k][j+1][i] = lhsY[4][k][j+1][i] + comz1;
    }
  }
#endif

  size_t kernel_y_solve_3_off[3] = { 1, 3, 1 };
  size_t kernel_y_solve_3_idx[3] = { gp0 - 2, gp1 - 6, gp2 - 2 };
  brisbane_kernel kernel_y_solve_3;
  brisbane_kernel_create("y_solve_3", &kernel_y_solve_3);
  brisbane_kernel_setmem(kernel_y_solve_3, 0, mem_lhsY, brisbane_rw);
  brisbane_kernel_setarg(kernel_y_solve_3, 1, sizeof(double), &comz1);
  brisbane_kernel_setarg(kernel_y_solve_3, 2, sizeof(double), &comz4);
  brisbane_kernel_setarg(kernel_y_solve_3, 3, sizeof(double), &comz6);

  brisbane_task task3;
  brisbane_task_create(&task3);
  brisbane_task_kernel(task3, kernel_y_solve_3, 3, kernel_y_solve_3_off, kernel_y_solve_3_idx);
  brisbane_task_submit(task3, brisbane_gpu, NULL, true);
#if 0
#ifdef SPEC_USE_INNER_SIMD
  #pragma omp target teams distribute parallel for private(i,j,k) collapse(2)
#else
  #pragma omp target teams distribute parallel for simd collapse(3)
#endif
  for (k = 1; k <= gp2-2; k++) {
    for (j = 3; j <= gp1-4; j++) {
#ifdef SPEC_USE_INNER_SIMD
    #pragma omp simd
#endif
      for (i = 1; i <= gp0-2; i++) {
        lhsY[0][k][j][i] = lhsY[0][k][j][i] + comz1;
        lhsY[1][k][j][i] = lhsY[1][k][j][i] - comz4;
        lhsY[2][k][j][i] = lhsY[2][k][j][i] + comz6;
        lhsY[3][k][j][i] = lhsY[3][k][j][i] - comz4;
        lhsY[4][k][j][i] = lhsY[4][k][j][i] + comz1;
      }
    }
  }
#endif

  j = gp1-3;

  size_t kernel_y_solve_4_off[2] = { 1, 1 };
  size_t kernel_y_solve_4_idx[2] = { gp0 - 2, gp2 - 2 };
  brisbane_kernel kernel_y_solve_4;
  brisbane_kernel_create("y_solve_4", &kernel_y_solve_4);
  brisbane_kernel_setmem(kernel_y_solve_4, 0, mem_lhsY, brisbane_rw);
  brisbane_kernel_setarg(kernel_y_solve_4, 1, sizeof(double), &comz1);
  brisbane_kernel_setarg(kernel_y_solve_4, 2, sizeof(double), &comz4);
  brisbane_kernel_setarg(kernel_y_solve_4, 3, sizeof(double), &comz5);
  brisbane_kernel_setarg(kernel_y_solve_4, 4, sizeof(double), &comz6);
  brisbane_kernel_setarg(kernel_y_solve_4, 5, sizeof(int), &j);

  brisbane_task task4;
  brisbane_task_create(&task4);
  brisbane_task_kernel(task4, kernel_y_solve_4, 2, kernel_y_solve_4_off, kernel_y_solve_4_idx);
  brisbane_task_submit(task4, brisbane_gpu, NULL, true);
#if 0
#ifdef SPEC_USE_INNER_SIMD
  #pragma omp target teams distribute parallel for private(i,k)
#else
  #pragma omp target teams distribute parallel for simd collapse(2)
#endif
  for (k = 1; k <= gp2-2; k++) {
#ifdef SPEC_USE_INNER_SIMD
    #pragma omp simd
#endif
    for (i = 1; i <= gp0-2; i++) {
      lhsY[0][k][j][i] = lhsY[0][k][j][i] + comz1;
      lhsY[1][k][j][i] = lhsY[1][k][j][i] - comz4;
      lhsY[2][k][j][i] = lhsY[2][k][j][i] + comz6;
      lhsY[3][k][j][i] = lhsY[3][k][j][i] - comz4;

      lhsY[0][k][j+1][i] = lhsY[0][k][j+1][i] + comz1;
      lhsY[1][k][j+1][i] = lhsY[1][k][j+1][i] - comz4;
      lhsY[2][k][j+1][i] = lhsY[2][k][j+1][i] + comz5;
    }
  }
#endif

    //---------------------------------------------------------------------
    // subsequently, for (the other two factors                    
    //---------------------------------------------------------------------
  size_t kernel_y_solve_5_off[3] = { 1, 1, 1 };
  size_t kernel_y_solve_5_idx[3] = { gp0 - 2, gp1 - 2, gp2 - 2 };
  brisbane_kernel kernel_y_solve_5;
  brisbane_kernel_create("y_solve_5", &kernel_y_solve_5);
  brisbane_kernel_setmem(kernel_y_solve_5, 0, mem_lhspY, brisbane_wr);
  brisbane_kernel_setmem(kernel_y_solve_5, 1, mem_lhsmY, brisbane_wr);
  brisbane_kernel_setmem(kernel_y_solve_5, 2, mem_lhsY, brisbane_rd);
  brisbane_kernel_setmem(kernel_y_solve_5, 3, mem_speed, brisbane_rd);
  brisbane_kernel_setarg(kernel_y_solve_5, 4, sizeof(double), &dtty2);

  brisbane_task task5;
  brisbane_task_create(&task5);
  brisbane_task_kernel(task5, kernel_y_solve_5, 3, kernel_y_solve_5_off, kernel_y_solve_5_idx);
  brisbane_task_submit(task5, brisbane_gpu, NULL, true);
#if 0
#ifdef SPEC_USE_INNER_SIMD
  #pragma omp target teams distribute parallel for private(i,j,k) collapse(2)
#else
  #pragma omp target teams distribute parallel for simd collapse(3)
#endif
  for (k = 1; k <= gp2-2; k++) {
    for (j = 1; j <= gp1-2; j++) {
#ifdef SPEC_USE_INNER_SIMD
      #pragma omp simd
#endif
      for (i = 1; i <= gp0-2; i++) {
        lhspY[0][k][j][i] = lhsY[0][k][j][i];
        lhspY[1][k][j][i] = lhsY[1][k][j][i] - dtty2 * speed[k][j-1][i];
        lhspY[2][k][j][i] = lhsY[2][k][j][i];
        lhspY[3][k][j][i] = lhsY[3][k][j][i] + dtty2 * speed[k][j+1][i];
        lhspY[4][k][j][i] = lhsY[4][k][j][i];
        lhsmY[0][k][j][i] = lhsY[0][k][j][i];
        lhsmY[1][k][j][i] = lhsY[1][k][j][i] + dtty2 * speed[k][j-1][i];
        lhsmY[2][k][j][i] = lhsY[2][k][j][i];
        lhsmY[3][k][j][i] = lhsY[3][k][j][i] - dtty2 * speed[k][j+1][i];
        lhsmY[4][k][j][i] = lhsY[4][k][j][i];
      }
    }
  }
#endif

    //---------------------------------------------------------------------
    // FORWARD ELIMINATION  
    //---------------------------------------------------------------------
  size_t kernel_y_solve_6_off[1] = { 1 };
  size_t kernel_y_solve_6_idx[1] = { gp2 - 2 };
  brisbane_kernel kernel_y_solve_6;
  brisbane_kernel_create("y_solve_6", &kernel_y_solve_6);
  brisbane_kernel_setmem(kernel_y_solve_6, 0, mem_lhsY, brisbane_rw);
  brisbane_kernel_setmem(kernel_y_solve_6, 1, mem_rhs, brisbane_rw);
  brisbane_kernel_setarg(kernel_y_solve_6, 2, sizeof(int), &gp0);
  brisbane_kernel_setarg(kernel_y_solve_6, 3, sizeof(int), &gp1);

  brisbane_task task6;
  brisbane_task_create(&task6);
  brisbane_task_kernel(task6, kernel_y_solve_6, 1, kernel_y_solve_6_off, kernel_y_solve_6_idx);
  brisbane_task_submit(task6, brisbane_gpu, NULL, true);
#if 0
  #pragma omp target teams distribute parallel for private(i,j,k,m,fac1,j1,j2) 
  for (k = 1; k <= gp2-2; k++) {
    for (j = 0; j <= gp1-3; j++) {
      j1 = j + 1;
      j2 = j + 2;
      for (i = 1; i <= gp0-2; i++) {
        fac1 = 1.0/lhsY[2][k][j][i];
        lhsY[3][k][j][i] = fac1*lhsY[3][k][j][i];
        lhsY[4][k][j][i] = fac1*lhsY[4][k][j][i];
        for (m = 0; m < 3; m++) {
          rhs[m][k][j][i] = fac1*rhs[m][k][j][i];
        }
        lhsY[2][k][j1][i] = lhsY[2][k][j1][i] - lhsY[1][k][j1][i]*lhsY[3][k][j][i];
        lhsY[3][k][j1][i] = lhsY[3][k][j1][i] - lhsY[1][k][j1][i]*lhsY[4][k][j][i];
        for (m = 0; m < 3; m++) {
          rhs[m][k][j1][i] = rhs[m][k][j1][i] - lhsY[1][k][j1][i]*rhs[m][k][j][i];
        }
        lhsY[1][k][j2][i] = lhsY[1][k][j2][i] - lhsY[0][k][j2][i]*lhsY[3][k][j][i];
        lhsY[2][k][j2][i] = lhsY[2][k][j2][i] - lhsY[0][k][j2][i]*lhsY[4][k][j][i];
        for (m = 0; m < 3; m++) {
          rhs[m][k][j2][i] = rhs[m][k][j2][i] - lhsY[0][k][j2][i]*rhs[m][k][j][i];
        }
      }
    }
  }
#endif

    //---------------------------------------------------------------------
    // The last two rows in this grid block are a bit different, 
    // since they for (not have two more rows available for the
    // elimination of off-diagonal entries
    //---------------------------------------------------------------------
  j  = gp1-2;
  j1 = gp1-1;

  size_t kernel_y_solve_7_off[2] = { 1, 1 };
  size_t kernel_y_solve_7_idx[2] = { gp0 - 2, gp2 - 2 };
  brisbane_kernel kernel_y_solve_7;
  brisbane_kernel_create("y_solve_7", &kernel_y_solve_7);
  brisbane_kernel_setmem(kernel_y_solve_7, 0, mem_lhsY, brisbane_rw);
  brisbane_kernel_setmem(kernel_y_solve_7, 1, mem_rhs, brisbane_rw);
  brisbane_kernel_setarg(kernel_y_solve_7, 2, sizeof(int), &j);
  brisbane_kernel_setarg(kernel_y_solve_7, 3, sizeof(int), &j1);

  brisbane_task task7;
  brisbane_task_create(&task7);
  brisbane_task_kernel(task7, kernel_y_solve_7, 2, kernel_y_solve_7_off, kernel_y_solve_7_idx);
  brisbane_task_submit(task7, brisbane_gpu, NULL, true);
#if 0
  #pragma omp target teams distribute parallel for private(i,k,m,fac1,fac2) collapse(2)
  for (k = 1; k <= gp2-2; k++) {
    for (i = 1; i <= gp0-2; i++) {
      fac1 = 1.0/lhsY[2][k][j][i];
      lhsY[3][k][j][i] = fac1*lhsY[3][k][j][i];
      lhsY[4][k][j][i] = fac1*lhsY[4][k][j][i];
      for (m = 0; m < 3; m++) {
        rhs[m][k][j][i] = fac1*rhs[m][k][j][i];
      }
      lhsY[2][k][j1][i] = lhsY[2][k][j1][i] - lhsY[1][k][j1][i]*lhsY[3][k][j][i];
      lhsY[3][k][j1][i] = lhsY[3][k][j1][i] - lhsY[1][k][j1][i]*lhsY[4][k][j][i];
      for (m = 0; m < 3; m++) {
        rhs[m][k][j1][i] = rhs[m][k][j1][i] - lhsY[1][k][j1][i]*rhs[m][k][j][i];
      }
      //---------------------------------------------------------------------
      // scale the last row immediately 
      //---------------------------------------------------------------------
      fac2 = 1.0/lhsY[2][k][j1][i];
      for (m = 0; m < 3; m++) {
        rhs[m][k][j1][i] = fac2*rhs[m][k][j1][i];
      }
    }
  }
#endif

    //---------------------------------------------------------------------
    // for (the u+c and the u-c factors                 
    //---------------------------------------------------------------------
  size_t kernel_y_solve_8_off[2] = { 1, 1 };
  size_t kernel_y_solve_8_idx[2] = { gp0 - 2, gp2 - 2 };
  brisbane_kernel kernel_y_solve_8;
  brisbane_kernel_create("y_solve_8", &kernel_y_solve_8);
  brisbane_kernel_setmem(kernel_y_solve_8, 0, mem_lhspY, brisbane_rw);
  brisbane_kernel_setmem(kernel_y_solve_8, 1, mem_lhsmY, brisbane_rw);
  brisbane_kernel_setmem(kernel_y_solve_8, 2, mem_rhs, brisbane_rw);
  brisbane_kernel_setarg(kernel_y_solve_8, 3, sizeof(int), &gp1);

  brisbane_task task8;
  brisbane_task_create(&task8);
  brisbane_task_kernel(task8, kernel_y_solve_8, 2, kernel_y_solve_8_off, kernel_y_solve_8_idx);
  brisbane_task_submit(task8, brisbane_gpu, NULL, true);
#if 0
#ifdef SPEC_USE_INNER_SIMD
  #pragma omp target teams distribute parallel for private(i,j,k,m)
#else
  #pragma omp target teams distribute parallel for simd private(j,m,fac1,j1,j2) collapse(2)
#endif
  for (k = 1; k <= gp2-2; k++) {
#ifdef SPEC_USE_INNER_SIMD
  #pragma omp simd private(fac1,j1,j2)
#endif
    for (i = 1; i <= gp0-2; i++) {
    for (j = 0; j <= gp1-3; j++) {
      j1 = j + 1;
      j2 = j + 2;
        m = 3;
        fac1 = 1.0/lhspY[2][k][j][i];
        lhspY[3][k][j][i]    = fac1*lhspY[3][k][j][i];
        lhspY[4][k][j][i]    = fac1*lhspY[4][k][j][i];
        rhs[m][k][j][i]  = fac1*rhs[m][k][j][i];
        lhspY[2][k][j1][i]   = lhspY[2][k][j1][i] - lhspY[1][k][j1][i]*lhspY[3][k][j][i];
        lhspY[3][k][j1][i]   = lhspY[3][k][j1][i] - lhspY[1][k][j1][i]*lhspY[4][k][j][i];
        rhs[m][k][j1][i] = rhs[m][k][j1][i] - lhspY[1][k][j1][i]*rhs[m][k][j][i];
        lhspY[1][k][j2][i]   = lhspY[1][k][j2][i] - lhspY[0][k][j2][i]*lhspY[3][k][j][i];
        lhspY[2][k][j2][i]   = lhspY[2][k][j2][i] - lhspY[0][k][j2][i]*lhspY[4][k][j][i];
        rhs[m][k][j2][i] = rhs[m][k][j2][i] - lhspY[0][k][j2][i]*rhs[m][k][j][i];

        m = 4;
        fac1 = 1.0/lhsmY[2][k][j][i];
        lhsmY[3][k][j][i]    = fac1*lhsmY[3][k][j][i];
        lhsmY[4][k][j][i]    = fac1*lhsmY[4][k][j][i];
        rhs[m][k][j][i]  = fac1*rhs[m][k][j][i];
        lhsmY[2][k][j1][i]   = lhsmY[2][k][j1][i] - lhsmY[1][k][j1][i]*lhsmY[3][k][j][i];
        lhsmY[3][k][j1][i]   = lhsmY[3][k][j1][i] - lhsmY[1][k][j1][i]*lhsmY[4][k][j][i];
        rhs[m][k][j1][i] = rhs[m][k][j1][i] - lhsmY[1][k][j1][i]*rhs[m][k][j][i];
        lhsmY[1][k][j2][i]   = lhsmY[1][k][j2][i] - lhsmY[0][k][j2][i]*lhsmY[3][k][j][i];
        lhsmY[2][k][j2][i]   = lhsmY[2][k][j2][i] - lhsmY[0][k][j2][i]*lhsmY[4][k][j][i];
        rhs[m][k][j2][i] = rhs[m][k][j2][i] - lhsmY[0][k][j2][i]*rhs[m][k][j][i];
      }
    }
  }
#endif

    //---------------------------------------------------------------------
    // And again the last two rows separately
    //---------------------------------------------------------------------
  j  = gp1-2;
  j1 = gp1-1;

  size_t kernel_y_solve_9_off[2] = { 1, 1 };
  size_t kernel_y_solve_9_idx[2] = { gp0 - 2, gp2 - 2 };
  brisbane_kernel kernel_y_solve_9;
  brisbane_kernel_create("y_solve_9", &kernel_y_solve_9);
  brisbane_kernel_setmem(kernel_y_solve_9, 0, mem_lhspY, brisbane_rw);
  brisbane_kernel_setmem(kernel_y_solve_9, 1, mem_lhsmY, brisbane_rw);
  brisbane_kernel_setmem(kernel_y_solve_9, 2, mem_rhs, brisbane_rw);
  brisbane_kernel_setarg(kernel_y_solve_9, 3, sizeof(int), &j);
  brisbane_kernel_setarg(kernel_y_solve_9, 4, sizeof(int), &j1);

  brisbane_task task9;
  brisbane_task_create(&task9);
  brisbane_task_kernel(task9, kernel_y_solve_9, 2, kernel_y_solve_9_off, kernel_y_solve_9_idx);
  brisbane_task_submit(task9, brisbane_gpu, NULL, true);
#if 0
#ifdef SPEC_USE_INNER_SIMD
  #pragma omp target teams distribute parallel for private(i,k,m,fac1)
#else
  #pragma omp target teams distribute parallel for simd private(m,fac1) collapse(2)
#endif
  for (k = 1; k <= gp2-2; k++) {
#ifdef SPEC_USE_INNER_SIMD
    #pragma omp simd private(m, fac1)
#endif
    for (i = 1; i <= gp0-2; i++) {
      m = 3;
      fac1 = 1.0/lhspY[2][k][j][i];
      lhspY[3][k][j][i]    = fac1*lhspY[3][k][j][i];
      lhspY[4][k][j][i]    = fac1*lhspY[4][k][j][i];
      rhs[m][k][j][i]  = fac1*rhs[m][k][j][i];
      lhspY[2][k][j1][i]   = lhspY[2][k][j1][i] - lhspY[1][k][j1][i]*lhspY[3][k][j][i];
      lhspY[3][k][j1][i]   = lhspY[3][k][j1][i] - lhspY[1][k][j1][i]*lhspY[4][k][j][i];
      rhs[m][k][j1][i] = rhs[m][k][j1][i] - lhspY[1][k][j1][i]*rhs[m][k][j][i];

      m = 4;
      fac1 = 1.0/lhsmY[2][k][j][i];
      lhsmY[3][k][j][i]    = fac1*lhsmY[3][k][j][i];
      lhsmY[4][k][j][i]    = fac1*lhsmY[4][k][j][i];
      rhs[m][k][j][i]  = fac1*rhs[m][k][j][i];
      lhsmY[2][k][j1][i]   = lhsmY[2][k][j1][i] - lhsmY[1][k][j1][i]*lhsmY[3][k][j][i];
      lhsmY[3][k][j1][i]   = lhsmY[3][k][j1][i] - lhsmY[1][k][j1][i]*lhsmY[4][k][j][i];
      rhs[m][k][j1][i] = rhs[m][k][j1][i] - lhsmY[1][k][j1][i]*rhs[m][k][j][i];

      //---------------------------------------------------------------------
      // Scale the last row immediately 
      //---------------------------------------------------------------------
      rhs[3][k][j1][i]   = rhs[3][k][j1][i]/lhspY[2][k][j1][i];
      rhs[4][k][j1][i]   = rhs[4][k][j1][i]/lhsmY[2][k][j1][i];
    }
  }
#endif

    //---------------------------------------------------------------------
    // BACKSUBSTITUTION 
    //---------------------------------------------------------------------
  j  = gp1-2;
  j1 = gp1-1;

  size_t kernel_y_solve_10_off[2] = { 1, 1 };
  size_t kernel_y_solve_10_idx[2] = { gp0 - 2, gp2 - 2 };
  brisbane_kernel kernel_y_solve_10;
  brisbane_kernel_create("y_solve_10", &kernel_y_solve_10);
  brisbane_kernel_setmem(kernel_y_solve_10, 0, mem_rhs, brisbane_rw);
  brisbane_kernel_setmem(kernel_y_solve_10, 1, mem_lhsY, brisbane_rw);
  brisbane_kernel_setmem(kernel_y_solve_10, 2, mem_lhspY, brisbane_rw);
  brisbane_kernel_setmem(kernel_y_solve_10, 3, mem_lhsmY, brisbane_rw);
  brisbane_kernel_setarg(kernel_y_solve_10, 4, sizeof(int), &j);
  brisbane_kernel_setarg(kernel_y_solve_10, 5, sizeof(int), &j1);

  brisbane_task task10;
  brisbane_task_create(&task10);
  brisbane_task_kernel(task10, kernel_y_solve_10, 2, kernel_y_solve_10_off, kernel_y_solve_10_idx);
  brisbane_task_submit(task10, brisbane_gpu, NULL, true);
#if 0
  #pragma omp target teams distribute parallel for private(i,k,m) collapse(2)
  for (k = 1; k <= gp2-2; k++) {
    for (i = 1; i <= gp0-2; i++) {
      for (m = 0; m < 3; m++) {
        rhs[m][k][j][i] = rhs[m][k][j][i] - lhsY[3][k][j][i]*rhs[m][k][j1][i];
      }

      rhs[3][k][j][i] = rhs[3][k][j][i] - lhspY[3][k][j][i]*rhs[3][k][j1][i];
      rhs[4][k][j][i] = rhs[4][k][j][i] - lhsmY[3][k][j][i]*rhs[4][k][j1][i];
    }
  }
#endif

    //---------------------------------------------------------------------
    // The first three factors
    //---------------------------------------------------------------------
  size_t kernel_y_solve_11_off[2] = { 1, 1 };
  size_t kernel_y_solve_11_idx[2] = { gp0 - 2, gp2 - 2 };
  brisbane_kernel kernel_y_solve_11;
  brisbane_kernel_create("y_solve_11", &kernel_y_solve_11);
  brisbane_kernel_setmem(kernel_y_solve_11, 0, mem_rhs, brisbane_rw);
  brisbane_kernel_setmem(kernel_y_solve_11, 1, mem_lhsY, brisbane_rw);
  brisbane_kernel_setmem(kernel_y_solve_11, 2, mem_lhspY, brisbane_rw);
  brisbane_kernel_setmem(kernel_y_solve_11, 3, mem_lhsmY, brisbane_rw);
  brisbane_kernel_setarg(kernel_y_solve_11, 4, sizeof(int), &gp1);

  brisbane_task task11;
  brisbane_task_create(&task11);
  brisbane_task_kernel(task11, kernel_y_solve_11, 2, kernel_y_solve_11_off, kernel_y_solve_11_idx);
  brisbane_task_submit(task11, brisbane_gpu, NULL, true);
#if 0
#ifdef SPEC_USE_INNER_SIMD
  #pragma omp target teams distribute parallel for private(i,j,k,m)
#else
  #pragma omp target teams distribute parallel for simd private(j,m,j1,j2) collapse(2)
#endif
  for (k = 1; k <= gp2-2; k++) {
#ifdef SPEC_USE_INNER_SIMD
  #pragma omp simd private(j1,j2)
#endif
    for (i = 1; i <= gp0-2; i++) {
    for (j = gp1-3; j >= 0; j--) {
      j1 = j + 1;
      j2 = j + 2;
        for (m = 0; m < 3; m++) {
          rhs[m][k][j][i] = rhs[m][k][j][i] - 
                            lhsY[3][k][j][i]*rhs[m][k][j1][i] -
                            lhsY[4][k][j][i]*rhs[m][k][j2][i];
        }

        //-------------------------------------------------------------------
        // And the remaining two
        //-------------------------------------------------------------------
        rhs[3][k][j][i] = rhs[3][k][j][i] - 
                          lhspY[3][k][j][i]*rhs[3][k][j1][i] -
                          lhspY[4][k][j][i]*rhs[3][k][j2][i];
        rhs[4][k][j][i] = rhs[4][k][j][i] - 
                          lhsmY[3][k][j][i]*rhs[4][k][j1][i] -
                          lhsmY[4][k][j][i]*rhs[4][k][j2][i];
      }
    }
  }
#endif

}/* end omp target data */
  brisbane_mem_release(mem_lhsY);
  brisbane_mem_release(mem_lhspY);
  brisbane_mem_release(mem_lhsmY);
  brisbane_mem_release(mem_rhoqY);
  
  pinvr();
}

void z_solve()
{
  int i, j, k, k1, k2, m;
  int gp21,gp22,gp23;
  double ru1, fac1, fac2;
  double lhsZ[5][ny2+1][IMAXP+1][IMAXP+1];
  double lhspZ[5][ny2+1][IMAXP+1][IMAXP+1];
  double lhsmZ[5][ny2+1][IMAXP+1][IMAXP+1];
  double rhosZ[ny2+1][IMAXP+1][PROBLEM_SIZE];

  int ni=nz2+1;
  gp21=grid_points[2]-1;
  gp22=grid_points[2]-2;
  gp23=grid_points[2]-3;

  brisbane_mem mem_lhsZ;
  brisbane_mem mem_lhspZ;
  brisbane_mem mem_lhsmZ;
  brisbane_mem mem_rhosZ;
  brisbane_mem_create(sizeof(double) * 5 * (ny2 + 1) * (IMAXP + 1) * (IMAXP + 1), &mem_lhsZ);
  brisbane_mem_create(sizeof(double) * 5 * (ny2 + 1) * (IMAXP + 1) * (IMAXP + 1), &mem_lhspZ);
  brisbane_mem_create(sizeof(double) * 5 * (ny2 + 1) * (IMAXP + 1) * (IMAXP + 1), &mem_lhsmZ);
  brisbane_mem_create(sizeof(double) * (ny2 + 1) * (IMAXP + 1) * (PROBLEM_SIZE), &mem_rhosZ);
  
#pragma omp target data map(alloc:lhsZ[:][:][:][:],lhspZ[:][:][:][:],lhsmZ[:][:][:][:],rhosZ[:][:][:]) //present(rho_i,ws,speed,rhs)
{
    size_t kernel_z_solve_0_off[2] = { 1, 1 };
    size_t kernel_z_solve_0_idx[2] = { nx2, ny2 };
    brisbane_kernel kernel_z_solve_0;
    brisbane_kernel_create("z_solve_0", &kernel_z_solve_0);
    brisbane_kernel_setmem(kernel_z_solve_0, 0, mem_lhsZ, brisbane_wr);
    brisbane_kernel_setmem(kernel_z_solve_0, 1, mem_lhspZ, brisbane_wr);
    brisbane_kernel_setmem(kernel_z_solve_0, 2, mem_lhsmZ, brisbane_wr);
    brisbane_kernel_setarg(kernel_z_solve_0, 3, sizeof(int), &ni);

    brisbane_task task0;
    brisbane_task_create(&task0);
    brisbane_task_kernel(task0, kernel_z_solve_0, 2, kernel_z_solve_0_off, kernel_z_solve_0_idx);
    brisbane_task_submit(task0, brisbane_gpu, NULL, true);
#if 0
    #pragma omp target teams distribute parallel for private(i,j,m) collapse(2)
    for (j = 1; j <= ny2; j++) {
      for (i = 1; i <= nx2; i++) {
      for (m = 0; m < 5; m++) {
        lhsZ[m][j][0][i] = 0.0;
        lhspZ[m][j][0][i] = 0.0;
        lhsmZ[m][j][0][i] = 0.0;
        lhsZ[m][j][ni][i] = 0.0;
        lhspZ[m][j][ni][i] = 0.0;
        lhsmZ[m][j][ni][i] = 0.0;
      }
      lhsZ[2][j][0][i] = 1.0;
      lhspZ[2][j][0][i] = 1.0;
      lhsmZ[2][j][0][i] = 1.0;
      lhsZ[2][j][ni][i] = 1.0;
      lhspZ[2][j][ni][i] = 1.0;
      lhsmZ[2][j][ni][i] = 1.0;
    }
  }
#endif

    //---------------------------------------------------------------------
    // Computes the left hand side for the three z-factors   
    //---------------------------------------------------------------------

    //---------------------------------------------------------------------
    // first fill the lhs for the u-eigenvalue                          
    //---------------------------------------------------------------------
    size_t kernel_z_solve_1_off[3] = { 0, 1, 1 };
    size_t kernel_z_solve_1_idx[3] = { nz2 + 2, nx2, ny2 };
    brisbane_kernel kernel_z_solve_1;
    brisbane_kernel_create("z_solve_1", &kernel_z_solve_1);
    brisbane_kernel_setmem(kernel_z_solve_1, 0, mem_rho_i, brisbane_rd);
    brisbane_kernel_setmem(kernel_z_solve_1, 1, mem_rhosZ, brisbane_wr);
    brisbane_kernel_setarg(kernel_z_solve_1, 2, sizeof(double), &dz1);
    brisbane_kernel_setarg(kernel_z_solve_1, 3, sizeof(double), &dz4);
    brisbane_kernel_setarg(kernel_z_solve_1, 4, sizeof(double), &dz5);
    brisbane_kernel_setarg(kernel_z_solve_1, 5, sizeof(double), &dzmax);
    brisbane_kernel_setarg(kernel_z_solve_1, 6, sizeof(double), &c1c5);
    brisbane_kernel_setarg(kernel_z_solve_1, 7, sizeof(double), &c3c4);
    brisbane_kernel_setarg(kernel_z_solve_1, 8, sizeof(double), &con43);

    brisbane_task task1;
    brisbane_task_create(&task1);
    brisbane_task_kernel(task1, kernel_z_solve_1, 3, kernel_z_solve_1_off, kernel_z_solve_1_idx);
    brisbane_task_submit(task1, brisbane_gpu, NULL, true);
#if 0
#ifdef SPEC_USE_INNER_SIMD
  #pragma omp target teams distribute parallel for private(i,j,k,ru1) collapse(2) 
#else
  #pragma omp target teams distribute parallel for simd private(ru1) collapse(3)
#endif
  for (j = 1; j <= ny2; j++) {
    for (i = 1; i <= nx2; i++) {
#ifdef SPEC_USE_INNER_SIMD
      #pragma omp simd
#endif
      for (k = 0; k <= nz2+1; k++) {
        ru1 = c3c4*rho_i[k][j][i];
        rhosZ[j][i][k] = max(max(dz4+con43*ru1, dz5+c1c5*ru1), max(dzmax+ru1, dz1));
      }
    }
  }
#endif

    size_t kernel_z_solve_2_off[3] = { 1, 1, 1 };
    size_t kernel_z_solve_2_idx[3] = { nz2, nx2, ny2 };
    brisbane_kernel kernel_z_solve_2;
    brisbane_kernel_create("z_solve_2", &kernel_z_solve_2);
    brisbane_kernel_setmem(kernel_z_solve_2, 0, mem_lhsZ, brisbane_rd);
    brisbane_kernel_setmem(kernel_z_solve_2, 1, mem_ws, brisbane_wr);
    brisbane_kernel_setmem(kernel_z_solve_2, 2, mem_rhosZ, brisbane_wr);
    brisbane_kernel_setarg(kernel_z_solve_2, 3, sizeof(double), &dttz1);
    brisbane_kernel_setarg(kernel_z_solve_2, 4, sizeof(double), &dttz2);
    brisbane_kernel_setarg(kernel_z_solve_2, 5, sizeof(double), &c2dttz1);

    brisbane_task task2;
    brisbane_task_create(&task2);
    brisbane_task_kernel(task2, kernel_z_solve_2, 3, kernel_z_solve_2_off, kernel_z_solve_2_idx);
    brisbane_task_submit(task2, brisbane_gpu, NULL, true);
#if 0
#ifdef SPEC_USE_INNER_SIMD
  #pragma omp target teams distribute parallel for private(i,j,k) collapse(2)
#else
  #pragma omp target teams distribute parallel for simd collapse(3)
#endif
  for (j = 1; j <= ny2; j++) {
    for (i = 1; i <= nx2; i++) {
#ifdef SPEC_USE_INNER_SIMD
    #pragma omp simd
#endif
    for (k = 1; k <= nz2; k++) {
        lhsZ[0][j][k][i] =  0.0;
        lhsZ[1][j][k][i] = -dttz2 * ws[k-1][j][i] - dttz1 * rhosZ[j][i][k-1];
        lhsZ[2][j][k][i] =  1.0 + c2dttz1 * rhosZ[j][i][k];
        lhsZ[3][j][k][i] =  dttz2 * ws[k+1][j][i] - dttz1 * rhosZ[j][i][k+1];
        lhsZ[4][j][k][i] =  0.0;
      }
    }
  }
#endif

    //---------------------------------------------------------------------
    // add fourth order dissipation                                  
    //---------------------------------------------------------------------
    size_t kernel_z_solve_3_off[2] = { 1, 1 };
    size_t kernel_z_solve_3_idx[2] = { nx2, ny2 };
    brisbane_kernel kernel_z_solve_3;
    brisbane_kernel_create("z_solve_3", &kernel_z_solve_3);
    brisbane_kernel_setmem(kernel_z_solve_3, 0, mem_lhsZ, brisbane_rw);
    brisbane_kernel_setarg(kernel_z_solve_3, 1, sizeof(double), &comz1);
    brisbane_kernel_setarg(kernel_z_solve_3, 2, sizeof(double), &comz4);
    brisbane_kernel_setarg(kernel_z_solve_3, 3, sizeof(double), &comz5);
    brisbane_kernel_setarg(kernel_z_solve_3, 4, sizeof(double), &comz6);

    brisbane_task task3;
    brisbane_task_create(&task3);
    brisbane_task_kernel(task3, kernel_z_solve_3, 2, kernel_z_solve_3_off, kernel_z_solve_3_idx);
    brisbane_task_submit(task3, brisbane_gpu, NULL, true);
#if 0
#ifdef SPEC_USE_INNER_SIMD
  #pragma omp target teams distribute parallel for private(i,j,k)
#else
  #pragma omp target teams distribute parallel for simd private(k) collapse(2)
#endif
  for (j = 1; j <= ny2; j++) {
#ifdef SPEC_USE_INNER_SIMD
    #pragma omp simd
#endif
    for (i = 1; i <= nx2; i++) {
      k = 1;
      lhsZ[2][j][k][i] = lhsZ[2][j][k][i] + comz5;
      lhsZ[3][j][k][i] = lhsZ[3][j][k][i] - comz4;
      lhsZ[4][j][k][i] = lhsZ[4][j][k][i] + comz1;

      k = 2;
      lhsZ[1][j][k][i] = lhsZ[1][j][k][i] - comz4;
      lhsZ[2][j][k][i] = lhsZ[2][j][k][i] + comz6;
      lhsZ[3][j][k][i] = lhsZ[3][j][k][i] - comz4;
      lhsZ[4][j][k][i] = lhsZ[4][j][k][i] + comz1;
    }
  }
#endif

    size_t kernel_z_solve_4_off[3] = { 1, 3, 1 };
    size_t kernel_z_solve_4_idx[3] = { nx2, nz2 - 4, ny2 };
    brisbane_kernel kernel_z_solve_4;
    brisbane_kernel_create("z_solve_4", &kernel_z_solve_4);
    brisbane_kernel_setmem(kernel_z_solve_4, 0, mem_lhsZ, brisbane_rw);
    brisbane_kernel_setarg(kernel_z_solve_4, 1, sizeof(double), &comz1);
    brisbane_kernel_setarg(kernel_z_solve_4, 2, sizeof(double), &comz4);
    brisbane_kernel_setarg(kernel_z_solve_4, 3, sizeof(double), &comz6);

    brisbane_task task4;
    brisbane_task_create(&task4);
    brisbane_task_kernel(task4, kernel_z_solve_4, 3, kernel_z_solve_4_off, kernel_z_solve_4_idx);
    brisbane_task_submit(task4, brisbane_gpu, NULL, true);
#if 0
#ifdef SPEC_USE_INNER_SIMD
 #pragma omp target teams distribute parallel for private(i,j,k) collapse(2)
#else
 #pragma omp target teams distribute parallel for simd collapse(3)
#endif
  for (j = 1; j <= ny2; j++) {
    for (k = 3; k <= nz2-2; k++) {
#ifdef SPEC_USE_INNER_SIMD
      #pragma omp simd
#endif
      for (i = 1; i <= nx2; i++) {
        lhsZ[0][j][k][i] = lhsZ[0][j][k][i] + comz1;
        lhsZ[1][j][k][i] = lhsZ[1][j][k][i] - comz4;
        lhsZ[2][j][k][i] = lhsZ[2][j][k][i] + comz6;
        lhsZ[3][j][k][i] = lhsZ[3][j][k][i] - comz4;
        lhsZ[4][j][k][i] = lhsZ[4][j][k][i] + comz1;
      }
    }
  }
#endif

    size_t kernel_z_solve_5_off[2] = { 1, 1 };
    size_t kernel_z_solve_5_idx[2] = { nx2, ny2 };
    brisbane_kernel kernel_z_solve_5;
    brisbane_kernel_create("z_solve_5", &kernel_z_solve_5);
    brisbane_kernel_setmem(kernel_z_solve_5, 0, mem_lhsZ, brisbane_rw);
    brisbane_kernel_setarg(kernel_z_solve_5, 1, sizeof(double), &comz1);
    brisbane_kernel_setarg(kernel_z_solve_5, 2, sizeof(double), &comz4);
    brisbane_kernel_setarg(kernel_z_solve_5, 3, sizeof(double), &comz5);
    brisbane_kernel_setarg(kernel_z_solve_5, 4, sizeof(double), &comz6);

    brisbane_task task5;
    brisbane_task_create(&task5);
    brisbane_task_kernel(task5, kernel_z_solve_5, 3, kernel_z_solve_5_off, kernel_z_solve_5_idx);
    brisbane_task_submit(task5, brisbane_gpu, NULL, true);
#if 0
#ifdef SPEC_USE_INNER_SIMD
 #pragma omp target teams distribute parallel for private(i,j,k)
#else
 #pragma omp target teams distribute parallel for simd private(k) collapse(2)
#endif
  for (j = 1; j <= ny2; j++) {
#ifdef SPEC_USE_INNER_SIMD
    #pragma omp simd
#endif
    for (i = 1; i <= nx2; i++) {
      k = nz2-1;
      lhsZ[0][j][k][i] = lhsZ[0][j][k][i] + comz1;
      lhsZ[1][j][k][i] = lhsZ[1][j][k][i] - comz4;
      lhsZ[2][j][k][i] = lhsZ[2][j][k][i] + comz6;
      lhsZ[3][j][k][i] = lhsZ[3][j][k][i] - comz4;

      k = nz2;
      lhsZ[0][j][k][i] = lhsZ[0][j][k][i] + comz1;
      lhsZ[1][j][k][i] = lhsZ[1][j][k][i] - comz4;
      lhsZ[2][j][k][i] = lhsZ[2][j][k][i] + comz5;
    }
  }
#endif

    //---------------------------------------------------------------------
    // subsequently, fill the other factors (u+c), (u-c) 
    //---------------------------------------------------------------------
    size_t kernel_z_solve_6_off[3] = { 1, 1, 1 };
    size_t kernel_z_solve_6_idx[3] = { nx2, nz2, ny2 };
    brisbane_kernel kernel_z_solve_6;
    brisbane_kernel_create("z_solve_6", &kernel_z_solve_6);
    brisbane_kernel_setmem(kernel_z_solve_6, 0, mem_lhsZ, brisbane_rd);
    brisbane_kernel_setmem(kernel_z_solve_6, 1, mem_lhspZ, brisbane_wr);
    brisbane_kernel_setmem(kernel_z_solve_6, 2, mem_lhsmZ, brisbane_wr);
    brisbane_kernel_setmem(kernel_z_solve_6, 3, mem_speed, brisbane_rd);
    brisbane_kernel_setarg(kernel_z_solve_6, 4, sizeof(double), &dttz2);

    brisbane_task task6;
    brisbane_task_create(&task6);
    brisbane_task_kernel(task6, kernel_z_solve_6, 3, kernel_z_solve_6_off, kernel_z_solve_6_idx);
    brisbane_task_submit(task6, brisbane_gpu, NULL, true);
#if 0
#ifdef SPEC_USE_INNER_SIMD
 #pragma omp target teams distribute parallel for private(i,j,k) collapse(2)
#else
 #pragma omp target teams distribute parallel for simd collapse(3)
#endif
  for (j = 1; j <= ny2; j++) {
    for (k = 1; k <= nz2; k++) {
#ifdef SPEC_USE_INNER_SIMD
      #pragma omp simd
#endif
      for (i = 1; i <= nx2; i++) {
        lhspZ[0][j][k][i] = lhsZ[0][j][k][i];
        lhspZ[1][j][k][i] = lhsZ[1][j][k][i] - dttz2 * speed[k-1][j][i];
        lhspZ[2][j][k][i] = lhsZ[2][j][k][i];
        lhspZ[3][j][k][i] = lhsZ[3][j][k][i] + dttz2 * speed[k+1][j][i];
        lhspZ[4][j][k][i] = lhsZ[4][j][k][i];
        lhsmZ[0][j][k][i] = lhsZ[0][j][k][i];
        lhsmZ[1][j][k][i] = lhsZ[1][j][k][i] + dttz2 * speed[k-1][j][i];
        lhsmZ[2][j][k][i] = lhsZ[2][j][k][i];
        lhsmZ[3][j][k][i] = lhsZ[3][j][k][i] - dttz2 * speed[k+1][j][i];
        lhsmZ[4][j][k][i] = lhsZ[4][j][k][i];
      }
    }
  }
#endif


    //---------------------------------------------------------------------
    // FORWARD ELIMINATION  
    //---------------------------------------------------------------------
    size_t kernel_z_solve_7_off[2] = { 1, 1 };
    size_t kernel_z_solve_7_idx[2] = { nx2, ny2 };
    brisbane_kernel kernel_z_solve_7;
    brisbane_kernel_create("z_solve_7", &kernel_z_solve_7);
    brisbane_kernel_setmem(kernel_z_solve_7, 0, mem_lhsZ, brisbane_rw);
    brisbane_kernel_setmem(kernel_z_solve_7, 1, mem_rhs, brisbane_rw);
    brisbane_kernel_setarg(kernel_z_solve_7, 2, sizeof(int), &gp23);

    brisbane_task task7;
    brisbane_task_create(&task7);
    brisbane_task_kernel(task7, kernel_z_solve_7, 2, kernel_z_solve_7_off, kernel_z_solve_7_idx);
    brisbane_task_submit(task7, brisbane_gpu, NULL, true);
#if 0
#ifdef SPEC_USE_INNER_SIMD
 #pragma omp target teams distribute parallel for private(i,j,k,m)
#else
 #pragma omp target teams distribute parallel for simd private(k,m,fac1,k1,k2) collapse(2)
#endif
  for (j = 1; j <= ny2; j++) {
#ifdef SPEC_USE_INNER_SIMD
  #pragma omp simd private(fac1,k1,k2)
#endif
    for (i = 1; i <= nx2; i++) {
    for (k = 0; k <= gp23; k++) {
      k1 = k + 1;
      k2 = k + 2;
        fac1 = 1.0/lhsZ[2][j][k][i];
        lhsZ[3][j][k][i] = fac1*lhsZ[3][j][k][i];
        lhsZ[4][j][k][i] = fac1*lhsZ[4][j][k][i];
        for (m = 0; m < 3; m++) {
          rhs[m][k][j][i] = fac1*rhs[m][k][j][i];
        }
        lhsZ[2][j][k1][i] = lhsZ[2][j][k1][i] - lhsZ[1][j][k1][i]*lhsZ[3][j][k][i];
        lhsZ[3][j][k1][i] = lhsZ[3][j][k1][i] - lhsZ[1][j][k1][i]*lhsZ[4][j][k][i];
        for (m = 0; m < 3; m++) {
          rhs[m][k1][j][i] = rhs[m][k1][j][i] - lhsZ[1][j][k1][i]*rhs[m][k][j][i];
        }
        lhsZ[1][j][k2][i] = lhsZ[1][j][k2][i] - lhsZ[0][j][k2][i]*lhsZ[3][j][k][i];
        lhsZ[2][j][k2][i] = lhsZ[2][j][k2][i] - lhsZ[0][j][k2][i]*lhsZ[4][j][k][i];
        for (m = 0; m < 3; m++) {
          rhs[m][k2][j][i] = rhs[m][k2][j][i] - lhsZ[0][j][k2][i]*rhs[m][k][j][i];
        }
      }
    }
  }
#endif

    //---------------------------------------------------------------------
    // The last two rows in this grid block are a bit different, 
    // since they for (not have two more rows available for the
    // elimination of off-diagonal entries
    //---------------------------------------------------------------------
    k  = gp22;
    k1 = gp21;

    size_t kernel_z_solve_8_off[2] = { 1, 1 };
    size_t kernel_z_solve_8_idx[2] = { nx2, ny2 };
    brisbane_kernel kernel_z_solve_8;
    brisbane_kernel_create("z_solve_8", &kernel_z_solve_8);
    brisbane_kernel_setmem(kernel_z_solve_8, 0, mem_lhsZ, brisbane_rw);
    brisbane_kernel_setmem(kernel_z_solve_8, 1, mem_rhs, brisbane_rw);
    brisbane_kernel_setarg(kernel_z_solve_8, 2, sizeof(int), &k);
    brisbane_kernel_setarg(kernel_z_solve_8, 3, sizeof(int), &k1);

    brisbane_task task8;
    brisbane_task_create(&task8);
    brisbane_task_kernel(task8, kernel_z_solve_8, 2, kernel_z_solve_8_off, kernel_z_solve_8_idx);
    brisbane_task_submit(task8, brisbane_gpu, NULL, true);
#if 0
  #pragma omp target teams distribute parallel for private(m,fac1,fac2) collapse(2)
  for (j = 1; j <= ny2; j++) {
    for (i = 1; i <= nx2; i++) {
      fac1 = 1.0/lhsZ[2][j][k][i];
      lhsZ[3][j][k][i] = fac1*lhsZ[3][j][k][i];
      lhsZ[4][j][k][i] = fac1*lhsZ[4][j][k][i];
      for (m = 0; m < 3; m++) {
        rhs[m][k][j][i] = fac1*rhs[m][k][j][i];
      }
      lhsZ[2][j][k1][i] = lhsZ[2][j][k1][i] - lhsZ[1][j][k1][i]*lhsZ[3][j][k][i];
      lhsZ[3][j][k1][i] = lhsZ[3][j][k1][i] - lhsZ[1][j][k1][i]*lhsZ[4][j][k][i];
      for (m = 0; m < 3; m++) {
        rhs[m][k1][j][i] = rhs[m][k1][j][i] - lhsZ[1][j][k1][i]*rhs[m][k][j][i];
      }

      //---------------------------------------------------------------------
      // scale the last row immediately
      //---------------------------------------------------------------------
      fac2 = 1.0/lhsZ[2][j][k1][i];
      for (m = 0; m < 3; m++) {
        rhs[m][k1][j][i] = fac2*rhs[m][k1][j][i];
      }
    }
  }
#endif

    //---------------------------------------------------------------------
    // for (the u+c and the u-c factors               
    //---------------------------------------------------------------------
    size_t kernel_z_solve_9_off[2] = { 1, 1 };
    size_t kernel_z_solve_9_idx[2] = { nx2, ny2 };
    brisbane_kernel kernel_z_solve_9;
    brisbane_kernel_create("z_solve_9", &kernel_z_solve_9);
    brisbane_kernel_setmem(kernel_z_solve_9, 0, mem_lhspZ, brisbane_rw);
    brisbane_kernel_setmem(kernel_z_solve_9, 1, mem_lhsmZ, brisbane_rw);
    brisbane_kernel_setmem(kernel_z_solve_9, 2, mem_rhs, brisbane_rw);
    brisbane_kernel_setarg(kernel_z_solve_9, 3, sizeof(int), &gp23);

    brisbane_task task9;
    brisbane_task_create(&task9);
    brisbane_task_kernel(task9, kernel_z_solve_9, 2, kernel_z_solve_9_off, kernel_z_solve_9_idx);
    brisbane_task_submit(task9, brisbane_gpu, NULL, true);
#if 0
#ifdef SPEC_USE_INNER_SIMD
  #pragma omp target teams distribute parallel for private(i,j,k,m)
#else
  #pragma omp target teams distribute parallel for simd private(k,m,fac1,k1,k2) collapse(2)
#endif
  for (j = 1; j <= ny2; j++) {
#ifdef SPEC_USE_INNER_SIMD
  #pragma omp simd private(fac1,k1,k2)
#endif
   for (i = 1; i <= nx2; i++) {
    for (k = 0; k <= gp23; k++) {
      k1 = k + 1;
      k2 = k + 2;
        m = 3;
        fac1 = 1.0/lhspZ[2][j][k][i];
        lhspZ[3][j][k][i]    = fac1*lhspZ[3][j][k][i];
        lhspZ[4][j][k][i]    = fac1*lhspZ[4][j][k][i];
        rhs[m][k][j][i]  = fac1*rhs[m][k][j][i];
        lhspZ[2][j][k1][i]   = lhspZ[2][j][k1][i] - lhspZ[1][j][k1][i]*lhspZ[3][j][k][i];
        lhspZ[3][j][k1][i]   = lhspZ[3][j][k1][i] - lhspZ[1][j][k1][i]*lhspZ[4][j][k][i];
        rhs[m][k1][j][i] = rhs[m][k1][j][i] - lhspZ[1][j][k1][i]*rhs[m][k][j][i];
        lhspZ[1][j][k2][i]   = lhspZ[1][j][k2][i] - lhspZ[0][j][k2][i]*lhspZ[3][j][k][i];
        lhspZ[2][j][k2][i]   = lhspZ[2][j][k2][i] - lhspZ[0][j][k2][i]*lhspZ[4][j][k][i];
        rhs[m][k2][j][i] = rhs[m][k2][j][i] - lhspZ[0][j][k2][i]*rhs[m][k][j][i];

        m = 4;
        fac1 = 1.0/lhsmZ[2][j][k][i];
        lhsmZ[3][j][k][i]    = fac1*lhsmZ[3][j][k][i];
        lhsmZ[4][j][k][i]    = fac1*lhsmZ[4][j][k][i];
        rhs[m][k][j][i]  = fac1*rhs[m][k][j][i];
        lhsmZ[2][j][k1][i]   = lhsmZ[2][j][k1][i] - lhsmZ[1][j][k1][i]*lhsmZ[3][j][k][i];
        lhsmZ[3][j][k1][i]   = lhsmZ[3][j][k1][i] - lhsmZ[1][j][k1][i]*lhsmZ[4][j][k][i];
        rhs[m][k1][j][i] = rhs[m][k1][j][i] - lhsmZ[1][j][k1][i]*rhs[m][k][j][i];
        lhsmZ[1][j][k2][i]   = lhsmZ[1][j][k2][i] - lhsmZ[0][j][k2][i]*lhsmZ[3][j][k][i];
        lhsmZ[2][j][k2][i]   = lhsmZ[2][j][k2][i] - lhsmZ[0][j][k2][i]*lhsmZ[4][j][k][i];
        rhs[m][k2][j][i] = rhs[m][k2][j][i] - lhsmZ[0][j][k2][i]*rhs[m][k][j][i];
      }
     }
    }
#endif

    //---------------------------------------------------------------------
    // And again the last two rows separately
    //---------------------------------------------------------------------
    k  = gp22;
    k1 = gp21;

    size_t kernel_z_solve_10_off[2] = { 1, 1 };
    size_t kernel_z_solve_10_idx[2] = { nx2, ny2 };
    brisbane_kernel kernel_z_solve_10;
    brisbane_kernel_create("z_solve_10", &kernel_z_solve_10);
    brisbane_kernel_setmem(kernel_z_solve_10, 0, mem_lhspZ, brisbane_rw);
    brisbane_kernel_setmem(kernel_z_solve_10, 1, mem_lhsmZ, brisbane_rw);
    brisbane_kernel_setmem(kernel_z_solve_10, 2, mem_rhs, brisbane_rw);
    brisbane_kernel_setarg(kernel_z_solve_10, 3, sizeof(int), &k);
    brisbane_kernel_setarg(kernel_z_solve_10, 4, sizeof(int), &k1);

    brisbane_task task10;
    brisbane_task_create(&task10);
    brisbane_task_kernel(task10, kernel_z_solve_10, 2, kernel_z_solve_10_off, kernel_z_solve_10_idx);
    brisbane_task_submit(task10, brisbane_gpu, NULL, true);
#if 0
#ifdef SPEC_USE_INNER_SIMD
  #pragma omp target teams distribute parallel for private(i,j,m,fac1)
#else
  #pragma omp target teams distribute parallel for simd private(m,fac1) collapse(2)
#endif
  for (j = 1; j <= ny2; j++) {
#ifdef SPEC_USE_INNER_SIMD
    #pragma omp simd private(m, fac1)
#endif
    for (i = 1; i <= nx2; i++) {
      m = 3;
      fac1 = 1.0/lhspZ[2][j][k][i];
      lhspZ[3][j][k][i]    = fac1*lhspZ[3][j][k][i];
      lhspZ[4][j][k][i]    = fac1*lhspZ[4][j][k][i];
      rhs[m][k][j][i]  = fac1*rhs[m][k][j][i];
      lhspZ[2][j][k1][i]   = lhspZ[2][j][k1][i] - lhspZ[1][j][k1][i]*lhspZ[3][j][k][i];
      lhspZ[3][j][k1][i]   = lhspZ[3][j][k1][i] - lhspZ[1][j][k1][i]*lhspZ[4][j][k][i];
      rhs[m][k1][j][i] = rhs[m][k1][j][i] - lhspZ[1][j][k1][i]*rhs[m][k][j][i];

      m = 4;
      fac1 = 1.0/lhsmZ[2][j][k][i];
      lhsmZ[3][j][k][i]    = fac1*lhsmZ[3][j][k][i];
      lhsmZ[4][j][k][i]    = fac1*lhsmZ[4][j][k][i];
      rhs[m][k][j][i]  = fac1*rhs[m][k][j][i];
      lhsmZ[2][j][k1][i]   = lhsmZ[2][j][k1][i] - lhsmZ[1][j][k1][i]*lhsmZ[3][j][k][i];
      lhsmZ[3][j][k1][i]   = lhsmZ[3][j][k1][i] - lhsmZ[1][j][k1][i]*lhsmZ[4][j][k][i];
      rhs[m][k1][j][i] = rhs[m][k1][j][i] - lhsmZ[1][j][k1][i]*rhs[m][k][j][i];

      //---------------------------------------------------------------------
      // Scale the last row immediately (some of this is overkill
      // if this is the last cell)
      //---------------------------------------------------------------------
      rhs[3][k1][j][i] = rhs[3][k1][j][i]/lhspZ[2][j][k1][i];
      rhs[4][k1][j][i] = rhs[4][k1][j][i]/lhsmZ[2][j][k1][i];
    }
  }
#endif


    //---------------------------------------------------------------------
    // BACKSUBSTITUTION 
    //---------------------------------------------------------------------
    k  = gp22;
    k1 = gp21;

    size_t kernel_z_solve_11_off[2] = { 1, 1 };
    size_t kernel_z_solve_11_idx[2] = { nx2, ny2 };
    brisbane_kernel kernel_z_solve_11;
    brisbane_kernel_create("z_solve_11", &kernel_z_solve_11);
    brisbane_kernel_setmem(kernel_z_solve_11, 0, mem_lhsZ, brisbane_rd);
    brisbane_kernel_setmem(kernel_z_solve_11, 1, mem_lhspZ, brisbane_rd);
    brisbane_kernel_setmem(kernel_z_solve_11, 2, mem_lhsmZ, brisbane_rd);
    brisbane_kernel_setmem(kernel_z_solve_11, 3, mem_rhs, brisbane_rw);
    brisbane_kernel_setarg(kernel_z_solve_11, 4, sizeof(int), &k);
    brisbane_kernel_setarg(kernel_z_solve_11, 5, sizeof(int), &k1);

    brisbane_task task11;
    brisbane_task_create(&task11);
    brisbane_task_kernel(task11, kernel_z_solve_11, 2, kernel_z_solve_11_off, kernel_z_solve_11_idx);
    brisbane_task_submit(task11, brisbane_gpu, NULL, true);
#if 0
  #pragma omp target teams distribute parallel for private(i,j,m) collapse(2)
  for (j = 1; j <= ny2; j++) {
    for (i = 1; i <= nx2; i++) {
      for (m = 0; m < 3; m++) {
        rhs[m][k][j][i] = rhs[m][k][j][i] - lhsZ[3][j][k][i]*rhs[m][k1][j][i];
      }

      rhs[3][k][j][i] = rhs[3][k][j][i] - lhspZ[3][j][k][i]*rhs[3][k1][j][i];
      rhs[4][k][j][i] = rhs[4][k][j][i] - lhsmZ[3][j][k][i]*rhs[4][k1][j][i];
    }
  }
#endif

    //---------------------------------------------------------------------
    // Whether or not this is the last processor, we always have
    // to complete the back-substitution 
    //---------------------------------------------------------------------

    //---------------------------------------------------------------------
    // The first three factors
    //---------------------------------------------------------------------
    size_t kernel_z_solve_12_off[2] = { 1, 1 };
    size_t kernel_z_solve_12_idx[2] = { nx2, ny2 };
    brisbane_kernel kernel_z_solve_12;
    brisbane_kernel_create("z_solve_12", &kernel_z_solve_12);
    brisbane_kernel_setmem(kernel_z_solve_12, 0, mem_lhsZ, brisbane_rd);
    brisbane_kernel_setmem(kernel_z_solve_12, 1, mem_lhspZ, brisbane_rd);
    brisbane_kernel_setmem(kernel_z_solve_12, 2, mem_lhsmZ, brisbane_rd);
    brisbane_kernel_setmem(kernel_z_solve_12, 3, mem_rhs, brisbane_rw);
    brisbane_kernel_setarg(kernel_z_solve_12, 4, sizeof(int), &gp23);

    brisbane_task task12;
    brisbane_task_create(&task12);
    brisbane_task_kernel(task12, kernel_z_solve_12, 2, kernel_z_solve_12_off, kernel_z_solve_12_idx);
    brisbane_task_submit(task12, brisbane_gpu, NULL, true);
#if 0
#ifdef SPEC_USE_INNER_SIMD
  #pragma omp target teams distribute parallel for private(i,j,k,m)
#else
  #pragma omp target teams distribute parallel for simd private(k,m,k1,k2) collapse(2)
#endif
  for (j = 1; j <= ny2; j++) {
#ifdef SPEC_USE_INNER_SIMD
  #pragma omp simd private(k1,k2)
#endif
    for (i = 1; i <= nx2; i++) {
    for (k = gp23; k >= 0; k--) {
      k1 = k + 1;
      k2 = k + 2;
        for (m = 0; m < 3; m++) {
          rhs[m][k][j][i] = rhs[m][k][j][i] - 
                            lhsZ[3][j][k][i]*rhs[m][k1][j][i] -
                            lhsZ[4][j][k][i]*rhs[m][k2][j][i];
        }

        //-------------------------------------------------------------------
        // And the remaining two
        //-------------------------------------------------------------------
        rhs[3][k][j][i] = rhs[3][k][j][i] - 
                          lhspZ[3][j][k][i]*rhs[3][k1][j][i] -
                          lhspZ[4][j][k][i]*rhs[3][k2][j][i];
        rhs[4][k][j][i] = rhs[4][k][j][i] - 
                          lhsmZ[3][j][k][i]*rhs[4][k1][j][i] -
                          lhsmZ[4][j][k][i]*rhs[4][k2][j][i];
      }
    }
  }
#endif

}/* end omp target data */
  
  brisbane_mem_release(mem_lhsZ);
  brisbane_mem_release(mem_lhspZ);
  brisbane_mem_release(mem_lhsmZ);
  brisbane_mem_release(mem_rhosZ);

  tzetar();
}
