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

#include <math.h>
#include "header-brisbane.h"

void compute_rhs()
{
  int i, j, k, m;
  double aux, rho_inv, uijk, up1, um1, vijk, vp1, vm1, wijk, wp1, wm1;
  int gp0, gp1, gp2;

  gp0 = grid_points[0];
  gp1 = grid_points[1];
  gp2 = grid_points[2];

  //---------------------------------------------------------------------
  // compute the reciprocal of density, and the kinetic energy, 
  // and the speed of sound. 
  //---------------------------------------------------------------------
//#pragma omp target //present(rho_i,u,qs,square,speed,rhs,forcing,us,vs,ws)
//{
/*get the value of rho_i,qs,square,us,vs,ws,speed*/
  size_t kernel_compute_rhs_0_off[3] = { 0, 0, 0 };
  size_t kernel_compute_rhs_0_idx[3] = { gp0, gp1, gp2 };
  brisbane_kernel kernel_compute_rhs_0;
  brisbane_kernel_create("compute_rhs_0", &kernel_compute_rhs_0);
  brisbane_kernel_setmem(kernel_compute_rhs_0, 0, mem_u, brisbane_r);
  brisbane_kernel_setmem(kernel_compute_rhs_0, 1, mem_rho_i, brisbane_w);
  brisbane_kernel_setmem(kernel_compute_rhs_0, 2, mem_us, brisbane_w);
  brisbane_kernel_setmem(kernel_compute_rhs_0, 3, mem_vs, brisbane_w);
  brisbane_kernel_setmem(kernel_compute_rhs_0, 4, mem_ws, brisbane_w);
  brisbane_kernel_setmem(kernel_compute_rhs_0, 5, mem_square, brisbane_rw);
  brisbane_kernel_setmem(kernel_compute_rhs_0, 6, mem_qs, brisbane_w);
  brisbane_kernel_setmem(kernel_compute_rhs_0, 7, mem_speed, brisbane_w);
  brisbane_kernel_setarg(kernel_compute_rhs_0, 8, sizeof(double), &c1c2);

  brisbane_task task0;
  brisbane_task_create(&task0);
  brisbane_task_h2d_full(task0, mem_u, u);
  brisbane_task_h2d_full(task0, mem_rho_i, rho_i);
  brisbane_task_h2d_full(task0, mem_us, us);
  brisbane_task_h2d_full(task0, mem_vs, vs);
  brisbane_task_h2d_full(task0, mem_ws, ws);
  brisbane_task_h2d_full(task0, mem_square, square);
  brisbane_task_h2d_full(task0, mem_qs, qs);
  brisbane_task_h2d_full(task0, mem_speed, speed);
  brisbane_task_kernel(task0, kernel_compute_rhs_0, 3, kernel_compute_rhs_0_off, kernel_compute_rhs_0_idx);
  brisbane_task_submit(task0, brisbane_gpu, NULL, true);
#if 0
#ifdef SPEC_USE_INNER_SIMD
  #pragma omp target teams distribute parallel for private(rho_inv,aux,i,j,k) collapse(2) 
#else
  #pragma omp target teams distribute parallel for simd private(rho_inv,aux) collapse(3)
#endif
  for (k = 0; k <= gp2-1; k++) {
    for (j = 0; j <= gp1-1; j++) {
#ifdef SPEC_USE_INNER_SIMD
      #pragma omp simd private(rho_inv,aux)
#endif
      for (i = 0; i <= gp0-1; i++) {
        rho_inv = 1.0/u[0][k][j][i];
        rho_i[k][j][i] = rho_inv;
        us[k][j][i] = u[1][k][j][i] * rho_inv;
        vs[k][j][i] = u[2][k][j][i] * rho_inv;
        ws[k][j][i] = u[3][k][j][i] * rho_inv;
        square[k][j][i] = 0.5* (
            u[1][k][j][i]*u[1][k][j][i] + 
            u[2][k][j][i]*u[2][k][j][i] +
            u[3][k][j][i]*u[3][k][j][i] ) * rho_inv;
        qs[k][j][i] = square[k][j][i] * rho_inv;
        //-------------------------------------------------------------------
        // (don't need speed and ainx until the lhs computation)
        //-------------------------------------------------------------------
        aux = c1c2*rho_inv* (u[4][k][j][i] - square[k][j][i]);
        speed[k][j][i] = sqrt(aux);
      }
    }
  }
#endif


  //---------------------------------------------------------------------
  // copy the exact forcing term to the right hand side;  because 
  // this forcing term is known, we can store it on the whole grid
  // including the boundary                   
  //---------------------------------------------------------------------
  size_t kernel_compute_rhs_1_off[2] = { 0, 0 };
  size_t kernel_compute_rhs_1_idx[2] = { gp1, gp2 };
  brisbane_kernel kernel_compute_rhs_1;
  brisbane_kernel_create("compute_rhs_1", &kernel_compute_rhs_1);
  brisbane_kernel_setmem(kernel_compute_rhs_1, 0, mem_rhs, brisbane_w);
  brisbane_kernel_setmem(kernel_compute_rhs_1, 1, mem_forcing, brisbane_r);
  brisbane_kernel_setarg(kernel_compute_rhs_1, 2, sizeof(int), &gp0);

  brisbane_task task1;
  brisbane_task_create(&task1);
  brisbane_task_h2d_full(task1, mem_forcing, forcing);
  brisbane_task_h2d_full(task1, mem_rhs, rhs);
  brisbane_task_kernel(task1, kernel_compute_rhs_1, 2, kernel_compute_rhs_1_off, kernel_compute_rhs_1_idx);
  brisbane_task_submit(task1, brisbane_gpu, NULL, true);
#if 0
    #pragma omp target teams distribute parallel for private(i,j,k,m) collapse(2)
    for (k = 0; k <= gp2-1; k++) {
      for (j = 0; j <= gp1-1; j++) {
        for (i = 0; i <= gp0-1; i++) {
        for (m = 0; m < 5; m++) {
          rhs[m][k][j][i] = forcing[m][k][j][i];
        }
      }
    }
  }
#endif

  //---------------------------------------------------------------------
  // compute xi-direction fluxes 
  //---------------------------------------------------------------------
  size_t kernel_compute_rhs_2_off[3] = { 1, 1, 1 };
  size_t kernel_compute_rhs_2_idx[3] = { nx2, ny2, nz2 };
  brisbane_kernel kernel_compute_rhs_2;
  brisbane_kernel_create("compute_rhs_2", &kernel_compute_rhs_2);
  brisbane_kernel_setmem(kernel_compute_rhs_2, 0, mem_us, brisbane_r);
  brisbane_kernel_setmem(kernel_compute_rhs_2, 1, mem_rhs, brisbane_rw);
  brisbane_kernel_setmem(kernel_compute_rhs_2, 2, mem_u, brisbane_r);
  brisbane_kernel_setmem(kernel_compute_rhs_2, 3, mem_square, brisbane_r);
  brisbane_kernel_setmem(kernel_compute_rhs_2, 4, mem_vs, brisbane_r);
  brisbane_kernel_setmem(kernel_compute_rhs_2, 5, mem_ws, brisbane_r);
  brisbane_kernel_setmem(kernel_compute_rhs_2, 6, mem_qs, brisbane_r);
  brisbane_kernel_setmem(kernel_compute_rhs_2, 7, mem_rho_i, brisbane_r);
  brisbane_kernel_setarg(kernel_compute_rhs_2, 8, sizeof(double), &dx1tx1);
  brisbane_kernel_setarg(kernel_compute_rhs_2, 9, sizeof(double), &dx2tx1);
  brisbane_kernel_setarg(kernel_compute_rhs_2, 10, sizeof(double), &dx3tx1);
  brisbane_kernel_setarg(kernel_compute_rhs_2, 11, sizeof(double), &dx4tx1);
  brisbane_kernel_setarg(kernel_compute_rhs_2, 12, sizeof(double), &dx5tx1);
  brisbane_kernel_setarg(kernel_compute_rhs_2, 13, sizeof(double), &tx2);
  brisbane_kernel_setarg(kernel_compute_rhs_2, 14, sizeof(double), &xxcon2);
  brisbane_kernel_setarg(kernel_compute_rhs_2, 15, sizeof(double), &xxcon3);
  brisbane_kernel_setarg(kernel_compute_rhs_2, 16, sizeof(double), &xxcon4);
  brisbane_kernel_setarg(kernel_compute_rhs_2, 17, sizeof(double), &xxcon5);
  brisbane_kernel_setarg(kernel_compute_rhs_2, 18, sizeof(double), &con43);
  brisbane_kernel_setarg(kernel_compute_rhs_2, 19, sizeof(double), &c1);
  brisbane_kernel_setarg(kernel_compute_rhs_2, 20, sizeof(double), &c2);

  brisbane_task task2;
  brisbane_task_create(&task2);
  brisbane_task_kernel(task2, kernel_compute_rhs_2, 3, kernel_compute_rhs_2_off, kernel_compute_rhs_2_idx);
  brisbane_task_submit(task2, brisbane_gpu, NULL, true);
#if 0
#ifdef SPEC_USE_INNER_SIMD
  #pragma omp target teams distribute parallel for private(i,j,k,uijk,up1,um1) collapse(2)
#else
  #pragma omp target teams distribute parallel for simd private(uijk,up1,um1) collapse(3)
#endif
  for (k = 1; k <= nz2; k++) {
    for (j = 1; j <= ny2; j++) {
#ifdef SPEC_USE_INNER_SIMD
      #pragma omp simd private(uijk,up1,um1)
#endif
      for (i = 1; i <= nx2; i++) {
        uijk = us[k][j][i];
        up1  = us[k][j][i+1];
        um1  = us[k][j][i-1];

        rhs[0][k][j][i] = rhs[0][k][j][i] + dx1tx1 * 
          (u[0][k][j][i+1] - 2.0*u[0][k][j][i] + u[0][k][j][i-1]) -
          tx2 * (u[1][k][j][i+1] - u[1][k][j][i-1]);

        rhs[1][k][j][i] = rhs[1][k][j][i] + dx2tx1 * 
          (u[1][k][j][i+1] - 2.0*u[1][k][j][i] + u[1][k][j][i-1]) +
          xxcon2*con43 * (up1 - 2.0*uijk + um1) -
          tx2 * (u[1][k][j][i+1]*up1 - u[1][k][j][i-1]*um1 +
                (u[4][k][j][i+1] - square[k][j][i+1] -
                 u[4][k][j][i-1] + square[k][j][i-1]) * c2);

        rhs[2][k][j][i] = rhs[2][k][j][i] + dx3tx1 * 
          (u[2][k][j][i+1] - 2.0*u[2][k][j][i] + u[2][k][j][i-1]) +
          xxcon2 * (vs[k][j][i+1] - 2.0*vs[k][j][i] + vs[k][j][i-1]) -
          tx2 * (u[2][k][j][i+1]*up1 - u[2][k][j][i-1]*um1);

        rhs[3][k][j][i] = rhs[3][k][j][i] + dx4tx1 * 
          (u[3][k][j][i+1] - 2.0*u[3][k][j][i] + u[3][k][j][i-1]) +
          xxcon2 * (ws[k][j][i+1] - 2.0*ws[k][j][i] + ws[k][j][i-1]) -
          tx2 * (u[3][k][j][i+1]*up1 - u[3][k][j][i-1]*um1);

        rhs[4][k][j][i] = rhs[4][k][j][i] + dx5tx1 * 
          (u[4][k][j][i+1] - 2.0*u[4][k][j][i] + u[4][k][j][i-1]) +
          xxcon3 * (qs[k][j][i+1] - 2.0*qs[k][j][i] + qs[k][j][i-1]) +
          xxcon4 * (up1*up1 -       2.0*uijk*uijk + um1*um1) +
          xxcon5 * (u[4][k][j][i+1]*rho_i[k][j][i+1] - 
                2.0*u[4][k][j][i]*rho_i[k][j][i] +
                    u[4][k][j][i-1]*rho_i[k][j][i-1]) -
          tx2 * ( (c1*u[4][k][j][i+1] - c2*square[k][j][i+1])*up1 -
                  (c1*u[4][k][j][i-1] - c2*square[k][j][i-1])*um1 );
      }
    }

  } /*end k*/
#endif
  
    //---------------------------------------------------------------------
    // add fourth order xi-direction dissipation               
    //---------------------------------------------------------------------
  i = 1;

  size_t kernel_compute_rhs_3_off[3] = { 0, 1, 1 };
  size_t kernel_compute_rhs_3_idx[3] = { 5, ny2, nz2 };
  brisbane_kernel kernel_compute_rhs_3;
  brisbane_kernel_create("compute_rhs_3", &kernel_compute_rhs_3);
  brisbane_kernel_setmem(kernel_compute_rhs_3, 0, mem_rhs, brisbane_rw);
  brisbane_kernel_setmem(kernel_compute_rhs_3, 1, mem_u, brisbane_r);
  brisbane_kernel_setarg(kernel_compute_rhs_3, 2, sizeof(double), &dssp);
  brisbane_kernel_setarg(kernel_compute_rhs_3, 3, sizeof(int), &i);

  brisbane_task task3;
  brisbane_task_create(&task3);
  brisbane_task_kernel(task3, kernel_compute_rhs_3, 3, kernel_compute_rhs_3_off, kernel_compute_rhs_3_idx);
  brisbane_task_submit(task3, brisbane_gpu, NULL, true);
#if 0
    #pragma omp target teams distribute parallel for private(j,k,m) collapse(3)
    for (k = 1; k <= nz2; k++){
      for (j = 1; j <= ny2; j++) {
      for (m = 0; m < 5; m++) {
        rhs[m][k][j][i] = rhs[m][k][j][i]- dssp * 
          (5.0*u[m][k][j][i] - 4.0*u[m][k][j][i+1] + u[m][k][j][i+2]);
      }
    }
  }
#endif
      
  i = 2;

  size_t kernel_compute_rhs_4_off[3] = { 1, 1, 0 };
  size_t kernel_compute_rhs_4_idx[3] = { ny2, nz2, 5 };
  brisbane_kernel kernel_compute_rhs_4;
  brisbane_kernel_create("compute_rhs_4", &kernel_compute_rhs_4);
  brisbane_kernel_setmem(kernel_compute_rhs_4, 0, mem_rhs, brisbane_rw);
  brisbane_kernel_setmem(kernel_compute_rhs_4, 1, mem_u, brisbane_r);
  brisbane_kernel_setarg(kernel_compute_rhs_4, 2, sizeof(double), &dssp);
  brisbane_kernel_setarg(kernel_compute_rhs_4, 3, sizeof(int), &i);

  brisbane_task task4;
  brisbane_task_create(&task4);
  brisbane_task_kernel(task4, kernel_compute_rhs_4, 3, kernel_compute_rhs_4_off, kernel_compute_rhs_4_idx);
  brisbane_task_submit(task4, brisbane_gpu, NULL, true);
#if 0
#ifdef SPEC_USE_INNER_SIMD
  #pragma omp target teams distribute parallel for private(j,k,m) collapse(2)
#else
  #pragma omp target teams distribute parallel for simd collapse(3)
#endif
  for (m = 0; m < 5; m++) {
    for (k = 1; k <= nz2; k++){
#ifdef SPEC_USE_INNER_SIMD
      #pragma omp simd
#endif
      for (j = 1; j <= ny2; j++) {
        rhs[m][k][j][i] = rhs[m][k][j][i] - dssp * 
          (-4.0*u[m][k][j][i-1] + 6.0*u[m][k][j][i] -
            4.0*u[m][k][j][i+1] + u[m][k][j][i+2]);
      }
    }
  }
#endif

  size_t kernel_compute_rhs_5_off[3] = { 3, 1, 1 };
  size_t kernel_compute_rhs_5_idx[3] = { nx2 - 4, ny2, nz2 };
  brisbane_kernel kernel_compute_rhs_5;
  brisbane_kernel_create("compute_rhs_5", &kernel_compute_rhs_5);
  brisbane_kernel_setmem(kernel_compute_rhs_5, 0, mem_rhs, brisbane_rw);
  brisbane_kernel_setmem(kernel_compute_rhs_5, 1, mem_u, brisbane_r);
  brisbane_kernel_setarg(kernel_compute_rhs_5, 2, sizeof(double), &dssp);

  brisbane_task task5;
  brisbane_task_create(&task5);
  brisbane_task_kernel(task5, kernel_compute_rhs_5, 3, kernel_compute_rhs_5_off, kernel_compute_rhs_5_idx);
  brisbane_task_submit(task5, brisbane_gpu, NULL, true);
#if 0
#ifdef SPEC_USE_INNER_SIMD    
    #pragma omp target teams distribute parallel for private(i,j,k,m) collapse(2)
#else
    #pragma omp target teams distribute parallel for simd collapse(4)
#endif
    for (k = 1; k <= nz2; k++){
      for (j = 1; j <= ny2; j++) {
#ifdef SPEC_USE_INNER_SIMD
      #pragma omp simd
#endif
        for (i = 3; i <= nx2-2; i++) {
        for (m = 0; m < 5; m++) {
          rhs[m][k][j][i] = rhs[m][k][j][i] - dssp * 
            ( u[m][k][j][i-2] - 4.0*u[m][k][j][i-1] + 
            6.0*u[m][k][j][i] - 4.0*u[m][k][j][i+1] + 
              u[m][k][j][i+2] );
        }
      }
    }
  }
#endif
    
  i = nx2-1;

  size_t kernel_compute_rhs_6_off[1] = { 1 };
  size_t kernel_compute_rhs_6_idx[1] = { nz2 };
  brisbane_kernel kernel_compute_rhs_6;
  brisbane_kernel_create("compute_rhs_6", &kernel_compute_rhs_6);
  brisbane_kernel_setmem(kernel_compute_rhs_6, 0, mem_rhs, brisbane_rw);
  brisbane_kernel_setmem(kernel_compute_rhs_6, 1, mem_u, brisbane_r);
  brisbane_kernel_setarg(kernel_compute_rhs_6, 2, sizeof(double), &dssp);
  brisbane_kernel_setarg(kernel_compute_rhs_6, 3, sizeof(int), &i);

  brisbane_task task6;
  brisbane_task_create(&task6);
  brisbane_task_kernel(task6, kernel_compute_rhs_6, 1, kernel_compute_rhs_6_off, kernel_compute_rhs_6_idx);
  brisbane_task_submit(task6, brisbane_gpu, NULL, true);
#if 0
    #pragma omp target teams distribute parallel for private(j,k,m)
    for (k = 1; k <= nz2; k++){
      for (j = 1; j <= ny2; j++) {
      for (m = 0; m < 5; m++) {
        rhs[m][k][j][i] = rhs[m][k][j][i] - dssp *
          ( u[m][k][j][i-2] - 4.0*u[m][k][j][i-1] + 
          6.0*u[m][k][j][i] - 4.0*u[m][k][j][i+1] );
      }
    }
  }
#endif
      
  i = nx2;

  size_t kernel_compute_rhs_7_off[1] = { 1 };
  size_t kernel_compute_rhs_7_idx[1] = { nz2 };
  brisbane_kernel kernel_compute_rhs_7;
  brisbane_kernel_create("compute_rhs_7", &kernel_compute_rhs_7);
  brisbane_kernel_setmem(kernel_compute_rhs_7, 0, mem_rhs, brisbane_rw);
  brisbane_kernel_setmem(kernel_compute_rhs_7, 1, mem_u, brisbane_r);
  brisbane_kernel_setarg(kernel_compute_rhs_7, 2, sizeof(double), &dssp);
  brisbane_kernel_setarg(kernel_compute_rhs_7, 3, sizeof(int), &i);

  brisbane_task task7;
  brisbane_task_create(&task7);
  brisbane_task_kernel(task7, kernel_compute_rhs_7, 1, kernel_compute_rhs_7_off, kernel_compute_rhs_7_idx);
  brisbane_task_submit(task7, brisbane_gpu, NULL, true);
#if 0
    #pragma omp target teams distribute parallel for private(j,k,m)
    for (k = 1; k <= nz2; k++){
      for (j = 1; j <= ny2; j++) {
      for (m = 0; m < 5; m++) {
        rhs[m][k][j][i] = rhs[m][k][j][i] - dssp *
          ( u[m][k][j][i-2] - 4.0*u[m][k][j][i-1] + 5.0*u[m][k][j][i] );
      }
    }
  }
#endif

  //---------------------------------------------------------------------
  // compute eta-direction fluxes 
  //---------------------------------------------------------------------

  size_t kernel_compute_rhs_8_off[3] = { 1, 1, 1 };
  size_t kernel_compute_rhs_8_idx[3] = { nx2, ny2, nz2 };
  brisbane_kernel kernel_compute_rhs_8;
  brisbane_kernel_create("compute_rhs_8", &kernel_compute_rhs_8);
  brisbane_kernel_setmem(kernel_compute_rhs_8, 0, mem_vs, brisbane_r);
  brisbane_kernel_setmem(kernel_compute_rhs_8, 1, mem_rhs, brisbane_rw);
  brisbane_kernel_setmem(kernel_compute_rhs_8, 2, mem_u, brisbane_r);
  brisbane_kernel_setmem(kernel_compute_rhs_8, 3, mem_us, brisbane_r);
  brisbane_kernel_setmem(kernel_compute_rhs_8, 4, mem_square, brisbane_r);
  brisbane_kernel_setmem(kernel_compute_rhs_8, 5, mem_ws, brisbane_r);
  brisbane_kernel_setmem(kernel_compute_rhs_8, 6, mem_qs, brisbane_r);
  brisbane_kernel_setmem(kernel_compute_rhs_8, 7, mem_rho_i, brisbane_r);
  brisbane_kernel_setarg(kernel_compute_rhs_8, 8, sizeof(double), &dy1ty1);
  brisbane_kernel_setarg(kernel_compute_rhs_8, 9, sizeof(double), &dy2ty1);
  brisbane_kernel_setarg(kernel_compute_rhs_8, 10, sizeof(double), &dy3ty1);
  brisbane_kernel_setarg(kernel_compute_rhs_8, 11, sizeof(double), &dy4ty1);
  brisbane_kernel_setarg(kernel_compute_rhs_8, 12, sizeof(double), &dy5ty1);
  brisbane_kernel_setarg(kernel_compute_rhs_8, 13, sizeof(double), &ty2);
  brisbane_kernel_setarg(kernel_compute_rhs_8, 14, sizeof(double), &yycon2);
  brisbane_kernel_setarg(kernel_compute_rhs_8, 15, sizeof(double), &yycon3);
  brisbane_kernel_setarg(kernel_compute_rhs_8, 16, sizeof(double), &yycon4);
  brisbane_kernel_setarg(kernel_compute_rhs_8, 17, sizeof(double), &yycon5);
  brisbane_kernel_setarg(kernel_compute_rhs_8, 18, sizeof(double), &con43);
  brisbane_kernel_setarg(kernel_compute_rhs_8, 19, sizeof(double), &c1);
  brisbane_kernel_setarg(kernel_compute_rhs_8, 20, sizeof(double), &c2);

  brisbane_task task8;
  brisbane_task_create(&task8);
  brisbane_task_kernel(task8, kernel_compute_rhs_8, 3, kernel_compute_rhs_8_off, kernel_compute_rhs_8_idx);
  brisbane_task_submit(task8, brisbane_gpu, NULL, true);
#if 0
#ifdef SPEC_USE_INNER_SIMD
  #pragma omp target teams distribute parallel for private(vijk,vp1,vm1,i,j,k) collapse(2)
#else
  #pragma omp target teams distribute parallel for simd private(vijk,vp1,vm1) collapse(3)
#endif
  for (k = 1; k <= nz2; k++) {
    for (j = 1; j <= ny2; j++) {
#ifdef SPEC_USE_INNER_SIMD
      #pragma omp simd private(vijk,vp1,vm1)
#endif
      for (i = 1; i <= nx2; i++) {
        vijk = vs[k][j][i];
        vp1  = vs[k][j+1][i];
        vm1  = vs[k][j-1][i];

        rhs[0][k][j][i] = rhs[0][k][j][i] + dy1ty1 * 
          (u[0][k][j+1][i] - 2.0*u[0][k][j][i] + u[0][k][j-1][i]) -
          ty2 * (u[2][k][j+1][i] - u[2][k][j-1][i]);

        rhs[1][k][j][i] = rhs[1][k][j][i] + dy2ty1 * 
          (u[1][k][j+1][i] - 2.0*u[1][k][j][i] + u[1][k][j-1][i]) +
          yycon2 * (us[k][j+1][i] - 2.0*us[k][j][i] + us[k][j-1][i]) -
          ty2 * (u[1][k][j+1][i]*vp1 - u[1][k][j-1][i]*vm1);

        rhs[2][k][j][i] = rhs[2][k][j][i] + dy3ty1 * 
          (u[2][k][j+1][i] - 2.0*u[2][k][j][i] + u[2][k][j-1][i]) +
          yycon2*con43 * (vp1 - 2.0*vijk + vm1) -
          ty2 * (u[2][k][j+1][i]*vp1 - u[2][k][j-1][i]*vm1 +
                (u[4][k][j+1][i] - square[k][j+1][i] - 
                 u[4][k][j-1][i] + square[k][j-1][i]) * c2);

        rhs[3][k][j][i] = rhs[3][k][j][i] + dy4ty1 * 
          (u[3][k][j+1][i] - 2.0*u[3][k][j][i] + u[3][k][j-1][i]) +
          yycon2 * (ws[k][j+1][i] - 2.0*ws[k][j][i] + ws[k][j-1][i]) -
          ty2 * (u[3][k][j+1][i]*vp1 - u[3][k][j-1][i]*vm1);

        rhs[4][k][j][i] = rhs[4][k][j][i] + dy5ty1 * 
          (u[4][k][j+1][i] - 2.0*u[4][k][j][i] + u[4][k][j-1][i]) +
          yycon3 * (qs[k][j+1][i] - 2.0*qs[k][j][i] + qs[k][j-1][i]) +
          yycon4 * (vp1*vp1       - 2.0*vijk*vijk + vm1*vm1) +
          yycon5 * (u[4][k][j+1][i]*rho_i[k][j+1][i] - 
                  2.0*u[4][k][j][i]*rho_i[k][j][i] +
                    u[4][k][j-1][i]*rho_i[k][j-1][i]) -
          ty2 * ((c1*u[4][k][j+1][i] - c2*square[k][j+1][i]) * vp1 -
                 (c1*u[4][k][j-1][i] - c2*square[k][j-1][i]) * vm1);
      }
    }
  }
#endif
    
	//---------------------------------------------------------------------
    // add fourth order eta-direction dissipation         
    //---------------------------------------------------------------------
  j = 1;

  size_t kernel_compute_rhs_9_off[3] = { 0, 1, 1 };
  size_t kernel_compute_rhs_9_idx[3] = { 5, nx2, nz2 };
  brisbane_kernel kernel_compute_rhs_9;
  brisbane_kernel_create("compute_rhs_9", &kernel_compute_rhs_9);
  brisbane_kernel_setmem(kernel_compute_rhs_9, 0, mem_rhs, brisbane_rw);
  brisbane_kernel_setmem(kernel_compute_rhs_9, 1, mem_u, brisbane_r);
  brisbane_kernel_setarg(kernel_compute_rhs_9, 2, sizeof(double), &dssp);
  brisbane_kernel_setarg(kernel_compute_rhs_9, 3, sizeof(int), &j);

  brisbane_task task9;
  brisbane_task_create(&task9);
  brisbane_task_kernel(task9, kernel_compute_rhs_9, 3, kernel_compute_rhs_9_off, kernel_compute_rhs_9_idx);
  brisbane_task_submit(task9, brisbane_gpu, NULL, true);
#if 0
    #pragma omp target teams distribute parallel for private(i,k,m) collapse(3)
    for (k = 1; k <= nz2; k++) {
      for (i = 1; i <= nx2; i++) {
      for (m = 0; m < 5; m++) {
        rhs[m][k][j][i] = rhs[m][k][j][i]- dssp * 
          ( 5.0*u[m][k][j][i] - 4.0*u[m][k][j+1][i] + u[m][k][j+2][i]);
      }
    }
  }
#endif

  j = 2;

  size_t kernel_compute_rhs_10_off[3] = { 0, 1, 1 };
  size_t kernel_compute_rhs_10_idx[3] = { 5, nx2, nz2 };
  brisbane_kernel kernel_compute_rhs_10;
  brisbane_kernel_create("compute_rhs_10", &kernel_compute_rhs_10);
  brisbane_kernel_setmem(kernel_compute_rhs_10, 0, mem_rhs, brisbane_rw);
  brisbane_kernel_setmem(kernel_compute_rhs_10, 1, mem_u, brisbane_r);
  brisbane_kernel_setarg(kernel_compute_rhs_10, 2, sizeof(double), &dssp);
  brisbane_kernel_setarg(kernel_compute_rhs_10, 3, sizeof(int), &j);

  brisbane_task task10;
  brisbane_task_create(&task10);
  brisbane_task_kernel(task10, kernel_compute_rhs_10, 3, kernel_compute_rhs_10_off, kernel_compute_rhs_10_idx);
  brisbane_task_submit(task10, brisbane_gpu, NULL, true);
#if 0
    #pragma omp target teams distribute parallel for private(i,k,m) collapse(3)
    for (k = 1; k <= nz2; k++) {
      for (i = 1; i <= nx2; i++) {
      for (m = 0; m < 5; m++) {
        rhs[m][k][j][i] = rhs[m][k][j][i] - dssp * 
          (-4.0*u[m][k][j-1][i] + 6.0*u[m][k][j][i] -
            4.0*u[m][k][j+1][i] + u[m][k][j+2][i]);
      }
    }
  }
#endif

  size_t kernel_compute_rhs_11_off[3] = { 1, 3, 1 };
  size_t kernel_compute_rhs_11_idx[3] = { nx2, ny2-4, nz2 };
  brisbane_kernel kernel_compute_rhs_11;
  brisbane_kernel_create("compute_rhs_11", &kernel_compute_rhs_11);
  brisbane_kernel_setmem(kernel_compute_rhs_11, 0, mem_rhs, brisbane_rw);
  brisbane_kernel_setmem(kernel_compute_rhs_11, 1, mem_u, brisbane_r);
  brisbane_kernel_setarg(kernel_compute_rhs_11, 2, sizeof(double), &dssp);

  brisbane_task task11;
  brisbane_task_create(&task11);
  brisbane_task_kernel(task11, kernel_compute_rhs_11, 3, kernel_compute_rhs_11_off, kernel_compute_rhs_11_idx);
  brisbane_task_submit(task11, brisbane_gpu, NULL, true);
#if 0
#ifdef SPEC_USE_INNER_SIMD
    #pragma omp target teams distribute parallel for private(i,j,k,m) collapse(2)
#else
    #pragma omp target teams distribute parallel for simd collapse(4)
#endif    
    for (k = 1; k <= nz2; k++) {
      for (j = 3; j <= ny2-2; j++) {
#ifdef SPEC_USE_INNER_SIMD
      #pragma omp simd
#endif
        for (i = 1; i <= nx2; i++) {
        for (m = 0; m < 5; m++) {
          rhs[m][k][j][i] = rhs[m][k][j][i] - dssp * 
            ( u[m][k][j-2][i] - 4.0*u[m][k][j-1][i] + 
            6.0*u[m][k][j][i] - 4.0*u[m][k][j+1][i] + 
              u[m][k][j+2][i] );
        }
      }
    }
  }
#endif
    
  j = ny2-1;

  size_t kernel_compute_rhs_12_off[3] = { 0, 1, 1 };
  size_t kernel_compute_rhs_12_idx[3] = { 5, nx2, nz2 };
  brisbane_kernel kernel_compute_rhs_12;
  brisbane_kernel_create("compute_rhs_12", &kernel_compute_rhs_12);
  brisbane_kernel_setmem(kernel_compute_rhs_12, 0, mem_rhs, brisbane_rw);
  brisbane_kernel_setmem(kernel_compute_rhs_12, 1, mem_u, brisbane_r);
  brisbane_kernel_setarg(kernel_compute_rhs_12, 2, sizeof(double), &dssp);
  brisbane_kernel_setarg(kernel_compute_rhs_12, 3, sizeof(int), &j);

  brisbane_task task12;
  brisbane_task_create(&task12);
  brisbane_task_kernel(task12, kernel_compute_rhs_12, 3, kernel_compute_rhs_12_off, kernel_compute_rhs_12_idx);
  brisbane_task_submit(task12, brisbane_gpu, NULL, true);
#if 0
    #pragma omp target teams distribute parallel for private(i,k,m) collapse(3) 
    for (k = 1; k <= nz2; k++) {
      for (i = 1; i <= nx2; i++) {
      for (m = 0; m < 5; m++) {
        rhs[m][k][j][i] = rhs[m][k][j][i] - dssp *
          ( u[m][k][j-2][i] - 4.0*u[m][k][j-1][i] + 
          6.0*u[m][k][j][i] - 4.0*u[m][k][j+1][i] );
      }
    }
  }
#endif
    
  j = ny2;

  size_t kernel_compute_rhs_13_off[3] = { 0, 1, 1 };
  size_t kernel_compute_rhs_13_idx[3] = { 5, nx2, nz2 };
  brisbane_kernel kernel_compute_rhs_13;
  brisbane_kernel_create("compute_rhs_13", &kernel_compute_rhs_13);
  brisbane_kernel_setmem(kernel_compute_rhs_13, 0, mem_rhs, brisbane_rw);
  brisbane_kernel_setmem(kernel_compute_rhs_13, 1, mem_u, brisbane_r);
  brisbane_kernel_setarg(kernel_compute_rhs_13, 2, sizeof(double), &dssp);
  brisbane_kernel_setarg(kernel_compute_rhs_13, 3, sizeof(int), &j);

  brisbane_task task13;
  brisbane_task_create(&task13);
  brisbane_task_kernel(task13, kernel_compute_rhs_13, 3, kernel_compute_rhs_13_off, kernel_compute_rhs_13_idx);
  brisbane_task_submit(task13, brisbane_gpu, NULL, true);
#if 0
    #pragma omp target teams distribute parallel for private(i,k,m) collapse(3) 
    for (k = 1; k <= nz2; k++) {
      for (i = 1; i <= nx2; i++) {
      for (m = 0; m < 5; m++) {
        rhs[m][k][j][i] = rhs[m][k][j][i] - dssp *
          ( u[m][k][j-2][i] - 4.0*u[m][k][j-1][i] + 5.0*u[m][k][j][i] );
      }
    }
  }
#endif

  //---------------------------------------------------------------------
  // compute zeta-direction fluxes 
  //---------------------------------------------------------------------
  size_t kernel_compute_rhs_14_off[3] = { 1, 1, 1 };
  size_t kernel_compute_rhs_14_idx[3] = { nx2, ny2, nz2 };
  brisbane_kernel kernel_compute_rhs_14;
  brisbane_kernel_create("compute_rhs_14", &kernel_compute_rhs_14);
  brisbane_kernel_setmem(kernel_compute_rhs_14, 0, mem_ws, brisbane_r);
  brisbane_kernel_setmem(kernel_compute_rhs_14, 1, mem_rhs, brisbane_rw);
  brisbane_kernel_setmem(kernel_compute_rhs_14, 2, mem_u, brisbane_r);
  brisbane_kernel_setmem(kernel_compute_rhs_14, 3, mem_us, brisbane_r);
  brisbane_kernel_setmem(kernel_compute_rhs_14, 4, mem_vs, brisbane_r);
  brisbane_kernel_setmem(kernel_compute_rhs_14, 5, mem_square, brisbane_r);
  brisbane_kernel_setmem(kernel_compute_rhs_14, 6, mem_qs, brisbane_r);
  brisbane_kernel_setmem(kernel_compute_rhs_14, 7, mem_rho_i, brisbane_r);
  brisbane_kernel_setarg(kernel_compute_rhs_14, 8, sizeof(double), &dz1tz1);
  brisbane_kernel_setarg(kernel_compute_rhs_14, 9, sizeof(double), &dz2tz1);
  brisbane_kernel_setarg(kernel_compute_rhs_14, 10, sizeof(double), &dz3tz1);
  brisbane_kernel_setarg(kernel_compute_rhs_14, 11, sizeof(double), &dz4tz1);
  brisbane_kernel_setarg(kernel_compute_rhs_14, 12, sizeof(double), &dz5tz1);
  brisbane_kernel_setarg(kernel_compute_rhs_14, 13, sizeof(double), &tz2);
  brisbane_kernel_setarg(kernel_compute_rhs_14, 14, sizeof(double), &zzcon2);
  brisbane_kernel_setarg(kernel_compute_rhs_14, 15, sizeof(double), &zzcon3);
  brisbane_kernel_setarg(kernel_compute_rhs_14, 16, sizeof(double), &zzcon4);
  brisbane_kernel_setarg(kernel_compute_rhs_14, 17, sizeof(double), &zzcon5);
  brisbane_kernel_setarg(kernel_compute_rhs_14, 18, sizeof(double), &con43);
  brisbane_kernel_setarg(kernel_compute_rhs_14, 19, sizeof(double), &c1);
  brisbane_kernel_setarg(kernel_compute_rhs_14, 20, sizeof(double), &c2);

  brisbane_task task14;
  brisbane_task_create(&task14);
  brisbane_task_kernel(task14, kernel_compute_rhs_14, 3, kernel_compute_rhs_14_off, kernel_compute_rhs_14_idx);
  brisbane_task_submit(task14, brisbane_gpu, NULL, true);
#if 0
#ifdef SPEC_USE_INNER_SIMD
  #pragma omp target teams distribute parallel for private(i,j,k,wijk,wp1,wm1) collapse(2)
#else
  #pragma omp target teams distribute parallel for simd private(wijk,wp1,wm1) collapse(3)
#endif
  for (k = 1; k <= nz2; k++) {
    for (j = 1; j <= ny2; j++) {
#ifdef SPEC_USE_INNER_SIMD
      #pragma omp simd private(wijk,wp1,wm1)
#endif
      for (i = 1; i <= nx2; i++) {
        wijk = ws[k][j][i];
        wp1  = ws[k+1][j][i];
        wm1  = ws[k-1][j][i];

        rhs[0][k][j][i] = rhs[0][k][j][i] + dz1tz1 * 
          (u[0][k+1][j][i] - 2.0*u[0][k][j][i] + u[0][k-1][j][i]) -
          tz2 * (u[3][k+1][j][i] - u[3][k-1][j][i]);

        rhs[1][k][j][i] = rhs[1][k][j][i] + dz2tz1 * 
          (u[1][k+1][j][i] - 2.0*u[1][k][j][i] + u[1][k-1][j][i]) +
          zzcon2 * (us[k+1][j][i] - 2.0*us[k][j][i] + us[k-1][j][i]) -
          tz2 * (u[1][k+1][j][i]*wp1 - u[1][k-1][j][i]*wm1);

        rhs[2][k][j][i] = rhs[2][k][j][i] + dz3tz1 * 
          (u[2][k+1][j][i] - 2.0*u[2][k][j][i] + u[2][k-1][j][i]) +
          zzcon2 * (vs[k+1][j][i] - 2.0*vs[k][j][i] + vs[k-1][j][i]) -
          tz2 * (u[2][k+1][j][i]*wp1 - u[2][k-1][j][i]*wm1);

        rhs[3][k][j][i] = rhs[3][k][j][i] + dz4tz1 * 
          (u[3][k+1][j][i] - 2.0*u[3][k][j][i] + u[3][k-1][j][i]) +
          zzcon2*con43 * (wp1 - 2.0*wijk + wm1) -
          tz2 * (u[3][k+1][j][i]*wp1 - u[3][k-1][j][i]*wm1 +
                (u[4][k+1][j][i] - square[k+1][j][i] - 
                 u[4][k-1][j][i] + square[k-1][j][i]) * c2);

        rhs[4][k][j][i] = rhs[4][k][j][i] + dz5tz1 * 
          (u[4][k+1][j][i] - 2.0*u[4][k][j][i] + u[4][k-1][j][i]) +
          zzcon3 * (qs[k+1][j][i] - 2.0*qs[k][j][i] + qs[k-1][j][i]) +
          zzcon4 * (wp1*wp1 - 2.0*wijk*wijk + wm1*wm1) +
          zzcon5 * (u[4][k+1][j][i]*rho_i[k+1][j][i] - 
                  2.0*u[4][k][j][i]*rho_i[k][j][i] +
                    u[4][k-1][j][i]*rho_i[k-1][j][i]) -
          tz2 * ((c1*u[4][k+1][j][i] - c2*square[k+1][j][i])*wp1 -
                 (c1*u[4][k-1][j][i] - c2*square[k-1][j][i])*wm1);
      }
    }
  }
#endif

  //---------------------------------------------------------------------
  // add fourth order zeta-direction dissipation                
  //---------------------------------------------------------------------
  k = 1;

  size_t kernel_compute_rhs_15_off[3] = { 0, 1, 1 };
  size_t kernel_compute_rhs_15_idx[3] = { 5, nx2, ny2 };
  brisbane_kernel kernel_compute_rhs_15;
  brisbane_kernel_create("compute_rhs_15", &kernel_compute_rhs_15);
  brisbane_kernel_setmem(kernel_compute_rhs_15, 0, mem_rhs, brisbane_rw);
  brisbane_kernel_setmem(kernel_compute_rhs_15, 1, mem_u, brisbane_r);
  brisbane_kernel_setarg(kernel_compute_rhs_15, 2, sizeof(double), &dssp);
  brisbane_kernel_setarg(kernel_compute_rhs_15, 3, sizeof(int), &k);

  brisbane_task task15;
  brisbane_task_create(&task15);
  brisbane_task_kernel(task15, kernel_compute_rhs_15, 3, kernel_compute_rhs_15_off, kernel_compute_rhs_15_idx);
  brisbane_task_submit(task15, brisbane_gpu, NULL, true);
#if 0
  #pragma omp target teams distribute parallel for private(i,j,m) collapse(3) 
    for (j = 1; j <= ny2; j++) {
      for (i = 1; i <= nx2; i++) {
      for (m = 0; m < 5; m++) {
        rhs[m][k][j][i] = rhs[m][k][j][i]- dssp * 
          (5.0*u[m][k][j][i] - 4.0*u[m][k+1][j][i] + u[m][k+2][j][i]);
      }
    }
  }
#endif

  k = 2;

  size_t kernel_compute_rhs_16_off[3] = { 0, 1, 1 };
  size_t kernel_compute_rhs_16_idx[3] = { 5, nx2, ny2 };
  brisbane_kernel kernel_compute_rhs_16;
  brisbane_kernel_create("compute_rhs_16", &kernel_compute_rhs_16);
  brisbane_kernel_setmem(kernel_compute_rhs_16, 0, mem_rhs, brisbane_rw);
  brisbane_kernel_setmem(kernel_compute_rhs_16, 1, mem_u, brisbane_r);
  brisbane_kernel_setarg(kernel_compute_rhs_16, 2, sizeof(double), &dssp);
  brisbane_kernel_setarg(kernel_compute_rhs_16, 3, sizeof(int), &k);

  brisbane_task task16;
  brisbane_task_create(&task16);
  brisbane_task_kernel(task16, kernel_compute_rhs_16, 3, kernel_compute_rhs_16_off, kernel_compute_rhs_16_idx);
  brisbane_task_submit(task16, brisbane_gpu, NULL, true);
#if 0
  #pragma omp target teams distribute parallel for private(i,j,m) collapse(3)
    for (j = 1; j <= ny2; j++) {
      for (i = 1; i <= nx2; i++) {
      for (m = 0; m < 5; m++) {
        rhs[m][k][j][i] = rhs[m][k][j][i] - dssp * 
          (-4.0*u[m][k-1][j][i] + 6.0*u[m][k][j][i] -
            4.0*u[m][k+1][j][i] + u[m][k+2][j][i]);
      }
    }
  }
#endif

  size_t kernel_compute_rhs_17_off[3] = { 1, 1, 3 };
  size_t kernel_compute_rhs_17_idx[3] = { nx2, ny2, nz2-4 };
  brisbane_kernel kernel_compute_rhs_17;
  brisbane_kernel_create("compute_rhs_17", &kernel_compute_rhs_17);
  brisbane_kernel_setmem(kernel_compute_rhs_17, 0, mem_rhs, brisbane_rw);
  brisbane_kernel_setmem(kernel_compute_rhs_17, 1, mem_u, brisbane_r);
  brisbane_kernel_setarg(kernel_compute_rhs_17, 2, sizeof(double), &dssp);

  brisbane_task task17;
  brisbane_task_create(&task17);
  brisbane_task_kernel(task17, kernel_compute_rhs_17, 3, kernel_compute_rhs_17_off, kernel_compute_rhs_17_idx);
  brisbane_task_submit(task17, brisbane_gpu, NULL, true);
#if 0
#ifdef SPEC_USE_INNER_SIMD
    #pragma omp target teams distribute parallel for private(i,j,k,m) collapse(2)
#else
    #pragma omp target teams distribute parallel for simd collapse(4)
#endif
    for (k = 3; k <= nz2-2; k++) {
      for (j = 1; j <= ny2; j++) {
#ifdef SPEC_USE_INNER_SIMD
      #pragma omp simd
#endif
        for (i = 1; i <= nx2; i++) {
        for (m = 0; m < 5; m++) {
          rhs[m][k][j][i] = rhs[m][k][j][i] - dssp * 
            ( u[m][k-2][j][i] - 4.0*u[m][k-1][j][i] + 
            6.0*u[m][k][j][i] - 4.0*u[m][k+1][j][i] + 
              u[m][k+2][j][i] );
        }
      }
    }
  }
#endif

  k = nz2-1;

  size_t kernel_compute_rhs_18_off[3] = { 0, 1, 1 };
  size_t kernel_compute_rhs_18_idx[3] = { 5, nx2, ny2 };
  brisbane_kernel kernel_compute_rhs_18;
  brisbane_kernel_create("compute_rhs_18", &kernel_compute_rhs_18);
  brisbane_kernel_setmem(kernel_compute_rhs_18, 0, mem_rhs, brisbane_rw);
  brisbane_kernel_setmem(kernel_compute_rhs_18, 1, mem_u, brisbane_r);
  brisbane_kernel_setarg(kernel_compute_rhs_18, 2, sizeof(double), &dssp);
  brisbane_kernel_setarg(kernel_compute_rhs_18, 3, sizeof(int), &k);

  brisbane_task task18;
  brisbane_task_create(&task18);
  brisbane_task_kernel(task18, kernel_compute_rhs_18, 3, kernel_compute_rhs_18_off, kernel_compute_rhs_18_idx);
  brisbane_task_submit(task18, brisbane_gpu, NULL, true);
#if 0
    #pragma omp target teams distribute parallel for private(i,j,m) collapse(3) 
    for (j = 1; j <= ny2; j++) {
      for (i = 1; i <= nx2; i++) {
      for (m = 0; m < 5; m++) {
        rhs[m][k][j][i] = rhs[m][k][j][i] - dssp *
          ( u[m][k-2][j][i] - 4.0*u[m][k-1][j][i] + 
          6.0*u[m][k][j][i] - 4.0*u[m][k+1][j][i] );
      }
    }
  }
#endif

  k = nz2;

  size_t kernel_compute_rhs_19_off[3] = { 0, 1, 1 };
  size_t kernel_compute_rhs_19_idx[3] = { 5, nx2, ny2 };
  brisbane_kernel kernel_compute_rhs_19;
  brisbane_kernel_create("compute_rhs_19", &kernel_compute_rhs_19);
  brisbane_kernel_setmem(kernel_compute_rhs_19, 0, mem_rhs, brisbane_rw);
  brisbane_kernel_setmem(kernel_compute_rhs_19, 1, mem_u, brisbane_r);
  brisbane_kernel_setarg(kernel_compute_rhs_19, 2, sizeof(double), &dssp);
  brisbane_kernel_setarg(kernel_compute_rhs_19, 3, sizeof(int), &k);

  brisbane_task task19;
  brisbane_task_create(&task19);
  brisbane_task_kernel(task19, kernel_compute_rhs_19, 3, kernel_compute_rhs_19_off, kernel_compute_rhs_19_idx);
  brisbane_task_submit(task19, brisbane_gpu, NULL, true);
#if 0
    #pragma omp target teams distribute parallel for private(i,j,m) collapse(3)
    for (j = 1; j <= ny2; j++) {
      for (i = 1; i <= nx2; i++) {
      for (m = 0; m < 5; m++) {
        rhs[m][k][j][i] = rhs[m][k][j][i] - dssp *
          ( u[m][k-2][j][i] - 4.0*u[m][k-1][j][i] + 5.0*u[m][k][j][i] );
      }
    }
  }
#endif

  size_t kernel_compute_rhs_20_off[3] = { 1, 1, 1 };
  size_t kernel_compute_rhs_20_idx[3] = { nx2, ny2, nz2 };
  brisbane_kernel kernel_compute_rhs_20;
  brisbane_kernel_create("compute_rhs_20", &kernel_compute_rhs_20);
  brisbane_kernel_setmem(kernel_compute_rhs_20, 0, mem_rhs, brisbane_rw);
  brisbane_kernel_setarg(kernel_compute_rhs_20, 1, sizeof(double), &dt);

  brisbane_task task20;
  brisbane_task_create(&task20);
  brisbane_task_kernel(task20, kernel_compute_rhs_20, 3, kernel_compute_rhs_20_off, kernel_compute_rhs_20_idx);
  brisbane_task_d2h_full(task20, mem_rhs, rhs);
  brisbane_task_submit(task20, brisbane_gpu, NULL, true);
#if 0
    #pragma omp target teams distribute parallel for private(i,j,k,m) collapse(3)
    for (k = 1; k <= nz2; k++) {
      for (j = 1; j <= ny2; j++) {
        for (i = 1; i <= nx2; i++) {
        for (m = 0; m < 5; m++) {
          rhs[m][k][j][i] = rhs[m][k][j][i] * dt;
        }
      }
    }
  }
#endif

//}/* end omp target data*/

}
