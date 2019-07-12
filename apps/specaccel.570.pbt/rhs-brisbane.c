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

void compute_rhs()
{
  int i, j, k, m;
  double rho_inv, uijk, up1, um1, vijk, vp1, vm1, wijk, wp1, wm1;
  int gp0, gp1, gp2;
  int gp01,gp11,gp21;
  int gp02,gp12,gp22;

  gp0 = grid_points[0];
  gp1 = grid_points[1];
  gp2 = grid_points[2];
  gp01 = grid_points[0]-1;
  gp11 = grid_points[1]-1;
  gp21 = grid_points[2]-1;
  gp02 = grid_points[0]-2;
  gp12 = grid_points[1]-2;
  gp22 = grid_points[2]-2;

  //---------------------------------------------------------------------
  // compute the reciprocal of density, and the kinetic energy,
  // and the speed of sound.
  //---------------------------------------------------------------------
  size_t kernel_compute_rhs_0_off[3] = { 0, 0, 0 };
  size_t kernel_compute_rhs_0_idx[3] = { gp01 + 1, gp11 + 1, gp21 + 1 };
  brisbane_kernel kernel_compute_rhs_0;
  brisbane_kernel_create("compute_rhs_0", &kernel_compute_rhs_0);
  brisbane_kernel_setmem(kernel_compute_rhs_0, 0, mem_u, brisbane_r);
  brisbane_kernel_setmem(kernel_compute_rhs_0, 1, mem_rho_i, brisbane_w);
  brisbane_kernel_setmem(kernel_compute_rhs_0, 2, mem_us, brisbane_w);
  brisbane_kernel_setmem(kernel_compute_rhs_0, 3, mem_vs, brisbane_w);
  brisbane_kernel_setmem(kernel_compute_rhs_0, 4, mem_ws, brisbane_w);
  brisbane_kernel_setmem(kernel_compute_rhs_0, 5, mem_square, brisbane_rw);
  brisbane_kernel_setmem(kernel_compute_rhs_0, 6, mem_qs, brisbane_w);

  brisbane_task task0;
  brisbane_task_create(&task0);
  brisbane_task_kernel(task0, kernel_compute_rhs_0, 3, kernel_compute_rhs_0_off, kernel_compute_rhs_0_idx);
  brisbane_task_submit(task0, brisbane_cpu, NULL, true);
#if 0
    #pragma omp target teams distribute parallel for collapse(2) private(i,j,k,rho_inv) 
    for (k = 0; k <= gp21; k++) {
      for (j = 0; j <= gp11; j++) {
        for (i = 0; i <= gp01; i++) {
          rho_inv = 1.0/u[k][j][i][0];
          rho_i[k][j][i] = rho_inv;
          us[k][j][i] = u[k][j][i][1] * rho_inv;
          vs[k][j][i] = u[k][j][i][2] * rho_inv;
          ws[k][j][i] = u[k][j][i][3] * rho_inv;
          square[k][j][i] = 0.5* (
            u[k][j][i][1]*u[k][j][i][1] +
            u[k][j][i][2]*u[k][j][i][2] +
            u[k][j][i][3]*u[k][j][i][3] ) * rho_inv;
          qs[k][j][i] = square[k][j][i] * rho_inv;
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
    size_t kernel_compute_rhs_1_idx[2] = { gp11 + 1, gp21 + 1 };
    brisbane_kernel kernel_compute_rhs_1;
    brisbane_kernel_create("compute_rhs_1", &kernel_compute_rhs_1);
    brisbane_kernel_setmem(kernel_compute_rhs_1, 0, mem_rhs, brisbane_w);
    brisbane_kernel_setmem(kernel_compute_rhs_1, 1, mem_forcing, brisbane_r);
    brisbane_kernel_setarg(kernel_compute_rhs_1, 2, sizeof(int), &gp01);

    brisbane_task task1;
    brisbane_task_create(&task1);
    brisbane_task_kernel(task1, kernel_compute_rhs_1, 2, kernel_compute_rhs_1_off, kernel_compute_rhs_1_idx);
    brisbane_task_submit(task1, brisbane_cpu, NULL, true);
#if 0
    #pragma omp target teams distribute parallel for collapse(2) private(i,j,k)
    for (k = 0; k <= gp21; k++) {
      for (j = 0; j <= gp11; j++) {
        for (i = 0; i <= gp01; i++) {
          rhs[k][j][i][0] = forcing[k][j][i][0];
          rhs[k][j][i][1] = forcing[k][j][i][1];
          rhs[k][j][i][2] = forcing[k][j][i][2];
          rhs[k][j][i][3] = forcing[k][j][i][3];
          rhs[k][j][i][4] = forcing[k][j][i][4];
        }
      }
    }
#endif

    //---------------------------------------------------------------------
    // compute xi-direction fluxes
    //---------------------------------------------------------------------
    size_t kernel_compute_rhs_2_off[1] = { 1 };
    size_t kernel_compute_rhs_2_idx[1] = { gp22 };
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
    brisbane_kernel_setarg(kernel_compute_rhs_2, 21, sizeof(double), &dssp);
    brisbane_kernel_setarg(kernel_compute_rhs_2, 22, sizeof(int), &gp0);
    brisbane_kernel_setarg(kernel_compute_rhs_2, 23, sizeof(int), &gp02);
    brisbane_kernel_setarg(kernel_compute_rhs_2, 24, sizeof(int), &gp12);

    brisbane_task task2;
    brisbane_task_create(&task2);
    brisbane_task_kernel(task2, kernel_compute_rhs_2, 1, kernel_compute_rhs_2_off, kernel_compute_rhs_2_idx);
    brisbane_task_submit(task2, brisbane_cpu, NULL, true);
#if 0
    #pragma omp target teams distribute parallel for private(uijk,up1,um1,i,j,k)
    for (k = 1; k <= gp22; k++) {
      for (j = 1; j <= gp12; j++) {
        #pragma omp simd private(uijk,up1,um1)
        for (i = 1; i <= gp02; i++) {
          uijk = us[k][j][i];
          up1  = us[k][j][i+1];
          um1  = us[k][j][i-1];

          rhs[k][j][i][0] = rhs[k][j][i][0] + dx1tx1 *
            (u[k][j][i+1][0] - 2.0*u[k][j][i][0] +
            u[k][j][i-1][0]) -
            tx2 * (u[k][j][i+1][1] - u[k][j][i-1][1]);

          rhs[k][j][i][1] = rhs[k][j][i][1] + dx2tx1 *
            (u[k][j][i+1][1] - 2.0*u[k][j][i][1] +
            u[k][j][i-1][1]) +
            xxcon2*con43 * (up1 - 2.0*uijk + um1) -
            tx2 * (u[k][j][i+1][1]*up1 -
            u[k][j][i-1][1]*um1 +
            (u[k][j][i+1][4]- square[k][j][i+1]-
          u[k][j][i-1][4]+ square[k][j][i-1])*
          c2);

          rhs[k][j][i][2] = rhs[k][j][i][2] + dx3tx1 *
            (u[k][j][i+1][2] - 2.0*u[k][j][i][2] +
            u[k][j][i-1][2]) +
            xxcon2 * (vs[k][j][i+1] - 2.0*vs[k][j][i] +
            vs[k][j][i-1]) -
            tx2 * (u[k][j][i+1][2]*up1 -
            u[k][j][i-1][2]*um1);

          rhs[k][j][i][3] = rhs[k][j][i][3] + dx4tx1 *
            (u[k][j][i+1][3] - 2.0*u[k][j][i][3] +
            u[k][j][i-1][3]) +
            xxcon2 * (ws[k][j][i+1] - 2.0*ws[k][j][i] +
            ws[k][j][i-1]) -
            tx2 * (u[k][j][i+1][3]*up1 -
            u[k][j][i-1][3]*um1);

          rhs[k][j][i][4] = rhs[k][j][i][4] + dx5tx1 *
            (u[k][j][i+1][4] - 2.0*u[k][j][i][4] +
            u[k][j][i-1][4]) +
            xxcon3 * (qs[k][j][i+1] - 2.0*qs[k][j][i] +
            qs[k][j][i-1]) +
            xxcon4 * (up1*up1 -       2.0*uijk*uijk +
            um1*um1) +
            xxcon5 * (u[k][j][i+1][4]*rho_i[k][j][i+1] -
            2.0*u[k][j][i][4]*rho_i[k][j][i] +
            u[k][j][i-1][4]*rho_i[k][j][i-1]) -
            tx2 * ( (c1*u[k][j][i+1][4] -
            c2*square[k][j][i+1])*up1 -
            (c1*u[k][j][i-1][4] -
            c2*square[k][j][i-1])*um1 );
        }
      }

      //---------------------------------------------------------------------
      // add fourth order xi-direction dissipation
      //---------------------------------------------------------------------
      for (j = 1; j <= gp12; j++) {
        i = 1;
        rhs[k][j][i][0] = rhs[k][j][i][0]- dssp *
          ( 5.0*u[k][j][i][0] - 4.0*u[k][j][i+1][0] +
          u[k][j][i+2][0]);
        rhs[k][j][i][1] = rhs[k][j][i][1]- dssp *
          ( 5.0*u[k][j][i][1] - 4.0*u[k][j][i+1][1] +
          u[k][j][i+2][1]);
        rhs[k][j][i][2] = rhs[k][j][i][2]- dssp *
          ( 5.0*u[k][j][i][2] - 4.0*u[k][j][i+1][2] +
          u[k][j][i+2][2]);
        rhs[k][j][i][3] = rhs[k][j][i][3]- dssp *
          ( 5.0*u[k][j][i][3] - 4.0*u[k][j][i+1][3] +
          u[k][j][i+2][3]);
        rhs[k][j][i][4] = rhs[k][j][i][4]- dssp *
          ( 5.0*u[k][j][i][4] - 4.0*u[k][j][i+1][4] +
          u[k][j][i+2][4]);

        i = 2;
        rhs[k][j][i][0] = rhs[k][j][i][0] - dssp *
          (-4.0*u[k][j][i-1][0] + 6.0*u[k][j][i][0] -
          4.0*u[k][j][i+1][0] + u[k][j][i+2][0]);
        rhs[k][j][i][1] = rhs[k][j][i][1] - dssp *
          (-4.0*u[k][j][i-1][1] + 6.0*u[k][j][i][1] -
          4.0*u[k][j][i+1][1] + u[k][j][i+2][1]);
        rhs[k][j][i][2] = rhs[k][j][i][2] - dssp *
          (-4.0*u[k][j][i-1][2] + 6.0*u[k][j][i][2] -
          4.0*u[k][j][i+1][2] + u[k][j][i+2][2]);
        rhs[k][j][i][3] = rhs[k][j][i][3] - dssp *
          (-4.0*u[k][j][i-1][3] + 6.0*u[k][j][i][3] -
          4.0*u[k][j][i+1][3] + u[k][j][i+2][3]);
        rhs[k][j][i][4] = rhs[k][j][i][4] - dssp *
          (-4.0*u[k][j][i-1][4] + 6.0*u[k][j][i][4] -
          4.0*u[k][j][i+1][4] + u[k][j][i+2][4]);
      }

      for (j = 1; j <= gp12; j++) {
        #pragma omp simd
        for (i = 3; i <= gp02-2; i++) {
          rhs[k][j][i][0] = rhs[k][j][i][0] - dssp*
          (  u[k][j][i-2][0] - 4.0*u[k][j][i-1][0] +
            6.0*u[k][j][i][0] - 4.0*u[k][j][i+1][0] +
            u[k][j][i+2][0] );
          rhs[k][j][i][1] = rhs[k][j][i][1] - dssp*
          (  u[k][j][i-2][1] - 4.0*u[k][j][i-1][1] +
            6.0*u[k][j][i][1] - 4.0*u[k][j][i+1][1] +
            u[k][j][i+2][1] );
          rhs[k][j][i][2] = rhs[k][j][i][2] - dssp*
          (  u[k][j][i-2][2] - 4.0*u[k][j][i-1][2] +
            6.0*u[k][j][i][2] - 4.0*u[k][j][i+1][2] +
            u[k][j][i+2][2] );
          rhs[k][j][i][3] = rhs[k][j][i][3] - dssp*
          (  u[k][j][i-2][3] - 4.0*u[k][j][i-1][3] +
            6.0*u[k][j][i][3] - 4.0*u[k][j][i+1][3] +
            u[k][j][i+2][3] );
          rhs[k][j][i][4] = rhs[k][j][i][4] - dssp*
          (  u[k][j][i-2][4] - 4.0*u[k][j][i-1][4] +
            6.0*u[k][j][i][4] - 4.0*u[k][j][i+1][4] +
            u[k][j][i+2][4] );
        }
      }

      for (j = 1; j <= gp12; j++) {
        i = gp0-3;
        rhs[k][j][i][0] = rhs[k][j][i][0] - dssp *
          ( u[k][j][i-2][0] - 4.0*u[k][j][i-1][0] +
          6.0*u[k][j][i][0] - 4.0*u[k][j][i+1][0] );
        rhs[k][j][i][1] = rhs[k][j][i][1] - dssp *
          ( u[k][j][i-2][1] - 4.0*u[k][j][i-1][1] +
          6.0*u[k][j][i][1] - 4.0*u[k][j][i+1][1] );
        rhs[k][j][i][2] = rhs[k][j][i][2] - dssp *
          ( u[k][j][i-2][2] - 4.0*u[k][j][i-1][2] +
          6.0*u[k][j][i][2] - 4.0*u[k][j][i+1][2] );
        rhs[k][j][i][3] = rhs[k][j][i][3] - dssp *
          ( u[k][j][i-2][3] - 4.0*u[k][j][i-1][3] +
          6.0*u[k][j][i][3] - 4.0*u[k][j][i+1][3] );
        rhs[k][j][i][4] = rhs[k][j][i][4] - dssp *
          ( u[k][j][i-2][4] - 4.0*u[k][j][i-1][4] +
          6.0*u[k][j][i][4] - 4.0*u[k][j][i+1][4] );

        i = gp02;
        rhs[k][j][i][0] = rhs[k][j][i][0] - dssp *
          ( u[k][j][i-2][0] - 4.*u[k][j][i-1][0] +
          5.*u[k][j][i][0] );
        rhs[k][j][i][1] = rhs[k][j][i][1] - dssp *
          ( u[k][j][i-2][1] - 4.*u[k][j][i-1][1] +
          5.*u[k][j][i][1] );
        rhs[k][j][i][2] = rhs[k][j][i][2] - dssp *
          ( u[k][j][i-2][2] - 4.*u[k][j][i-1][2] +
          5.*u[k][j][i][2] );
        rhs[k][j][i][3] = rhs[k][j][i][3] - dssp *
          ( u[k][j][i-2][3] - 4.*u[k][j][i-1][3] +
          5.*u[k][j][i][3] );
        rhs[k][j][i][4] = rhs[k][j][i][4] - dssp *
          ( u[k][j][i-2][4] - 4.*u[k][j][i-1][4] +
          5.*u[k][j][i][4] );
      }
    }
#endif

    //---------------------------------------------------------------------
    // compute eta-direction fluxes
    //---------------------------------------------------------------------
    size_t kernel_compute_rhs_3_off[1] = { 1 };
    size_t kernel_compute_rhs_3_idx[1] = { gp22 };
    brisbane_kernel kernel_compute_rhs_3;
    brisbane_kernel_create("compute_rhs_3", &kernel_compute_rhs_3);
    brisbane_kernel_setmem(kernel_compute_rhs_3, 0, mem_vs, brisbane_r);
    brisbane_kernel_setmem(kernel_compute_rhs_3, 1, mem_rhs, brisbane_rw);
    brisbane_kernel_setmem(kernel_compute_rhs_3, 2, mem_u, brisbane_r);
    brisbane_kernel_setmem(kernel_compute_rhs_3, 3, mem_us, brisbane_r);
    brisbane_kernel_setmem(kernel_compute_rhs_3, 4, mem_square, brisbane_r);
    brisbane_kernel_setmem(kernel_compute_rhs_3, 5, mem_ws, brisbane_r);
    brisbane_kernel_setmem(kernel_compute_rhs_3, 6, mem_qs, brisbane_r);
    brisbane_kernel_setmem(kernel_compute_rhs_3, 7, mem_rho_i, brisbane_r);
    brisbane_kernel_setarg(kernel_compute_rhs_3, 8, sizeof(double), &dy1ty1);
    brisbane_kernel_setarg(kernel_compute_rhs_3, 9, sizeof(double), &dy2ty1);
    brisbane_kernel_setarg(kernel_compute_rhs_3, 10, sizeof(double), &dy3ty1);
    brisbane_kernel_setarg(kernel_compute_rhs_3, 11, sizeof(double), &dy4ty1);
    brisbane_kernel_setarg(kernel_compute_rhs_3, 12, sizeof(double), &dy5ty1);
    brisbane_kernel_setarg(kernel_compute_rhs_3, 13, sizeof(double), &ty2);
    brisbane_kernel_setarg(kernel_compute_rhs_3, 14, sizeof(double), &yycon2);
    brisbane_kernel_setarg(kernel_compute_rhs_3, 15, sizeof(double), &yycon3);
    brisbane_kernel_setarg(kernel_compute_rhs_3, 16, sizeof(double), &yycon4);
    brisbane_kernel_setarg(kernel_compute_rhs_3, 17, sizeof(double), &yycon5);
    brisbane_kernel_setarg(kernel_compute_rhs_3, 18, sizeof(double), &con43);
    brisbane_kernel_setarg(kernel_compute_rhs_3, 19, sizeof(double), &c1);
    brisbane_kernel_setarg(kernel_compute_rhs_3, 20, sizeof(double), &c2);
    brisbane_kernel_setarg(kernel_compute_rhs_3, 21, sizeof(double), &dssp);
    brisbane_kernel_setarg(kernel_compute_rhs_3, 22, sizeof(int), &gp1);
    brisbane_kernel_setarg(kernel_compute_rhs_3, 23, sizeof(int), &gp02);
    brisbane_kernel_setarg(kernel_compute_rhs_3, 24, sizeof(int), &gp12);

    brisbane_task task3;
    brisbane_task_create(&task3);
    brisbane_task_kernel(task3, kernel_compute_rhs_3, 1, kernel_compute_rhs_3_off, kernel_compute_rhs_3_idx);
    brisbane_task_submit(task3, brisbane_cpu, NULL, true);
#if 0
    #pragma omp target teams distribute parallel for private(vijk,vp1,vm1,i,j,k) 
    for (k = 1; k <= gp22; k++) {
      for (j = 1; j <= gp12; j++) {
        #pragma omp simd private(vijk,vp1,vm1)
        for (i = 1; i <= gp02; i++) {
          vijk = vs[k][j][i];
          vp1  = vs[k][j+1][i];
          vm1  = vs[k][j-1][i];
          rhs[k][j][i][0] = rhs[k][j][i][0] + dy1ty1 *
            (u[k][j+1][i][0] - 2.0*u[k][j][i][0] +
            u[k][j-1][i][0]) -
            ty2 * (u[k][j+1][i][2] - u[k][j-1][i][2]);
          rhs[k][j][i][1] = rhs[k][j][i][1] + dy2ty1 *
            (u[k][j+1][i][1] - 2.0*u[k][j][i][1] +
            u[k][j-1][i][1]) +
            yycon2 * (us[k][j+1][i] - 2.0*us[k][j][i] +
            us[k][j-1][i]) -
            ty2 * (u[k][j+1][i][1]*vp1 -
            u[k][j-1][i][1]*vm1);
          rhs[k][j][i][2] = rhs[k][j][i][2] + dy3ty1 *
            (u[k][j+1][i][2] - 2.0*u[k][j][i][2] +
            u[k][j-1][i][2]) +
            yycon2*con43 * (vp1 - 2.0*vijk + vm1) -
            ty2 * (u[k][j+1][i][2]*vp1 -
            u[k][j-1][i][2]*vm1 +
            (u[k][j+1][i][4] - square[k][j+1][i] -
            u[k][j-1][i][4] + square[k][j-1][i])
          *c2);
          rhs[k][j][i][3] = rhs[k][j][i][3] + dy4ty1 *
            (u[k][j+1][i][3] - 2.0*u[k][j][i][3] +
            u[k][j-1][i][3]) +
            yycon2 * (ws[k][j+1][i] - 2.0*ws[k][j][i] +
            ws[k][j-1][i]) -
            ty2 * (u[k][j+1][i][3]*vp1 -
            u[k][j-1][i][3]*vm1);
          rhs[k][j][i][4] = rhs[k][j][i][4] + dy5ty1 *
            (u[k][j+1][i][4] - 2.0*u[k][j][i][4] +
            u[k][j-1][i][4]) +
            yycon3 * (qs[k][j+1][i] - 2.0*qs[k][j][i] +
            qs[k][j-1][i]) +
            yycon4 * (vp1*vp1       - 2.0*vijk*vijk +
            vm1*vm1) +
            yycon5 * (u[k][j+1][i][4]*rho_i[k][j+1][i] -
            2.0*u[k][j][i][4]*rho_i[k][j][i] +
            u[k][j-1][i][4]*rho_i[k][j-1][i]) -
            ty2 * ((c1*u[k][j+1][i][4] -
            c2*square[k][j+1][i]) * vp1 -
            (c1*u[k][j-1][i][4] -
            c2*square[k][j-1][i]) * vm1);
        }
      }

      //---------------------------------------------------------------------
      // add fourth order eta-direction dissipation
      //---------------------------------------------------------------------
      j = 1;
      #pragma omp simd
      for (i = 1; i <= gp02; i++) {
        rhs[k][j][i][0] = rhs[k][j][i][0]- dssp *
          ( 5.0*u[k][j][i][0] - 4.0*u[k][j+1][i][0] +
          u[k][j+2][i][0]);
        rhs[k][j][i][1] = rhs[k][j][i][1]- dssp *
          ( 5.0*u[k][j][i][1] - 4.0*u[k][j+1][i][1] +
          u[k][j+2][i][1]);
        rhs[k][j][i][2] = rhs[k][j][i][2]- dssp *
          ( 5.0*u[k][j][i][2] - 4.0*u[k][j+1][i][2] +
          u[k][j+2][i][2]);
        rhs[k][j][i][3] = rhs[k][j][i][3]- dssp *
          ( 5.0*u[k][j][i][3] - 4.0*u[k][j+1][i][3] +
          u[k][j+2][i][3]);
        rhs[k][j][i][4] = rhs[k][j][i][4]- dssp *
          ( 5.0*u[k][j][i][4] - 4.0*u[k][j+1][i][4] +
          u[k][j+2][i][4]);
      }

      j = 2;
      #pragma omp simd
      for (i = 1; i <= gp02; i++) {
        rhs[k][j][i][0] = rhs[k][j][i][0] - dssp *
          (-4.0*u[k][j-1][i][0] + 6.0*u[k][j][i][0] -
          4.0*u[k][j+1][i][0] + u[k][j+2][i][0]);
        rhs[k][j][i][1] = rhs[k][j][i][1] - dssp *
          (-4.0*u[k][j-1][i][1] + 6.0*u[k][j][i][1] -
          4.0*u[k][j+1][i][1] + u[k][j+2][i][1]);
        rhs[k][j][i][2] = rhs[k][j][i][2] - dssp *
          (-4.0*u[k][j-1][i][2] + 6.0*u[k][j][i][2] -
          4.0*u[k][j+1][i][2] + u[k][j+2][i][2]);
        rhs[k][j][i][3] = rhs[k][j][i][3] - dssp *
          (-4.0*u[k][j-1][i][3] + 6.0*u[k][j][i][3] -
          4.0*u[k][j+1][i][3] + u[k][j+2][i][3]);
        rhs[k][j][i][4] = rhs[k][j][i][4] - dssp *
          (-4.0*u[k][j-1][i][4] + 6.0*u[k][j][i][4] -
          4.0*u[k][j+1][i][4] + u[k][j+2][i][4]);
      }

      for (j = 3; j <= gp1-4; j++) {
        #pragma omp simd
        for (i = 1; i <= gp02; i++) {
          rhs[k][j][i][0] = rhs[k][j][i][0] - dssp *
            (  u[k][j-2][i][0] - 4.0*u[k][j-1][i][0] +
            6.0*u[k][j][i][0] - 4.0*u[k][j+1][i][0] +
            u[k][j+2][i][0] );
          rhs[k][j][i][1] = rhs[k][j][i][1] - dssp *
            (  u[k][j-2][i][1] - 4.0*u[k][j-1][i][1] +
            6.0*u[k][j][i][1] - 4.0*u[k][j+1][i][1] +
            u[k][j+2][i][1] );
          rhs[k][j][i][2] = rhs[k][j][i][2] - dssp *
            (  u[k][j-2][i][2] - 4.0*u[k][j-1][i][2] +
            6.0*u[k][j][i][2] - 4.0*u[k][j+1][i][2] +
            u[k][j+2][i][2] );
          rhs[k][j][i][3] = rhs[k][j][i][3] - dssp *
            (  u[k][j-2][i][3] - 4.0*u[k][j-1][i][3] +
            6.0*u[k][j][i][3] - 4.0*u[k][j+1][i][3] +
            u[k][j+2][i][3] );
          rhs[k][j][i][4] = rhs[k][j][i][4] - dssp *
            (  u[k][j-2][i][4] - 4.0*u[k][j-1][i][4] +
            6.0*u[k][j][i][4] - 4.0*u[k][j+1][i][4] +
            u[k][j+2][i][4] );
        }
      }

      j = gp1-3;
      #pragma omp simd
      for (i = 1; i <= gp02; i++) {
        rhs[k][j][i][0] = rhs[k][j][i][0] - dssp *
          ( u[k][j-2][i][0] - 4.0*u[k][j-1][i][0] +
          6.0*u[k][j][i][0] - 4.0*u[k][j+1][i][0] );
        rhs[k][j][i][1] = rhs[k][j][i][1] - dssp *
          ( u[k][j-2][i][1] - 4.0*u[k][j-1][i][1] +
          6.0*u[k][j][i][1] - 4.0*u[k][j+1][i][1] );
        rhs[k][j][i][2] = rhs[k][j][i][2] - dssp *
          ( u[k][j-2][i][2] - 4.0*u[k][j-1][i][2] +
          6.0*u[k][j][i][2] - 4.0*u[k][j+1][i][2] );
        rhs[k][j][i][3] = rhs[k][j][i][3] - dssp *
          ( u[k][j-2][i][3] - 4.0*u[k][j-1][i][3] +
          6.0*u[k][j][i][3] - 4.0*u[k][j+1][i][3] );
        rhs[k][j][i][4] = rhs[k][j][i][4] - dssp *
          ( u[k][j-2][i][4] - 4.0*u[k][j-1][i][4] +
          6.0*u[k][j][i][4] - 4.0*u[k][j+1][i][4] );
      }

      j = gp12;
      #pragma omp simd
      for (i = 1; i <= gp02; i++) {
        rhs[k][j][i][0] = rhs[k][j][i][0] - dssp *
          ( u[k][j-2][i][0] - 4.*u[k][j-1][i][0] +
          5.*u[k][j][i][0] );
        rhs[k][j][i][1] = rhs[k][j][i][1] - dssp *
          ( u[k][j-2][i][1] - 4.*u[k][j-1][i][1] +
          5.*u[k][j][i][1] );
        rhs[k][j][i][2] = rhs[k][j][i][2] - dssp *
          ( u[k][j-2][i][2] - 4.*u[k][j-1][i][2] +
          5.*u[k][j][i][2] );
        rhs[k][j][i][3] = rhs[k][j][i][3] - dssp *
          ( u[k][j-2][i][3] - 4.*u[k][j-1][i][3] +
          5.*u[k][j][i][3] );
        rhs[k][j][i][4] = rhs[k][j][i][4] - dssp *
          ( u[k][j-2][i][4] - 4.*u[k][j-1][i][4] +
          5.*u[k][j][i][4] );
      }
    }
#endif

    //---------------------------------------------------------------------
    // compute zeta-direction fluxes
    //---------------------------------------------------------------------
    size_t kernel_compute_rhs_4_off[2] = { 1, 1 };
    size_t kernel_compute_rhs_4_idx[2] = { gp12, gp22 };
    brisbane_kernel kernel_compute_rhs_4;
    brisbane_kernel_create("compute_rhs_4", &kernel_compute_rhs_4);
    brisbane_kernel_setmem(kernel_compute_rhs_4, 0, mem_ws, brisbane_r);
    brisbane_kernel_setmem(kernel_compute_rhs_4, 1, mem_rhs, brisbane_rw);
    brisbane_kernel_setmem(kernel_compute_rhs_4, 2, mem_u, brisbane_r);
    brisbane_kernel_setmem(kernel_compute_rhs_4, 3, mem_us, brisbane_r);
    brisbane_kernel_setmem(kernel_compute_rhs_4, 4, mem_vs, brisbane_r);
    brisbane_kernel_setmem(kernel_compute_rhs_4, 5, mem_square, brisbane_r);
    brisbane_kernel_setmem(kernel_compute_rhs_4, 6, mem_qs, brisbane_r);
    brisbane_kernel_setmem(kernel_compute_rhs_4, 7, mem_rho_i, brisbane_r);
    brisbane_kernel_setarg(kernel_compute_rhs_4, 8, sizeof(double), &dz1tz1);
    brisbane_kernel_setarg(kernel_compute_rhs_4, 9, sizeof(double), &dz2tz1);
    brisbane_kernel_setarg(kernel_compute_rhs_4, 10, sizeof(double), &dz3tz1);
    brisbane_kernel_setarg(kernel_compute_rhs_4, 11, sizeof(double), &dz4tz1);
    brisbane_kernel_setarg(kernel_compute_rhs_4, 12, sizeof(double), &dz5tz1);
    brisbane_kernel_setarg(kernel_compute_rhs_4, 13, sizeof(double), &tz2);
    brisbane_kernel_setarg(kernel_compute_rhs_4, 14, sizeof(double), &zzcon2);
    brisbane_kernel_setarg(kernel_compute_rhs_4, 15, sizeof(double), &zzcon3);
    brisbane_kernel_setarg(kernel_compute_rhs_4, 16, sizeof(double), &zzcon4);
    brisbane_kernel_setarg(kernel_compute_rhs_4, 17, sizeof(double), &zzcon5);
    brisbane_kernel_setarg(kernel_compute_rhs_4, 18, sizeof(double), &con43);
    brisbane_kernel_setarg(kernel_compute_rhs_4, 19, sizeof(double), &c1);
    brisbane_kernel_setarg(kernel_compute_rhs_4, 20, sizeof(double), &c2);
    brisbane_kernel_setarg(kernel_compute_rhs_4, 21, sizeof(int), &gp02);

    brisbane_task task4;
    brisbane_task_create(&task4);
    brisbane_task_kernel(task4, kernel_compute_rhs_4, 2, kernel_compute_rhs_4_off, kernel_compute_rhs_4_idx);
    brisbane_task_submit(task4, brisbane_cpu, NULL, true);
#if 0
#ifdef SPEC_USE_INNER_SIMD
    #pragma omp target teams distribute parallel for collapse(2) private(wijk,wp1,wm1,i,j,k)
#else
    #pragma omp target teams distribute parallel for simd collapse(3) private(wijk,wp1,wm1)
#endif
    for (k = 1; k <= gp22; k++) {
      for (j = 1; j <= gp12; j++) {
#ifdef SPEC_USE_INNER_SIMD
        #pragma omp simd private(wijk,wp1,wm1)
#endif
        for (i = 1; i <= gp02; i++) {
          wijk = ws[k][j][i];
          wp1  = ws[k+1][j][i];
          wm1  = ws[k-1][j][i];

          rhs[k][j][i][0] = rhs[k][j][i][0] + dz1tz1 *
            (u[k+1][j][i][0] - 2.0*u[k][j][i][0] +
            u[k-1][j][i][0]) -
            tz2 * (u[k+1][j][i][3] - u[k-1][j][i][3]);
          rhs[k][j][i][1] = rhs[k][j][i][1] + dz2tz1 *
            (u[k+1][j][i][1] - 2.0*u[k][j][i][1] +
            u[k-1][j][i][1]) +
            zzcon2 * (us[k+1][j][i] - 2.0*us[k][j][i] +
            us[k-1][j][i]) -
            tz2 * (u[k+1][j][i][1]*wp1 -
            u[k-1][j][i][1]*wm1);
          rhs[k][j][i][2] = rhs[k][j][i][2] + dz3tz1 *
            (u[k+1][j][i][2] - 2.0*u[k][j][i][2] +
            u[k-1][j][i][2]) +
            zzcon2 * (vs[k+1][j][i] - 2.0*vs[k][j][i] +
            vs[k-1][j][i]) -
            tz2 * (u[k+1][j][i][2]*wp1 -
            u[k-1][j][i][2]*wm1);
          rhs[k][j][i][3] = rhs[k][j][i][3] + dz4tz1 *
            (u[k+1][j][i][3] - 2.0*u[k][j][i][3] +
            u[k-1][j][i][3]) +
            zzcon2*con43 * (wp1 - 2.0*wijk + wm1) -
            tz2 * (u[k+1][j][i][3]*wp1 -
            u[k-1][j][i][3]*wm1 +
            (u[k+1][j][i][4] - square[k+1][j][i] -
            u[k-1][j][i][4] + square[k-1][j][i])
          *c2);
          rhs[k][j][i][4] = rhs[k][j][i][4] + dz5tz1 *
            (u[k+1][j][i][4] - 2.0*u[k][j][i][4] +
            u[k-1][j][i][4]) +
            zzcon3 * (qs[k+1][j][i] - 2.0*qs[k][j][i] +
            qs[k-1][j][i]) +
            zzcon4 * (wp1*wp1 - 2.0*wijk*wijk +
            wm1*wm1) +
            zzcon5 * (u[k+1][j][i][4]*rho_i[k+1][j][i] -
            2.0*u[k][j][i][4]*rho_i[k][j][i] +
            u[k-1][j][i][4]*rho_i[k-1][j][i]) -
            tz2 * ( (c1*u[k+1][j][i][4] -
            c2*square[k+1][j][i])*wp1 -
            (c1*u[k-1][j][i][4] -
            c2*square[k-1][j][i])*wm1);
        }
      }
    }
#endif

    //---------------------------------------------------------------------
    // add fourth order zeta-direction dissipation
    //---------------------------------------------------------------------
    k = 1;

    size_t kernel_compute_rhs_5_off[2] = { 1, 1 };
    size_t kernel_compute_rhs_5_idx[2] = { gp02, gp12 };
    brisbane_kernel kernel_compute_rhs_5;
    brisbane_kernel_create("compute_rhs_5", &kernel_compute_rhs_5);
    brisbane_kernel_setmem(kernel_compute_rhs_5, 0, mem_rhs, brisbane_rw);
    brisbane_kernel_setmem(kernel_compute_rhs_5, 1, mem_u, brisbane_r);
    brisbane_kernel_setarg(kernel_compute_rhs_5, 2, sizeof(double), &dssp);
    brisbane_kernel_setarg(kernel_compute_rhs_5, 3, sizeof(int), &k);

    brisbane_task task5;
    brisbane_task_create(&task5);
    brisbane_task_kernel(task5, kernel_compute_rhs_5, 2, kernel_compute_rhs_5_off, kernel_compute_rhs_5_idx);
    brisbane_task_submit(task5, brisbane_cpu, NULL, true);
#if 0
#ifdef SPEC_USE_INNER_SIMD
    #pragma omp target teams distribute parallel for private(i,j)
#else
    #pragma omp target teams distribute parallel for simd collapse(2)
#endif
    for (j = 1; j <= gp12; j++) {
#ifdef SPEC_USE_INNER_SIMD
      #pragma omp simd
#endif
      for (i = 1; i <= gp02; i++) {
        rhs[k][j][i][0] = rhs[k][j][i][0]- dssp *
          ( 5.0*u[k][j][i][0] - 4.0*u[k+1][j][i][0] +
          u[k+2][j][i][0]);
        rhs[k][j][i][1] = rhs[k][j][i][1]- dssp *
          ( 5.0*u[k][j][i][1] - 4.0*u[k+1][j][i][1] +
          u[k+2][j][i][1]);
        rhs[k][j][i][2] = rhs[k][j][i][2]- dssp *
          ( 5.0*u[k][j][i][2] - 4.0*u[k+1][j][i][2] +
          u[k+2][j][i][2]);
        rhs[k][j][i][3] = rhs[k][j][i][3]- dssp *
          ( 5.0*u[k][j][i][3] - 4.0*u[k+1][j][i][3] +
          u[k+2][j][i][3]);
        rhs[k][j][i][4] = rhs[k][j][i][4]- dssp *
          ( 5.0*u[k][j][i][4] - 4.0*u[k+1][j][i][4] +
          u[k+2][j][i][4]);
      }
    }
#endif

    k = 2;

    size_t kernel_compute_rhs_6_off[2] = { 1, 1 };
    size_t kernel_compute_rhs_6_idx[2] = { gp02, gp12 };
    brisbane_kernel kernel_compute_rhs_6;
    brisbane_kernel_create("compute_rhs_6", &kernel_compute_rhs_6);
    brisbane_kernel_setmem(kernel_compute_rhs_6, 0, mem_rhs, brisbane_rw);
    brisbane_kernel_setmem(kernel_compute_rhs_6, 1, mem_u, brisbane_r);
    brisbane_kernel_setarg(kernel_compute_rhs_6, 2, sizeof(double), &dssp);
    brisbane_kernel_setarg(kernel_compute_rhs_6, 3, sizeof(int), &k);

    brisbane_task task6;
    brisbane_task_create(&task6);
    brisbane_task_kernel(task6, kernel_compute_rhs_6, 2, kernel_compute_rhs_6_off, kernel_compute_rhs_6_idx);
    brisbane_task_submit(task6, brisbane_cpu, NULL, true);
#if 0
#ifdef SPEC_USE_INNER_SIMD
    #pragma omp target teams distribute parallel for private(i,j)
#else
    #pragma omp target teams distribute parallel for simd collapse(2)
#endif
    for (j = 1; j <= gp12; j++) {
#ifdef SPEC_USE_INNER_SIMD
      #pragma omp simd
#endif
      for (i = 1; i <= gp02; i++){
        rhs[k][j][i][0] = rhs[k][j][i][0] - dssp *
          (-4.0*u[k-1][j][i][0] + 6.0*u[k][j][i][0] -
          4.0*u[k+1][j][i][0] + u[k+2][j][i][0]);
        rhs[k][j][i][1] = rhs[k][j][i][1] - dssp *
          (-4.0*u[k-1][j][i][1] + 6.0*u[k][j][i][1] -
          4.0*u[k+1][j][i][1] + u[k+2][j][i][1]);
        rhs[k][j][i][2] = rhs[k][j][i][2] - dssp *
          (-4.0*u[k-1][j][i][2] + 6.0*u[k][j][i][2] -
          4.0*u[k+1][j][i][2] + u[k+2][j][i][2]);
        rhs[k][j][i][3] = rhs[k][j][i][3] - dssp *
          (-4.0*u[k-1][j][i][3] + 6.0*u[k][j][i][3] -
          4.0*u[k+1][j][i][3] + u[k+2][j][i][3]);
        rhs[k][j][i][4] = rhs[k][j][i][4] - dssp *
          (-4.0*u[k-1][j][i][4] + 6.0*u[k][j][i][4] -
          4.0*u[k+1][j][i][4] + u[k+2][j][i][4]);
      }
    }
#endif

    size_t kernel_compute_rhs_7_off[3] = { 1, 1, 3 };
    size_t kernel_compute_rhs_7_idx[3] = { gp02, gp12, gp2 - 6 };
    brisbane_kernel kernel_compute_rhs_7;
    brisbane_kernel_create("compute_rhs_7", &kernel_compute_rhs_7);
    brisbane_kernel_setmem(kernel_compute_rhs_7, 0, mem_rhs, brisbane_rw);
    brisbane_kernel_setmem(kernel_compute_rhs_7, 1, mem_u, brisbane_r);
    brisbane_kernel_setarg(kernel_compute_rhs_7, 2, sizeof(double), &dssp);

    brisbane_task task7;
    brisbane_task_create(&task7);
    brisbane_task_kernel(task7, kernel_compute_rhs_7, 3, kernel_compute_rhs_7_off, kernel_compute_rhs_7_idx);
    brisbane_task_submit(task7, brisbane_cpu, NULL, true);
#if 0
#ifdef SPEC_USE_INNER_SIMD
    #pragma omp target teams distribute parallel for collapse(2) private(i,j,k)
#else
    #pragma omp target teams distribute parallel for simd collapse(3)
#endif
    for (k = 3; k <= gp2-4; k++) {
      for (j = 1; j <= gp12; j++) {
#ifdef SPEC_USE_INNER_SIMD
        #pragma omp simd
#endif
        for (i = 1; i <= gp02; i++) {
          rhs[k][j][i][0] = rhs[k][j][i][0] - dssp *
            (  u[k-2][j][i][0] - 4.0*u[k-1][j][i][0] +
            6.0*u[k][j][i][0] - 4.0*u[k+1][j][i][0] +
            u[k+2][j][i][0] );
          rhs[k][j][i][1] = rhs[k][j][i][1] - dssp *
            (  u[k-2][j][i][1] - 4.0*u[k-1][j][i][1] +
            6.0*u[k][j][i][1] - 4.0*u[k+1][j][i][1] +
            u[k+2][j][i][1] );
          rhs[k][j][i][2] = rhs[k][j][i][2] - dssp *
            (  u[k-2][j][i][2] - 4.0*u[k-1][j][i][2] +
            6.0*u[k][j][i][2] - 4.0*u[k+1][j][i][2] +
            u[k+2][j][i][2] );
          rhs[k][j][i][3] = rhs[k][j][i][3] - dssp *
            (  u[k-2][j][i][3] - 4.0*u[k-1][j][i][3] +
            6.0*u[k][j][i][3] - 4.0*u[k+1][j][i][3] +
            u[k+2][j][i][3] );
          rhs[k][j][i][4] = rhs[k][j][i][4] - dssp *
            (  u[k-2][j][i][4] - 4.0*u[k-1][j][i][4] +
            6.0*u[k][j][i][4] - 4.0*u[k+1][j][i][4] +
            u[k+2][j][i][4] );
        }
      }
    }
#endif

    k = gp2-3;

    size_t kernel_compute_rhs_8_off[2] = { 1, 1 };
    size_t kernel_compute_rhs_8_idx[2] = { gp02, gp12 };
    brisbane_kernel kernel_compute_rhs_8;
    brisbane_kernel_create("compute_rhs_8", &kernel_compute_rhs_8);
    brisbane_kernel_setmem(kernel_compute_rhs_8, 0, mem_rhs, brisbane_rw);
    brisbane_kernel_setmem(kernel_compute_rhs_8, 1, mem_u, brisbane_r);
    brisbane_kernel_setarg(kernel_compute_rhs_8, 2, sizeof(double), &dssp);
    brisbane_kernel_setarg(kernel_compute_rhs_8, 3, sizeof(int), &k);

    brisbane_task task8;
    brisbane_task_create(&task8);
    brisbane_task_kernel(task8, kernel_compute_rhs_8, 2, kernel_compute_rhs_8_off, kernel_compute_rhs_8_idx);
    brisbane_task_submit(task8, brisbane_cpu, NULL, true);
#if 0
#ifdef SPEC_USE_INNER_SIMD
    #pragma omp target teams distribute parallel for private(i,j)
#else
    #pragma omp target teams distribute parallel for simd collapse(2)
#endif
    for (j = 1; j <= gp12; j++) {
#ifdef SPEC_USE_INNER_SIMD
      #pragma omp simd
#endif
      for (i = 1; i <= gp02; i++) {
        rhs[k][j][i][0] = rhs[k][j][i][0] - dssp *
          ( u[k-2][j][i][0] - 4.0*u[k-1][j][i][0] +
          6.0*u[k][j][i][0] - 4.0*u[k+1][j][i][0] );
        rhs[k][j][i][1] = rhs[k][j][i][1] - dssp *
          ( u[k-2][j][i][1] - 4.0*u[k-1][j][i][1] +
          6.0*u[k][j][i][1] - 4.0*u[k+1][j][i][1] );
        rhs[k][j][i][2] = rhs[k][j][i][2] - dssp *
          ( u[k-2][j][i][2] - 4.0*u[k-1][j][i][2] +
          6.0*u[k][j][i][2] - 4.0*u[k+1][j][i][2] );
        rhs[k][j][i][3] = rhs[k][j][i][3] - dssp *
          ( u[k-2][j][i][3] - 4.0*u[k-1][j][i][3] +
          6.0*u[k][j][i][3] - 4.0*u[k+1][j][i][3] );
        rhs[k][j][i][4] = rhs[k][j][i][4] - dssp *
          ( u[k-2][j][i][4] - 4.0*u[k-1][j][i][4] +
          6.0*u[k][j][i][4] - 4.0*u[k+1][j][i][4] );
      }
    }
#endif

    k = gp22;

    size_t kernel_compute_rhs_9_off[2] = { 1, 1 };
    size_t kernel_compute_rhs_9_idx[2] = { gp02, gp12 };
    brisbane_kernel kernel_compute_rhs_9;
    brisbane_kernel_create("compute_rhs_9", &kernel_compute_rhs_9);
    brisbane_kernel_setmem(kernel_compute_rhs_9, 0, mem_rhs, brisbane_rw);
    brisbane_kernel_setmem(kernel_compute_rhs_9, 1, mem_u, brisbane_r);
    brisbane_kernel_setarg(kernel_compute_rhs_9, 2, sizeof(double), &dssp);
    brisbane_kernel_setarg(kernel_compute_rhs_9, 3, sizeof(int), &k);

    brisbane_task task9;
    brisbane_task_create(&task9);
    brisbane_task_kernel(task9, kernel_compute_rhs_9, 2, kernel_compute_rhs_9_off, kernel_compute_rhs_9_idx);
    brisbane_task_submit(task9, brisbane_cpu, NULL, true);
#if 0
#ifdef SPEC_USE_INNER_SIMD
    #pragma omp target teams distribute parallel for private(i,j)
#else
    #pragma omp target teams distribute parallel for simd collapse(2)
#endif
    for (j = 1; j <= gp12; j++) {
#ifdef SPEC_USE_INNER_SIMD
      #pragma omp simd
#endif
      for (i = 1; i <= gp02; i++) {
        rhs[k][j][i][0] = rhs[k][j][i][0] - dssp *
          ( u[k-2][j][i][0] - 4.*u[k-1][j][i][0] +
          5.*u[k][j][i][0] );
        rhs[k][j][i][1] = rhs[k][j][i][1] - dssp *
          ( u[k-2][j][i][1] - 4.*u[k-1][j][i][1] +
          5.*u[k][j][i][1] );
        rhs[k][j][i][2] = rhs[k][j][i][2] - dssp *
          ( u[k-2][j][i][2] - 4.*u[k-1][j][i][2] +
          5.*u[k][j][i][2] );
        rhs[k][j][i][3] = rhs[k][j][i][3] - dssp *
          ( u[k-2][j][i][3] - 4.*u[k-1][j][i][3] +
          5.*u[k][j][i][3] );
        rhs[k][j][i][4] = rhs[k][j][i][4] - dssp *
          ( u[k-2][j][i][4] - 4.*u[k-1][j][i][4] +
          5.*u[k][j][i][4] );
      }
    }
#endif

    size_t kernel_compute_rhs_10_off[3] = { 1, 1, 1 };
    size_t kernel_compute_rhs_10_idx[3] = { gp02, gp12, gp22 };
    brisbane_kernel kernel_compute_rhs_10;
    brisbane_kernel_create("compute_rhs_10", &kernel_compute_rhs_10);
    brisbane_kernel_setmem(kernel_compute_rhs_10, 0, mem_rhs, brisbane_rw);
    brisbane_kernel_setarg(kernel_compute_rhs_10, 1, sizeof(double), &dt);

    brisbane_task task10;
    brisbane_task_create(&task10);
    brisbane_task_kernel(task10, kernel_compute_rhs_10, 3, kernel_compute_rhs_10_off, kernel_compute_rhs_10_idx);
    brisbane_task_submit(task10, brisbane_cpu, NULL, true);
#if 0
#ifdef SPEC_USE_INNER_SIMD
    #pragma omp target teams distribute parallel for collapse(2) private(i,j,k)
#else
    #pragma omp target teams distribute parallel for simd collapse(3)
#endif
    for (k = 1; k <= gp22; k++) {
      for (j = 1; j <= gp12; j++) {
#ifdef SPEC_USE_INNER_SIMD
    #pragma omp simd
#endif
        for (i = 1; i <= gp02; i++) {
          rhs[k][j][i][0] = rhs[k][j][i][0] * dt;
          rhs[k][j][i][1] = rhs[k][j][i][1] * dt;
          rhs[k][j][i][2] = rhs[k][j][i][2] * dt;
          rhs[k][j][i][3] = rhs[k][j][i][3] * dt;
          rhs[k][j][i][4] = rhs[k][j][i][4] * dt;
        }
      }
    }
#endif
}
