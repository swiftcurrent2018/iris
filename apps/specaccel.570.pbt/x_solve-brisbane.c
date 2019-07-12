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
#include "work_lhs.h"
//#include "timers.h"

//---------------------------------------------------------------------
//
// Performs line solves in X direction by first factoring
// the block-tridiagonal matrix into an upper triangular matrix,
// and then performing back substitution to solve for the unknow
// vectors of each line.
//
// Make sure we treat elements zero to cell_size in the direction
// of the sweep.
//
//---------------------------------------------------------------------
void x_solve()
{
  int i, j, k, m, n, isize, z;
  //  double pivot, coeff;
  int gp22, gp12;
  //  double temp1, temp2, temp3;
  double fjacX[5][5][PROBLEM_SIZE+1][JMAXP-1][KMAX-1];
  double njacX[5][5][PROBLEM_SIZE+1][JMAXP-1][KMAX-1];
  double lhsX[5][5][3][PROBLEM_SIZE][JMAXP-1][KMAX-1];
  double pivot,coeff,temp1, temp2, temp3;

  gp22 = grid_points[2]-2;
  gp12 = grid_points[1]-2;

  //---------------------------------------------------------------------
  // This function computes the left hand side in the xi-direction
  //---------------------------------------------------------------------

  isize = grid_points[0]-1;

  brisbane_mem mem_fjacX;
  brisbane_mem mem_njacX;
  brisbane_mem mem_lhsX;
  brisbane_mem_create(sizeof(double) * 5 * 5 * (PROBLEM_SIZE + 1) * (JMAXP - 1) * (KMAX - 1), &mem_fjacX);
  brisbane_mem_create(sizeof(double) * 5 * 5 * (PROBLEM_SIZE + 1) * (JMAXP - 1) * (KMAX - 1), &mem_njacX);
  brisbane_mem_create(sizeof(double) * 5 * 5 * 3 * (PROBLEM_SIZE) * (JMAXP - 1) * (KMAX - 1), &mem_lhsX);

  //---------------------------------------------------------------------
  // determine a (labeled f) and n jacobians
  //---------------------------------------------------------------------
  #pragma omp target data map(alloc:fjacX[:][:][:][:][:],njacX[:][:][:][:][:],lhsX[:][:][:][:][:][:])
  //present(rho_i,u,qs,rhs,square)
  {
    size_t kernel_x_solve_0_off[2] = { 1, 0 };
    size_t kernel_x_solve_0_idx[2] = { gp12, isize + 1 };
    brisbane_kernel kernel_x_solve_0;
    brisbane_kernel_create("x_solve_0", &kernel_x_solve_0);
    brisbane_kernel_setmem(kernel_x_solve_0, 0, mem_rho_i, brisbane_r);
    brisbane_kernel_setmem(kernel_x_solve_0, 1, mem_fjacX, brisbane_w);
    brisbane_kernel_setmem(kernel_x_solve_0, 2, mem_njacX, brisbane_w);
    brisbane_kernel_setmem(kernel_x_solve_0, 3, mem_u, brisbane_r);
    brisbane_kernel_setmem(kernel_x_solve_0, 4, mem_qs, brisbane_r);
    brisbane_kernel_setmem(kernel_x_solve_0, 5, mem_square, brisbane_r);
    brisbane_kernel_setarg(kernel_x_solve_0, 6, sizeof(double), &c1);
    brisbane_kernel_setarg(kernel_x_solve_0, 7, sizeof(double), &c2);
    brisbane_kernel_setarg(kernel_x_solve_0, 8, sizeof(double), &c3c4);
    brisbane_kernel_setarg(kernel_x_solve_0, 9, sizeof(double), &c1345);
    brisbane_kernel_setarg(kernel_x_solve_0, 10, sizeof(double), &con43);
    brisbane_kernel_setarg(kernel_x_solve_0, 11, sizeof(int), &gp22);

    brisbane_task task0;
    brisbane_task_create(&task0);
    brisbane_task_kernel(task0, kernel_x_solve_0, 2, kernel_x_solve_0_off, kernel_x_solve_0_idx);
    brisbane_task_submit(task0, brisbane_cpu, NULL, true);
#if 0
    #pragma omp target teams distribute parallel for collapse(2) private(temp1,temp2,temp3,i,j,k)
    for (i = 0; i <= isize; i++) {
      for (j = 1; j <= gp12; j++) {
        for (k = 1; k <= gp22; k++) {
          temp1 = rho_i[k][j][i];
          temp2 = temp1 * temp1;
          temp3 = temp1 * temp2;
          //-------------------------------------------------------------------
          //
          //-------------------------------------------------------------------
          fjacX[0][0][i][j][k] = 0.0;
          fjacX[0][1][i][j][k] = 1.0;
          fjacX[0][2][i][j][k] = 0.0;
          fjacX[0][3][i][j][k] = 0.0;
          fjacX[0][4][i][j][k] = 0.0;

          fjacX[1][0][i][j][k] = -(u[k][j][i][1] * temp2 * u[k][j][i][1])
            + c2 * qs[k][j][i];
          fjacX[1][1][i][j][k] = ( 2.0 - c2 ) * ( u[k][j][i][1] / u[k][j][i][0] );
          fjacX[1][2][i][j][k] = - c2 * ( u[k][j][i][2] * temp1 );
          fjacX[1][3][i][j][k] = - c2 * ( u[k][j][i][3] * temp1 );
          fjacX[1][4][i][j][k] = c2;

          fjacX[2][0][i][j][k] = - ( u[k][j][i][1]*u[k][j][i][2] ) * temp2;
          fjacX[2][1][i][j][k] = u[k][j][i][2] * temp1;
          fjacX[2][2][i][j][k] = u[k][j][i][1] * temp1;
          fjacX[2][3][i][j][k] = 0.0;
          fjacX[2][4][i][j][k] = 0.0;

          fjacX[3][0][i][j][k] = - ( u[k][j][i][1]*u[k][j][i][3] ) * temp2;
          fjacX[3][1][i][j][k] = u[k][j][i][3] * temp1;
          fjacX[3][2][i][j][k] = 0.0;
          fjacX[3][3][i][j][k] = u[k][j][i][1] * temp1;
          fjacX[3][4][i][j][k] = 0.0;

          fjacX[4][0][i][j][k] = ( c2 * 2.0 * square[k][j][i] - c1 * u[k][j][i][4] )
            * ( u[k][j][i][1] * temp2 );
          fjacX[4][1][i][j][k] = c1 *  u[k][j][i][4] * temp1
            - c2 * ( u[k][j][i][1]*u[k][j][i][1] * temp2 + qs[k][j][i] );
          fjacX[4][2][i][j][k] = - c2 * ( u[k][j][i][2]*u[k][j][i][1] ) * temp2;
          fjacX[4][3][i][j][k] = - c2 * ( u[k][j][i][3]*u[k][j][i][1] ) * temp2;
          fjacX[4][4][i][j][k] = c1 * ( u[k][j][i][1] * temp1 );

          njacX[0][0][i][j][k] = 0.0;
          njacX[0][1][i][j][k] = 0.0;
          njacX[0][2][i][j][k] = 0.0;
          njacX[0][3][i][j][k] = 0.0;
          njacX[0][4][i][j][k] = 0.0;

          njacX[1][0][i][j][k] = - con43 * c3c4 * temp2 * u[k][j][i][1];
          njacX[1][1][i][j][k] =   con43 * c3c4 * temp1;
          njacX[1][2][i][j][k] =   0.0;
          njacX[1][3][i][j][k] =   0.0;
          njacX[1][4][i][j][k] =   0.0;

          njacX[2][0][i][j][k] = - c3c4 * temp2 * u[k][j][i][2];
          njacX[2][1][i][j][k] =   0.0;
          njacX[2][2][i][j][k] =   c3c4 * temp1;
          njacX[2][3][i][j][k] =   0.0;
          njacX[2][4][i][j][k] =   0.0;

          njacX[3][0][i][j][k] = - c3c4 * temp2 * u[k][j][i][3];
          njacX[3][1][i][j][k] =   0.0;
          njacX[3][2][i][j][k] =   0.0;
          njacX[3][3][i][j][k] =   c3c4 * temp1;
          njacX[3][4][i][j][k] =   0.0;

          njacX[4][0][i][j][k] = - ( con43 * c3c4
            - c1345 ) * temp3 * (u[k][j][i][1]*u[k][j][i][1])
            - ( c3c4 - c1345 ) * temp3 * (u[k][j][i][2]*u[k][j][i][2])
            - ( c3c4 - c1345 ) * temp3 * (u[k][j][i][3]*u[k][j][i][3])
            - c1345 * temp2 * u[k][j][i][4];

          njacX[4][1][i][j][k] = ( con43 * c3c4
            - c1345 ) * temp2 * u[k][j][i][1];
          njacX[4][2][i][j][k] = ( c3c4 - c1345 ) * temp2 * u[k][j][i][2];
          njacX[4][3][i][j][k] = ( c3c4 - c1345 ) * temp2 * u[k][j][i][3];
          njacX[4][4][i][j][k] = ( c1345 ) * temp1;
        }
      }
    }
#endif

    //---------------------------------------------------------------------
    // now jacobians set, so form left hand side in x direction
    //---------------------------------------------------------------------
    //    lhsX[k][j]init(lhsX[k][j], isize);
    // zero the whole left hand side for starters
    size_t kernel_x_solve_1_off[3] = { 0, 1, 1 };
    size_t kernel_x_solve_1_idx[3] = { 5, gp12, gp22 };
    brisbane_kernel kernel_x_solve_1;
    brisbane_kernel_create("x_solve_1", &kernel_x_solve_1);
    brisbane_kernel_setmem(kernel_x_solve_1, 0, mem_lhsX, brisbane_w);
    brisbane_kernel_setarg(kernel_x_solve_1, 1, sizeof(int), &isize);

    brisbane_task task1;
    brisbane_task_create(&task1);
    brisbane_task_kernel(task1, kernel_x_solve_1, 3, kernel_x_solve_1_off, kernel_x_solve_1_idx);
    brisbane_task_submit(task1, brisbane_cpu, NULL, true);
#if 0
#ifdef SPEC_USE_INNER_SIMD
  #pragma omp target teams distribute parallel for collapse(3) private(k,j,n,m)
#else
  #pragma omp target teams distribute parallel for simd collapse(4)
#endif
  for (k = 1; k <= gp22; k++) {
     for (j = 1; j <= gp12; j++) {
       for (n = 0; n < 5; n++) {
#ifdef SPEC_USE_INNER_SIMD
  #pragma omp simd
#endif
         for (m = 0; m < 5; m++){
            lhsX[m][n][0][0][j][k] = 0.0;
            lhsX[m][n][1][0][j][k] = 0.0;
            lhsX[m][n][2][0][j][k] = 0.0;
            lhsX[m][n][0][isize][j][k] = 0.0;
            lhsX[m][n][1][isize][j][k] = 0.0;
            lhsX[m][n][2][isize][j][k] = 0.0;
          }
        }
      }
    }
#endif

    // next, set all diagonal values to 1. This is overkill, but convenient
    size_t kernel_x_solve_2_off[2] = { 1, 1 };
    size_t kernel_x_solve_2_idx[2] = { gp12, gp22 };
    brisbane_kernel kernel_x_solve_2;
    brisbane_kernel_create("x_solve_2", &kernel_x_solve_2);
    brisbane_kernel_setmem(kernel_x_solve_2, 0, mem_lhsX, brisbane_w);
    brisbane_kernel_setarg(kernel_x_solve_2, 1, sizeof(int), &isize);

    brisbane_task task2;
    brisbane_task_create(&task2);
    brisbane_task_kernel(task2, kernel_x_solve_2, 2, kernel_x_solve_2_off, kernel_x_solve_2_idx);
    brisbane_task_submit(task2, brisbane_cpu, NULL, true);
#if 0
#ifdef SPEC_USE_INNER_SIMD
  #pragma omp target teams distribute parallel for private(k,j) // collapse(2)
#else
  #pragma omp target teams distribute parallel for simd collapse(2)
#endif
  for (k = 1; k <= gp22; k++) {
#ifdef SPEC_USE_INNER_SIMD
  #pragma omp simd
#endif
      for (j = 1; j <= gp12; j++) {
    		lhsX[0][0][1][0][j][k] = 1.0;
    		lhsX[0][0][1][isize][j][k] = 1.0;
    		lhsX[1][1][1][0][j][k] = 1.0;
    		lhsX[1][1][1][isize][j][k] = 1.0;
    		lhsX[2][2][1][0][j][k] = 1.0;
    		lhsX[2][2][1][isize][j][k] = 1.0;
    		lhsX[3][3][1][0][j][k] = 1.0;
    		lhsX[3][3][1][isize][j][k] = 1.0;
    		lhsX[4][4][1][0][j][k] = 1.0;
    		lhsX[4][4][1][isize][j][k] = 1.0;
      }
    }
#endif

    size_t kernel_x_solve_3_off[3] = { 1, 1, 1 };
    size_t kernel_x_solve_3_idx[3] = { gp22, gp12, isize - 1 };
    brisbane_kernel kernel_x_solve_3;
    brisbane_kernel_create("x_solve_3", &kernel_x_solve_3);
    brisbane_kernel_setmem(kernel_x_solve_3, 0, mem_lhsX, brisbane_w);
    brisbane_kernel_setmem(kernel_x_solve_3, 1, mem_fjacX, brisbane_r);
    brisbane_kernel_setmem(kernel_x_solve_3, 2, mem_njacX, brisbane_r);
    brisbane_kernel_setarg(kernel_x_solve_3, 3, sizeof(double), &dt);
    brisbane_kernel_setarg(kernel_x_solve_3, 4, sizeof(double), &tx1);
    brisbane_kernel_setarg(kernel_x_solve_3, 5, sizeof(double), &tx2);
    brisbane_kernel_setarg(kernel_x_solve_3, 6, sizeof(double), &dx1);
    brisbane_kernel_setarg(kernel_x_solve_3, 7, sizeof(double), &dx2);
    brisbane_kernel_setarg(kernel_x_solve_3, 8, sizeof(double), &dx3);
    brisbane_kernel_setarg(kernel_x_solve_3, 9, sizeof(double), &dx4);
    brisbane_kernel_setarg(kernel_x_solve_3, 10, sizeof(double), &dx5);

    brisbane_task task3;
    brisbane_task_create(&task3);
    brisbane_task_kernel(task3, kernel_x_solve_3, 3, kernel_x_solve_3_off, kernel_x_solve_3_idx);
    brisbane_task_submit(task3, brisbane_cpu, NULL, true);
#if 0
#ifdef SPEC_USE_INNER_SIMD
  #pragma omp target teams distribute parallel for collapse(2) private(j,k)
#else
  #pragma omp target teams distribute parallel for simd collapse(3) private(temp1,temp2)
#endif
  for (i = 1; i <= isize-1; i++) {
    for (j = 1; j <= gp12; j++) {
#ifdef SPEC_USE_INNER_SIMD
      #pragma omp simd private(temp1, temp2)
#endif
      for (k = 1; k <= gp22; k++) {
        temp1 = dt * tx1;
        temp2 = dt * tx2;

        lhsX[0][0][AA][i][j][k] = - temp2 * fjacX[0][0][i-1][j][k]
          - temp1 * njacX[0][0][i-1][j][k]
          - temp1 * dx1; 
        lhsX[0][1][AA][i][j][k] = - temp2 * fjacX[0][1][i-1][j][k]
          - temp1 * njacX[0][1][i-1][j][k];
        lhsX[0][2][AA][i][j][k] = - temp2 * fjacX[0][2][i-1][j][k]
          - temp1 * njacX[0][2][i-1][j][k];
        lhsX[0][3][AA][i][j][k] = - temp2 * fjacX[0][3][i-1][j][k]
          - temp1 * njacX[0][3][i-1][j][k];
        lhsX[0][4][AA][i][j][k] = - temp2 * fjacX[0][4][i-1][j][k]
          - temp1 * njacX[0][4][i-1][j][k];

        lhsX[1][0][AA][i][j][k] = - temp2 * fjacX[1][0][i-1][j][k]
          - temp1 * njacX[1][0][i-1][j][k];
        lhsX[1][1][AA][i][j][k] = - temp2 * fjacX[1][1][i-1][j][k]
          - temp1 * njacX[1][1][i-1][j][k]
          - temp1 * dx2;
        lhsX[1][2][AA][i][j][k] = - temp2 * fjacX[1][2][i-1][j][k]
          - temp1 * njacX[1][2][i-1][j][k];
        lhsX[1][3][AA][i][j][k] = - temp2 * fjacX[1][3][i-1][j][k]
          - temp1 * njacX[1][3][i-1][j][k];
        lhsX[1][4][AA][i][j][k] = - temp2 * fjacX[1][4][i-1][j][k]
          - temp1 * njacX[1][4][i-1][j][k];

        lhsX[2][0][AA][i][j][k] = - temp2 * fjacX[2][0][i-1][j][k]
          - temp1 * njacX[2][0][i-1][j][k];
        lhsX[2][1][AA][i][j][k] = - temp2 * fjacX[2][1][i-1][j][k]
          - temp1 * njacX[2][1][i-1][j][k];
        lhsX[2][2][AA][i][j][k] = - temp2 * fjacX[2][2][i-1][j][k]
          - temp1 * njacX[2][2][i-1][j][k]
          - temp1 * dx3;
        lhsX[2][3][AA][i][j][k] = - temp2 * fjacX[2][3][i-1][j][k]
          - temp1 * njacX[2][3][i-1][j][k];
        lhsX[2][4][AA][i][j][k] = - temp2 * fjacX[2][4][i-1][j][k]
          - temp1 * njacX[2][4][i-1][j][k];

        lhsX[3][0][AA][i][j][k] = - temp2 * fjacX[3][0][i-1][j][k]
          - temp1 * njacX[3][0][i-1][j][k];
        lhsX[3][1][AA][i][j][k] = - temp2 * fjacX[3][1][i-1][j][k]
          - temp1 * njacX[3][1][i-1][j][k];
        lhsX[3][2][AA][i][j][k] = - temp2 * fjacX[3][2][i-1][j][k]
          - temp1 * njacX[3][2][i-1][j][k];
        lhsX[3][3][AA][i][j][k] = - temp2 * fjacX[3][3][i-1][j][k]
          - temp1 * njacX[3][3][i-1][j][k]
          - temp1 * dx4;
        lhsX[3][4][AA][i][j][k] = - temp2 * fjacX[3][4][i-1][j][k]
          - temp1 * njacX[3][4][i-1][j][k];

        lhsX[4][0][AA][i][j][k] = - temp2 * fjacX[4][0][i-1][j][k]
          - temp1 * njacX[4][0][i-1][j][k];
        lhsX[4][1][AA][i][j][k] = - temp2 * fjacX[4][1][i-1][j][k]
          - temp1 * njacX[4][1][i-1][j][k];
        lhsX[4][2][AA][i][j][k] = - temp2 * fjacX[4][2][i-1][j][k]
          - temp1 * njacX[4][2][i-1][j][k];
        lhsX[4][3][AA][i][j][k] = - temp2 * fjacX[4][3][i-1][j][k]
          - temp1 * njacX[4][3][i-1][j][k];
        lhsX[4][4][AA][i][j][k] = - temp2 * fjacX[4][4][i-1][j][k]
          - temp1 * njacX[4][4][i-1][j][k]
          - temp1 * dx5;

          lhsX[0][0][BB][i][j][k] = 1.0
          + temp1 * 2.0 * njacX[0][0][i][j][k]
          + temp1 * 2.0 * dx1;
        lhsX[0][1][BB][i][j][k] = temp1 * 2.0 * njacX[0][1][i][j][k];
        lhsX[0][2][BB][i][j][k] = temp1 * 2.0 * njacX[0][2][i][j][k];
        lhsX[0][3][BB][i][j][k] = temp1 * 2.0 * njacX[0][3][i][j][k];
        lhsX[0][4][BB][i][j][k] = temp1 * 2.0 * njacX[0][4][i][j][k];

        lhsX[1][0][BB][i][j][k] = temp1 * 2.0 * njacX[1][0][i][j][k];
          lhsX[1][1][BB][i][j][k] = 1.0
          + temp1 * 2.0 * njacX[1][1][i][j][k]
          + temp1 * 2.0 * dx2;
        lhsX[1][2][BB][i][j][k] = temp1 * 2.0 * njacX[1][2][i][j][k];
        lhsX[1][3][BB][i][j][k] = temp1 * 2.0 * njacX[1][3][i][j][k];
        lhsX[1][4][BB][i][j][k] = temp1 * 2.0 * njacX[1][4][i][j][k];

        lhsX[2][0][BB][i][j][k] = temp1 * 2.0 * njacX[2][0][i][j][k];
        lhsX[2][1][BB][i][j][k] = temp1 * 2.0 * njacX[2][1][i][j][k];
          lhsX[2][2][BB][i][j][k] = 1.0
          + temp1 * 2.0 * njacX[2][2][i][j][k]
          + temp1 * 2.0 * dx3;
        lhsX[2][3][BB][i][j][k] = temp1 * 2.0 * njacX[2][3][i][j][k];
        lhsX[2][4][BB][i][j][k] = temp1 * 2.0 * njacX[2][4][i][j][k];

        lhsX[3][0][BB][i][j][k] = temp1 * 2.0 * njacX[3][0][i][j][k];
        lhsX[3][1][BB][i][j][k] = temp1 * 2.0 * njacX[3][1][i][j][k];
        lhsX[3][2][BB][i][j][k] = temp1 * 2.0 * njacX[3][2][i][j][k];
          lhsX[3][3][BB][i][j][k] = 1.0
          + temp1 * 2.0 * njacX[3][3][i][j][k]
          + temp1 * 2.0 * dx4;
        lhsX[3][4][BB][i][j][k] = temp1 * 2.0 * njacX[3][4][i][j][k];

        lhsX[4][0][BB][i][j][k] = temp1 * 2.0 * njacX[4][0][i][j][k];
        lhsX[4][1][BB][i][j][k] = temp1 * 2.0 * njacX[4][1][i][j][k];
        lhsX[4][2][BB][i][j][k] = temp1 * 2.0 * njacX[4][2][i][j][k];
        lhsX[4][3][BB][i][j][k] = temp1 * 2.0 * njacX[4][3][i][j][k];
          lhsX[4][4][BB][i][j][k] = 1.0
          + temp1 * 2.0 * njacX[4][4][i][j][k]
          + temp1 * 2.0 * dx5;

        lhsX[0][0][CC][i][j][k] =  temp2 * fjacX[0][0][i+1][j][k]
          - temp1 * njacX[0][0][i+1][j][k]
          - temp1 * dx1;
        lhsX[0][1][CC][i][j][k] =  temp2 * fjacX[0][1][i+1][j][k]
          - temp1 * njacX[0][1][i+1][j][k];
        lhsX[0][2][CC][i][j][k] =  temp2 * fjacX[0][2][i+1][j][k]
          - temp1 * njacX[0][2][i+1][j][k];
        lhsX[0][3][CC][i][j][k] =  temp2 * fjacX[0][3][i+1][j][k]
          - temp1 * njacX[0][3][i+1][j][k];
        lhsX[0][4][CC][i][j][k] =  temp2 * fjacX[0][4][i+1][j][k]
          - temp1 * njacX[0][4][i+1][j][k];

        lhsX[1][0][CC][i][j][k] =  temp2 * fjacX[1][0][i+1][j][k]
          - temp1 * njacX[1][0][i+1][j][k];
        lhsX[1][1][CC][i][j][k] =  temp2 * fjacX[1][1][i+1][j][k]
          - temp1 * njacX[1][1][i+1][j][k]
          - temp1 * dx2;
        lhsX[1][2][CC][i][j][k] =  temp2 * fjacX[1][2][i+1][j][k]
          - temp1 * njacX[1][2][i+1][j][k];
        lhsX[1][3][CC][i][j][k] =  temp2 * fjacX[1][3][i+1][j][k]
          - temp1 * njacX[1][3][i+1][j][k];
        lhsX[1][4][CC][i][j][k] =  temp2 * fjacX[1][4][i+1][j][k]
          - temp1 * njacX[1][4][i+1][j][k];

        lhsX[2][0][CC][i][j][k] =  temp2 * fjacX[2][0][i+1][j][k]
          - temp1 * njacX[2][0][i+1][j][k];
        lhsX[2][1][CC][i][j][k] =  temp2 * fjacX[2][1][i+1][j][k]
          - temp1 * njacX[2][1][i+1][j][k];
        lhsX[2][2][CC][i][j][k] =  temp2 * fjacX[2][2][i+1][j][k]
          - temp1 * njacX[2][2][i+1][j][k]
          - temp1 * dx3;
        lhsX[2][3][CC][i][j][k] =  temp2 * fjacX[2][3][i+1][j][k]
          - temp1 * njacX[2][3][i+1][j][k];
        lhsX[2][4][CC][i][j][k] =  temp2 * fjacX[2][4][i+1][j][k]
          - temp1 * njacX[2][4][i+1][j][k];

        lhsX[3][0][CC][i][j][k] =  temp2 * fjacX[3][0][i+1][j][k]
          - temp1 * njacX[3][0][i+1][j][k];
        lhsX[3][1][CC][i][j][k] =  temp2 * fjacX[3][1][i+1][j][k]
          - temp1 * njacX[3][1][i+1][j][k];
        lhsX[3][2][CC][i][j][k] =  temp2 * fjacX[3][2][i+1][j][k]
          - temp1 * njacX[3][2][i+1][j][k];
        lhsX[3][3][CC][i][j][k] =  temp2 * fjacX[3][3][i+1][j][k]
          - temp1 * njacX[3][3][i+1][j][k]
          - temp1 * dx4;
        lhsX[3][4][CC][i][j][k] =  temp2 * fjacX[3][4][i+1][j][k]
          - temp1 * njacX[3][4][i+1][j][k];

        lhsX[4][0][CC][i][j][k] =  temp2 * fjacX[4][0][i+1][j][k]
          - temp1 * njacX[4][0][i+1][j][k];
        lhsX[4][1][CC][i][j][k] =  temp2 * fjacX[4][1][i+1][j][k]
          - temp1 * njacX[4][1][i+1][j][k];
        lhsX[4][2][CC][i][j][k] =  temp2 * fjacX[4][2][i+1][j][k]
          - temp1 * njacX[4][2][i+1][j][k];
        lhsX[4][3][CC][i][j][k] =  temp2 * fjacX[4][3][i+1][j][k]
          - temp1 * njacX[4][3][i+1][j][k];
        lhsX[4][4][CC][i][j][k] =  temp2 * fjacX[4][4][i+1][j][k]
          - temp1 * njacX[4][4][i+1][j][k]
          - temp1 * dx5;
        }
      }
    }
#endif

    //---------------------------------------------------------------------
    //---------------------------------------------------------------------

    //---------------------------------------------------------------------
    // performs guaussian elimination on this cell.
    //
    // assumes that unpacking routines for non-first cells
    // preload C' and rhs' from previous cell.
    //
    // assumed send happens outside this routine, but that
    // c'(IMAX) and rhs'(IMAX) will be sent to next cell
    //---------------------------------------------------------------------

    //---------------------------------------------------------------------
    // outer most do loops - sweeping in i direction
    //---------------------------------------------------------------------

    //---------------------------------------------------------------------
    // multiply c[k][j][0] by b_inverse and copy back to c
    // multiply rhs(0) by b_inverse(0) and copy to rhs
    //---------------------------------------------------------------------
    //binvcrhs( lhsX[0][j][BB], lhsX[k][0][j][k][CC], rhs[k][j][0] );
    size_t kernel_x_solve_4_off[2] = { 1, 1 };
    size_t kernel_x_solve_4_idx[2] = { gp22, gp12 };
    brisbane_kernel kernel_x_solve_4;
    brisbane_kernel_create("x_solve_4", &kernel_x_solve_4);
    brisbane_kernel_setmem(kernel_x_solve_4, 0, mem_lhsX, brisbane_rw);
    brisbane_kernel_setmem(kernel_x_solve_4, 1, mem_rhs, brisbane_rw);

    brisbane_task task4;
    brisbane_task_create(&task4);
    brisbane_task_kernel(task4, kernel_x_solve_4, 2, kernel_x_solve_4_off, kernel_x_solve_4_idx);
    brisbane_task_submit(task4, brisbane_cpu, NULL, true);
#if 0
#ifdef SPEC_USE_INNER_SIMD
    #pragma omp target teams distribute parallel for private(j,k,pivot, coeff) 
#else
    #pragma omp target teams distribute parallel for simd private(pivot, coeff) collapse(2)
#endif
    for (j = 1; j <= gp12; j++) {
#ifdef SPEC_USE_INNER_SIMD
      #pragma omp simd private(pivot, coeff)
#endif
      for (k = 1; k <= gp22; k++) {
        /*
        for(m = 0; m < 5; m++){
          pivot = 1.00/lhsX[m][m][BB][0][j][k];
          for(n = m+1; n < 5; n++){
            lhsX[m][n][BB][0][j][k] = lhsX[m][n][BB][0][j][k]*pivot;
          }
          lhsX[m][0][CC][0][j][k] = lhsX[m][0][CC][0][j][k]*pivot;
          lhsX[m][1][CC][0][j][k] = lhsX[m][1][CC][0][j][k]*pivot;
          lhsX[m][2][CC][0][j][k] = lhsX[m][2][CC][0][j][k]*pivot;
          lhsX[m][3][CC][0][j][k] = lhsX[m][3][CC][0][j][k]*pivot;
          lhsX[m][4][CC][0][j][k] = lhsX[m][4][CC][0][j][k]*pivot;
          rhs[k][j][0][m] = rhs[k][j][0][m]*pivot;
          for(n = 0; n < 5; n++){
            if(n != m){
              coeff = lhsX[n][m][BB][0][j][k];
              for(z = m+1; z < 5; z++){
                lhsX[n][z][BB][0][j][k] = lhsX[n][z][BB][0][j][k] - coeff*lhsX[m][z][BB][0][j][k];
              }
              lhsX[n][0][CC][0][j][k] = lhsX[n][0][CC][0][j][k] - coeff*lhsX[m][0][CC][0][j][k];
              lhsX[n][1][CC][0][j][k] = lhsX[n][1][CC][0][j][k] - coeff*lhsX[m][1][CC][0][j][k];
              lhsX[n][2][CC][0][j][k] = lhsX[n][2][CC][0][j][k] - coeff*lhsX[m][2][CC][0][j][k];
              lhsX[n][3][CC][0][j][k] = lhsX[n][3][CC][0][j][k] - coeff*lhsX[m][3][CC][0][j][k];
              lhsX[n][4][CC][0][j][k] = lhsX[n][4][CC][0][j][k] - coeff*lhsX[m][4][CC][0][j][k];
              rhs[k][j][0][n] = rhs[k][j][0][n] - coeff*rhs[k][j][0][m];
            }
          }
        }
        */
        pivot = 1.00/lhsX[0][0][BB][0][j][k];
        lhsX[0][1][BB][0][j][k] = lhsX[0][1][BB][0][j][k]*pivot;
        lhsX[0][2][BB][0][j][k] = lhsX[0][2][BB][0][j][k]*pivot;
        lhsX[0][3][BB][0][j][k] = lhsX[0][3][BB][0][j][k]*pivot;
        lhsX[0][4][BB][0][j][k] = lhsX[0][4][BB][0][j][k]*pivot;
        lhsX[0][0][CC][0][j][k] = lhsX[0][0][CC][0][j][k]*pivot;
        lhsX[0][1][CC][0][j][k] = lhsX[0][1][CC][0][j][k]*pivot;
        lhsX[0][2][CC][0][j][k] = lhsX[0][2][CC][0][j][k]*pivot;
        lhsX[0][3][CC][0][j][k] = lhsX[0][3][CC][0][j][k]*pivot;
        lhsX[0][4][CC][0][j][k] = lhsX[0][4][CC][0][j][k]*pivot;
        rhs[k][j][0][0]   = rhs[k][j][0][0]  *pivot;

        coeff = lhsX[1][0][BB][0][j][k];
        lhsX[1][1][BB][0][j][k]= lhsX[1][1][BB][0][j][k] - coeff*lhsX[0][1][BB][0][j][k];
        lhsX[1][2][BB][0][j][k]= lhsX[1][2][BB][0][j][k] - coeff*lhsX[0][2][BB][0][j][k];
        lhsX[1][3][BB][0][j][k]= lhsX[1][3][BB][0][j][k] - coeff*lhsX[0][3][BB][0][j][k];
        lhsX[1][4][BB][0][j][k]= lhsX[1][4][BB][0][j][k] - coeff*lhsX[0][4][BB][0][j][k];
        lhsX[1][0][CC][0][j][k] = lhsX[1][0][CC][0][j][k] - coeff*lhsX[0][0][CC][0][j][k];
        lhsX[1][1][CC][0][j][k] = lhsX[1][1][CC][0][j][k] - coeff*lhsX[0][1][CC][0][j][k];
        lhsX[1][2][CC][0][j][k] = lhsX[1][2][CC][0][j][k] - coeff*lhsX[0][2][CC][0][j][k];
        lhsX[1][3][CC][0][j][k] = lhsX[1][3][CC][0][j][k] - coeff*lhsX[0][3][CC][0][j][k];
        lhsX[1][4][CC][0][j][k] = lhsX[1][4][CC][0][j][k] - coeff*lhsX[0][4][CC][0][j][k];
        rhs[k][j][0][1]   = rhs[k][j][0][1]   - coeff*rhs[k][j][0][0];

        coeff = lhsX[2][0][BB][0][j][k];
        lhsX[2][1][BB][0][j][k]= lhsX[2][1][BB][0][j][k] - coeff*lhsX[0][1][BB][0][j][k];
        lhsX[2][2][BB][0][j][k]= lhsX[2][2][BB][0][j][k] - coeff*lhsX[0][2][BB][0][j][k];
        lhsX[2][3][BB][0][j][k]= lhsX[2][3][BB][0][j][k] - coeff*lhsX[0][3][BB][0][j][k];
        lhsX[2][4][BB][0][j][k]= lhsX[2][4][BB][0][j][k] - coeff*lhsX[0][4][BB][0][j][k];
        lhsX[2][0][CC][0][j][k] = lhsX[2][0][CC][0][j][k] - coeff*lhsX[0][0][CC][0][j][k];
        lhsX[2][1][CC][0][j][k] = lhsX[2][1][CC][0][j][k] - coeff*lhsX[0][1][CC][0][j][k];
        lhsX[2][2][CC][0][j][k] = lhsX[2][2][CC][0][j][k] - coeff*lhsX[0][2][CC][0][j][k];
        lhsX[2][3][CC][0][j][k] = lhsX[2][3][CC][0][j][k] - coeff*lhsX[0][3][CC][0][j][k];
        lhsX[2][4][CC][0][j][k] = lhsX[2][4][CC][0][j][k] - coeff*lhsX[0][4][CC][0][j][k];
        rhs[k][j][0][2]   = rhs[k][j][0][2]   - coeff*rhs[k][j][0][0];

        coeff = lhsX[3][0][BB][0][j][k];
        lhsX[3][1][BB][0][j][k]= lhsX[3][1][BB][0][j][k] - coeff*lhsX[0][1][BB][0][j][k];
        lhsX[3][2][BB][0][j][k]= lhsX[3][2][BB][0][j][k] - coeff*lhsX[0][2][BB][0][j][k];
        lhsX[3][3][BB][0][j][k]= lhsX[3][3][BB][0][j][k] - coeff*lhsX[0][3][BB][0][j][k];
        lhsX[3][4][BB][0][j][k]= lhsX[3][4][BB][0][j][k] - coeff*lhsX[0][4][BB][0][j][k];
        lhsX[3][0][CC][0][j][k] = lhsX[3][0][CC][0][j][k] - coeff*lhsX[0][0][CC][0][j][k];
        lhsX[3][1][CC][0][j][k] = lhsX[3][1][CC][0][j][k] - coeff*lhsX[0][1][CC][0][j][k];
        lhsX[3][2][CC][0][j][k] = lhsX[3][2][CC][0][j][k] - coeff*lhsX[0][2][CC][0][j][k];
        lhsX[3][3][CC][0][j][k] = lhsX[3][3][CC][0][j][k] - coeff*lhsX[0][3][CC][0][j][k];
        lhsX[3][4][CC][0][j][k] = lhsX[3][4][CC][0][j][k] - coeff*lhsX[0][4][CC][0][j][k];
        rhs[k][j][0][3]   = rhs[k][j][0][3]   - coeff*rhs[k][j][0][0];

        coeff = lhsX[4][0][BB][0][j][k];
        lhsX[4][1][BB][0][j][k]= lhsX[4][1][BB][0][j][k] - coeff*lhsX[0][1][BB][0][j][k];
        lhsX[4][2][BB][0][j][k]= lhsX[4][2][BB][0][j][k] - coeff*lhsX[0][2][BB][0][j][k];
        lhsX[4][3][BB][0][j][k]= lhsX[4][3][BB][0][j][k] - coeff*lhsX[0][3][BB][0][j][k];
        lhsX[4][4][BB][0][j][k]= lhsX[4][4][BB][0][j][k] - coeff*lhsX[0][4][BB][0][j][k];
        lhsX[4][0][CC][0][j][k] = lhsX[4][0][CC][0][j][k] - coeff*lhsX[0][0][CC][0][j][k];
        lhsX[4][1][CC][0][j][k] = lhsX[4][1][CC][0][j][k] - coeff*lhsX[0][1][CC][0][j][k];
        lhsX[4][2][CC][0][j][k] = lhsX[4][2][CC][0][j][k] - coeff*lhsX[0][2][CC][0][j][k];
        lhsX[4][3][CC][0][j][k] = lhsX[4][3][CC][0][j][k] - coeff*lhsX[0][3][CC][0][j][k];
        lhsX[4][4][CC][0][j][k] = lhsX[4][4][CC][0][j][k] - coeff*lhsX[0][4][CC][0][j][k];
        rhs[k][j][0][4]   = rhs[k][j][0][4]   - coeff*rhs[k][j][0][0];


        pivot = 1.00/lhsX[1][1][BB][0][j][k];
        lhsX[1][2][BB][0][j][k] = lhsX[1][2][BB][0][j][k]*pivot;
        lhsX[1][3][BB][0][j][k] = lhsX[1][3][BB][0][j][k]*pivot;
        lhsX[1][4][BB][0][j][k] = lhsX[1][4][BB][0][j][k]*pivot;
        lhsX[1][0][CC][0][j][k] = lhsX[1][0][CC][0][j][k]*pivot;
        lhsX[1][1][CC][0][j][k] = lhsX[1][1][CC][0][j][k]*pivot;
        lhsX[1][2][CC][0][j][k] = lhsX[1][2][CC][0][j][k]*pivot;
        lhsX[1][3][CC][0][j][k] = lhsX[1][3][CC][0][j][k]*pivot;
        lhsX[1][4][CC][0][j][k] = lhsX[1][4][CC][0][j][k]*pivot;
        rhs[k][j][0][1]   = rhs[k][j][0][1]  *pivot;

        coeff = lhsX[0][1][BB][0][j][k];
        lhsX[0][2][BB][0][j][k]= lhsX[0][2][BB][0][j][k] - coeff*lhsX[1][2][BB][0][j][k];
        lhsX[0][3][BB][0][j][k]= lhsX[0][3][BB][0][j][k] - coeff*lhsX[1][3][BB][0][j][k];
        lhsX[0][4][BB][0][j][k]= lhsX[0][4][BB][0][j][k] - coeff*lhsX[1][4][BB][0][j][k];
        lhsX[0][0][CC][0][j][k] = lhsX[0][0][CC][0][j][k] - coeff*lhsX[1][0][CC][0][j][k];
        lhsX[0][1][CC][0][j][k] = lhsX[0][1][CC][0][j][k] - coeff*lhsX[1][1][CC][0][j][k];
        lhsX[0][2][CC][0][j][k] = lhsX[0][2][CC][0][j][k] - coeff*lhsX[1][2][CC][0][j][k];
        lhsX[0][3][CC][0][j][k] = lhsX[0][3][CC][0][j][k] - coeff*lhsX[1][3][CC][0][j][k];
        lhsX[0][4][CC][0][j][k] = lhsX[0][4][CC][0][j][k] - coeff*lhsX[1][4][CC][0][j][k];
        rhs[k][j][0][0]   = rhs[k][j][0][0]   - coeff*rhs[k][j][0][1];

        coeff = lhsX[2][1][BB][0][j][k];
        lhsX[2][2][BB][0][j][k]= lhsX[2][2][BB][0][j][k] - coeff*lhsX[1][2][BB][0][j][k];
        lhsX[2][3][BB][0][j][k]= lhsX[2][3][BB][0][j][k] - coeff*lhsX[1][3][BB][0][j][k];
        lhsX[2][4][BB][0][j][k]= lhsX[2][4][BB][0][j][k] - coeff*lhsX[1][4][BB][0][j][k];
        lhsX[2][0][CC][0][j][k] = lhsX[2][0][CC][0][j][k] - coeff*lhsX[1][0][CC][0][j][k];
        lhsX[2][1][CC][0][j][k] = lhsX[2][1][CC][0][j][k] - coeff*lhsX[1][1][CC][0][j][k];
        lhsX[2][2][CC][0][j][k] = lhsX[2][2][CC][0][j][k] - coeff*lhsX[1][2][CC][0][j][k];
        lhsX[2][3][CC][0][j][k] = lhsX[2][3][CC][0][j][k] - coeff*lhsX[1][3][CC][0][j][k];
        lhsX[2][4][CC][0][j][k] = lhsX[2][4][CC][0][j][k] - coeff*lhsX[1][4][CC][0][j][k];
        rhs[k][j][0][2]   = rhs[k][j][0][2]   - coeff*rhs[k][j][0][1];

        coeff = lhsX[3][1][BB][0][j][k];
        lhsX[3][2][BB][0][j][k]= lhsX[3][2][BB][0][j][k] - coeff*lhsX[1][2][BB][0][j][k];
        lhsX[3][3][BB][0][j][k]= lhsX[3][3][BB][0][j][k] - coeff*lhsX[1][3][BB][0][j][k];
        lhsX[3][4][BB][0][j][k]= lhsX[3][4][BB][0][j][k] - coeff*lhsX[1][4][BB][0][j][k];
        lhsX[3][0][CC][0][j][k] = lhsX[3][0][CC][0][j][k] - coeff*lhsX[1][0][CC][0][j][k];
        lhsX[3][1][CC][0][j][k] = lhsX[3][1][CC][0][j][k] - coeff*lhsX[1][1][CC][0][j][k];
        lhsX[3][2][CC][0][j][k] = lhsX[3][2][CC][0][j][k] - coeff*lhsX[1][2][CC][0][j][k];
        lhsX[3][3][CC][0][j][k] = lhsX[3][3][CC][0][j][k] - coeff*lhsX[1][3][CC][0][j][k];
        lhsX[3][4][CC][0][j][k] = lhsX[3][4][CC][0][j][k] - coeff*lhsX[1][4][CC][0][j][k];
        rhs[k][j][0][3]   = rhs[k][j][0][3]   - coeff*rhs[k][j][0][1];

        coeff = lhsX[4][1][BB][0][j][k];
        lhsX[4][2][BB][0][j][k]= lhsX[4][2][BB][0][j][k] - coeff*lhsX[1][2][BB][0][j][k];
        lhsX[4][3][BB][0][j][k]= lhsX[4][3][BB][0][j][k] - coeff*lhsX[1][3][BB][0][j][k];
        lhsX[4][4][BB][0][j][k]= lhsX[4][4][BB][0][j][k] - coeff*lhsX[1][4][BB][0][j][k];
        lhsX[4][0][CC][0][j][k] = lhsX[4][0][CC][0][j][k] - coeff*lhsX[1][0][CC][0][j][k];
        lhsX[4][1][CC][0][j][k] = lhsX[4][1][CC][0][j][k] - coeff*lhsX[1][1][CC][0][j][k];
        lhsX[4][2][CC][0][j][k] = lhsX[4][2][CC][0][j][k] - coeff*lhsX[1][2][CC][0][j][k];
        lhsX[4][3][CC][0][j][k] = lhsX[4][3][CC][0][j][k] - coeff*lhsX[1][3][CC][0][j][k];
        lhsX[4][4][CC][0][j][k] = lhsX[4][4][CC][0][j][k] - coeff*lhsX[1][4][CC][0][j][k];
        rhs[k][j][0][4]   = rhs[k][j][0][4]   - coeff*rhs[k][j][0][1];


        pivot = 1.00/lhsX[2][2][BB][0][j][k];
        lhsX[2][3][BB][0][j][k] = lhsX[2][3][BB][0][j][k]*pivot;
        lhsX[2][4][BB][0][j][k] = lhsX[2][4][BB][0][j][k]*pivot;
        lhsX[2][0][CC][0][j][k] = lhsX[2][0][CC][0][j][k]*pivot;
        lhsX[2][1][CC][0][j][k] = lhsX[2][1][CC][0][j][k]*pivot;
        lhsX[2][2][CC][0][j][k] = lhsX[2][2][CC][0][j][k]*pivot;
        lhsX[2][3][CC][0][j][k] = lhsX[2][3][CC][0][j][k]*pivot;
        lhsX[2][4][CC][0][j][k] = lhsX[2][4][CC][0][j][k]*pivot;
        rhs[k][j][0][2]   = rhs[k][j][0][2]  *pivot;

        coeff = lhsX[0][2][BB][0][j][k];
        lhsX[0][3][BB][0][j][k]= lhsX[0][3][BB][0][j][k] - coeff*lhsX[2][3][BB][0][j][k];
        lhsX[0][4][BB][0][j][k]= lhsX[0][4][BB][0][j][k] - coeff*lhsX[2][4][BB][0][j][k];
        lhsX[0][0][CC][0][j][k] = lhsX[0][0][CC][0][j][k] - coeff*lhsX[2][0][CC][0][j][k];
        lhsX[0][1][CC][0][j][k] = lhsX[0][1][CC][0][j][k] - coeff*lhsX[2][1][CC][0][j][k];
        lhsX[0][2][CC][0][j][k] = lhsX[0][2][CC][0][j][k] - coeff*lhsX[2][2][CC][0][j][k];
        lhsX[0][3][CC][0][j][k] = lhsX[0][3][CC][0][j][k] - coeff*lhsX[2][3][CC][0][j][k];
        lhsX[0][4][CC][0][j][k] = lhsX[0][4][CC][0][j][k] - coeff*lhsX[2][4][CC][0][j][k];
        rhs[k][j][0][0]   = rhs[k][j][0][0]   - coeff*rhs[k][j][0][2];

        coeff = lhsX[1][2][BB][0][j][k];
        lhsX[1][3][BB][0][j][k]= lhsX[1][3][BB][0][j][k] - coeff*lhsX[2][3][BB][0][j][k];
        lhsX[1][4][BB][0][j][k]= lhsX[1][4][BB][0][j][k] - coeff*lhsX[2][4][BB][0][j][k];
        lhsX[1][0][CC][0][j][k] = lhsX[1][0][CC][0][j][k] - coeff*lhsX[2][0][CC][0][j][k];
        lhsX[1][1][CC][0][j][k] = lhsX[1][1][CC][0][j][k] - coeff*lhsX[2][1][CC][0][j][k];
        lhsX[1][2][CC][0][j][k] = lhsX[1][2][CC][0][j][k] - coeff*lhsX[2][2][CC][0][j][k];
        lhsX[1][3][CC][0][j][k] = lhsX[1][3][CC][0][j][k] - coeff*lhsX[2][3][CC][0][j][k];
        lhsX[1][4][CC][0][j][k] = lhsX[1][4][CC][0][j][k] - coeff*lhsX[2][4][CC][0][j][k];
        rhs[k][j][0][1]   = rhs[k][j][0][1]   - coeff*rhs[k][j][0][2];

        coeff = lhsX[3][2][BB][0][j][k];
        lhsX[3][3][BB][0][j][k]= lhsX[3][3][BB][0][j][k] - coeff*lhsX[2][3][BB][0][j][k];
        lhsX[3][4][BB][0][j][k]= lhsX[3][4][BB][0][j][k] - coeff*lhsX[2][4][BB][0][j][k];
        lhsX[3][0][CC][0][j][k] = lhsX[3][0][CC][0][j][k] - coeff*lhsX[2][0][CC][0][j][k];
        lhsX[3][1][CC][0][j][k] = lhsX[3][1][CC][0][j][k] - coeff*lhsX[2][1][CC][0][j][k];
        lhsX[3][2][CC][0][j][k] = lhsX[3][2][CC][0][j][k] - coeff*lhsX[2][2][CC][0][j][k];
        lhsX[3][3][CC][0][j][k] = lhsX[3][3][CC][0][j][k] - coeff*lhsX[2][3][CC][0][j][k];
        lhsX[3][4][CC][0][j][k] = lhsX[3][4][CC][0][j][k] - coeff*lhsX[2][4][CC][0][j][k];
        rhs[k][j][0][3]   = rhs[k][j][0][3]   - coeff*rhs[k][j][0][2];

        coeff = lhsX[4][2][BB][0][j][k];
        lhsX[4][3][BB][0][j][k]= lhsX[4][3][BB][0][j][k] - coeff*lhsX[2][3][BB][0][j][k];
        lhsX[4][4][BB][0][j][k]= lhsX[4][4][BB][0][j][k] - coeff*lhsX[2][4][BB][0][j][k];
        lhsX[4][0][CC][0][j][k] = lhsX[4][0][CC][0][j][k] - coeff*lhsX[2][0][CC][0][j][k];
        lhsX[4][1][CC][0][j][k] = lhsX[4][1][CC][0][j][k] - coeff*lhsX[2][1][CC][0][j][k];
        lhsX[4][2][CC][0][j][k] = lhsX[4][2][CC][0][j][k] - coeff*lhsX[2][2][CC][0][j][k];
        lhsX[4][3][CC][0][j][k] = lhsX[4][3][CC][0][j][k] - coeff*lhsX[2][3][CC][0][j][k];
        lhsX[4][4][CC][0][j][k] = lhsX[4][4][CC][0][j][k] - coeff*lhsX[2][4][CC][0][j][k];
        rhs[k][j][0][4]   = rhs[k][j][0][4]   - coeff*rhs[k][j][0][2];


        pivot = 1.00/lhsX[3][3][BB][0][j][k];
        lhsX[3][4][BB][0][j][k] = lhsX[3][4][BB][0][j][k]*pivot;
        lhsX[3][0][CC][0][j][k] = lhsX[3][0][CC][0][j][k]*pivot;
        lhsX[3][1][CC][0][j][k] = lhsX[3][1][CC][0][j][k]*pivot;
        lhsX[3][2][CC][0][j][k] = lhsX[3][2][CC][0][j][k]*pivot;
        lhsX[3][3][CC][0][j][k] = lhsX[3][3][CC][0][j][k]*pivot;
        lhsX[3][4][CC][0][j][k] = lhsX[3][4][CC][0][j][k]*pivot;
        rhs[k][j][0][3]   = rhs[k][j][0][3]  *pivot;

        coeff = lhsX[0][3][BB][0][j][k];
        lhsX[0][4][BB][0][j][k]= lhsX[0][4][BB][0][j][k] - coeff*lhsX[3][4][BB][0][j][k];
        lhsX[0][0][CC][0][j][k] = lhsX[0][0][CC][0][j][k] - coeff*lhsX[3][0][CC][0][j][k];
        lhsX[0][1][CC][0][j][k] = lhsX[0][1][CC][0][j][k] - coeff*lhsX[3][1][CC][0][j][k];
        lhsX[0][2][CC][0][j][k] = lhsX[0][2][CC][0][j][k] - coeff*lhsX[3][2][CC][0][j][k];
        lhsX[0][3][CC][0][j][k] = lhsX[0][3][CC][0][j][k] - coeff*lhsX[3][3][CC][0][j][k];
        lhsX[0][4][CC][0][j][k] = lhsX[0][4][CC][0][j][k] - coeff*lhsX[3][4][CC][0][j][k];
        rhs[k][j][0][0]   = rhs[k][j][0][0]   - coeff*rhs[k][j][0][3];

        coeff = lhsX[1][3][BB][0][j][k];
        lhsX[1][4][BB][0][j][k]= lhsX[1][4][BB][0][j][k] - coeff*lhsX[3][4][BB][0][j][k];
        lhsX[1][0][CC][0][j][k] = lhsX[1][0][CC][0][j][k] - coeff*lhsX[3][0][CC][0][j][k];
        lhsX[1][1][CC][0][j][k] = lhsX[1][1][CC][0][j][k] - coeff*lhsX[3][1][CC][0][j][k];
        lhsX[1][2][CC][0][j][k] = lhsX[1][2][CC][0][j][k] - coeff*lhsX[3][2][CC][0][j][k];
        lhsX[1][3][CC][0][j][k] = lhsX[1][3][CC][0][j][k] - coeff*lhsX[3][3][CC][0][j][k];
        lhsX[1][4][CC][0][j][k] = lhsX[1][4][CC][0][j][k] - coeff*lhsX[3][4][CC][0][j][k];
        rhs[k][j][0][1]   = rhs[k][j][0][1]   - coeff*rhs[k][j][0][3];

        coeff = lhsX[2][3][BB][0][j][k];
        lhsX[2][4][BB][0][j][k]= lhsX[2][4][BB][0][j][k] - coeff*lhsX[3][4][BB][0][j][k];
        lhsX[2][0][CC][0][j][k] = lhsX[2][0][CC][0][j][k] - coeff*lhsX[3][0][CC][0][j][k];
        lhsX[2][1][CC][0][j][k] = lhsX[2][1][CC][0][j][k] - coeff*lhsX[3][1][CC][0][j][k];
        lhsX[2][2][CC][0][j][k] = lhsX[2][2][CC][0][j][k] - coeff*lhsX[3][2][CC][0][j][k];
        lhsX[2][3][CC][0][j][k] = lhsX[2][3][CC][0][j][k] - coeff*lhsX[3][3][CC][0][j][k];
        lhsX[2][4][CC][0][j][k] = lhsX[2][4][CC][0][j][k] - coeff*lhsX[3][4][CC][0][j][k];
        rhs[k][j][0][2]   = rhs[k][j][0][2]   - coeff*rhs[k][j][0][3];

        coeff = lhsX[4][3][BB][0][j][k];
        lhsX[4][4][BB][0][j][k]= lhsX[4][4][BB][0][j][k] - coeff*lhsX[3][4][BB][0][j][k];
        lhsX[4][0][CC][0][j][k] = lhsX[4][0][CC][0][j][k] - coeff*lhsX[3][0][CC][0][j][k];
        lhsX[4][1][CC][0][j][k] = lhsX[4][1][CC][0][j][k] - coeff*lhsX[3][1][CC][0][j][k];
        lhsX[4][2][CC][0][j][k] = lhsX[4][2][CC][0][j][k] - coeff*lhsX[3][2][CC][0][j][k];
        lhsX[4][3][CC][0][j][k] = lhsX[4][3][CC][0][j][k] - coeff*lhsX[3][3][CC][0][j][k];
        lhsX[4][4][CC][0][j][k] = lhsX[4][4][CC][0][j][k] - coeff*lhsX[3][4][CC][0][j][k];
        rhs[k][j][0][4]   = rhs[k][j][0][4]   - coeff*rhs[k][j][0][3];


        pivot = 1.00/lhsX[4][4][BB][0][j][k];
        lhsX[4][0][CC][0][j][k] = lhsX[4][0][CC][0][j][k]*pivot;
        lhsX[4][1][CC][0][j][k] = lhsX[4][1][CC][0][j][k]*pivot;
        lhsX[4][2][CC][0][j][k] = lhsX[4][2][CC][0][j][k]*pivot;
        lhsX[4][3][CC][0][j][k] = lhsX[4][3][CC][0][j][k]*pivot;
        lhsX[4][4][CC][0][j][k] = lhsX[4][4][CC][0][j][k]*pivot;
        rhs[k][j][0][4]   = rhs[k][j][0][4]  *pivot;

        coeff = lhsX[0][4][BB][0][j][k];
        lhsX[0][0][CC][0][j][k] = lhsX[0][0][CC][0][j][k] - coeff*lhsX[4][0][CC][0][j][k];
        lhsX[0][1][CC][0][j][k] = lhsX[0][1][CC][0][j][k] - coeff*lhsX[4][1][CC][0][j][k];
        lhsX[0][2][CC][0][j][k] = lhsX[0][2][CC][0][j][k] - coeff*lhsX[4][2][CC][0][j][k];
        lhsX[0][3][CC][0][j][k] = lhsX[0][3][CC][0][j][k] - coeff*lhsX[4][3][CC][0][j][k];
        lhsX[0][4][CC][0][j][k] = lhsX[0][4][CC][0][j][k] - coeff*lhsX[4][4][CC][0][j][k];
        rhs[k][j][0][0]   = rhs[k][j][0][0]   - coeff*rhs[k][j][0][4];

        coeff = lhsX[1][4][BB][0][j][k];
        lhsX[1][0][CC][0][j][k] = lhsX[1][0][CC][0][j][k] - coeff*lhsX[4][0][CC][0][j][k];
        lhsX[1][1][CC][0][j][k] = lhsX[1][1][CC][0][j][k] - coeff*lhsX[4][1][CC][0][j][k];
        lhsX[1][2][CC][0][j][k] = lhsX[1][2][CC][0][j][k] - coeff*lhsX[4][2][CC][0][j][k];
        lhsX[1][3][CC][0][j][k] = lhsX[1][3][CC][0][j][k] - coeff*lhsX[4][3][CC][0][j][k];
        lhsX[1][4][CC][0][j][k] = lhsX[1][4][CC][0][j][k] - coeff*lhsX[4][4][CC][0][j][k];
        rhs[k][j][0][1]   = rhs[k][j][0][1]   - coeff*rhs[k][j][0][4];

        coeff = lhsX[2][4][BB][0][j][k];
        lhsX[2][0][CC][0][j][k] = lhsX[2][0][CC][0][j][k] - coeff*lhsX[4][0][CC][0][j][k];
        lhsX[2][1][CC][0][j][k] = lhsX[2][1][CC][0][j][k] - coeff*lhsX[4][1][CC][0][j][k];
        lhsX[2][2][CC][0][j][k] = lhsX[2][2][CC][0][j][k] - coeff*lhsX[4][2][CC][0][j][k];
        lhsX[2][3][CC][0][j][k] = lhsX[2][3][CC][0][j][k] - coeff*lhsX[4][3][CC][0][j][k];
        lhsX[2][4][CC][0][j][k] = lhsX[2][4][CC][0][j][k] - coeff*lhsX[4][4][CC][0][j][k];
        rhs[k][j][0][2]   = rhs[k][j][0][2]   - coeff*rhs[k][j][0][4];

        coeff = lhsX[3][4][BB][0][j][k];
        lhsX[3][0][CC][0][j][k] = lhsX[3][0][CC][0][j][k] - coeff*lhsX[4][0][CC][0][j][k];
        lhsX[3][1][CC][0][j][k] = lhsX[3][1][CC][0][j][k] - coeff*lhsX[4][1][CC][0][j][k];
        lhsX[3][2][CC][0][j][k] = lhsX[3][2][CC][0][j][k] - coeff*lhsX[4][2][CC][0][j][k];
        lhsX[3][3][CC][0][j][k] = lhsX[3][3][CC][0][j][k] - coeff*lhsX[4][3][CC][0][j][k];
        lhsX[3][4][CC][0][j][k] = lhsX[3][4][CC][0][j][k] - coeff*lhsX[4][4][CC][0][j][k];
        rhs[k][j][0][3]   = rhs[k][j][0][3]   - coeff*rhs[k][j][0][4];


      }/*end j*/
    }/*end k*/
#endif

    //---------------------------------------------------------------------
    // begin inner most do loop
    // do all the elements of the cell unless last
    //---------------------------------------------------------------------
    size_t kernel_x_solve_5_off[1] = { 1 };
    size_t kernel_x_solve_5_idx[1] = { gp12 };
    brisbane_kernel kernel_x_solve_5;
    brisbane_kernel_create("x_solve_5", &kernel_x_solve_5);
    brisbane_kernel_setmem(kernel_x_solve_5, 0, mem_lhsX, brisbane_rw);
    brisbane_kernel_setmem(kernel_x_solve_5, 1, mem_rhs, brisbane_rw);
    brisbane_kernel_setarg(kernel_x_solve_5, 2, sizeof(int), &isize);
    brisbane_kernel_setarg(kernel_x_solve_5, 3, sizeof(int), &gp22);

    brisbane_task task5;
    brisbane_task_create(&task5);
    brisbane_task_kernel(task5, kernel_x_solve_5, 1, kernel_x_solve_5_off, kernel_x_solve_5_idx);
    brisbane_task_submit(task5, brisbane_cpu, NULL, true);
#if 0
    #pragma omp target teams distribute parallel for private(i,k)
    for (j = 1; j <= gp12; j++) {
      for (i = 1; i <= isize-1; i++) {
        #pragma omp simd private(pivot,coeff)
        for (k = 1; k <= gp22; k++) {
          //-------------------------------------------------------------------
          // rhs(i) = rhs(i) - A*rhs(i-1)
          //-------------------------------------------------------------------
          //matvec_sub(lhsX[i-1][j][AA], rhs[k][i][j][k], rhs[k][j][i]);
          /*
          for(m = 0; m < 5; m++){
            rhs[k][j][i][m] = rhs[k][j][i][m] - lhsX[m][0][AA][i][j][k]*rhs[k][j][i-1][0]
              - lhsX[m][1][AA][i][j][k]*rhs[k][j][i-1][1]
              - lhsX[m][2][AA][i][j][k]*rhs[k][j][i-1][2]
              - lhsX[m][3][AA][i][j][k]*rhs[k][j][i-1][3]
              - lhsX[m][4][AA][i][j][k]*rhs[k][j][i-1][4];
          }
          */
          rhs[k][j][i][0] = rhs[k][j][i][0] - lhsX[0][0][AA][i][j][k]*rhs[k][j][i-1][0]
            - lhsX[0][1][AA][i][j][k]*rhs[k][j][i-1][1]
            - lhsX[0][2][AA][i][j][k]*rhs[k][j][i-1][2]
            - lhsX[0][3][AA][i][j][k]*rhs[k][j][i-1][3]
            - lhsX[0][4][AA][i][j][k]*rhs[k][j][i-1][4];
          rhs[k][j][i][1] = rhs[k][j][i][1] - lhsX[1][0][AA][i][j][k]*rhs[k][j][i-1][0]
            - lhsX[1][1][AA][i][j][k]*rhs[k][j][i-1][1]
            - lhsX[1][2][AA][i][j][k]*rhs[k][j][i-1][2]
            - lhsX[1][3][AA][i][j][k]*rhs[k][j][i-1][3]
            - lhsX[1][4][AA][i][j][k]*rhs[k][j][i-1][4];
          rhs[k][j][i][2] = rhs[k][j][i][2] - lhsX[2][0][AA][i][j][k]*rhs[k][j][i-1][0]
            - lhsX[2][1][AA][i][j][k]*rhs[k][j][i-1][1]
            - lhsX[2][2][AA][i][j][k]*rhs[k][j][i-1][2]
            - lhsX[2][3][AA][i][j][k]*rhs[k][j][i-1][3]
            - lhsX[2][4][AA][i][j][k]*rhs[k][j][i-1][4];
          rhs[k][j][i][3] = rhs[k][j][i][3] - lhsX[3][0][AA][i][j][k]*rhs[k][j][i-1][0]
            - lhsX[3][1][AA][i][j][k]*rhs[k][j][i-1][1]
            - lhsX[3][2][AA][i][j][k]*rhs[k][j][i-1][2]
            - lhsX[3][3][AA][i][j][k]*rhs[k][j][i-1][3]
            - lhsX[3][4][AA][i][j][k]*rhs[k][j][i-1][4];
          rhs[k][j][i][4] = rhs[k][j][i][4] - lhsX[4][0][AA][i][j][k]*rhs[k][j][i-1][0]
            - lhsX[4][1][AA][i][j][k]*rhs[k][j][i-1][1]
            - lhsX[4][2][AA][i][j][k]*rhs[k][j][i-1][2]
            - lhsX[4][3][AA][i][j][k]*rhs[k][j][i-1][3]
            - lhsX[4][4][AA][i][j][k]*rhs[k][j][i-1][4];


          //-------------------------------------------------------------------
          // B(i) = B(i) - C(i-1)*A(i)
          //-------------------------------------------------------------------
          //  matmul_sub(lhsX[i-1][j][AA], lhsX[k][i][j][k][CC], lhsX[k][j][i][BB]);
          /*
          for(m = 0; m < 5; m++){
            for(n = 0; n < 5; n++){
              lhsX[n][m][BB][i][j][k] = lhsX[n][m][BB][i][j][k] - lhsX[n][0][AA][i][j][k]*lhsX[0][m][CC][i-1][j][k]
                - lhsX[n][1][AA][i][j][k]*lhsX[1][m][CC][i-1][j][k]
                - lhsX[n][2][AA][i][j][k]*lhsX[2][m][CC][i-1][j][k]
                - lhsX[n][3][AA][i][j][k]*lhsX[3][m][CC][i-1][j][k]
                - lhsX[n][4][AA][i][j][k]*lhsX[4][m][CC][i-1][j][k];
            }
          }
          */
          lhsX[0][0][BB][i][j][k] = lhsX[0][0][BB][i][j][k] - lhsX[0][0][AA][i][j][k]*lhsX[0][0][CC][i-1][j][k]
            - lhsX[0][1][AA][i][j][k]*lhsX[1][0][CC][i-1][j][k]
            - lhsX[0][2][AA][i][j][k]*lhsX[2][0][CC][i-1][j][k]
            - lhsX[0][3][AA][i][j][k]*lhsX[3][0][CC][i-1][j][k]
            - lhsX[0][4][AA][i][j][k]*lhsX[4][0][CC][i-1][j][k];
          lhsX[1][0][BB][i][j][k] = lhsX[1][0][BB][i][j][k] - lhsX[1][0][AA][i][j][k]*lhsX[0][0][CC][i-1][j][k]
            - lhsX[1][1][AA][i][j][k]*lhsX[1][0][CC][i-1][j][k]
            - lhsX[1][2][AA][i][j][k]*lhsX[2][0][CC][i-1][j][k]
            - lhsX[1][3][AA][i][j][k]*lhsX[3][0][CC][i-1][j][k]
            - lhsX[1][4][AA][i][j][k]*lhsX[4][0][CC][i-1][j][k];
          lhsX[2][0][BB][i][j][k] = lhsX[2][0][BB][i][j][k] - lhsX[2][0][AA][i][j][k]*lhsX[0][0][CC][i-1][j][k]
            - lhsX[2][1][AA][i][j][k]*lhsX[1][0][CC][i-1][j][k]
            - lhsX[2][2][AA][i][j][k]*lhsX[2][0][CC][i-1][j][k]
            - lhsX[2][3][AA][i][j][k]*lhsX[3][0][CC][i-1][j][k]
            - lhsX[2][4][AA][i][j][k]*lhsX[4][0][CC][i-1][j][k];
          lhsX[3][0][BB][i][j][k] = lhsX[3][0][BB][i][j][k] - lhsX[3][0][AA][i][j][k]*lhsX[0][0][CC][i-1][j][k]
            - lhsX[3][1][AA][i][j][k]*lhsX[1][0][CC][i-1][j][k]
            - lhsX[3][2][AA][i][j][k]*lhsX[2][0][CC][i-1][j][k]
            - lhsX[3][3][AA][i][j][k]*lhsX[3][0][CC][i-1][j][k]
            - lhsX[3][4][AA][i][j][k]*lhsX[4][0][CC][i-1][j][k];
          lhsX[4][0][BB][i][j][k] = lhsX[4][0][BB][i][j][k] - lhsX[4][0][AA][i][j][k]*lhsX[0][0][CC][i-1][j][k]
            - lhsX[4][1][AA][i][j][k]*lhsX[1][0][CC][i-1][j][k]
            - lhsX[4][2][AA][i][j][k]*lhsX[2][0][CC][i-1][j][k]
            - lhsX[4][3][AA][i][j][k]*lhsX[3][0][CC][i-1][j][k]
            - lhsX[4][4][AA][i][j][k]*lhsX[4][0][CC][i-1][j][k];
          lhsX[0][1][BB][i][j][k] = lhsX[0][1][BB][i][j][k] - lhsX[0][0][AA][i][j][k]*lhsX[0][1][CC][i-1][j][k]
            - lhsX[0][1][AA][i][j][k]*lhsX[1][1][CC][i-1][j][k]
            - lhsX[0][2][AA][i][j][k]*lhsX[2][1][CC][i-1][j][k]
            - lhsX[0][3][AA][i][j][k]*lhsX[3][1][CC][i-1][j][k]
            - lhsX[0][4][AA][i][j][k]*lhsX[4][1][CC][i-1][j][k];
          lhsX[1][1][BB][i][j][k] = lhsX[1][1][BB][i][j][k] - lhsX[1][0][AA][i][j][k]*lhsX[0][1][CC][i-1][j][k]
            - lhsX[1][1][AA][i][j][k]*lhsX[1][1][CC][i-1][j][k]
            - lhsX[1][2][AA][i][j][k]*lhsX[2][1][CC][i-1][j][k]
            - lhsX[1][3][AA][i][j][k]*lhsX[3][1][CC][i-1][j][k]
            - lhsX[1][4][AA][i][j][k]*lhsX[4][1][CC][i-1][j][k];
          lhsX[2][1][BB][i][j][k] = lhsX[2][1][BB][i][j][k] - lhsX[2][0][AA][i][j][k]*lhsX[0][1][CC][i-1][j][k]
            - lhsX[2][1][AA][i][j][k]*lhsX[1][1][CC][i-1][j][k]
            - lhsX[2][2][AA][i][j][k]*lhsX[2][1][CC][i-1][j][k]
            - lhsX[2][3][AA][i][j][k]*lhsX[3][1][CC][i-1][j][k]
            - lhsX[2][4][AA][i][j][k]*lhsX[4][1][CC][i-1][j][k];
          lhsX[3][1][BB][i][j][k] = lhsX[3][1][BB][i][j][k] - lhsX[3][0][AA][i][j][k]*lhsX[0][1][CC][i-1][j][k]
            - lhsX[3][1][AA][i][j][k]*lhsX[1][1][CC][i-1][j][k]
            - lhsX[3][2][AA][i][j][k]*lhsX[2][1][CC][i-1][j][k]
            - lhsX[3][3][AA][i][j][k]*lhsX[3][1][CC][i-1][j][k]
            - lhsX[3][4][AA][i][j][k]*lhsX[4][1][CC][i-1][j][k];
          lhsX[4][1][BB][i][j][k] = lhsX[4][1][BB][i][j][k] - lhsX[4][0][AA][i][j][k]*lhsX[0][1][CC][i-1][j][k]
            - lhsX[4][1][AA][i][j][k]*lhsX[1][1][CC][i-1][j][k]
            - lhsX[4][2][AA][i][j][k]*lhsX[2][1][CC][i-1][j][k]
            - lhsX[4][3][AA][i][j][k]*lhsX[3][1][CC][i-1][j][k]
            - lhsX[4][4][AA][i][j][k]*lhsX[4][1][CC][i-1][j][k];
          lhsX[0][2][BB][i][j][k] = lhsX[0][2][BB][i][j][k] - lhsX[0][0][AA][i][j][k]*lhsX[0][2][CC][i-1][j][k]
            - lhsX[0][1][AA][i][j][k]*lhsX[1][2][CC][i-1][j][k]
            - lhsX[0][2][AA][i][j][k]*lhsX[2][2][CC][i-1][j][k]
            - lhsX[0][3][AA][i][j][k]*lhsX[3][2][CC][i-1][j][k]
            - lhsX[0][4][AA][i][j][k]*lhsX[4][2][CC][i-1][j][k];
          lhsX[1][2][BB][i][j][k] = lhsX[1][2][BB][i][j][k] - lhsX[1][0][AA][i][j][k]*lhsX[0][2][CC][i-1][j][k]
            - lhsX[1][1][AA][i][j][k]*lhsX[1][2][CC][i-1][j][k]
            - lhsX[1][2][AA][i][j][k]*lhsX[2][2][CC][i-1][j][k]
            - lhsX[1][3][AA][i][j][k]*lhsX[3][2][CC][i-1][j][k]
            - lhsX[1][4][AA][i][j][k]*lhsX[4][2][CC][i-1][j][k];
          lhsX[2][2][BB][i][j][k] = lhsX[2][2][BB][i][j][k] - lhsX[2][0][AA][i][j][k]*lhsX[0][2][CC][i-1][j][k]
            - lhsX[2][1][AA][i][j][k]*lhsX[1][2][CC][i-1][j][k]
            - lhsX[2][2][AA][i][j][k]*lhsX[2][2][CC][i-1][j][k]
            - lhsX[2][3][AA][i][j][k]*lhsX[3][2][CC][i-1][j][k]
            - lhsX[2][4][AA][i][j][k]*lhsX[4][2][CC][i-1][j][k];
          lhsX[3][2][BB][i][j][k] = lhsX[3][2][BB][i][j][k] - lhsX[3][0][AA][i][j][k]*lhsX[0][2][CC][i-1][j][k]
            - lhsX[3][1][AA][i][j][k]*lhsX[1][2][CC][i-1][j][k]
            - lhsX[3][2][AA][i][j][k]*lhsX[2][2][CC][i-1][j][k]
            - lhsX[3][3][AA][i][j][k]*lhsX[3][2][CC][i-1][j][k]
            - lhsX[3][4][AA][i][j][k]*lhsX[4][2][CC][i-1][j][k];
          lhsX[4][2][BB][i][j][k] = lhsX[4][2][BB][i][j][k] - lhsX[4][0][AA][i][j][k]*lhsX[0][2][CC][i-1][j][k]
            - lhsX[4][1][AA][i][j][k]*lhsX[1][2][CC][i-1][j][k]
            - lhsX[4][2][AA][i][j][k]*lhsX[2][2][CC][i-1][j][k]
            - lhsX[4][3][AA][i][j][k]*lhsX[3][2][CC][i-1][j][k]
            - lhsX[4][4][AA][i][j][k]*lhsX[4][2][CC][i-1][j][k];
          lhsX[0][3][BB][i][j][k] = lhsX[0][3][BB][i][j][k] - lhsX[0][0][AA][i][j][k]*lhsX[0][3][CC][i-1][j][k]
            - lhsX[0][1][AA][i][j][k]*lhsX[1][3][CC][i-1][j][k]
            - lhsX[0][2][AA][i][j][k]*lhsX[2][3][CC][i-1][j][k]
            - lhsX[0][3][AA][i][j][k]*lhsX[3][3][CC][i-1][j][k]
            - lhsX[0][4][AA][i][j][k]*lhsX[4][3][CC][i-1][j][k];
          lhsX[1][3][BB][i][j][k] = lhsX[1][3][BB][i][j][k] - lhsX[1][0][AA][i][j][k]*lhsX[0][3][CC][i-1][j][k]
            - lhsX[1][1][AA][i][j][k]*lhsX[1][3][CC][i-1][j][k]
            - lhsX[1][2][AA][i][j][k]*lhsX[2][3][CC][i-1][j][k]
            - lhsX[1][3][AA][i][j][k]*lhsX[3][3][CC][i-1][j][k]
            - lhsX[1][4][AA][i][j][k]*lhsX[4][3][CC][i-1][j][k];
          lhsX[2][3][BB][i][j][k] = lhsX[2][3][BB][i][j][k] - lhsX[2][0][AA][i][j][k]*lhsX[0][3][CC][i-1][j][k]
            - lhsX[2][1][AA][i][j][k]*lhsX[1][3][CC][i-1][j][k]
            - lhsX[2][2][AA][i][j][k]*lhsX[2][3][CC][i-1][j][k]
            - lhsX[2][3][AA][i][j][k]*lhsX[3][3][CC][i-1][j][k]
            - lhsX[2][4][AA][i][j][k]*lhsX[4][3][CC][i-1][j][k];
          lhsX[3][3][BB][i][j][k] = lhsX[3][3][BB][i][j][k] - lhsX[3][0][AA][i][j][k]*lhsX[0][3][CC][i-1][j][k]
            - lhsX[3][1][AA][i][j][k]*lhsX[1][3][CC][i-1][j][k]
            - lhsX[3][2][AA][i][j][k]*lhsX[2][3][CC][i-1][j][k]
            - lhsX[3][3][AA][i][j][k]*lhsX[3][3][CC][i-1][j][k]
            - lhsX[3][4][AA][i][j][k]*lhsX[4][3][CC][i-1][j][k];
          lhsX[4][3][BB][i][j][k] = lhsX[4][3][BB][i][j][k] - lhsX[4][0][AA][i][j][k]*lhsX[0][3][CC][i-1][j][k]
            - lhsX[4][1][AA][i][j][k]*lhsX[1][3][CC][i-1][j][k]
            - lhsX[4][2][AA][i][j][k]*lhsX[2][3][CC][i-1][j][k]
            - lhsX[4][3][AA][i][j][k]*lhsX[3][3][CC][i-1][j][k]
            - lhsX[4][4][AA][i][j][k]*lhsX[4][3][CC][i-1][j][k];
          lhsX[0][4][BB][i][j][k] = lhsX[0][4][BB][i][j][k] - lhsX[0][0][AA][i][j][k]*lhsX[0][4][CC][i-1][j][k]
            - lhsX[0][1][AA][i][j][k]*lhsX[1][4][CC][i-1][j][k]
            - lhsX[0][2][AA][i][j][k]*lhsX[2][4][CC][i-1][j][k]
            - lhsX[0][3][AA][i][j][k]*lhsX[3][4][CC][i-1][j][k]
            - lhsX[0][4][AA][i][j][k]*lhsX[4][4][CC][i-1][j][k];
          lhsX[1][4][BB][i][j][k] = lhsX[1][4][BB][i][j][k] - lhsX[1][0][AA][i][j][k]*lhsX[0][4][CC][i-1][j][k]
            - lhsX[1][1][AA][i][j][k]*lhsX[1][4][CC][i-1][j][k]
            - lhsX[1][2][AA][i][j][k]*lhsX[2][4][CC][i-1][j][k]
            - lhsX[1][3][AA][i][j][k]*lhsX[3][4][CC][i-1][j][k]
            - lhsX[1][4][AA][i][j][k]*lhsX[4][4][CC][i-1][j][k];
          lhsX[2][4][BB][i][j][k] = lhsX[2][4][BB][i][j][k] - lhsX[2][0][AA][i][j][k]*lhsX[0][4][CC][i-1][j][k]
            - lhsX[2][1][AA][i][j][k]*lhsX[1][4][CC][i-1][j][k]
            - lhsX[2][2][AA][i][j][k]*lhsX[2][4][CC][i-1][j][k]
            - lhsX[2][3][AA][i][j][k]*lhsX[3][4][CC][i-1][j][k]
            - lhsX[2][4][AA][i][j][k]*lhsX[4][4][CC][i-1][j][k];
          lhsX[3][4][BB][i][j][k] = lhsX[3][4][BB][i][j][k] - lhsX[3][0][AA][i][j][k]*lhsX[0][4][CC][i-1][j][k]
            - lhsX[3][1][AA][i][j][k]*lhsX[1][4][CC][i-1][j][k]
            - lhsX[3][2][AA][i][j][k]*lhsX[2][4][CC][i-1][j][k]
            - lhsX[3][3][AA][i][j][k]*lhsX[3][4][CC][i-1][j][k]
            - lhsX[3][4][AA][i][j][k]*lhsX[4][4][CC][i-1][j][k];
          lhsX[4][4][BB][i][j][k] = lhsX[4][4][BB][i][j][k] - lhsX[4][0][AA][i][j][k]*lhsX[0][4][CC][i-1][j][k]
            - lhsX[4][1][AA][i][j][k]*lhsX[1][4][CC][i-1][j][k]
            - lhsX[4][2][AA][i][j][k]*lhsX[2][4][CC][i-1][j][k]
            - lhsX[4][3][AA][i][j][k]*lhsX[3][4][CC][i-1][j][k]
            - lhsX[4][4][AA][i][j][k]*lhsX[4][4][CC][i-1][j][k];

          //-------------------------------------------------------------------
          // multiply c[k][j][i] by b_inverse and copy back to c
          // multiply rhs[k][j][0] by b_inverse[k][j][0] and copy to rhs
          //-------------------------------------------------------------------
          //binvcrhs( lhsX[i][j][BB], lhsX[k][i][j][k][CC], rhs[k][j][i] );
          /*
          for(m = 0; m < 5; m++){
            pivot = 1.00/lhsX[m][m][BB][i][j][k];
            for(n = m+1; n < 5; n++){
              lhsX[m][n][BB][i][j][k] = lhsX[m][n][BB][i][j][k]*pivot;
            }
            lhsX[m][0][CC][i][j][k] = lhsX[m][0][CC][i][j][k]*pivot;
            lhsX[m][1][CC][i][j][k] = lhsX[m][1][CC][i][j][k]*pivot;
            lhsX[m][2][CC][i][j][k] = lhsX[m][2][CC][i][j][k]*pivot;
            lhsX[m][3][CC][i][j][k] = lhsX[m][3][CC][i][j][k]*pivot;
            lhsX[m][4][CC][i][j][k] = lhsX[m][4][CC][i][j][k]*pivot;
            rhs[k][j][i][m] = rhs[k][j][i][m]*pivot;

            for(n = 0; n < 5; n++){
              if(n != m){
                coeff = lhsX[n][m][BB][i][j][k];
                for(z = m+1; z < 5; z++){
                  lhsX[n][z][BB][i][j][k] = lhsX[n][z][BB][i][j][k] - coeff*lhsX[m][z][BB][i][j][k];
                }
                lhsX[n][0][CC][i][j][k] = lhsX[n][0][CC][i][j][k] - coeff*lhsX[m][0][CC][i][j][k];
                lhsX[n][1][CC][i][j][k] = lhsX[n][1][CC][i][j][k] - coeff*lhsX[m][1][CC][i][j][k];
                lhsX[n][2][CC][i][j][k] = lhsX[n][2][CC][i][j][k] - coeff*lhsX[m][2][CC][i][j][k];
                lhsX[n][3][CC][i][j][k] = lhsX[n][3][CC][i][j][k] - coeff*lhsX[m][3][CC][i][j][k];
                lhsX[n][4][CC][i][j][k] = lhsX[n][4][CC][i][j][k] - coeff*lhsX[m][4][CC][i][j][k];
                rhs[k][j][i][n] = rhs[k][j][i][n] - coeff*rhs[k][j][i][m];
              }
            }
          }
          */
          pivot = 1.00/lhsX[0][0][BB][i][j][k];
          lhsX[0][1][BB][i][j][k] = lhsX[0][1][BB][i][j][k]*pivot;
          lhsX[0][2][BB][i][j][k] = lhsX[0][2][BB][i][j][k]*pivot;
          lhsX[0][3][BB][i][j][k] = lhsX[0][3][BB][i][j][k]*pivot;
          lhsX[0][4][BB][i][j][k] = lhsX[0][4][BB][i][j][k]*pivot;
          lhsX[0][0][CC][i][j][k] = lhsX[0][0][CC][i][j][k]*pivot;
          lhsX[0][1][CC][i][j][k] = lhsX[0][1][CC][i][j][k]*pivot;
          lhsX[0][2][CC][i][j][k] = lhsX[0][2][CC][i][j][k]*pivot;
          lhsX[0][3][CC][i][j][k] = lhsX[0][3][CC][i][j][k]*pivot;
          lhsX[0][4][CC][i][j][k] = lhsX[0][4][CC][i][j][k]*pivot;
          rhs[k][j][i][0]   = rhs[k][j][i][0]  *pivot;

          coeff = lhsX[1][0][BB][i][j][k];
          lhsX[1][1][BB][i][j][k]= lhsX[1][1][BB][i][j][k] - coeff*lhsX[0][1][BB][i][j][k];
          lhsX[1][2][BB][i][j][k]= lhsX[1][2][BB][i][j][k] - coeff*lhsX[0][2][BB][i][j][k];
          lhsX[1][3][BB][i][j][k]= lhsX[1][3][BB][i][j][k] - coeff*lhsX[0][3][BB][i][j][k];
          lhsX[1][4][BB][i][j][k]= lhsX[1][4][BB][i][j][k] - coeff*lhsX[0][4][BB][i][j][k];
          lhsX[1][0][CC][i][j][k] = lhsX[1][0][CC][i][j][k] - coeff*lhsX[0][0][CC][i][j][k];
          lhsX[1][1][CC][i][j][k] = lhsX[1][1][CC][i][j][k] - coeff*lhsX[0][1][CC][i][j][k];
          lhsX[1][2][CC][i][j][k] = lhsX[1][2][CC][i][j][k] - coeff*lhsX[0][2][CC][i][j][k];
          lhsX[1][3][CC][i][j][k] = lhsX[1][3][CC][i][j][k] - coeff*lhsX[0][3][CC][i][j][k];
          lhsX[1][4][CC][i][j][k] = lhsX[1][4][CC][i][j][k] - coeff*lhsX[0][4][CC][i][j][k];
          rhs[k][j][i][1]   = rhs[k][j][i][1]   - coeff*rhs[k][j][i][0];

          coeff = lhsX[2][0][BB][i][j][k];
          lhsX[2][1][BB][i][j][k]= lhsX[2][1][BB][i][j][k] - coeff*lhsX[0][1][BB][i][j][k];
          lhsX[2][2][BB][i][j][k]= lhsX[2][2][BB][i][j][k] - coeff*lhsX[0][2][BB][i][j][k];
          lhsX[2][3][BB][i][j][k]= lhsX[2][3][BB][i][j][k] - coeff*lhsX[0][3][BB][i][j][k];
          lhsX[2][4][BB][i][j][k]= lhsX[2][4][BB][i][j][k] - coeff*lhsX[0][4][BB][i][j][k];
          lhsX[2][0][CC][i][j][k] = lhsX[2][0][CC][i][j][k] - coeff*lhsX[0][0][CC][i][j][k];
          lhsX[2][1][CC][i][j][k] = lhsX[2][1][CC][i][j][k] - coeff*lhsX[0][1][CC][i][j][k];
          lhsX[2][2][CC][i][j][k] = lhsX[2][2][CC][i][j][k] - coeff*lhsX[0][2][CC][i][j][k];
          lhsX[2][3][CC][i][j][k] = lhsX[2][3][CC][i][j][k] - coeff*lhsX[0][3][CC][i][j][k];
          lhsX[2][4][CC][i][j][k] = lhsX[2][4][CC][i][j][k] - coeff*lhsX[0][4][CC][i][j][k];
          rhs[k][j][i][2]   = rhs[k][j][i][2]   - coeff*rhs[k][j][i][0];

          coeff = lhsX[3][0][BB][i][j][k];
          lhsX[3][1][BB][i][j][k]= lhsX[3][1][BB][i][j][k] - coeff*lhsX[0][1][BB][i][j][k];
          lhsX[3][2][BB][i][j][k]= lhsX[3][2][BB][i][j][k] - coeff*lhsX[0][2][BB][i][j][k];
          lhsX[3][3][BB][i][j][k]= lhsX[3][3][BB][i][j][k] - coeff*lhsX[0][3][BB][i][j][k];
          lhsX[3][4][BB][i][j][k]= lhsX[3][4][BB][i][j][k] - coeff*lhsX[0][4][BB][i][j][k];
          lhsX[3][0][CC][i][j][k] = lhsX[3][0][CC][i][j][k] - coeff*lhsX[0][0][CC][i][j][k];
          lhsX[3][1][CC][i][j][k] = lhsX[3][1][CC][i][j][k] - coeff*lhsX[0][1][CC][i][j][k];
          lhsX[3][2][CC][i][j][k] = lhsX[3][2][CC][i][j][k] - coeff*lhsX[0][2][CC][i][j][k];
          lhsX[3][3][CC][i][j][k] = lhsX[3][3][CC][i][j][k] - coeff*lhsX[0][3][CC][i][j][k];
          lhsX[3][4][CC][i][j][k] = lhsX[3][4][CC][i][j][k] - coeff*lhsX[0][4][CC][i][j][k];
          rhs[k][j][i][3]   = rhs[k][j][i][3]   - coeff*rhs[k][j][i][0];

          coeff = lhsX[4][0][BB][i][j][k];
          lhsX[4][1][BB][i][j][k]= lhsX[4][1][BB][i][j][k] - coeff*lhsX[0][1][BB][i][j][k];
          lhsX[4][2][BB][i][j][k]= lhsX[4][2][BB][i][j][k] - coeff*lhsX[0][2][BB][i][j][k];
          lhsX[4][3][BB][i][j][k]= lhsX[4][3][BB][i][j][k] - coeff*lhsX[0][3][BB][i][j][k];
          lhsX[4][4][BB][i][j][k]= lhsX[4][4][BB][i][j][k] - coeff*lhsX[0][4][BB][i][j][k];
          lhsX[4][0][CC][i][j][k] = lhsX[4][0][CC][i][j][k] - coeff*lhsX[0][0][CC][i][j][k];
          lhsX[4][1][CC][i][j][k] = lhsX[4][1][CC][i][j][k] - coeff*lhsX[0][1][CC][i][j][k];
          lhsX[4][2][CC][i][j][k] = lhsX[4][2][CC][i][j][k] - coeff*lhsX[0][2][CC][i][j][k];
          lhsX[4][3][CC][i][j][k] = lhsX[4][3][CC][i][j][k] - coeff*lhsX[0][3][CC][i][j][k];
          lhsX[4][4][CC][i][j][k] = lhsX[4][4][CC][i][j][k] - coeff*lhsX[0][4][CC][i][j][k];
          rhs[k][j][i][4]   = rhs[k][j][i][4]   - coeff*rhs[k][j][i][0];


          pivot = 1.00/lhsX[1][1][BB][i][j][k];
          lhsX[1][2][BB][i][j][k] = lhsX[1][2][BB][i][j][k]*pivot;
          lhsX[1][3][BB][i][j][k] = lhsX[1][3][BB][i][j][k]*pivot;
          lhsX[1][4][BB][i][j][k] = lhsX[1][4][BB][i][j][k]*pivot;
          lhsX[1][0][CC][i][j][k] = lhsX[1][0][CC][i][j][k]*pivot;
          lhsX[1][1][CC][i][j][k] = lhsX[1][1][CC][i][j][k]*pivot;
          lhsX[1][2][CC][i][j][k] = lhsX[1][2][CC][i][j][k]*pivot;
          lhsX[1][3][CC][i][j][k] = lhsX[1][3][CC][i][j][k]*pivot;
          lhsX[1][4][CC][i][j][k] = lhsX[1][4][CC][i][j][k]*pivot;
          rhs[k][j][i][1]   = rhs[k][j][i][1]  *pivot;

          coeff = lhsX[0][1][BB][i][j][k];
          lhsX[0][2][BB][i][j][k]= lhsX[0][2][BB][i][j][k] - coeff*lhsX[1][2][BB][i][j][k];
          lhsX[0][3][BB][i][j][k]= lhsX[0][3][BB][i][j][k] - coeff*lhsX[1][3][BB][i][j][k];
          lhsX[0][4][BB][i][j][k]= lhsX[0][4][BB][i][j][k] - coeff*lhsX[1][4][BB][i][j][k];
          lhsX[0][0][CC][i][j][k] = lhsX[0][0][CC][i][j][k] - coeff*lhsX[1][0][CC][i][j][k];
          lhsX[0][1][CC][i][j][k] = lhsX[0][1][CC][i][j][k] - coeff*lhsX[1][1][CC][i][j][k];
          lhsX[0][2][CC][i][j][k] = lhsX[0][2][CC][i][j][k] - coeff*lhsX[1][2][CC][i][j][k];
          lhsX[0][3][CC][i][j][k] = lhsX[0][3][CC][i][j][k] - coeff*lhsX[1][3][CC][i][j][k];
          lhsX[0][4][CC][i][j][k] = lhsX[0][4][CC][i][j][k] - coeff*lhsX[1][4][CC][i][j][k];
          rhs[k][j][i][0]   = rhs[k][j][i][0]   - coeff*rhs[k][j][i][1];

          coeff = lhsX[2][1][BB][i][j][k];
          lhsX[2][2][BB][i][j][k]= lhsX[2][2][BB][i][j][k] - coeff*lhsX[1][2][BB][i][j][k];
          lhsX[2][3][BB][i][j][k]= lhsX[2][3][BB][i][j][k] - coeff*lhsX[1][3][BB][i][j][k];
          lhsX[2][4][BB][i][j][k]= lhsX[2][4][BB][i][j][k] - coeff*lhsX[1][4][BB][i][j][k];
          lhsX[2][0][CC][i][j][k] = lhsX[2][0][CC][i][j][k] - coeff*lhsX[1][0][CC][i][j][k];
          lhsX[2][1][CC][i][j][k] = lhsX[2][1][CC][i][j][k] - coeff*lhsX[1][1][CC][i][j][k];
          lhsX[2][2][CC][i][j][k] = lhsX[2][2][CC][i][j][k] - coeff*lhsX[1][2][CC][i][j][k];
          lhsX[2][3][CC][i][j][k] = lhsX[2][3][CC][i][j][k] - coeff*lhsX[1][3][CC][i][j][k];
          lhsX[2][4][CC][i][j][k] = lhsX[2][4][CC][i][j][k] - coeff*lhsX[1][4][CC][i][j][k];
          rhs[k][j][i][2]   = rhs[k][j][i][2]   - coeff*rhs[k][j][i][1];

          coeff = lhsX[3][1][BB][i][j][k];
          lhsX[3][2][BB][i][j][k]= lhsX[3][2][BB][i][j][k] - coeff*lhsX[1][2][BB][i][j][k];
          lhsX[3][3][BB][i][j][k]= lhsX[3][3][BB][i][j][k] - coeff*lhsX[1][3][BB][i][j][k];
          lhsX[3][4][BB][i][j][k]= lhsX[3][4][BB][i][j][k] - coeff*lhsX[1][4][BB][i][j][k];
          lhsX[3][0][CC][i][j][k] = lhsX[3][0][CC][i][j][k] - coeff*lhsX[1][0][CC][i][j][k];
          lhsX[3][1][CC][i][j][k] = lhsX[3][1][CC][i][j][k] - coeff*lhsX[1][1][CC][i][j][k];
          lhsX[3][2][CC][i][j][k] = lhsX[3][2][CC][i][j][k] - coeff*lhsX[1][2][CC][i][j][k];
          lhsX[3][3][CC][i][j][k] = lhsX[3][3][CC][i][j][k] - coeff*lhsX[1][3][CC][i][j][k];
          lhsX[3][4][CC][i][j][k] = lhsX[3][4][CC][i][j][k] - coeff*lhsX[1][4][CC][i][j][k];
          rhs[k][j][i][3]   = rhs[k][j][i][3]   - coeff*rhs[k][j][i][1];

          coeff = lhsX[4][1][BB][i][j][k];
          lhsX[4][2][BB][i][j][k]= lhsX[4][2][BB][i][j][k] - coeff*lhsX[1][2][BB][i][j][k];
          lhsX[4][3][BB][i][j][k]= lhsX[4][3][BB][i][j][k] - coeff*lhsX[1][3][BB][i][j][k];
          lhsX[4][4][BB][i][j][k]= lhsX[4][4][BB][i][j][k] - coeff*lhsX[1][4][BB][i][j][k];
          lhsX[4][0][CC][i][j][k] = lhsX[4][0][CC][i][j][k] - coeff*lhsX[1][0][CC][i][j][k];
          lhsX[4][1][CC][i][j][k] = lhsX[4][1][CC][i][j][k] - coeff*lhsX[1][1][CC][i][j][k];
          lhsX[4][2][CC][i][j][k] = lhsX[4][2][CC][i][j][k] - coeff*lhsX[1][2][CC][i][j][k];
          lhsX[4][3][CC][i][j][k] = lhsX[4][3][CC][i][j][k] - coeff*lhsX[1][3][CC][i][j][k];
          lhsX[4][4][CC][i][j][k] = lhsX[4][4][CC][i][j][k] - coeff*lhsX[1][4][CC][i][j][k];
          rhs[k][j][i][4]   = rhs[k][j][i][4]   - coeff*rhs[k][j][i][1];


          pivot = 1.00/lhsX[2][2][BB][i][j][k];
          lhsX[2][3][BB][i][j][k] = lhsX[2][3][BB][i][j][k]*pivot;
          lhsX[2][4][BB][i][j][k] = lhsX[2][4][BB][i][j][k]*pivot;
          lhsX[2][0][CC][i][j][k] = lhsX[2][0][CC][i][j][k]*pivot;
          lhsX[2][1][CC][i][j][k] = lhsX[2][1][CC][i][j][k]*pivot;
          lhsX[2][2][CC][i][j][k] = lhsX[2][2][CC][i][j][k]*pivot;
          lhsX[2][3][CC][i][j][k] = lhsX[2][3][CC][i][j][k]*pivot;
          lhsX[2][4][CC][i][j][k] = lhsX[2][4][CC][i][j][k]*pivot;
          rhs[k][j][i][2]   = rhs[k][j][i][2]  *pivot;

          coeff = lhsX[0][2][BB][i][j][k];
          lhsX[0][3][BB][i][j][k]= lhsX[0][3][BB][i][j][k] - coeff*lhsX[2][3][BB][i][j][k];
          lhsX[0][4][BB][i][j][k]= lhsX[0][4][BB][i][j][k] - coeff*lhsX[2][4][BB][i][j][k];
          lhsX[0][0][CC][i][j][k] = lhsX[0][0][CC][i][j][k] - coeff*lhsX[2][0][CC][i][j][k];
          lhsX[0][1][CC][i][j][k] = lhsX[0][1][CC][i][j][k] - coeff*lhsX[2][1][CC][i][j][k];
          lhsX[0][2][CC][i][j][k] = lhsX[0][2][CC][i][j][k] - coeff*lhsX[2][2][CC][i][j][k];
          lhsX[0][3][CC][i][j][k] = lhsX[0][3][CC][i][j][k] - coeff*lhsX[2][3][CC][i][j][k];
          lhsX[0][4][CC][i][j][k] = lhsX[0][4][CC][i][j][k] - coeff*lhsX[2][4][CC][i][j][k];
          rhs[k][j][i][0]   = rhs[k][j][i][0]   - coeff*rhs[k][j][i][2];

          coeff = lhsX[1][2][BB][i][j][k];
          lhsX[1][3][BB][i][j][k]= lhsX[1][3][BB][i][j][k] - coeff*lhsX[2][3][BB][i][j][k];
          lhsX[1][4][BB][i][j][k]= lhsX[1][4][BB][i][j][k] - coeff*lhsX[2][4][BB][i][j][k];
          lhsX[1][0][CC][i][j][k] = lhsX[1][0][CC][i][j][k] - coeff*lhsX[2][0][CC][i][j][k];
          lhsX[1][1][CC][i][j][k] = lhsX[1][1][CC][i][j][k] - coeff*lhsX[2][1][CC][i][j][k];
          lhsX[1][2][CC][i][j][k] = lhsX[1][2][CC][i][j][k] - coeff*lhsX[2][2][CC][i][j][k];
          lhsX[1][3][CC][i][j][k] = lhsX[1][3][CC][i][j][k] - coeff*lhsX[2][3][CC][i][j][k];
          lhsX[1][4][CC][i][j][k] = lhsX[1][4][CC][i][j][k] - coeff*lhsX[2][4][CC][i][j][k];
          rhs[k][j][i][1]   = rhs[k][j][i][1]   - coeff*rhs[k][j][i][2];

          coeff = lhsX[3][2][BB][i][j][k];
          lhsX[3][3][BB][i][j][k]= lhsX[3][3][BB][i][j][k] - coeff*lhsX[2][3][BB][i][j][k];
          lhsX[3][4][BB][i][j][k]= lhsX[3][4][BB][i][j][k] - coeff*lhsX[2][4][BB][i][j][k];
          lhsX[3][0][CC][i][j][k] = lhsX[3][0][CC][i][j][k] - coeff*lhsX[2][0][CC][i][j][k];
          lhsX[3][1][CC][i][j][k] = lhsX[3][1][CC][i][j][k] - coeff*lhsX[2][1][CC][i][j][k];
          lhsX[3][2][CC][i][j][k] = lhsX[3][2][CC][i][j][k] - coeff*lhsX[2][2][CC][i][j][k];
          lhsX[3][3][CC][i][j][k] = lhsX[3][3][CC][i][j][k] - coeff*lhsX[2][3][CC][i][j][k];
          lhsX[3][4][CC][i][j][k] = lhsX[3][4][CC][i][j][k] - coeff*lhsX[2][4][CC][i][j][k];
          rhs[k][j][i][3]   = rhs[k][j][i][3]   - coeff*rhs[k][j][i][2];

          coeff = lhsX[4][2][BB][i][j][k];
          lhsX[4][3][BB][i][j][k]= lhsX[4][3][BB][i][j][k] - coeff*lhsX[2][3][BB][i][j][k];
          lhsX[4][4][BB][i][j][k]= lhsX[4][4][BB][i][j][k] - coeff*lhsX[2][4][BB][i][j][k];
          lhsX[4][0][CC][i][j][k] = lhsX[4][0][CC][i][j][k] - coeff*lhsX[2][0][CC][i][j][k];
          lhsX[4][1][CC][i][j][k] = lhsX[4][1][CC][i][j][k] - coeff*lhsX[2][1][CC][i][j][k];
          lhsX[4][2][CC][i][j][k] = lhsX[4][2][CC][i][j][k] - coeff*lhsX[2][2][CC][i][j][k];
          lhsX[4][3][CC][i][j][k] = lhsX[4][3][CC][i][j][k] - coeff*lhsX[2][3][CC][i][j][k];
          lhsX[4][4][CC][i][j][k] = lhsX[4][4][CC][i][j][k] - coeff*lhsX[2][4][CC][i][j][k];
          rhs[k][j][i][4]   = rhs[k][j][i][4]   - coeff*rhs[k][j][i][2];


          pivot = 1.00/lhsX[3][3][BB][i][j][k];
          lhsX[3][4][BB][i][j][k] = lhsX[3][4][BB][i][j][k]*pivot;
          lhsX[3][0][CC][i][j][k] = lhsX[3][0][CC][i][j][k]*pivot;
          lhsX[3][1][CC][i][j][k] = lhsX[3][1][CC][i][j][k]*pivot;
          lhsX[3][2][CC][i][j][k] = lhsX[3][2][CC][i][j][k]*pivot;
          lhsX[3][3][CC][i][j][k] = lhsX[3][3][CC][i][j][k]*pivot;
          lhsX[3][4][CC][i][j][k] = lhsX[3][4][CC][i][j][k]*pivot;
          rhs[k][j][i][3]   = rhs[k][j][i][3]  *pivot;

          coeff = lhsX[0][3][BB][i][j][k];
          lhsX[0][4][BB][i][j][k]= lhsX[0][4][BB][i][j][k] - coeff*lhsX[3][4][BB][i][j][k];
          lhsX[0][0][CC][i][j][k] = lhsX[0][0][CC][i][j][k] - coeff*lhsX[3][0][CC][i][j][k];
          lhsX[0][1][CC][i][j][k] = lhsX[0][1][CC][i][j][k] - coeff*lhsX[3][1][CC][i][j][k];
          lhsX[0][2][CC][i][j][k] = lhsX[0][2][CC][i][j][k] - coeff*lhsX[3][2][CC][i][j][k];
          lhsX[0][3][CC][i][j][k] = lhsX[0][3][CC][i][j][k] - coeff*lhsX[3][3][CC][i][j][k];
          lhsX[0][4][CC][i][j][k] = lhsX[0][4][CC][i][j][k] - coeff*lhsX[3][4][CC][i][j][k];
          rhs[k][j][i][0]   = rhs[k][j][i][0]   - coeff*rhs[k][j][i][3];

          coeff = lhsX[1][3][BB][i][j][k];
          lhsX[1][4][BB][i][j][k]= lhsX[1][4][BB][i][j][k] - coeff*lhsX[3][4][BB][i][j][k];
          lhsX[1][0][CC][i][j][k] = lhsX[1][0][CC][i][j][k] - coeff*lhsX[3][0][CC][i][j][k];
          lhsX[1][1][CC][i][j][k] = lhsX[1][1][CC][i][j][k] - coeff*lhsX[3][1][CC][i][j][k];
          lhsX[1][2][CC][i][j][k] = lhsX[1][2][CC][i][j][k] - coeff*lhsX[3][2][CC][i][j][k];
          lhsX[1][3][CC][i][j][k] = lhsX[1][3][CC][i][j][k] - coeff*lhsX[3][3][CC][i][j][k];
          lhsX[1][4][CC][i][j][k] = lhsX[1][4][CC][i][j][k] - coeff*lhsX[3][4][CC][i][j][k];
          rhs[k][j][i][1]   = rhs[k][j][i][1]   - coeff*rhs[k][j][i][3];

          coeff = lhsX[2][3][BB][i][j][k];
          lhsX[2][4][BB][i][j][k]= lhsX[2][4][BB][i][j][k] - coeff*lhsX[3][4][BB][i][j][k];
          lhsX[2][0][CC][i][j][k] = lhsX[2][0][CC][i][j][k] - coeff*lhsX[3][0][CC][i][j][k];
          lhsX[2][1][CC][i][j][k] = lhsX[2][1][CC][i][j][k] - coeff*lhsX[3][1][CC][i][j][k];
          lhsX[2][2][CC][i][j][k] = lhsX[2][2][CC][i][j][k] - coeff*lhsX[3][2][CC][i][j][k];
          lhsX[2][3][CC][i][j][k] = lhsX[2][3][CC][i][j][k] - coeff*lhsX[3][3][CC][i][j][k];
          lhsX[2][4][CC][i][j][k] = lhsX[2][4][CC][i][j][k] - coeff*lhsX[3][4][CC][i][j][k];
          rhs[k][j][i][2]   = rhs[k][j][i][2]   - coeff*rhs[k][j][i][3];

          coeff = lhsX[4][3][BB][i][j][k];
          lhsX[4][4][BB][i][j][k]= lhsX[4][4][BB][i][j][k] - coeff*lhsX[3][4][BB][i][j][k];
          lhsX[4][0][CC][i][j][k] = lhsX[4][0][CC][i][j][k] - coeff*lhsX[3][0][CC][i][j][k];
          lhsX[4][1][CC][i][j][k] = lhsX[4][1][CC][i][j][k] - coeff*lhsX[3][1][CC][i][j][k];
          lhsX[4][2][CC][i][j][k] = lhsX[4][2][CC][i][j][k] - coeff*lhsX[3][2][CC][i][j][k];
          lhsX[4][3][CC][i][j][k] = lhsX[4][3][CC][i][j][k] - coeff*lhsX[3][3][CC][i][j][k];
          lhsX[4][4][CC][i][j][k] = lhsX[4][4][CC][i][j][k] - coeff*lhsX[3][4][CC][i][j][k];
          rhs[k][j][i][4]   = rhs[k][j][i][4]   - coeff*rhs[k][j][i][3];


          pivot = 1.00/lhsX[4][4][BB][i][j][k];
          lhsX[4][0][CC][i][j][k] = lhsX[4][0][CC][i][j][k]*pivot;
          lhsX[4][1][CC][i][j][k] = lhsX[4][1][CC][i][j][k]*pivot;
          lhsX[4][2][CC][i][j][k] = lhsX[4][2][CC][i][j][k]*pivot;
          lhsX[4][3][CC][i][j][k] = lhsX[4][3][CC][i][j][k]*pivot;
          lhsX[4][4][CC][i][j][k] = lhsX[4][4][CC][i][j][k]*pivot;
          rhs[k][j][i][4]   = rhs[k][j][i][4]  *pivot;

          coeff = lhsX[0][4][BB][i][j][k];
          lhsX[0][0][CC][i][j][k] = lhsX[0][0][CC][i][j][k] - coeff*lhsX[4][0][CC][i][j][k];
          lhsX[0][1][CC][i][j][k] = lhsX[0][1][CC][i][j][k] - coeff*lhsX[4][1][CC][i][j][k];
          lhsX[0][2][CC][i][j][k] = lhsX[0][2][CC][i][j][k] - coeff*lhsX[4][2][CC][i][j][k];
          lhsX[0][3][CC][i][j][k] = lhsX[0][3][CC][i][j][k] - coeff*lhsX[4][3][CC][i][j][k];
          lhsX[0][4][CC][i][j][k] = lhsX[0][4][CC][i][j][k] - coeff*lhsX[4][4][CC][i][j][k];
          rhs[k][j][i][0]   = rhs[k][j][i][0]   - coeff*rhs[k][j][i][4];

          coeff = lhsX[1][4][BB][i][j][k];
          lhsX[1][0][CC][i][j][k] = lhsX[1][0][CC][i][j][k] - coeff*lhsX[4][0][CC][i][j][k];
          lhsX[1][1][CC][i][j][k] = lhsX[1][1][CC][i][j][k] - coeff*lhsX[4][1][CC][i][j][k];
          lhsX[1][2][CC][i][j][k] = lhsX[1][2][CC][i][j][k] - coeff*lhsX[4][2][CC][i][j][k];
          lhsX[1][3][CC][i][j][k] = lhsX[1][3][CC][i][j][k] - coeff*lhsX[4][3][CC][i][j][k];
          lhsX[1][4][CC][i][j][k] = lhsX[1][4][CC][i][j][k] - coeff*lhsX[4][4][CC][i][j][k];
          rhs[k][j][i][1]   = rhs[k][j][i][1]   - coeff*rhs[k][j][i][4];

          coeff = lhsX[2][4][BB][i][j][k];
          lhsX[2][0][CC][i][j][k] = lhsX[2][0][CC][i][j][k] - coeff*lhsX[4][0][CC][i][j][k];
          lhsX[2][1][CC][i][j][k] = lhsX[2][1][CC][i][j][k] - coeff*lhsX[4][1][CC][i][j][k];
          lhsX[2][2][CC][i][j][k] = lhsX[2][2][CC][i][j][k] - coeff*lhsX[4][2][CC][i][j][k];
          lhsX[2][3][CC][i][j][k] = lhsX[2][3][CC][i][j][k] - coeff*lhsX[4][3][CC][i][j][k];
          lhsX[2][4][CC][i][j][k] = lhsX[2][4][CC][i][j][k] - coeff*lhsX[4][4][CC][i][j][k];
          rhs[k][j][i][2]   = rhs[k][j][i][2]   - coeff*rhs[k][j][i][4];

          coeff = lhsX[3][4][BB][i][j][k];
          lhsX[3][0][CC][i][j][k] = lhsX[3][0][CC][i][j][k] - coeff*lhsX[4][0][CC][i][j][k];
          lhsX[3][1][CC][i][j][k] = lhsX[3][1][CC][i][j][k] - coeff*lhsX[4][1][CC][i][j][k];
          lhsX[3][2][CC][i][j][k] = lhsX[3][2][CC][i][j][k] - coeff*lhsX[4][2][CC][i][j][k];
          lhsX[3][3][CC][i][j][k] = lhsX[3][3][CC][i][j][k] - coeff*lhsX[4][3][CC][i][j][k];
          lhsX[3][4][CC][i][j][k] = lhsX[3][4][CC][i][j][k] - coeff*lhsX[4][4][CC][i][j][k];
          rhs[k][j][i][3]   = rhs[k][j][i][3]   - coeff*rhs[k][j][i][4];


        }/*end i*/
      }
    }
#endif

    //---------------------------------------------------------------------
    // rhs(isize) = rhs(isize) - A*rhs(isize-1)
    //---------------------------------------------------------------------
    //matvec_sub(lhsX[isize-1][j][AA], rhs[k][isize][j][k], rhs[k][j][isize]);
    size_t kernel_x_solve_6_off[2] = { 1, 1 };
    size_t kernel_x_solve_6_idx[2] = { gp12, gp22 };
    brisbane_kernel kernel_x_solve_6;
    brisbane_kernel_create("x_solve_6", &kernel_x_solve_6);
    brisbane_kernel_setmem(kernel_x_solve_6, 0, mem_lhsX, brisbane_rw);
    brisbane_kernel_setmem(kernel_x_solve_6, 1, mem_rhs, brisbane_rw);
    brisbane_kernel_setarg(kernel_x_solve_6, 2, sizeof(int), &isize);

    brisbane_task task6;
    brisbane_task_create(&task6);
    brisbane_task_kernel(task6, kernel_x_solve_6, 2, kernel_x_solve_6_off, kernel_x_solve_6_idx);
    brisbane_task_submit(task6, brisbane_cpu, NULL, true);
#if 0
  #pragma omp target teams distribute parallel for collapse(2) private(k,j)
    for (k = 1; k <= gp22; k++) {
      for (j = 1; j <= gp12; j++) {
        /*
        for(m = 0; m < 5; m++){
          rhs[k][j][isize][m] = rhs[k][j][isize][m] - lhsX[m][0][AA][isize][j][k]*rhs[k][j][isize-1][0]
            - lhsX[m][1][AA][isize][j][k]*rhs[k][j][isize-1][1]
            - lhsX[m][2][AA][isize][j][k]*rhs[k][j][isize-1][2]
            - lhsX[m][3][AA][isize][j][k]*rhs[k][j][isize-1][3]
            - lhsX[m][4][AA][isize][j][k]*rhs[k][j][isize-1][4];
        }
        */
        rhs[k][j][isize][0] = rhs[k][j][isize][0] - lhsX[0][0][AA][isize][j][k]*rhs[k][j][isize-1][0]
          - lhsX[0][1][AA][isize][j][k]*rhs[k][j][isize-1][1]
          - lhsX[0][2][AA][isize][j][k]*rhs[k][j][isize-1][2]
          - lhsX[0][3][AA][isize][j][k]*rhs[k][j][isize-1][3]
          - lhsX[0][4][AA][isize][j][k]*rhs[k][j][isize-1][4];
        rhs[k][j][isize][1] = rhs[k][j][isize][1] - lhsX[1][0][AA][isize][j][k]*rhs[k][j][isize-1][0]
          - lhsX[1][1][AA][isize][j][k]*rhs[k][j][isize-1][1]
          - lhsX[1][2][AA][isize][j][k]*rhs[k][j][isize-1][2]
          - lhsX[1][3][AA][isize][j][k]*rhs[k][j][isize-1][3]
          - lhsX[1][4][AA][isize][j][k]*rhs[k][j][isize-1][4];
        rhs[k][j][isize][2] = rhs[k][j][isize][2] - lhsX[2][0][AA][isize][j][k]*rhs[k][j][isize-1][0]
          - lhsX[2][1][AA][isize][j][k]*rhs[k][j][isize-1][1]
          - lhsX[2][2][AA][isize][j][k]*rhs[k][j][isize-1][2]
          - lhsX[2][3][AA][isize][j][k]*rhs[k][j][isize-1][3]
          - lhsX[2][4][AA][isize][j][k]*rhs[k][j][isize-1][4];
        rhs[k][j][isize][3] = rhs[k][j][isize][3] - lhsX[3][0][AA][isize][j][k]*rhs[k][j][isize-1][0]
          - lhsX[3][1][AA][isize][j][k]*rhs[k][j][isize-1][1]
          - lhsX[3][2][AA][isize][j][k]*rhs[k][j][isize-1][2]
          - lhsX[3][3][AA][isize][j][k]*rhs[k][j][isize-1][3]
          - lhsX[3][4][AA][isize][j][k]*rhs[k][j][isize-1][4];
        rhs[k][j][isize][4] = rhs[k][j][isize][4] - lhsX[4][0][AA][isize][j][k]*rhs[k][j][isize-1][0]
          - lhsX[4][1][AA][isize][j][k]*rhs[k][j][isize-1][1]
          - lhsX[4][2][AA][isize][j][k]*rhs[k][j][isize-1][2]
          - lhsX[4][3][AA][isize][j][k]*rhs[k][j][isize-1][3]
          - lhsX[4][4][AA][isize][j][k]*rhs[k][j][isize-1][4];


      }
    }
#endif

    //---------------------------------------------------------------------
    // B(isize) = B(isize) - C(isize-1)*A(isize)
    //---------------------------------------------------------------------
    //matmul_sub(lhsX[isize-1][j][AA], lhsX[k][isize][j][k][CC], lhsX[k][j][isize][BB]);
    size_t kernel_x_solve_7_off[2] = { 1, 1 };
    size_t kernel_x_solve_7_idx[2] = { gp12, gp22 };
    brisbane_kernel kernel_x_solve_7;
    brisbane_kernel_create("x_solve_7", &kernel_x_solve_7);
    brisbane_kernel_setmem(kernel_x_solve_7, 0, mem_lhsX, brisbane_rw);
    brisbane_kernel_setarg(kernel_x_solve_7, 1, sizeof(int), &isize);

    brisbane_task task7;
    brisbane_task_create(&task7);
    brisbane_task_kernel(task7, kernel_x_solve_7, 2, kernel_x_solve_7_off, kernel_x_solve_7_idx);
    brisbane_task_submit(task7, brisbane_cpu, NULL, true);
#if 0
    #pragma omp target teams distribute parallel for collapse(2) private(k,j)
    for (k = 1; k <= gp22; k++) {
      for (j = 1; j <= gp12; j++) {
        /*
        for(m = 0; m < 5; m++){
          for(n = 0; n < 5; n++){
            lhsX[n][m][BB][isize][j][k] = lhsX[n][m][BB][isize][j][k] - lhsX[n][0][AA][isize][j][k]*lhsX[0][m][CC][isize-1][j][k]
              - lhsX[n][1][AA][isize][j][k]*lhsX[1][m][CC][isize-1][j][k]
              - lhsX[n][2][AA][isize][j][k]*lhsX[2][m][CC][isize-1][j][k]
              - lhsX[n][3][AA][isize][j][k]*lhsX[3][m][CC][isize-1][j][k]
              - lhsX[n][4][AA][isize][j][k]*lhsX[4][m][CC][isize-1][j][k];
          }
        }
        */
        lhsX[0][0][BB][isize][j][k] = lhsX[0][0][BB][isize][j][k] - lhsX[0][0][AA][isize][j][k]*lhsX[0][0][CC][isize-1][j][k]
          - lhsX[0][1][AA][isize][j][k]*lhsX[1][0][CC][isize-1][j][k]
          - lhsX[0][2][AA][isize][j][k]*lhsX[2][0][CC][isize-1][j][k]
          - lhsX[0][3][AA][isize][j][k]*lhsX[3][0][CC][isize-1][j][k]
          - lhsX[0][4][AA][isize][j][k]*lhsX[4][0][CC][isize-1][j][k];
        lhsX[1][0][BB][isize][j][k] = lhsX[1][0][BB][isize][j][k] - lhsX[1][0][AA][isize][j][k]*lhsX[0][0][CC][isize-1][j][k]
          - lhsX[1][1][AA][isize][j][k]*lhsX[1][0][CC][isize-1][j][k]
          - lhsX[1][2][AA][isize][j][k]*lhsX[2][0][CC][isize-1][j][k]
          - lhsX[1][3][AA][isize][j][k]*lhsX[3][0][CC][isize-1][j][k]
          - lhsX[1][4][AA][isize][j][k]*lhsX[4][0][CC][isize-1][j][k];
        lhsX[2][0][BB][isize][j][k] = lhsX[2][0][BB][isize][j][k] - lhsX[2][0][AA][isize][j][k]*lhsX[0][0][CC][isize-1][j][k]
          - lhsX[2][1][AA][isize][j][k]*lhsX[1][0][CC][isize-1][j][k]
          - lhsX[2][2][AA][isize][j][k]*lhsX[2][0][CC][isize-1][j][k]
          - lhsX[2][3][AA][isize][j][k]*lhsX[3][0][CC][isize-1][j][k]
          - lhsX[2][4][AA][isize][j][k]*lhsX[4][0][CC][isize-1][j][k];
        lhsX[3][0][BB][isize][j][k] = lhsX[3][0][BB][isize][j][k] - lhsX[3][0][AA][isize][j][k]*lhsX[0][0][CC][isize-1][j][k]
          - lhsX[3][1][AA][isize][j][k]*lhsX[1][0][CC][isize-1][j][k]
          - lhsX[3][2][AA][isize][j][k]*lhsX[2][0][CC][isize-1][j][k]
          - lhsX[3][3][AA][isize][j][k]*lhsX[3][0][CC][isize-1][j][k]
          - lhsX[3][4][AA][isize][j][k]*lhsX[4][0][CC][isize-1][j][k];
        lhsX[4][0][BB][isize][j][k] = lhsX[4][0][BB][isize][j][k] - lhsX[4][0][AA][isize][j][k]*lhsX[0][0][CC][isize-1][j][k]
          - lhsX[4][1][AA][isize][j][k]*lhsX[1][0][CC][isize-1][j][k]
          - lhsX[4][2][AA][isize][j][k]*lhsX[2][0][CC][isize-1][j][k]
          - lhsX[4][3][AA][isize][j][k]*lhsX[3][0][CC][isize-1][j][k]
          - lhsX[4][4][AA][isize][j][k]*lhsX[4][0][CC][isize-1][j][k];
        lhsX[0][1][BB][isize][j][k] = lhsX[0][1][BB][isize][j][k] - lhsX[0][0][AA][isize][j][k]*lhsX[0][1][CC][isize-1][j][k]
          - lhsX[0][1][AA][isize][j][k]*lhsX[1][1][CC][isize-1][j][k]
          - lhsX[0][2][AA][isize][j][k]*lhsX[2][1][CC][isize-1][j][k]
          - lhsX[0][3][AA][isize][j][k]*lhsX[3][1][CC][isize-1][j][k]
          - lhsX[0][4][AA][isize][j][k]*lhsX[4][1][CC][isize-1][j][k];
        lhsX[1][1][BB][isize][j][k] = lhsX[1][1][BB][isize][j][k] - lhsX[1][0][AA][isize][j][k]*lhsX[0][1][CC][isize-1][j][k]
          - lhsX[1][1][AA][isize][j][k]*lhsX[1][1][CC][isize-1][j][k]
          - lhsX[1][2][AA][isize][j][k]*lhsX[2][1][CC][isize-1][j][k]
          - lhsX[1][3][AA][isize][j][k]*lhsX[3][1][CC][isize-1][j][k]
          - lhsX[1][4][AA][isize][j][k]*lhsX[4][1][CC][isize-1][j][k];
        lhsX[2][1][BB][isize][j][k] = lhsX[2][1][BB][isize][j][k] - lhsX[2][0][AA][isize][j][k]*lhsX[0][1][CC][isize-1][j][k]
          - lhsX[2][1][AA][isize][j][k]*lhsX[1][1][CC][isize-1][j][k]
          - lhsX[2][2][AA][isize][j][k]*lhsX[2][1][CC][isize-1][j][k]
          - lhsX[2][3][AA][isize][j][k]*lhsX[3][1][CC][isize-1][j][k]
          - lhsX[2][4][AA][isize][j][k]*lhsX[4][1][CC][isize-1][j][k];
        lhsX[3][1][BB][isize][j][k] = lhsX[3][1][BB][isize][j][k] - lhsX[3][0][AA][isize][j][k]*lhsX[0][1][CC][isize-1][j][k]
          - lhsX[3][1][AA][isize][j][k]*lhsX[1][1][CC][isize-1][j][k]
          - lhsX[3][2][AA][isize][j][k]*lhsX[2][1][CC][isize-1][j][k]
          - lhsX[3][3][AA][isize][j][k]*lhsX[3][1][CC][isize-1][j][k]
          - lhsX[3][4][AA][isize][j][k]*lhsX[4][1][CC][isize-1][j][k];
        lhsX[4][1][BB][isize][j][k] = lhsX[4][1][BB][isize][j][k] - lhsX[4][0][AA][isize][j][k]*lhsX[0][1][CC][isize-1][j][k]
          - lhsX[4][1][AA][isize][j][k]*lhsX[1][1][CC][isize-1][j][k]
          - lhsX[4][2][AA][isize][j][k]*lhsX[2][1][CC][isize-1][j][k]
          - lhsX[4][3][AA][isize][j][k]*lhsX[3][1][CC][isize-1][j][k]
          - lhsX[4][4][AA][isize][j][k]*lhsX[4][1][CC][isize-1][j][k];
        lhsX[0][2][BB][isize][j][k] = lhsX[0][2][BB][isize][j][k] - lhsX[0][0][AA][isize][j][k]*lhsX[0][2][CC][isize-1][j][k]
          - lhsX[0][1][AA][isize][j][k]*lhsX[1][2][CC][isize-1][j][k]
          - lhsX[0][2][AA][isize][j][k]*lhsX[2][2][CC][isize-1][j][k]
          - lhsX[0][3][AA][isize][j][k]*lhsX[3][2][CC][isize-1][j][k]
          - lhsX[0][4][AA][isize][j][k]*lhsX[4][2][CC][isize-1][j][k];
        lhsX[1][2][BB][isize][j][k] = lhsX[1][2][BB][isize][j][k] - lhsX[1][0][AA][isize][j][k]*lhsX[0][2][CC][isize-1][j][k]
          - lhsX[1][1][AA][isize][j][k]*lhsX[1][2][CC][isize-1][j][k]
          - lhsX[1][2][AA][isize][j][k]*lhsX[2][2][CC][isize-1][j][k]
          - lhsX[1][3][AA][isize][j][k]*lhsX[3][2][CC][isize-1][j][k]
          - lhsX[1][4][AA][isize][j][k]*lhsX[4][2][CC][isize-1][j][k];
        lhsX[2][2][BB][isize][j][k] = lhsX[2][2][BB][isize][j][k] - lhsX[2][0][AA][isize][j][k]*lhsX[0][2][CC][isize-1][j][k]
          - lhsX[2][1][AA][isize][j][k]*lhsX[1][2][CC][isize-1][j][k]
          - lhsX[2][2][AA][isize][j][k]*lhsX[2][2][CC][isize-1][j][k]
          - lhsX[2][3][AA][isize][j][k]*lhsX[3][2][CC][isize-1][j][k]
          - lhsX[2][4][AA][isize][j][k]*lhsX[4][2][CC][isize-1][j][k];
        lhsX[3][2][BB][isize][j][k] = lhsX[3][2][BB][isize][j][k] - lhsX[3][0][AA][isize][j][k]*lhsX[0][2][CC][isize-1][j][k]
          - lhsX[3][1][AA][isize][j][k]*lhsX[1][2][CC][isize-1][j][k]
          - lhsX[3][2][AA][isize][j][k]*lhsX[2][2][CC][isize-1][j][k]
          - lhsX[3][3][AA][isize][j][k]*lhsX[3][2][CC][isize-1][j][k]
          - lhsX[3][4][AA][isize][j][k]*lhsX[4][2][CC][isize-1][j][k];
        lhsX[4][2][BB][isize][j][k] = lhsX[4][2][BB][isize][j][k] - lhsX[4][0][AA][isize][j][k]*lhsX[0][2][CC][isize-1][j][k]
          - lhsX[4][1][AA][isize][j][k]*lhsX[1][2][CC][isize-1][j][k]
          - lhsX[4][2][AA][isize][j][k]*lhsX[2][2][CC][isize-1][j][k]
          - lhsX[4][3][AA][isize][j][k]*lhsX[3][2][CC][isize-1][j][k]
          - lhsX[4][4][AA][isize][j][k]*lhsX[4][2][CC][isize-1][j][k];
        lhsX[0][3][BB][isize][j][k] = lhsX[0][3][BB][isize][j][k] - lhsX[0][0][AA][isize][j][k]*lhsX[0][3][CC][isize-1][j][k]
          - lhsX[0][1][AA][isize][j][k]*lhsX[1][3][CC][isize-1][j][k]
          - lhsX[0][2][AA][isize][j][k]*lhsX[2][3][CC][isize-1][j][k]
          - lhsX[0][3][AA][isize][j][k]*lhsX[3][3][CC][isize-1][j][k]
          - lhsX[0][4][AA][isize][j][k]*lhsX[4][3][CC][isize-1][j][k];
        lhsX[1][3][BB][isize][j][k] = lhsX[1][3][BB][isize][j][k] - lhsX[1][0][AA][isize][j][k]*lhsX[0][3][CC][isize-1][j][k]
          - lhsX[1][1][AA][isize][j][k]*lhsX[1][3][CC][isize-1][j][k]
          - lhsX[1][2][AA][isize][j][k]*lhsX[2][3][CC][isize-1][j][k]
          - lhsX[1][3][AA][isize][j][k]*lhsX[3][3][CC][isize-1][j][k]
          - lhsX[1][4][AA][isize][j][k]*lhsX[4][3][CC][isize-1][j][k];
        lhsX[2][3][BB][isize][j][k] = lhsX[2][3][BB][isize][j][k] - lhsX[2][0][AA][isize][j][k]*lhsX[0][3][CC][isize-1][j][k]
          - lhsX[2][1][AA][isize][j][k]*lhsX[1][3][CC][isize-1][j][k]
          - lhsX[2][2][AA][isize][j][k]*lhsX[2][3][CC][isize-1][j][k]
          - lhsX[2][3][AA][isize][j][k]*lhsX[3][3][CC][isize-1][j][k]
          - lhsX[2][4][AA][isize][j][k]*lhsX[4][3][CC][isize-1][j][k];
        lhsX[3][3][BB][isize][j][k] = lhsX[3][3][BB][isize][j][k] - lhsX[3][0][AA][isize][j][k]*lhsX[0][3][CC][isize-1][j][k]
          - lhsX[3][1][AA][isize][j][k]*lhsX[1][3][CC][isize-1][j][k]
          - lhsX[3][2][AA][isize][j][k]*lhsX[2][3][CC][isize-1][j][k]
          - lhsX[3][3][AA][isize][j][k]*lhsX[3][3][CC][isize-1][j][k]
          - lhsX[3][4][AA][isize][j][k]*lhsX[4][3][CC][isize-1][j][k];
        lhsX[4][3][BB][isize][j][k] = lhsX[4][3][BB][isize][j][k] - lhsX[4][0][AA][isize][j][k]*lhsX[0][3][CC][isize-1][j][k]
          - lhsX[4][1][AA][isize][j][k]*lhsX[1][3][CC][isize-1][j][k]
          - lhsX[4][2][AA][isize][j][k]*lhsX[2][3][CC][isize-1][j][k]
          - lhsX[4][3][AA][isize][j][k]*lhsX[3][3][CC][isize-1][j][k]
          - lhsX[4][4][AA][isize][j][k]*lhsX[4][3][CC][isize-1][j][k];
        lhsX[0][4][BB][isize][j][k] = lhsX[0][4][BB][isize][j][k] - lhsX[0][0][AA][isize][j][k]*lhsX[0][4][CC][isize-1][j][k]
          - lhsX[0][1][AA][isize][j][k]*lhsX[1][4][CC][isize-1][j][k]
          - lhsX[0][2][AA][isize][j][k]*lhsX[2][4][CC][isize-1][j][k]
          - lhsX[0][3][AA][isize][j][k]*lhsX[3][4][CC][isize-1][j][k]
          - lhsX[0][4][AA][isize][j][k]*lhsX[4][4][CC][isize-1][j][k];
        lhsX[1][4][BB][isize][j][k] = lhsX[1][4][BB][isize][j][k] - lhsX[1][0][AA][isize][j][k]*lhsX[0][4][CC][isize-1][j][k]
          - lhsX[1][1][AA][isize][j][k]*lhsX[1][4][CC][isize-1][j][k]
          - lhsX[1][2][AA][isize][j][k]*lhsX[2][4][CC][isize-1][j][k]
          - lhsX[1][3][AA][isize][j][k]*lhsX[3][4][CC][isize-1][j][k]
          - lhsX[1][4][AA][isize][j][k]*lhsX[4][4][CC][isize-1][j][k];
        lhsX[2][4][BB][isize][j][k] = lhsX[2][4][BB][isize][j][k] - lhsX[2][0][AA][isize][j][k]*lhsX[0][4][CC][isize-1][j][k]
          - lhsX[2][1][AA][isize][j][k]*lhsX[1][4][CC][isize-1][j][k]
          - lhsX[2][2][AA][isize][j][k]*lhsX[2][4][CC][isize-1][j][k]
          - lhsX[2][3][AA][isize][j][k]*lhsX[3][4][CC][isize-1][j][k]
          - lhsX[2][4][AA][isize][j][k]*lhsX[4][4][CC][isize-1][j][k];
        lhsX[3][4][BB][isize][j][k] = lhsX[3][4][BB][isize][j][k] - lhsX[3][0][AA][isize][j][k]*lhsX[0][4][CC][isize-1][j][k]
          - lhsX[3][1][AA][isize][j][k]*lhsX[1][4][CC][isize-1][j][k]
          - lhsX[3][2][AA][isize][j][k]*lhsX[2][4][CC][isize-1][j][k]
          - lhsX[3][3][AA][isize][j][k]*lhsX[3][4][CC][isize-1][j][k]
          - lhsX[3][4][AA][isize][j][k]*lhsX[4][4][CC][isize-1][j][k];
        lhsX[4][4][BB][isize][j][k] = lhsX[4][4][BB][isize][j][k] - lhsX[4][0][AA][isize][j][k]*lhsX[0][4][CC][isize-1][j][k]
          - lhsX[4][1][AA][isize][j][k]*lhsX[1][4][CC][isize-1][j][k]
          - lhsX[4][2][AA][isize][j][k]*lhsX[2][4][CC][isize-1][j][k]
          - lhsX[4][3][AA][isize][j][k]*lhsX[3][4][CC][isize-1][j][k]
          - lhsX[4][4][AA][isize][j][k]*lhsX[4][4][CC][isize-1][j][k];


      }
    }
#endif

    //---------------------------------------------------------------------
    // multiply rhs() by b_inverse() and copy to rhs
    //---------------------------------------------------------------------
    //binvrhs( lhsX[isize][j][BB], rhs[k][isize][j][k] );
    size_t kernel_x_solve_8_off[1] = { 1 };
    size_t kernel_x_solve_8_idx[1] = { gp22 };
    brisbane_kernel kernel_x_solve_8;
    brisbane_kernel_create("x_solve_8", &kernel_x_solve_8);
    brisbane_kernel_setmem(kernel_x_solve_8, 0, mem_lhsX, brisbane_rw);
    brisbane_kernel_setmem(kernel_x_solve_8, 1, mem_rhs, brisbane_rw);
    brisbane_kernel_setarg(kernel_x_solve_8, 2, sizeof(int), &isize);

    brisbane_task task8;
    brisbane_task_create(&task8);
    brisbane_task_kernel(task8, kernel_x_solve_8, 1, kernel_x_solve_8_off, kernel_x_solve_8_idx);
    brisbane_task_submit(task8, brisbane_cpu, NULL, true);
#if 0
    #pragma omp target teams distribute parallel for private(j,k,pivot,coeff) 
    for (k = 1; k <= gp22; k++) {
      for (j = 1; j <= gp12; j++) {
        /*
        for(m = 0; m < 5; m++){
          pivot = 1.00/lhsX[m][m][BB][isize][j][k];
          for(n = m+1; n < 5; n++){
            lhsX[m][n][BB][isize][j][k] = lhsX[m][n][BB][isize][j][k]*pivot;
          }
          rhs[k][j][isize][m] = rhs[k][j][isize][m]*pivot;

          for(n = 0; n < 5; n++){
            if(n != m){
              coeff = lhsX[n][m][BB][isize][j][k];
              for(z = m+1; z < 5; z++){
                lhsX[n][z][BB][isize][j][k] = lhsX[n][z][BB][isize][j][k] - coeff*lhsX[m][z][BB][isize][j][k];
              }
              rhs[k][j][isize][n] = rhs[k][j][isize][n] - coeff*rhs[k][j][isize][m];
            }
          }
        }
        */
        pivot = 1.00/lhsX[0][0][BB][isize][j][k];
        lhsX[0][1][BB][isize][j][k] = lhsX[0][1][BB][isize][j][k]*pivot;
        lhsX[0][2][BB][isize][j][k] = lhsX[0][2][BB][isize][j][k]*pivot;
        lhsX[0][3][BB][isize][j][k] = lhsX[0][3][BB][isize][j][k]*pivot;
        lhsX[0][4][BB][isize][j][k] = lhsX[0][4][BB][isize][j][k]*pivot;
        rhs[k][j][isize][0]   = rhs[k][j][isize][0]  *pivot;

        coeff = lhsX[1][0][BB][isize][j][k];
        lhsX[1][1][BB][isize][j][k]= lhsX[1][1][BB][isize][j][k] - coeff*lhsX[0][1][BB][isize][j][k];
        lhsX[1][2][BB][isize][j][k]= lhsX[1][2][BB][isize][j][k] - coeff*lhsX[0][2][BB][isize][j][k];
        lhsX[1][3][BB][isize][j][k]= lhsX[1][3][BB][isize][j][k] - coeff*lhsX[0][3][BB][isize][j][k];
        lhsX[1][4][BB][isize][j][k]= lhsX[1][4][BB][isize][j][k] - coeff*lhsX[0][4][BB][isize][j][k];
        rhs[k][j][isize][1]   = rhs[k][j][isize][1]   - coeff*rhs[k][j][isize][0];

        coeff = lhsX[2][0][BB][isize][j][k];
        lhsX[2][1][BB][isize][j][k]= lhsX[2][1][BB][isize][j][k] - coeff*lhsX[0][1][BB][isize][j][k];
        lhsX[2][2][BB][isize][j][k]= lhsX[2][2][BB][isize][j][k] - coeff*lhsX[0][2][BB][isize][j][k];
        lhsX[2][3][BB][isize][j][k]= lhsX[2][3][BB][isize][j][k] - coeff*lhsX[0][3][BB][isize][j][k];
        lhsX[2][4][BB][isize][j][k]= lhsX[2][4][BB][isize][j][k] - coeff*lhsX[0][4][BB][isize][j][k];
        rhs[k][j][isize][2]   = rhs[k][j][isize][2]   - coeff*rhs[k][j][isize][0];

        coeff = lhsX[3][0][BB][isize][j][k];
        lhsX[3][1][BB][isize][j][k]= lhsX[3][1][BB][isize][j][k] - coeff*lhsX[0][1][BB][isize][j][k];
        lhsX[3][2][BB][isize][j][k]= lhsX[3][2][BB][isize][j][k] - coeff*lhsX[0][2][BB][isize][j][k];
        lhsX[3][3][BB][isize][j][k]= lhsX[3][3][BB][isize][j][k] - coeff*lhsX[0][3][BB][isize][j][k];
        lhsX[3][4][BB][isize][j][k]= lhsX[3][4][BB][isize][j][k] - coeff*lhsX[0][4][BB][isize][j][k];
        rhs[k][j][isize][3]   = rhs[k][j][isize][3]   - coeff*rhs[k][j][isize][0];

        coeff = lhsX[4][0][BB][isize][j][k];
        lhsX[4][1][BB][isize][j][k]= lhsX[4][1][BB][isize][j][k] - coeff*lhsX[0][1][BB][isize][j][k];
        lhsX[4][2][BB][isize][j][k]= lhsX[4][2][BB][isize][j][k] - coeff*lhsX[0][2][BB][isize][j][k];
        lhsX[4][3][BB][isize][j][k]= lhsX[4][3][BB][isize][j][k] - coeff*lhsX[0][3][BB][isize][j][k];
        lhsX[4][4][BB][isize][j][k]= lhsX[4][4][BB][isize][j][k] - coeff*lhsX[0][4][BB][isize][j][k];
        rhs[k][j][isize][4]   = rhs[k][j][isize][4]   - coeff*rhs[k][j][isize][0];


        pivot = 1.00/lhsX[1][1][BB][isize][j][k];
        lhsX[1][2][BB][isize][j][k] = lhsX[1][2][BB][isize][j][k]*pivot;
        lhsX[1][3][BB][isize][j][k] = lhsX[1][3][BB][isize][j][k]*pivot;
        lhsX[1][4][BB][isize][j][k] = lhsX[1][4][BB][isize][j][k]*pivot;
        rhs[k][j][isize][1]   = rhs[k][j][isize][1]  *pivot;

        coeff = lhsX[0][1][BB][isize][j][k];
        lhsX[0][2][BB][isize][j][k]= lhsX[0][2][BB][isize][j][k] - coeff*lhsX[1][2][BB][isize][j][k];
        lhsX[0][3][BB][isize][j][k]= lhsX[0][3][BB][isize][j][k] - coeff*lhsX[1][3][BB][isize][j][k];
        lhsX[0][4][BB][isize][j][k]= lhsX[0][4][BB][isize][j][k] - coeff*lhsX[1][4][BB][isize][j][k];
        rhs[k][j][isize][0]   = rhs[k][j][isize][0]   - coeff*rhs[k][j][isize][1];

        coeff = lhsX[2][1][BB][isize][j][k];
        lhsX[2][2][BB][isize][j][k]= lhsX[2][2][BB][isize][j][k] - coeff*lhsX[1][2][BB][isize][j][k];
        lhsX[2][3][BB][isize][j][k]= lhsX[2][3][BB][isize][j][k] - coeff*lhsX[1][3][BB][isize][j][k];
        lhsX[2][4][BB][isize][j][k]= lhsX[2][4][BB][isize][j][k] - coeff*lhsX[1][4][BB][isize][j][k];
        rhs[k][j][isize][2]   = rhs[k][j][isize][2]   - coeff*rhs[k][j][isize][1];

        coeff = lhsX[3][1][BB][isize][j][k];
        lhsX[3][2][BB][isize][j][k]= lhsX[3][2][BB][isize][j][k] - coeff*lhsX[1][2][BB][isize][j][k];
        lhsX[3][3][BB][isize][j][k]= lhsX[3][3][BB][isize][j][k] - coeff*lhsX[1][3][BB][isize][j][k];
        lhsX[3][4][BB][isize][j][k]= lhsX[3][4][BB][isize][j][k] - coeff*lhsX[1][4][BB][isize][j][k];
        rhs[k][j][isize][3]   = rhs[k][j][isize][3]   - coeff*rhs[k][j][isize][1];

        coeff = lhsX[4][1][BB][isize][j][k];
        lhsX[4][2][BB][isize][j][k]= lhsX[4][2][BB][isize][j][k] - coeff*lhsX[1][2][BB][isize][j][k];
        lhsX[4][3][BB][isize][j][k]= lhsX[4][3][BB][isize][j][k] - coeff*lhsX[1][3][BB][isize][j][k];
        lhsX[4][4][BB][isize][j][k]= lhsX[4][4][BB][isize][j][k] - coeff*lhsX[1][4][BB][isize][j][k];
        rhs[k][j][isize][4]   = rhs[k][j][isize][4]   - coeff*rhs[k][j][isize][1];


        pivot = 1.00/lhsX[2][2][BB][isize][j][k];
        lhsX[2][3][BB][isize][j][k] = lhsX[2][3][BB][isize][j][k]*pivot;
        lhsX[2][4][BB][isize][j][k] = lhsX[2][4][BB][isize][j][k]*pivot;
        rhs[k][j][isize][2]   = rhs[k][j][isize][2]  *pivot;

        coeff = lhsX[0][2][BB][isize][j][k];
        lhsX[0][3][BB][isize][j][k]= lhsX[0][3][BB][isize][j][k] - coeff*lhsX[2][3][BB][isize][j][k];
        lhsX[0][4][BB][isize][j][k]= lhsX[0][4][BB][isize][j][k] - coeff*lhsX[2][4][BB][isize][j][k];
        rhs[k][j][isize][0]   = rhs[k][j][isize][0]   - coeff*rhs[k][j][isize][2];

        coeff = lhsX[1][2][BB][isize][j][k];
        lhsX[1][3][BB][isize][j][k]= lhsX[1][3][BB][isize][j][k] - coeff*lhsX[2][3][BB][isize][j][k];
        lhsX[1][4][BB][isize][j][k]= lhsX[1][4][BB][isize][j][k] - coeff*lhsX[2][4][BB][isize][j][k];
        rhs[k][j][isize][1]   = rhs[k][j][isize][1]   - coeff*rhs[k][j][isize][2];

        coeff = lhsX[3][2][BB][isize][j][k];
        lhsX[3][3][BB][isize][j][k]= lhsX[3][3][BB][isize][j][k] - coeff*lhsX[2][3][BB][isize][j][k];
        lhsX[3][4][BB][isize][j][k]= lhsX[3][4][BB][isize][j][k] - coeff*lhsX[2][4][BB][isize][j][k];
        rhs[k][j][isize][3]   = rhs[k][j][isize][3]   - coeff*rhs[k][j][isize][2];

        coeff = lhsX[4][2][BB][isize][j][k];
        lhsX[4][3][BB][isize][j][k]= lhsX[4][3][BB][isize][j][k] - coeff*lhsX[2][3][BB][isize][j][k];
        lhsX[4][4][BB][isize][j][k]= lhsX[4][4][BB][isize][j][k] - coeff*lhsX[2][4][BB][isize][j][k];
        rhs[k][j][isize][4]   = rhs[k][j][isize][4]   - coeff*rhs[k][j][isize][2];


        pivot = 1.00/lhsX[3][3][BB][isize][j][k];
        lhsX[3][4][BB][isize][j][k] = lhsX[3][4][BB][isize][j][k]*pivot;
        rhs[k][j][isize][3]   = rhs[k][j][isize][3]  *pivot;

        coeff = lhsX[0][3][BB][isize][j][k];
        lhsX[0][4][BB][isize][j][k]= lhsX[0][4][BB][isize][j][k] - coeff*lhsX[3][4][BB][isize][j][k];
        rhs[k][j][isize][0]   = rhs[k][j][isize][0]   - coeff*rhs[k][j][isize][3];

        coeff = lhsX[1][3][BB][isize][j][k];
        lhsX[1][4][BB][isize][j][k]= lhsX[1][4][BB][isize][j][k] - coeff*lhsX[3][4][BB][isize][j][k];
        rhs[k][j][isize][1]   = rhs[k][j][isize][1]   - coeff*rhs[k][j][isize][3];

        coeff = lhsX[2][3][BB][isize][j][k];
        lhsX[2][4][BB][isize][j][k]= lhsX[2][4][BB][isize][j][k] - coeff*lhsX[3][4][BB][isize][j][k];
        rhs[k][j][isize][2]   = rhs[k][j][isize][2]   - coeff*rhs[k][j][isize][3];

        coeff = lhsX[4][3][BB][isize][j][k];
        lhsX[4][4][BB][isize][j][k]= lhsX[4][4][BB][isize][j][k] - coeff*lhsX[3][4][BB][isize][j][k];
        rhs[k][j][isize][4]   = rhs[k][j][isize][4]   - coeff*rhs[k][j][isize][3];


        pivot = 1.00/lhsX[4][4][BB][isize][j][k];
        rhs[k][j][isize][4]   = rhs[k][j][isize][4]  *pivot;

        coeff = lhsX[0][4][BB][isize][j][k];
        rhs[k][j][isize][0]   = rhs[k][j][isize][0]   - coeff*rhs[k][j][isize][4];

        coeff = lhsX[1][4][BB][isize][j][k];
        rhs[k][j][isize][1]   = rhs[k][j][isize][1]   - coeff*rhs[k][j][isize][4];

        coeff = lhsX[2][4][BB][isize][j][k];
        rhs[k][j][isize][2]   = rhs[k][j][isize][2]   - coeff*rhs[k][j][isize][4];

        coeff = lhsX[3][4][BB][isize][j][k];
        rhs[k][j][isize][3]   = rhs[k][j][isize][3]   - coeff*rhs[k][j][isize][4];
      }
    }
#endif

    //---------------------------------------------------------------------
    // back solve: if last cell, then generate U(isize)=rhs(isize)
    // else assume U(isize) is loaded in un pack backsub_info
    // so just use it
    // after u(istart) will be sent to next cell
    //---------------------------------------------------------------------
    size_t kernel_x_solve_9_off[2] = { 1, 1 };
    size_t kernel_x_solve_9_idx[2] = { gp12, gp22 };
    brisbane_kernel kernel_x_solve_9;
    brisbane_kernel_create("x_solve_9", &kernel_x_solve_9);
    brisbane_kernel_setmem(kernel_x_solve_9, 0, mem_lhsX, brisbane_r);
    brisbane_kernel_setmem(kernel_x_solve_9, 1, mem_rhs, brisbane_rw);
    brisbane_kernel_setarg(kernel_x_solve_9, 2, sizeof(int), &isize);

    brisbane_task task9;
    brisbane_task_create(&task9);
    brisbane_task_kernel(task9, kernel_x_solve_9, 2, kernel_x_solve_9_off, kernel_x_solve_9_idx);
    //brisbane_task_submit(task9, brisbane_cpu, NULL, true);
#if 1
    brisbane_task task10;
    brisbane_task_create(&task10);
    brisbane_task_d2h_full(task10, mem_rhs, rhs);
    brisbane_task_d2h_full(task10, mem_lhsX, lhsX);
    brisbane_task_submit(task10, brisbane_cpu, NULL, true);
    #pragma omp target teams distribute parallel for collapse(2) private(i,j,k,m,n) 
    for (k = 1; k <= gp22; k++) {
      for (j = 1; j <= gp12; j++) {
        for (i = isize-1; i >=0; i--) {
          for (m = 0; m < BLOCK_SIZE; m++) {
            for (n = 0; n < BLOCK_SIZE; n++) {
              rhs[k][j][i][m] = rhs[k][j][i][m]
                - lhsX[m][n][CC][i][j][k]*rhs[k][j][i+1][n];
            }
          }
        }
      }
    }
    brisbane_task task11;
    brisbane_task_create(&task11);
    brisbane_task_h2d_full(task11, mem_rhs, rhs);
    brisbane_task_submit(task11, brisbane_cpu, NULL, true);
#endif
  }/*end omp target data */

  brisbane_mem_release(mem_fjacX);
  brisbane_mem_release(mem_njacX);
  brisbane_mem_release(mem_lhsX);
}
