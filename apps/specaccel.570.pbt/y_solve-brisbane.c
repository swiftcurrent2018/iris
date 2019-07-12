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
// Performs line solves in Y direction by first factoring
// the block-tridiagonal matrix into an upper triangular matrix,
// and then performing back substitution to solve for the unknow
// vectors of each line.
//
// Make sure we treat elements zero to cell_size in the direction
// of the sweep.
//---------------------------------------------------------------------
void y_solve()
{
  int i, j, k, m, n, jsize, z;
  double pivot, coeff;
  int gp22, gp02;
  double fjacY[5][5][PROBLEM_SIZE+1][IMAXP-1][KMAX-1];
  double njacY[5][5][PROBLEM_SIZE+1][IMAXP-1][KMAX-1];
  double lhsY[5][5][3][PROBLEM_SIZE][IMAXP-1][KMAX-1];
  double temp1, temp2, temp3;

  gp22 = grid_points[2]-2;
  gp02 = grid_points[0]-2;

  //---------------------------------------------------------------------
  // This function computes the left hand side for the three y-factors
  //---------------------------------------------------------------------

  jsize = grid_points[1]-1;

  brisbane_mem mem_fjacY;
  brisbane_mem mem_njacY;
  brisbane_mem mem_lhsY;
  brisbane_mem_create(sizeof(double) * 5 * 5 * (PROBLEM_SIZE + 1) * (IMAXP - 1) * (KMAX - 1), &mem_fjacY);
  brisbane_mem_create(sizeof(double) * 5 * 5 * (PROBLEM_SIZE + 1) * (IMAXP - 1) * (KMAX - 1), &mem_njacY);
  brisbane_mem_create(sizeof(double) * 5 * 5 * 3 * (PROBLEM_SIZE) * (IMAXP - 1) * (KMAX - 1), &mem_lhsY);

  //---------------------------------------------------------------------
  // Compute the indices for storing the tri-diagonal matrix;
  // determine a (labeled f) and n jacobians for cell c
  //---------------------------------------------------------------------
  #pragma omp target data map(alloc:lhsY[:][:][:][:][:][:],fjacY[:][:][:][:][:],njacY[:][:][:][:][:]) //present(rho_i,u,qs,rhs,square)
  {
    size_t kernel_y_solve_0_off[2] = { 1, 0 };
    size_t kernel_y_solve_0_idx[2] = { gp02, jsize + 1 };
    brisbane_kernel kernel_y_solve_0;
    brisbane_kernel_create("y_solve_0", &kernel_y_solve_0);
    brisbane_kernel_setmem(kernel_y_solve_0, 0, mem_rho_i, brisbane_r);
    brisbane_kernel_setmem(kernel_y_solve_0, 1, mem_fjacY, brisbane_w);
    brisbane_kernel_setmem(kernel_y_solve_0, 2, mem_njacY, brisbane_w);
    brisbane_kernel_setmem(kernel_y_solve_0, 3, mem_u, brisbane_r);
    brisbane_kernel_setmem(kernel_y_solve_0, 4, mem_qs, brisbane_r);
    brisbane_kernel_setmem(kernel_y_solve_0, 5, mem_square, brisbane_r);
    brisbane_kernel_setarg(kernel_y_solve_0, 6, sizeof(double), &c1);
    brisbane_kernel_setarg(kernel_y_solve_0, 7, sizeof(double), &c2);
    brisbane_kernel_setarg(kernel_y_solve_0, 8, sizeof(double), &c3c4);
    brisbane_kernel_setarg(kernel_y_solve_0, 9, sizeof(double), &c1345);
    brisbane_kernel_setarg(kernel_y_solve_0, 10, sizeof(double), &con43);
    brisbane_kernel_setarg(kernel_y_solve_0, 11, sizeof(int), &gp22);

    brisbane_task task0;
    brisbane_task_create(&task0);
    brisbane_task_kernel(task0, kernel_y_solve_0, 2, kernel_y_solve_0_off, kernel_y_solve_0_idx);
    brisbane_task_submit(task0, brisbane_cpu, NULL, true);
#if 0
    #pragma omp target teams distribute parallel for collapse(2) private(i,j,k,temp1,temp2,temp3) 
    for (j = 0; j <= jsize; j++) {
      for (i = 1; i <= gp02; i++) {
        for (k = 1; k <= gp22; k++) {
          temp1 = rho_i[k][j][i];
          temp2 = temp1 * temp1;
          temp3 = temp1 * temp2;

          fjacY[0][0][j][i][k] = 0.0;
          fjacY[0][1][j][i][k] = 0.0;
          fjacY[0][2][j][i][k] = 1.0;
          fjacY[0][3][j][i][k] = 0.0;
          fjacY[0][4][j][i][k] = 0.0;

          fjacY[1][0][j][i][k] = - ( u[k][j][i][1]*u[k][j][i][2] ) * temp2;
          fjacY[1][1][j][i][k] = u[k][j][i][2] * temp1;
          fjacY[1][2][j][i][k] = u[k][j][i][1] * temp1;
          fjacY[1][3][j][i][k] = 0.0;
          fjacY[1][4][j][i][k] = 0.0;

          fjacY[2][0][j][i][k] = - ( u[k][j][i][2]*u[k][j][i][2]*temp2)
            + c2 * qs[k][j][i];
          fjacY[2][1][j][i][k] = - c2 *  u[k][j][i][1] * temp1;
          fjacY[2][2][j][i][k] = ( 2.0 - c2 ) *  u[k][j][i][2] * temp1;
          fjacY[2][3][j][i][k] = - c2 * u[k][j][i][3] * temp1;
          fjacY[2][4][j][i][k] = c2;

          fjacY[3][0][j][i][k] = - ( u[k][j][i][2]*u[k][j][i][3] ) * temp2;
          fjacY[3][1][j][i][k] = 0.0;
          fjacY[3][2][j][i][k] = u[k][j][i][3] * temp1;
          fjacY[3][3][j][i][k] = u[k][j][i][2] * temp1;
          fjacY[3][4][j][i][k] = 0.0;

          fjacY[4][0][j][i][k] = ( c2 * 2.0 * square[k][j][i] - c1 * u[k][j][i][4] )
            * u[k][j][i][2] * temp2;
          fjacY[4][1][j][i][k] = - c2 * u[k][j][i][1]*u[k][j][i][2] * temp2;
          fjacY[4][2][j][i][k] = c1 * u[k][j][i][4] * temp1
            - c2 * ( qs[k][j][i] + u[k][j][i][2]*u[k][j][i][2] * temp2 );
          fjacY[4][3][j][i][k] = - c2 * ( u[k][j][i][2]*u[k][j][i][3] ) * temp2;
          fjacY[4][4][j][i][k] = c1 * u[k][j][i][2] * temp1;

          njacY[0][0][j][i][k] = 0.0;
          njacY[0][1][j][i][k] = 0.0;
          njacY[0][2][j][i][k] = 0.0;
          njacY[0][3][j][i][k] = 0.0;
          njacY[0][4][j][i][k] = 0.0;

          njacY[1][0][j][i][k] = - c3c4 * temp2 * u[k][j][i][1];
          njacY[1][1][j][i][k] =   c3c4 * temp1;
          njacY[1][2][j][i][k] =   0.0;
          njacY[1][3][j][i][k] =   0.0;
          njacY[1][4][j][i][k] =   0.0;

          njacY[2][0][j][i][k] = - con43 * c3c4 * temp2 * u[k][j][i][2];
          njacY[2][1][j][i][k] =   0.0;
          njacY[2][2][j][i][k] =   con43 * c3c4 * temp1;
          njacY[2][3][j][i][k] =   0.0;
          njacY[2][4][j][i][k] =   0.0;

          njacY[3][0][j][i][k] = - c3c4 * temp2 * u[k][j][i][3];
          njacY[3][1][j][i][k] =   0.0;
          njacY[3][2][j][i][k] =   0.0;
          njacY[3][3][j][i][k] =   c3c4 * temp1;
          njacY[3][4][j][i][k] =   0.0;

          njacY[4][0][j][i][k] = - (  c3c4
            - c1345 ) * temp3 * (u[k][j][i][1]*u[k][j][i][1])
            - ( con43 * c3c4
            - c1345 ) * temp3 * (u[k][j][i][2]*u[k][j][i][2])
            - ( c3c4 - c1345 ) * temp3 * (u[k][j][i][3]*u[k][j][i][3])
            - c1345 * temp2 * u[k][j][i][4];

          njacY[4][1][j][i][k] = (  c3c4 - c1345 ) * temp2 * u[k][j][i][1];
          njacY[4][2][j][i][k] = ( con43 * c3c4 - c1345 ) * temp2 * u[k][j][i][2];
          njacY[4][3][j][i][k] = ( c3c4 - c1345 ) * temp2 * u[k][j][i][3];
          njacY[4][4][j][i][k] = ( c1345 ) * temp1;
        }
      }
    }
#endif

    //---------------------------------------------------------------------
    // now joacobians set, so form left hand side in y direction
    //---------------------------------------------------------------------
    //lhsY[k][i]init(lhsY[k][i], jsize);
    // zero the whole left hand side for starters
    size_t kernel_y_solve_1_off[3] = { 1, 0, 0 };
    size_t kernel_y_solve_1_idx[3] = { gp02, 5, 5 };
    brisbane_kernel kernel_y_solve_1;
    brisbane_kernel_create("y_solve_1", &kernel_y_solve_1);
    brisbane_kernel_setmem(kernel_y_solve_1, 0, mem_lhsY, brisbane_w);
    brisbane_kernel_setarg(kernel_y_solve_1, 1, sizeof(int), &jsize);
    brisbane_kernel_setarg(kernel_y_solve_1, 2, sizeof(int), &gp22);

    brisbane_task task1;
    brisbane_task_create(&task1);
    brisbane_task_kernel(task1, kernel_y_solve_1, 3, kernel_y_solve_1_off, kernel_y_solve_1_idx);
    brisbane_task_submit(task1, brisbane_cpu, NULL, true);
#if 0
#ifdef SPEC_USE_INNER_SIMD
      #pragma omp target teams distribute parallel for private(i,k) collapse(3)
#else
      #pragma omp target teams distribute parallel for simd collapse(4)
#endif
    for (m = 0; m < 5; m++) {
      for (n = 0; n < 5; n++) {
      for (i = 1; i <= gp02; i++) {
#ifdef SPEC_USE_INNER_SIMD
        #pragma omp simd
#endif
          for (k = 1; k <= gp22; k++) {
            lhsY[m][n][0][0][i][k] = 0.0;
            lhsY[m][n][1][0][i][k] = 0.0;
            lhsY[m][n][2][0][i][k] = 0.0;
            lhsY[m][n][0][jsize][i][k] = 0.0;
            lhsY[m][n][1][jsize][i][k] = 0.0;
            lhsY[m][n][2][jsize][i][k] = 0.0;
          }
        }
      }
    }
#endif

    // next, set all diagonal values to 1. This is overkill, but convenient
    size_t kernel_y_solve_2_off[3] = { 1, 1, 0 };
    size_t kernel_y_solve_2_idx[3] = { gp22, gp02, 5 };
    brisbane_kernel kernel_y_solve_2;
    brisbane_kernel_create("y_solve_2", &kernel_y_solve_2);
    brisbane_kernel_setmem(kernel_y_solve_2, 0, mem_lhsY, brisbane_w);
    brisbane_kernel_setarg(kernel_y_solve_2, 1, sizeof(int), &jsize);

    brisbane_task task2;
    brisbane_task_create(&task2);
    brisbane_task_kernel(task2, kernel_y_solve_2, 3, kernel_y_solve_2_off, kernel_y_solve_2_idx);
    brisbane_task_submit(task2, brisbane_cpu, NULL, true);
#if 0
#ifdef SPEC_USE_INNER_SIMD
      #pragma omp target teams distribute parallel for private(i,k) collapse(2)
#else
      #pragma omp target teams distribute parallel for simd collapse(3)
#endif
    for (m = 0; m < 5; m++){
      for (i = 1; i <= gp02; i++) {
#ifdef SPEC_USE_INNER_SIMD
        #pragma omp simd
#endif
        for (k = 1; k <= gp22; k++) {
          lhsY[m][m][1][0][i][k] = 1.0;
          lhsY[m][m][1][jsize][i][k] = 1.0;
        }
      }
    }
#endif

    size_t kernel_y_solve_3_off[3] = { 1, 1, 1 };
    size_t kernel_y_solve_3_idx[3] = { gp22, gp02, jsize - 1 };
    brisbane_kernel kernel_y_solve_3;
    brisbane_kernel_create("y_solve_3", &kernel_y_solve_3);
    brisbane_kernel_setmem(kernel_y_solve_3, 0, mem_lhsY, brisbane_w);
    brisbane_kernel_setmem(kernel_y_solve_3, 1, mem_fjacY, brisbane_r);
    brisbane_kernel_setmem(kernel_y_solve_3, 2, mem_njacY, brisbane_r);
    brisbane_kernel_setarg(kernel_y_solve_3, 3, sizeof(double), &dtty1);
    brisbane_kernel_setarg(kernel_y_solve_3, 4, sizeof(double), &dtty2);
    brisbane_kernel_setarg(kernel_y_solve_3, 5, sizeof(double), &dy1);
    brisbane_kernel_setarg(kernel_y_solve_3, 6, sizeof(double), &dy2);
    brisbane_kernel_setarg(kernel_y_solve_3, 7, sizeof(double), &dy3);
    brisbane_kernel_setarg(kernel_y_solve_3, 8, sizeof(double), &dy4);
    brisbane_kernel_setarg(kernel_y_solve_3, 9, sizeof(double), &dy5);

    brisbane_task task3;
    brisbane_task_create(&task3);
    brisbane_task_kernel(task3, kernel_y_solve_3, 3, kernel_y_solve_3_off, kernel_y_solve_3_idx);
    brisbane_task_submit(task3, brisbane_cpu, NULL, true);
#if 0
#ifdef SPEC_USE_INNER_SIMD
    #pragma omp target teams distribute parallel for collapse(2) private(i,j,k)
#else
    #pragma omp target teams distribute parallel for simd collapse(3)
#endif
    for (j = 1; j <= jsize-1; j++) {
      for (i = 1; i <= gp02; i++) {
#ifdef SPEC_USE_INNER_SIMD
        #pragma omp simd
#endif
        for (k = 1; k <= gp22; k++) {
          lhsY[0][0][AA][j][i][k] = - dtty2 * fjacY[0][0][j-1][i][k]
            - dtty1 * njacY[0][0][j-1][i][k]
            - dtty1 * dy1;
          lhsY[0][1][AA][j][i][k] = - dtty2 * fjacY[0][1][j-1][i][k]
            - dtty1 * njacY[0][1][j-1][i][k];
          lhsY[0][2][AA][j][i][k] = - dtty2 * fjacY[0][2][j-1][i][k]
            - dtty1 * njacY[0][2][j-1][i][k];
          lhsY[0][3][AA][j][i][k] = - dtty2 * fjacY[0][3][j-1][i][k]
            - dtty1 * njacY[0][3][j-1][i][k];
          lhsY[0][4][AA][j][i][k] = - dtty2 * fjacY[0][4][j-1][i][k]
            - dtty1 * njacY[0][4][j-1][i][k];

          lhsY[1][0][AA][j][i][k] = - dtty2 * fjacY[1][0][j-1][i][k]
            - dtty1 * njacY[1][0][j-1][i][k];
          lhsY[1][1][AA][j][i][k] = - dtty2 * fjacY[1][1][j-1][i][k]
            - dtty1 * njacY[1][1][j-1][i][k]
            - dtty1 * dy2;
          lhsY[1][2][AA][j][i][k] = - dtty2 * fjacY[1][2][j-1][i][k]
            - dtty1 * njacY[1][2][j-1][i][k];
          lhsY[1][3][AA][j][i][k] = - dtty2 * fjacY[1][3][j-1][i][k]
            - dtty1 * njacY[1][3][j-1][i][k];
          lhsY[1][4][AA][j][i][k] = - dtty2 * fjacY[1][4][j-1][i][k]
            - dtty1 * njacY[1][4][j-1][i][k];

          lhsY[2][0][AA][j][i][k] = - dtty2 * fjacY[2][0][j-1][i][k]
            - dtty1 * njacY[2][0][j-1][i][k];
          lhsY[2][1][AA][j][i][k] = - dtty2 * fjacY[2][1][j-1][i][k]
            - dtty1 * njacY[2][1][j-1][i][k];
          lhsY[2][2][AA][j][i][k] = - dtty2 * fjacY[2][2][j-1][i][k]
            - dtty1 * njacY[2][2][j-1][i][k]
            - dtty1 * dy3;
          lhsY[2][3][AA][j][i][k] = - dtty2 * fjacY[2][3][j-1][i][k]
            - dtty1 * njacY[2][3][j-1][i][k];
          lhsY[2][4][AA][j][i][k] = - dtty2 * fjacY[2][4][j-1][i][k]
            - dtty1 * njacY[2][4][j-1][i][k];

          lhsY[3][0][AA][j][i][k] = - dtty2 * fjacY[3][0][j-1][i][k]
            - dtty1 * njacY[3][0][j-1][i][k];
          lhsY[3][1][AA][j][i][k] = - dtty2 * fjacY[3][1][j-1][i][k]
            - dtty1 * njacY[3][1][j-1][i][k];
          lhsY[3][2][AA][j][i][k] = - dtty2 * fjacY[3][2][j-1][i][k]
            - dtty1 * njacY[3][2][j-1][i][k];
          lhsY[3][3][AA][j][i][k] = - dtty2 * fjacY[3][3][j-1][i][k]
            - dtty1 * njacY[3][3][j-1][i][k]
            - dtty1 * dy4;
          lhsY[3][4][AA][j][i][k] = - dtty2 * fjacY[3][4][j-1][i][k]
            - dtty1 * njacY[3][4][j-1][i][k];

          lhsY[4][0][AA][j][i][k] = - dtty2 * fjacY[4][0][j-1][i][k]
            - dtty1 * njacY[4][0][j-1][i][k];
          lhsY[4][1][AA][j][i][k] = - dtty2 * fjacY[4][1][j-1][i][k]
            - dtty1 * njacY[4][1][j-1][i][k];
          lhsY[4][2][AA][j][i][k] = - dtty2 * fjacY[4][2][j-1][i][k]
            - dtty1 * njacY[4][2][j-1][i][k];
          lhsY[4][3][AA][j][i][k] = - dtty2 * fjacY[4][3][j-1][i][k]
            - dtty1 * njacY[4][3][j-1][i][k];
          lhsY[4][4][AA][j][i][k] = - dtty2 * fjacY[4][4][j-1][i][k]
            - dtty1 * njacY[4][4][j-1][i][k]
            - dtty1 * dy5;

          lhsY[0][0][BB][j][i][k] = 1.0
            + dtty1 * 2.0 * njacY[0][0][j][i][k]
            + dtty1 * 2.0 * dy1;
          lhsY[0][1][BB][j][i][k] = dtty1 * 2.0 * njacY[0][1][j][i][k];
          lhsY[0][2][BB][j][i][k] = dtty1 * 2.0 * njacY[0][2][j][i][k];
          lhsY[0][3][BB][j][i][k] = dtty1 * 2.0 * njacY[0][3][j][i][k];
          lhsY[0][4][BB][j][i][k] = dtty1 * 2.0 * njacY[0][4][j][i][k];

          lhsY[1][0][BB][j][i][k] = dtty1 * 2.0 * njacY[1][0][j][i][k];
          lhsY[1][1][BB][j][i][k] = 1.0
            + dtty1 * 2.0 * njacY[1][1][j][i][k]
            + dtty1 * 2.0 * dy2;
          lhsY[1][2][BB][j][i][k] = dtty1 * 2.0 * njacY[1][2][j][i][k];
          lhsY[1][3][BB][j][i][k] = dtty1 * 2.0 * njacY[1][3][j][i][k];
          lhsY[1][4][BB][j][i][k] = dtty1 * 2.0 * njacY[1][4][j][i][k];

          lhsY[2][0][BB][j][i][k] = dtty1 * 2.0 * njacY[2][0][j][i][k];
          lhsY[2][1][BB][j][i][k] = dtty1 * 2.0 * njacY[2][1][j][i][k];
          lhsY[2][2][BB][j][i][k] = 1.0
            + dtty1 * 2.0 * njacY[2][2][j][i][k]
            + dtty1 * 2.0 * dy3;
          lhsY[2][3][BB][j][i][k] = dtty1 * 2.0 * njacY[2][3][j][i][k];
          lhsY[2][4][BB][j][i][k] = dtty1 * 2.0 * njacY[2][4][j][i][k];

          lhsY[3][0][BB][j][i][k] = dtty1 * 2.0 * njacY[3][0][j][i][k];
          lhsY[3][1][BB][j][i][k] = dtty1 * 2.0 * njacY[3][1][j][i][k];
          lhsY[3][2][BB][j][i][k] = dtty1 * 2.0 * njacY[3][2][j][i][k];
          lhsY[3][3][BB][j][i][k] = 1.0
            + dtty1 * 2.0 * njacY[3][3][j][i][k]
            + dtty1 * 2.0 * dy4;
          lhsY[3][4][BB][j][i][k] = dtty1 * 2.0 * njacY[3][4][j][i][k];

          lhsY[4][0][BB][j][i][k] = dtty1 * 2.0 * njacY[4][0][j][i][k];
          lhsY[4][1][BB][j][i][k] = dtty1 * 2.0 * njacY[4][1][j][i][k];
          lhsY[4][2][BB][j][i][k] = dtty1 * 2.0 * njacY[4][2][j][i][k];
          lhsY[4][3][BB][j][i][k] = dtty1 * 2.0 * njacY[4][3][j][i][k];
          lhsY[4][4][BB][j][i][k] = 1.0
            + dtty1 * 2.0 * njacY[4][4][j][i][k]
            + dtty1 * 2.0 * dy5;

          lhsY[0][0][CC][j][i][k] =  dtty2 * fjacY[0][0][j+1][i][k]
            - dtty1 * njacY[0][0][j+1][i][k]
            - dtty1 * dy1;
          lhsY[0][1][CC][j][i][k] =  dtty2 * fjacY[0][1][j+1][i][k]
            - dtty1 * njacY[0][1][j+1][i][k];
          lhsY[0][2][CC][j][i][k] =  dtty2 * fjacY[0][2][j+1][i][k]
            - dtty1 * njacY[0][2][j+1][i][k];
          lhsY[0][3][CC][j][i][k] =  dtty2 * fjacY[0][3][j+1][i][k]
            - dtty1 * njacY[0][3][j+1][i][k];
          lhsY[0][4][CC][j][i][k] =  dtty2 * fjacY[0][4][j+1][i][k]
            - dtty1 * njacY[0][4][j+1][i][k];

          lhsY[1][0][CC][j][i][k] =  dtty2 * fjacY[1][0][j+1][i][k]
            - dtty1 * njacY[1][0][j+1][i][k];
          lhsY[1][1][CC][j][i][k] =  dtty2 * fjacY[1][1][j+1][i][k]
            - dtty1 * njacY[1][1][j+1][i][k]
            - dtty1 * dy2;
          lhsY[1][2][CC][j][i][k] =  dtty2 * fjacY[1][2][j+1][i][k]
            - dtty1 * njacY[1][2][j+1][i][k];
          lhsY[1][3][CC][j][i][k] =  dtty2 * fjacY[1][3][j+1][i][k]
            - dtty1 * njacY[1][3][j+1][i][k];
          lhsY[1][4][CC][j][i][k] =  dtty2 * fjacY[1][4][j+1][i][k]
            - dtty1 * njacY[1][4][j+1][i][k];

          lhsY[2][0][CC][j][i][k] =  dtty2 * fjacY[2][0][j+1][i][k]
            - dtty1 * njacY[2][0][j+1][i][k];
          lhsY[2][1][CC][j][i][k] =  dtty2 * fjacY[2][1][j+1][i][k]
            - dtty1 * njacY[2][1][j+1][i][k];
          lhsY[2][2][CC][j][i][k] =  dtty2 * fjacY[2][2][j+1][i][k]
            - dtty1 * njacY[2][2][j+1][i][k]
            - dtty1 * dy3;
          lhsY[2][3][CC][j][i][k] =  dtty2 * fjacY[2][3][j+1][i][k]
            - dtty1 * njacY[2][3][j+1][i][k];
          lhsY[2][4][CC][j][i][k] =  dtty2 * fjacY[2][4][j+1][i][k]
            - dtty1 * njacY[2][4][j+1][i][k];

          lhsY[3][0][CC][j][i][k] =  dtty2 * fjacY[3][0][j+1][i][k]
            - dtty1 * njacY[3][0][j+1][i][k];
          lhsY[3][1][CC][j][i][k] =  dtty2 * fjacY[3][1][j+1][i][k]
            - dtty1 * njacY[3][1][j+1][i][k];
          lhsY[3][2][CC][j][i][k] =  dtty2 * fjacY[3][2][j+1][i][k]
            - dtty1 * njacY[3][2][j+1][i][k];
          lhsY[3][3][CC][j][i][k] =  dtty2 * fjacY[3][3][j+1][i][k]
            - dtty1 * njacY[3][3][j+1][i][k]
            - dtty1 * dy4;
          lhsY[3][4][CC][j][i][k] =  dtty2 * fjacY[3][4][j+1][i][k]
            - dtty1 * njacY[3][4][j+1][i][k];

          lhsY[4][0][CC][j][i][k] =  dtty2 * fjacY[4][0][j+1][i][k]
            - dtty1 * njacY[4][0][j+1][i][k];
          lhsY[4][1][CC][j][i][k] =  dtty2 * fjacY[4][1][j+1][i][k]
            - dtty1 * njacY[4][1][j+1][i][k];
          lhsY[4][2][CC][j][i][k] =  dtty2 * fjacY[4][2][j+1][i][k]
            - dtty1 * njacY[4][2][j+1][i][k];
          lhsY[4][3][CC][j][i][k] =  dtty2 * fjacY[4][3][j+1][i][k]
            - dtty1 * njacY[4][3][j+1][i][k];
          lhsY[4][4][CC][j][i][k] =  dtty2 * fjacY[4][4][j+1][i][k]
            - dtty1 * njacY[4][4][j+1][i][k]
            - dtty1 * dy5;
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
    // c'(JMAX) and rhs'(JMAX) will be sent to next cell
    //---------------------------------------------------------------------

    //---------------------------------------------------------------------
    // multiply c[k][0][i] by b_inverse and copy back to c
    // multiply rhs(0) by b_inverse(0) and copy to rhs
    //---------------------------------------------------------------------
    //binvcrhs( lhsY[0][i][BB], lhsY[k][0][i][k][CC], rhs[k][0][i] );
    size_t kernel_y_solve_4_off[2] = { 1, 1 };
    size_t kernel_y_solve_4_idx[2] = { gp22, gp02 };
    brisbane_kernel kernel_y_solve_4;
    brisbane_kernel_create("y_solve_4", &kernel_y_solve_4);
    brisbane_kernel_setmem(kernel_y_solve_4, 0, mem_lhsY, brisbane_rw);
    brisbane_kernel_setmem(kernel_y_solve_4, 1, mem_rhs, brisbane_rw);

    brisbane_task task4;
    brisbane_task_create(&task4);
    brisbane_task_kernel(task4, kernel_y_solve_4, 2, kernel_y_solve_4_off, kernel_y_solve_4_idx);
    brisbane_task_submit(task4, brisbane_cpu, NULL, true);
#if 0
#ifdef SPEC_USE_INNER_SIMD
    #pragma omp target teams distribute parallel for private(i,k,pivot, coeff)
#else
    #pragma omp target teams distribute parallel for simd private(pivot, coeff) collapse(2)
#endif
    for (i = 1; i <= gp02; i++) {
#ifdef SPEC_USE_INNER_SIMD
      #pragma omp simd private(pivot, coeff)
#endif
      for (k = 1; k <= gp22; k++) {
        /*
        for(m = 0; m < 5; m++){
          pivot = 1.00/lhsY[m][m][BB][0][i][k];
          for(n = m+1; n < 5; n++){
            lhsY[m][n][BB][0][i][k] = lhsY[m][n][BB][0][i][k]*pivot;
          }
          lhsY[m][0][CC][0][i][k] = lhsY[m][0][CC][0][i][k]*pivot;
          lhsY[m][1][CC][0][i][k] = lhsY[m][1][CC][0][i][k]*pivot;
          lhsY[m][2][CC][0][i][k] = lhsY[m][2][CC][0][i][k]*pivot;
          lhsY[m][3][CC][0][i][k] = lhsY[m][3][CC][0][i][k]*pivot;
          lhsY[m][4][CC][0][i][k] = lhsY[m][4][CC][0][i][k]*pivot;
          rhs[k][0][i][m] = rhs[k][0][i][m]*pivot;

          for(n = 0; n < 5; n++){
            if(n != m){
              coeff = lhsY[n][m][BB][0][i][k];
              for(z = m+1; z < 5; z++){
                lhsY[n][z][BB][0][i][k] = lhsY[n][z][BB][0][i][k] - coeff*lhsY[m][z][BB][0][i][k];
              }
              lhsY[n][0][CC][0][i][k] = lhsY[n][0][CC][0][i][k] - coeff*lhsY[m][0][CC][0][i][k];
              lhsY[n][1][CC][0][i][k] = lhsY[n][1][CC][0][i][k] - coeff*lhsY[m][1][CC][0][i][k];
              lhsY[n][2][CC][0][i][k] = lhsY[n][2][CC][0][i][k] - coeff*lhsY[m][2][CC][0][i][k];
              lhsY[n][3][CC][0][i][k] = lhsY[n][3][CC][0][i][k] - coeff*lhsY[m][3][CC][0][i][k];
              lhsY[n][4][CC][0][i][k] = lhsY[n][4][CC][0][i][k] - coeff*lhsY[m][4][CC][0][i][k];
              rhs[k][0][i][n] = rhs[k][0][i][n] - coeff*rhs[k][0][i][m];
            }
          }
        }
        */
        pivot = 1.00/lhsY[0][0][BB][0][i][k];
        lhsY[0][1][BB][0][i][k] = lhsY[0][1][BB][0][i][k]*pivot;
        lhsY[0][2][BB][0][i][k] = lhsY[0][2][BB][0][i][k]*pivot;
        lhsY[0][3][BB][0][i][k] = lhsY[0][3][BB][0][i][k]*pivot;
        lhsY[0][4][BB][0][i][k] = lhsY[0][4][BB][0][i][k]*pivot;
        lhsY[0][0][CC][0][i][k] = lhsY[0][0][CC][0][i][k]*pivot;
        lhsY[0][1][CC][0][i][k] = lhsY[0][1][CC][0][i][k]*pivot;
        lhsY[0][2][CC][0][i][k] = lhsY[0][2][CC][0][i][k]*pivot;
        lhsY[0][3][CC][0][i][k] = lhsY[0][3][CC][0][i][k]*pivot;
        lhsY[0][4][CC][0][i][k] = lhsY[0][4][CC][0][i][k]*pivot;
        rhs[k][0][i][0]   = rhs[k][0][i][0]  *pivot;

        coeff = lhsY[1][0][BB][0][i][k];
        lhsY[1][1][BB][0][i][k]= lhsY[1][1][BB][0][i][k] - coeff*lhsY[0][1][BB][0][i][k];
        lhsY[1][2][BB][0][i][k]= lhsY[1][2][BB][0][i][k] - coeff*lhsY[0][2][BB][0][i][k];
        lhsY[1][3][BB][0][i][k]= lhsY[1][3][BB][0][i][k] - coeff*lhsY[0][3][BB][0][i][k];
        lhsY[1][4][BB][0][i][k]= lhsY[1][4][BB][0][i][k] - coeff*lhsY[0][4][BB][0][i][k];
        lhsY[1][0][CC][0][i][k] = lhsY[1][0][CC][0][i][k] - coeff*lhsY[0][0][CC][0][i][k];
        lhsY[1][1][CC][0][i][k] = lhsY[1][1][CC][0][i][k] - coeff*lhsY[0][1][CC][0][i][k];
        lhsY[1][2][CC][0][i][k] = lhsY[1][2][CC][0][i][k] - coeff*lhsY[0][2][CC][0][i][k];
        lhsY[1][3][CC][0][i][k] = lhsY[1][3][CC][0][i][k] - coeff*lhsY[0][3][CC][0][i][k];
        lhsY[1][4][CC][0][i][k] = lhsY[1][4][CC][0][i][k] - coeff*lhsY[0][4][CC][0][i][k];
        rhs[k][0][i][1]   = rhs[k][0][i][1]   - coeff*rhs[k][0][i][0];

        coeff = lhsY[2][0][BB][0][i][k];
        lhsY[2][1][BB][0][i][k]= lhsY[2][1][BB][0][i][k] - coeff*lhsY[0][1][BB][0][i][k];
        lhsY[2][2][BB][0][i][k]= lhsY[2][2][BB][0][i][k] - coeff*lhsY[0][2][BB][0][i][k];
        lhsY[2][3][BB][0][i][k]= lhsY[2][3][BB][0][i][k] - coeff*lhsY[0][3][BB][0][i][k];
        lhsY[2][4][BB][0][i][k]= lhsY[2][4][BB][0][i][k] - coeff*lhsY[0][4][BB][0][i][k];
        lhsY[2][0][CC][0][i][k] = lhsY[2][0][CC][0][i][k] - coeff*lhsY[0][0][CC][0][i][k];
        lhsY[2][1][CC][0][i][k] = lhsY[2][1][CC][0][i][k] - coeff*lhsY[0][1][CC][0][i][k];
        lhsY[2][2][CC][0][i][k] = lhsY[2][2][CC][0][i][k] - coeff*lhsY[0][2][CC][0][i][k];
        lhsY[2][3][CC][0][i][k] = lhsY[2][3][CC][0][i][k] - coeff*lhsY[0][3][CC][0][i][k];
        lhsY[2][4][CC][0][i][k] = lhsY[2][4][CC][0][i][k] - coeff*lhsY[0][4][CC][0][i][k];
        rhs[k][0][i][2]   = rhs[k][0][i][2]   - coeff*rhs[k][0][i][0];

        coeff = lhsY[3][0][BB][0][i][k];
        lhsY[3][1][BB][0][i][k]= lhsY[3][1][BB][0][i][k] - coeff*lhsY[0][1][BB][0][i][k];
        lhsY[3][2][BB][0][i][k]= lhsY[3][2][BB][0][i][k] - coeff*lhsY[0][2][BB][0][i][k];
        lhsY[3][3][BB][0][i][k]= lhsY[3][3][BB][0][i][k] - coeff*lhsY[0][3][BB][0][i][k];
        lhsY[3][4][BB][0][i][k]= lhsY[3][4][BB][0][i][k] - coeff*lhsY[0][4][BB][0][i][k];
        lhsY[3][0][CC][0][i][k] = lhsY[3][0][CC][0][i][k] - coeff*lhsY[0][0][CC][0][i][k];
        lhsY[3][1][CC][0][i][k] = lhsY[3][1][CC][0][i][k] - coeff*lhsY[0][1][CC][0][i][k];
        lhsY[3][2][CC][0][i][k] = lhsY[3][2][CC][0][i][k] - coeff*lhsY[0][2][CC][0][i][k];
        lhsY[3][3][CC][0][i][k] = lhsY[3][3][CC][0][i][k] - coeff*lhsY[0][3][CC][0][i][k];
        lhsY[3][4][CC][0][i][k] = lhsY[3][4][CC][0][i][k] - coeff*lhsY[0][4][CC][0][i][k];
        rhs[k][0][i][3]   = rhs[k][0][i][3]   - coeff*rhs[k][0][i][0];

        coeff = lhsY[4][0][BB][0][i][k];
        lhsY[4][1][BB][0][i][k]= lhsY[4][1][BB][0][i][k] - coeff*lhsY[0][1][BB][0][i][k];
        lhsY[4][2][BB][0][i][k]= lhsY[4][2][BB][0][i][k] - coeff*lhsY[0][2][BB][0][i][k];
        lhsY[4][3][BB][0][i][k]= lhsY[4][3][BB][0][i][k] - coeff*lhsY[0][3][BB][0][i][k];
        lhsY[4][4][BB][0][i][k]= lhsY[4][4][BB][0][i][k] - coeff*lhsY[0][4][BB][0][i][k];
        lhsY[4][0][CC][0][i][k] = lhsY[4][0][CC][0][i][k] - coeff*lhsY[0][0][CC][0][i][k];
        lhsY[4][1][CC][0][i][k] = lhsY[4][1][CC][0][i][k] - coeff*lhsY[0][1][CC][0][i][k];
        lhsY[4][2][CC][0][i][k] = lhsY[4][2][CC][0][i][k] - coeff*lhsY[0][2][CC][0][i][k];
        lhsY[4][3][CC][0][i][k] = lhsY[4][3][CC][0][i][k] - coeff*lhsY[0][3][CC][0][i][k];
        lhsY[4][4][CC][0][i][k] = lhsY[4][4][CC][0][i][k] - coeff*lhsY[0][4][CC][0][i][k];
        rhs[k][0][i][4]   = rhs[k][0][i][4]   - coeff*rhs[k][0][i][0];


        pivot = 1.00/lhsY[1][1][BB][0][i][k];
        lhsY[1][2][BB][0][i][k] = lhsY[1][2][BB][0][i][k]*pivot;
        lhsY[1][3][BB][0][i][k] = lhsY[1][3][BB][0][i][k]*pivot;
        lhsY[1][4][BB][0][i][k] = lhsY[1][4][BB][0][i][k]*pivot;
        lhsY[1][0][CC][0][i][k] = lhsY[1][0][CC][0][i][k]*pivot;
        lhsY[1][1][CC][0][i][k] = lhsY[1][1][CC][0][i][k]*pivot;
        lhsY[1][2][CC][0][i][k] = lhsY[1][2][CC][0][i][k]*pivot;
        lhsY[1][3][CC][0][i][k] = lhsY[1][3][CC][0][i][k]*pivot;
        lhsY[1][4][CC][0][i][k] = lhsY[1][4][CC][0][i][k]*pivot;
        rhs[k][0][i][1]   = rhs[k][0][i][1]  *pivot;

        coeff = lhsY[0][1][BB][0][i][k];
        lhsY[0][2][BB][0][i][k]= lhsY[0][2][BB][0][i][k] - coeff*lhsY[1][2][BB][0][i][k];
        lhsY[0][3][BB][0][i][k]= lhsY[0][3][BB][0][i][k] - coeff*lhsY[1][3][BB][0][i][k];
        lhsY[0][4][BB][0][i][k]= lhsY[0][4][BB][0][i][k] - coeff*lhsY[1][4][BB][0][i][k];
        lhsY[0][0][CC][0][i][k] = lhsY[0][0][CC][0][i][k] - coeff*lhsY[1][0][CC][0][i][k];
        lhsY[0][1][CC][0][i][k] = lhsY[0][1][CC][0][i][k] - coeff*lhsY[1][1][CC][0][i][k];
        lhsY[0][2][CC][0][i][k] = lhsY[0][2][CC][0][i][k] - coeff*lhsY[1][2][CC][0][i][k];
        lhsY[0][3][CC][0][i][k] = lhsY[0][3][CC][0][i][k] - coeff*lhsY[1][3][CC][0][i][k];
        lhsY[0][4][CC][0][i][k] = lhsY[0][4][CC][0][i][k] - coeff*lhsY[1][4][CC][0][i][k];
        rhs[k][0][i][0]   = rhs[k][0][i][0]   - coeff*rhs[k][0][i][1];

        coeff = lhsY[2][1][BB][0][i][k];
        lhsY[2][2][BB][0][i][k]= lhsY[2][2][BB][0][i][k] - coeff*lhsY[1][2][BB][0][i][k];
        lhsY[2][3][BB][0][i][k]= lhsY[2][3][BB][0][i][k] - coeff*lhsY[1][3][BB][0][i][k];
        lhsY[2][4][BB][0][i][k]= lhsY[2][4][BB][0][i][k] - coeff*lhsY[1][4][BB][0][i][k];
        lhsY[2][0][CC][0][i][k] = lhsY[2][0][CC][0][i][k] - coeff*lhsY[1][0][CC][0][i][k];
        lhsY[2][1][CC][0][i][k] = lhsY[2][1][CC][0][i][k] - coeff*lhsY[1][1][CC][0][i][k];
        lhsY[2][2][CC][0][i][k] = lhsY[2][2][CC][0][i][k] - coeff*lhsY[1][2][CC][0][i][k];
        lhsY[2][3][CC][0][i][k] = lhsY[2][3][CC][0][i][k] - coeff*lhsY[1][3][CC][0][i][k];
        lhsY[2][4][CC][0][i][k] = lhsY[2][4][CC][0][i][k] - coeff*lhsY[1][4][CC][0][i][k];
        rhs[k][0][i][2]   = rhs[k][0][i][2]   - coeff*rhs[k][0][i][1];

        coeff = lhsY[3][1][BB][0][i][k];
        lhsY[3][2][BB][0][i][k]= lhsY[3][2][BB][0][i][k] - coeff*lhsY[1][2][BB][0][i][k];
        lhsY[3][3][BB][0][i][k]= lhsY[3][3][BB][0][i][k] - coeff*lhsY[1][3][BB][0][i][k];
        lhsY[3][4][BB][0][i][k]= lhsY[3][4][BB][0][i][k] - coeff*lhsY[1][4][BB][0][i][k];
        lhsY[3][0][CC][0][i][k] = lhsY[3][0][CC][0][i][k] - coeff*lhsY[1][0][CC][0][i][k];
        lhsY[3][1][CC][0][i][k] = lhsY[3][1][CC][0][i][k] - coeff*lhsY[1][1][CC][0][i][k];
        lhsY[3][2][CC][0][i][k] = lhsY[3][2][CC][0][i][k] - coeff*lhsY[1][2][CC][0][i][k];
        lhsY[3][3][CC][0][i][k] = lhsY[3][3][CC][0][i][k] - coeff*lhsY[1][3][CC][0][i][k];
        lhsY[3][4][CC][0][i][k] = lhsY[3][4][CC][0][i][k] - coeff*lhsY[1][4][CC][0][i][k];
        rhs[k][0][i][3]   = rhs[k][0][i][3]   - coeff*rhs[k][0][i][1];

        coeff = lhsY[4][1][BB][0][i][k];
        lhsY[4][2][BB][0][i][k]= lhsY[4][2][BB][0][i][k] - coeff*lhsY[1][2][BB][0][i][k];
        lhsY[4][3][BB][0][i][k]= lhsY[4][3][BB][0][i][k] - coeff*lhsY[1][3][BB][0][i][k];
        lhsY[4][4][BB][0][i][k]= lhsY[4][4][BB][0][i][k] - coeff*lhsY[1][4][BB][0][i][k];
        lhsY[4][0][CC][0][i][k] = lhsY[4][0][CC][0][i][k] - coeff*lhsY[1][0][CC][0][i][k];
        lhsY[4][1][CC][0][i][k] = lhsY[4][1][CC][0][i][k] - coeff*lhsY[1][1][CC][0][i][k];
        lhsY[4][2][CC][0][i][k] = lhsY[4][2][CC][0][i][k] - coeff*lhsY[1][2][CC][0][i][k];
        lhsY[4][3][CC][0][i][k] = lhsY[4][3][CC][0][i][k] - coeff*lhsY[1][3][CC][0][i][k];
        lhsY[4][4][CC][0][i][k] = lhsY[4][4][CC][0][i][k] - coeff*lhsY[1][4][CC][0][i][k];
        rhs[k][0][i][4]   = rhs[k][0][i][4]   - coeff*rhs[k][0][i][1];


        pivot = 1.00/lhsY[2][2][BB][0][i][k];
        lhsY[2][3][BB][0][i][k] = lhsY[2][3][BB][0][i][k]*pivot;
        lhsY[2][4][BB][0][i][k] = lhsY[2][4][BB][0][i][k]*pivot;
        lhsY[2][0][CC][0][i][k] = lhsY[2][0][CC][0][i][k]*pivot;
        lhsY[2][1][CC][0][i][k] = lhsY[2][1][CC][0][i][k]*pivot;
        lhsY[2][2][CC][0][i][k] = lhsY[2][2][CC][0][i][k]*pivot;
        lhsY[2][3][CC][0][i][k] = lhsY[2][3][CC][0][i][k]*pivot;
        lhsY[2][4][CC][0][i][k] = lhsY[2][4][CC][0][i][k]*pivot;
        rhs[k][0][i][2]   = rhs[k][0][i][2]  *pivot;

        coeff = lhsY[0][2][BB][0][i][k];
        lhsY[0][3][BB][0][i][k]= lhsY[0][3][BB][0][i][k] - coeff*lhsY[2][3][BB][0][i][k];
        lhsY[0][4][BB][0][i][k]= lhsY[0][4][BB][0][i][k] - coeff*lhsY[2][4][BB][0][i][k];
        lhsY[0][0][CC][0][i][k] = lhsY[0][0][CC][0][i][k] - coeff*lhsY[2][0][CC][0][i][k];
        lhsY[0][1][CC][0][i][k] = lhsY[0][1][CC][0][i][k] - coeff*lhsY[2][1][CC][0][i][k];
        lhsY[0][2][CC][0][i][k] = lhsY[0][2][CC][0][i][k] - coeff*lhsY[2][2][CC][0][i][k];
        lhsY[0][3][CC][0][i][k] = lhsY[0][3][CC][0][i][k] - coeff*lhsY[2][3][CC][0][i][k];
        lhsY[0][4][CC][0][i][k] = lhsY[0][4][CC][0][i][k] - coeff*lhsY[2][4][CC][0][i][k];
        rhs[k][0][i][0]   = rhs[k][0][i][0]   - coeff*rhs[k][0][i][2];

        coeff = lhsY[1][2][BB][0][i][k];
        lhsY[1][3][BB][0][i][k]= lhsY[1][3][BB][0][i][k] - coeff*lhsY[2][3][BB][0][i][k];
        lhsY[1][4][BB][0][i][k]= lhsY[1][4][BB][0][i][k] - coeff*lhsY[2][4][BB][0][i][k];
        lhsY[1][0][CC][0][i][k] = lhsY[1][0][CC][0][i][k] - coeff*lhsY[2][0][CC][0][i][k];
        lhsY[1][1][CC][0][i][k] = lhsY[1][1][CC][0][i][k] - coeff*lhsY[2][1][CC][0][i][k];
        lhsY[1][2][CC][0][i][k] = lhsY[1][2][CC][0][i][k] - coeff*lhsY[2][2][CC][0][i][k];
        lhsY[1][3][CC][0][i][k] = lhsY[1][3][CC][0][i][k] - coeff*lhsY[2][3][CC][0][i][k];
        lhsY[1][4][CC][0][i][k] = lhsY[1][4][CC][0][i][k] - coeff*lhsY[2][4][CC][0][i][k];
        rhs[k][0][i][1]   = rhs[k][0][i][1]   - coeff*rhs[k][0][i][2];

        coeff = lhsY[3][2][BB][0][i][k];
        lhsY[3][3][BB][0][i][k]= lhsY[3][3][BB][0][i][k] - coeff*lhsY[2][3][BB][0][i][k];
        lhsY[3][4][BB][0][i][k]= lhsY[3][4][BB][0][i][k] - coeff*lhsY[2][4][BB][0][i][k];
        lhsY[3][0][CC][0][i][k] = lhsY[3][0][CC][0][i][k] - coeff*lhsY[2][0][CC][0][i][k];
        lhsY[3][1][CC][0][i][k] = lhsY[3][1][CC][0][i][k] - coeff*lhsY[2][1][CC][0][i][k];
        lhsY[3][2][CC][0][i][k] = lhsY[3][2][CC][0][i][k] - coeff*lhsY[2][2][CC][0][i][k];
        lhsY[3][3][CC][0][i][k] = lhsY[3][3][CC][0][i][k] - coeff*lhsY[2][3][CC][0][i][k];
        lhsY[3][4][CC][0][i][k] = lhsY[3][4][CC][0][i][k] - coeff*lhsY[2][4][CC][0][i][k];
        rhs[k][0][i][3]   = rhs[k][0][i][3]   - coeff*rhs[k][0][i][2];

        coeff = lhsY[4][2][BB][0][i][k];
        lhsY[4][3][BB][0][i][k]= lhsY[4][3][BB][0][i][k] - coeff*lhsY[2][3][BB][0][i][k];
        lhsY[4][4][BB][0][i][k]= lhsY[4][4][BB][0][i][k] - coeff*lhsY[2][4][BB][0][i][k];
        lhsY[4][0][CC][0][i][k] = lhsY[4][0][CC][0][i][k] - coeff*lhsY[2][0][CC][0][i][k];
        lhsY[4][1][CC][0][i][k] = lhsY[4][1][CC][0][i][k] - coeff*lhsY[2][1][CC][0][i][k];
        lhsY[4][2][CC][0][i][k] = lhsY[4][2][CC][0][i][k] - coeff*lhsY[2][2][CC][0][i][k];
        lhsY[4][3][CC][0][i][k] = lhsY[4][3][CC][0][i][k] - coeff*lhsY[2][3][CC][0][i][k];
        lhsY[4][4][CC][0][i][k] = lhsY[4][4][CC][0][i][k] - coeff*lhsY[2][4][CC][0][i][k];
        rhs[k][0][i][4]   = rhs[k][0][i][4]   - coeff*rhs[k][0][i][2];


        pivot = 1.00/lhsY[3][3][BB][0][i][k];
        lhsY[3][4][BB][0][i][k] = lhsY[3][4][BB][0][i][k]*pivot;
        lhsY[3][0][CC][0][i][k] = lhsY[3][0][CC][0][i][k]*pivot;
        lhsY[3][1][CC][0][i][k] = lhsY[3][1][CC][0][i][k]*pivot;
        lhsY[3][2][CC][0][i][k] = lhsY[3][2][CC][0][i][k]*pivot;
        lhsY[3][3][CC][0][i][k] = lhsY[3][3][CC][0][i][k]*pivot;
        lhsY[3][4][CC][0][i][k] = lhsY[3][4][CC][0][i][k]*pivot;
        rhs[k][0][i][3]   = rhs[k][0][i][3]  *pivot;

        coeff = lhsY[0][3][BB][0][i][k];
        lhsY[0][4][BB][0][i][k]= lhsY[0][4][BB][0][i][k] - coeff*lhsY[3][4][BB][0][i][k];
        lhsY[0][0][CC][0][i][k] = lhsY[0][0][CC][0][i][k] - coeff*lhsY[3][0][CC][0][i][k];
        lhsY[0][1][CC][0][i][k] = lhsY[0][1][CC][0][i][k] - coeff*lhsY[3][1][CC][0][i][k];
        lhsY[0][2][CC][0][i][k] = lhsY[0][2][CC][0][i][k] - coeff*lhsY[3][2][CC][0][i][k];
        lhsY[0][3][CC][0][i][k] = lhsY[0][3][CC][0][i][k] - coeff*lhsY[3][3][CC][0][i][k];
        lhsY[0][4][CC][0][i][k] = lhsY[0][4][CC][0][i][k] - coeff*lhsY[3][4][CC][0][i][k];
        rhs[k][0][i][0]   = rhs[k][0][i][0]   - coeff*rhs[k][0][i][3];

        coeff = lhsY[1][3][BB][0][i][k];
        lhsY[1][4][BB][0][i][k]= lhsY[1][4][BB][0][i][k] - coeff*lhsY[3][4][BB][0][i][k];
        lhsY[1][0][CC][0][i][k] = lhsY[1][0][CC][0][i][k] - coeff*lhsY[3][0][CC][0][i][k];
        lhsY[1][1][CC][0][i][k] = lhsY[1][1][CC][0][i][k] - coeff*lhsY[3][1][CC][0][i][k];
        lhsY[1][2][CC][0][i][k] = lhsY[1][2][CC][0][i][k] - coeff*lhsY[3][2][CC][0][i][k];
        lhsY[1][3][CC][0][i][k] = lhsY[1][3][CC][0][i][k] - coeff*lhsY[3][3][CC][0][i][k];
        lhsY[1][4][CC][0][i][k] = lhsY[1][4][CC][0][i][k] - coeff*lhsY[3][4][CC][0][i][k];
        rhs[k][0][i][1]   = rhs[k][0][i][1]   - coeff*rhs[k][0][i][3];

        coeff = lhsY[2][3][BB][0][i][k];
        lhsY[2][4][BB][0][i][k]= lhsY[2][4][BB][0][i][k] - coeff*lhsY[3][4][BB][0][i][k];
        lhsY[2][0][CC][0][i][k] = lhsY[2][0][CC][0][i][k] - coeff*lhsY[3][0][CC][0][i][k];
        lhsY[2][1][CC][0][i][k] = lhsY[2][1][CC][0][i][k] - coeff*lhsY[3][1][CC][0][i][k];
        lhsY[2][2][CC][0][i][k] = lhsY[2][2][CC][0][i][k] - coeff*lhsY[3][2][CC][0][i][k];
        lhsY[2][3][CC][0][i][k] = lhsY[2][3][CC][0][i][k] - coeff*lhsY[3][3][CC][0][i][k];
        lhsY[2][4][CC][0][i][k] = lhsY[2][4][CC][0][i][k] - coeff*lhsY[3][4][CC][0][i][k];
        rhs[k][0][i][2]   = rhs[k][0][i][2]   - coeff*rhs[k][0][i][3];

        coeff = lhsY[4][3][BB][0][i][k];
        lhsY[4][4][BB][0][i][k]= lhsY[4][4][BB][0][i][k] - coeff*lhsY[3][4][BB][0][i][k];
        lhsY[4][0][CC][0][i][k] = lhsY[4][0][CC][0][i][k] - coeff*lhsY[3][0][CC][0][i][k];
        lhsY[4][1][CC][0][i][k] = lhsY[4][1][CC][0][i][k] - coeff*lhsY[3][1][CC][0][i][k];
        lhsY[4][2][CC][0][i][k] = lhsY[4][2][CC][0][i][k] - coeff*lhsY[3][2][CC][0][i][k];
        lhsY[4][3][CC][0][i][k] = lhsY[4][3][CC][0][i][k] - coeff*lhsY[3][3][CC][0][i][k];
        lhsY[4][4][CC][0][i][k] = lhsY[4][4][CC][0][i][k] - coeff*lhsY[3][4][CC][0][i][k];
        rhs[k][0][i][4]   = rhs[k][0][i][4]   - coeff*rhs[k][0][i][3];


        pivot = 1.00/lhsY[4][4][BB][0][i][k];
        lhsY[4][0][CC][0][i][k] = lhsY[4][0][CC][0][i][k]*pivot;
        lhsY[4][1][CC][0][i][k] = lhsY[4][1][CC][0][i][k]*pivot;
        lhsY[4][2][CC][0][i][k] = lhsY[4][2][CC][0][i][k]*pivot;
        lhsY[4][3][CC][0][i][k] = lhsY[4][3][CC][0][i][k]*pivot;
        lhsY[4][4][CC][0][i][k] = lhsY[4][4][CC][0][i][k]*pivot;
        rhs[k][0][i][4]   = rhs[k][0][i][4]  *pivot;

        coeff = lhsY[0][4][BB][0][i][k];
        lhsY[0][0][CC][0][i][k] = lhsY[0][0][CC][0][i][k] - coeff*lhsY[4][0][CC][0][i][k];
        lhsY[0][1][CC][0][i][k] = lhsY[0][1][CC][0][i][k] - coeff*lhsY[4][1][CC][0][i][k];
        lhsY[0][2][CC][0][i][k] = lhsY[0][2][CC][0][i][k] - coeff*lhsY[4][2][CC][0][i][k];
        lhsY[0][3][CC][0][i][k] = lhsY[0][3][CC][0][i][k] - coeff*lhsY[4][3][CC][0][i][k];
        lhsY[0][4][CC][0][i][k] = lhsY[0][4][CC][0][i][k] - coeff*lhsY[4][4][CC][0][i][k];
        rhs[k][0][i][0]   = rhs[k][0][i][0]   - coeff*rhs[k][0][i][4];

        coeff = lhsY[1][4][BB][0][i][k];
        lhsY[1][0][CC][0][i][k] = lhsY[1][0][CC][0][i][k] - coeff*lhsY[4][0][CC][0][i][k];
        lhsY[1][1][CC][0][i][k] = lhsY[1][1][CC][0][i][k] - coeff*lhsY[4][1][CC][0][i][k];
        lhsY[1][2][CC][0][i][k] = lhsY[1][2][CC][0][i][k] - coeff*lhsY[4][2][CC][0][i][k];
        lhsY[1][3][CC][0][i][k] = lhsY[1][3][CC][0][i][k] - coeff*lhsY[4][3][CC][0][i][k];
        lhsY[1][4][CC][0][i][k] = lhsY[1][4][CC][0][i][k] - coeff*lhsY[4][4][CC][0][i][k];
        rhs[k][0][i][1]   = rhs[k][0][i][1]   - coeff*rhs[k][0][i][4];

        coeff = lhsY[2][4][BB][0][i][k];
        lhsY[2][0][CC][0][i][k] = lhsY[2][0][CC][0][i][k] - coeff*lhsY[4][0][CC][0][i][k];
        lhsY[2][1][CC][0][i][k] = lhsY[2][1][CC][0][i][k] - coeff*lhsY[4][1][CC][0][i][k];
        lhsY[2][2][CC][0][i][k] = lhsY[2][2][CC][0][i][k] - coeff*lhsY[4][2][CC][0][i][k];
        lhsY[2][3][CC][0][i][k] = lhsY[2][3][CC][0][i][k] - coeff*lhsY[4][3][CC][0][i][k];
        lhsY[2][4][CC][0][i][k] = lhsY[2][4][CC][0][i][k] - coeff*lhsY[4][4][CC][0][i][k];
        rhs[k][0][i][2]   = rhs[k][0][i][2]   - coeff*rhs[k][0][i][4];

        coeff = lhsY[3][4][BB][0][i][k];
        lhsY[3][0][CC][0][i][k] = lhsY[3][0][CC][0][i][k] - coeff*lhsY[4][0][CC][0][i][k];
        lhsY[3][1][CC][0][i][k] = lhsY[3][1][CC][0][i][k] - coeff*lhsY[4][1][CC][0][i][k];
        lhsY[3][2][CC][0][i][k] = lhsY[3][2][CC][0][i][k] - coeff*lhsY[4][2][CC][0][i][k];
        lhsY[3][3][CC][0][i][k] = lhsY[3][3][CC][0][i][k] - coeff*lhsY[4][3][CC][0][i][k];
        lhsY[3][4][CC][0][i][k] = lhsY[3][4][CC][0][i][k] - coeff*lhsY[4][4][CC][0][i][k];
        rhs[k][0][i][3]   = rhs[k][0][i][3]   - coeff*rhs[k][0][i][4];


      }/*end i*/
    }/*end k*/
#endif

    //---------------------------------------------------------------------
    // begin inner most do loop
    // do all the elements of the cell unless last
    //---------------------------------------------------------------------
    size_t kernel_y_solve_5_off[1] = { 1 };
    size_t kernel_y_solve_5_idx[1] = { gp02 };
    brisbane_kernel kernel_y_solve_5;
    brisbane_kernel_create("y_solve_5", &kernel_y_solve_5);
    brisbane_kernel_setmem(kernel_y_solve_5, 0, mem_lhsY, brisbane_rw);
    brisbane_kernel_setmem(kernel_y_solve_5, 1, mem_rhs, brisbane_rw);
    brisbane_kernel_setarg(kernel_y_solve_5, 2, sizeof(int), &jsize);
    brisbane_kernel_setarg(kernel_y_solve_5, 3, sizeof(int), &gp22);

    brisbane_task task5;
    brisbane_task_create(&task5);
    brisbane_task_kernel(task5, kernel_y_solve_5, 1, kernel_y_solve_5_off, kernel_y_solve_5_idx);
    brisbane_task_submit(task5, brisbane_cpu, NULL, true);
#if 0
    #pragma omp target teams distribute parallel for private(j,k)
    for (i = 1; i <= gp02; i++) {
      for (j = 1; j <= jsize-1; j++) {
        #pragma omp simd private(pivot,coeff)
        for (k = 1; k <= gp22; k++) {
          //-------------------------------------------------------------------
          // subtract A*lhsY[k][i]_vector(j-1) from lhsY[k][i]_vector(j)
          //
          // rhs(j) = rhs(j) - A*rhs(j-1)
          //-------------------------------------------------------------------
          //matvec_sub(lhsY[i][j-1][AA], rhs[k][j][i][k], rhs[k][j][i]);
          /*
          for(m = 0; m < 5; m++){
            rhs[k][j][i][m] = rhs[k][j][i][m] - lhsY[m][0][AA][j][i][k]*rhs[k][j-1][i][0]
              - lhsY[m][1][AA][j][i][k]*rhs[k][j-1][i][1]
              - lhsY[m][2][AA][j][i][k]*rhs[k][j-1][i][2]
              - lhsY[m][3][AA][j][i][k]*rhs[k][j-1][i][3]
              - lhsY[m][4][AA][j][i][k]*rhs[k][j-1][i][4];
          }
          */

          rhs[k][j][i][0] = rhs[k][j][i][0] - lhsY[0][0][AA][j][i][k]*rhs[k][j-1][i][0]
            - lhsY[0][1][AA][j][i][k]*rhs[k][j-1][i][1]
            - lhsY[0][2][AA][j][i][k]*rhs[k][j-1][i][2]
            - lhsY[0][3][AA][j][i][k]*rhs[k][j-1][i][3]
            - lhsY[0][4][AA][j][i][k]*rhs[k][j-1][i][4];
          rhs[k][j][i][1] = rhs[k][j][i][1] - lhsY[1][0][AA][j][i][k]*rhs[k][j-1][i][0]
            - lhsY[1][1][AA][j][i][k]*rhs[k][j-1][i][1]
            - lhsY[1][2][AA][j][i][k]*rhs[k][j-1][i][2]
            - lhsY[1][3][AA][j][i][k]*rhs[k][j-1][i][3]
            - lhsY[1][4][AA][j][i][k]*rhs[k][j-1][i][4];
          rhs[k][j][i][2] = rhs[k][j][i][2] - lhsY[2][0][AA][j][i][k]*rhs[k][j-1][i][0]
            - lhsY[2][1][AA][j][i][k]*rhs[k][j-1][i][1]
            - lhsY[2][2][AA][j][i][k]*rhs[k][j-1][i][2]
            - lhsY[2][3][AA][j][i][k]*rhs[k][j-1][i][3]
            - lhsY[2][4][AA][j][i][k]*rhs[k][j-1][i][4];
          rhs[k][j][i][3] = rhs[k][j][i][3] - lhsY[3][0][AA][j][i][k]*rhs[k][j-1][i][0]
            - lhsY[3][1][AA][j][i][k]*rhs[k][j-1][i][1]
            - lhsY[3][2][AA][j][i][k]*rhs[k][j-1][i][2]
            - lhsY[3][3][AA][j][i][k]*rhs[k][j-1][i][3]
            - lhsY[3][4][AA][j][i][k]*rhs[k][j-1][i][4];
          rhs[k][j][i][4] = rhs[k][j][i][4] - lhsY[4][0][AA][j][i][k]*rhs[k][j-1][i][0]
            - lhsY[4][1][AA][j][i][k]*rhs[k][j-1][i][1]
            - lhsY[4][2][AA][j][i][k]*rhs[k][j-1][i][2]
            - lhsY[4][3][AA][j][i][k]*rhs[k][j-1][i][3]
            - lhsY[4][4][AA][j][i][k]*rhs[k][j-1][i][4];

          //-------------------------------------------------------------------
          // B(j) = B(j) - C(j-1)*A(j)
          //-------------------------------------------------------------------
          //  matmul_sub(lhsY[j-1][i][AA], lhsY[k][j][i][k][CC], lhsY[k][i][j][BB]);
          /*
          for(m = 0; m < 5; m++){
            for(n = 0; n < 5; n++){
              lhsY[n][m][BB][j][i][k] = lhsY[n][m][BB][j][i][k] - lhsY[n][0][AA][j][i][k]*lhsY[0][m][CC][j-1][i][k]
                - lhsY[n][1][AA][j][i][k]*lhsY[1][m][CC][j-1][i][k]
                - lhsY[n][2][AA][j][i][k]*lhsY[2][m][CC][j-1][i][k]
                - lhsY[n][3][AA][j][i][k]*lhsY[3][m][CC][j-1][i][k]
                - lhsY[n][4][AA][j][i][k]*lhsY[4][m][CC][j-1][i][k];
            }
          }
          */

          lhsY[0][0][BB][j][i][k] = lhsY[0][0][BB][j][i][k] - lhsY[0][0][AA][j][i][k]*lhsY[0][0][CC][j-1][i][k]
            - lhsY[0][1][AA][j][i][k]*lhsY[1][0][CC][j-1][i][k]
            - lhsY[0][2][AA][j][i][k]*lhsY[2][0][CC][j-1][i][k]
            - lhsY[0][3][AA][j][i][k]*lhsY[3][0][CC][j-1][i][k]
            - lhsY[0][4][AA][j][i][k]*lhsY[4][0][CC][j-1][i][k];
          lhsY[1][0][BB][j][i][k] = lhsY[1][0][BB][j][i][k] - lhsY[1][0][AA][j][i][k]*lhsY[0][0][CC][j-1][i][k]
            - lhsY[1][1][AA][j][i][k]*lhsY[1][0][CC][j-1][i][k]
            - lhsY[1][2][AA][j][i][k]*lhsY[2][0][CC][j-1][i][k]
            - lhsY[1][3][AA][j][i][k]*lhsY[3][0][CC][j-1][i][k]
            - lhsY[1][4][AA][j][i][k]*lhsY[4][0][CC][j-1][i][k];
          lhsY[2][0][BB][j][i][k] = lhsY[2][0][BB][j][i][k] - lhsY[2][0][AA][j][i][k]*lhsY[0][0][CC][j-1][i][k]
            - lhsY[2][1][AA][j][i][k]*lhsY[1][0][CC][j-1][i][k]
            - lhsY[2][2][AA][j][i][k]*lhsY[2][0][CC][j-1][i][k]
            - lhsY[2][3][AA][j][i][k]*lhsY[3][0][CC][j-1][i][k]
            - lhsY[2][4][AA][j][i][k]*lhsY[4][0][CC][j-1][i][k];
          lhsY[3][0][BB][j][i][k] = lhsY[3][0][BB][j][i][k] - lhsY[3][0][AA][j][i][k]*lhsY[0][0][CC][j-1][i][k]
            - lhsY[3][1][AA][j][i][k]*lhsY[1][0][CC][j-1][i][k]
            - lhsY[3][2][AA][j][i][k]*lhsY[2][0][CC][j-1][i][k]
            - lhsY[3][3][AA][j][i][k]*lhsY[3][0][CC][j-1][i][k]
            - lhsY[3][4][AA][j][i][k]*lhsY[4][0][CC][j-1][i][k];
          lhsY[4][0][BB][j][i][k] = lhsY[4][0][BB][j][i][k] - lhsY[4][0][AA][j][i][k]*lhsY[0][0][CC][j-1][i][k]
            - lhsY[4][1][AA][j][i][k]*lhsY[1][0][CC][j-1][i][k]
            - lhsY[4][2][AA][j][i][k]*lhsY[2][0][CC][j-1][i][k]
            - lhsY[4][3][AA][j][i][k]*lhsY[3][0][CC][j-1][i][k]
            - lhsY[4][4][AA][j][i][k]*lhsY[4][0][CC][j-1][i][k];
          lhsY[0][1][BB][j][i][k] = lhsY[0][1][BB][j][i][k] - lhsY[0][0][AA][j][i][k]*lhsY[0][1][CC][j-1][i][k]
            - lhsY[0][1][AA][j][i][k]*lhsY[1][1][CC][j-1][i][k]
            - lhsY[0][2][AA][j][i][k]*lhsY[2][1][CC][j-1][i][k]
            - lhsY[0][3][AA][j][i][k]*lhsY[3][1][CC][j-1][i][k]
            - lhsY[0][4][AA][j][i][k]*lhsY[4][1][CC][j-1][i][k];
          lhsY[1][1][BB][j][i][k] = lhsY[1][1][BB][j][i][k] - lhsY[1][0][AA][j][i][k]*lhsY[0][1][CC][j-1][i][k]
            - lhsY[1][1][AA][j][i][k]*lhsY[1][1][CC][j-1][i][k]
            - lhsY[1][2][AA][j][i][k]*lhsY[2][1][CC][j-1][i][k]
            - lhsY[1][3][AA][j][i][k]*lhsY[3][1][CC][j-1][i][k]
            - lhsY[1][4][AA][j][i][k]*lhsY[4][1][CC][j-1][i][k];
          lhsY[2][1][BB][j][i][k] = lhsY[2][1][BB][j][i][k] - lhsY[2][0][AA][j][i][k]*lhsY[0][1][CC][j-1][i][k]
            - lhsY[2][1][AA][j][i][k]*lhsY[1][1][CC][j-1][i][k]
            - lhsY[2][2][AA][j][i][k]*lhsY[2][1][CC][j-1][i][k]
            - lhsY[2][3][AA][j][i][k]*lhsY[3][1][CC][j-1][i][k]
            - lhsY[2][4][AA][j][i][k]*lhsY[4][1][CC][j-1][i][k];
          lhsY[3][1][BB][j][i][k] = lhsY[3][1][BB][j][i][k] - lhsY[3][0][AA][j][i][k]*lhsY[0][1][CC][j-1][i][k]
            - lhsY[3][1][AA][j][i][k]*lhsY[1][1][CC][j-1][i][k]
            - lhsY[3][2][AA][j][i][k]*lhsY[2][1][CC][j-1][i][k]
            - lhsY[3][3][AA][j][i][k]*lhsY[3][1][CC][j-1][i][k]
            - lhsY[3][4][AA][j][i][k]*lhsY[4][1][CC][j-1][i][k];
          lhsY[4][1][BB][j][i][k] = lhsY[4][1][BB][j][i][k] - lhsY[4][0][AA][j][i][k]*lhsY[0][1][CC][j-1][i][k]
            - lhsY[4][1][AA][j][i][k]*lhsY[1][1][CC][j-1][i][k]
            - lhsY[4][2][AA][j][i][k]*lhsY[2][1][CC][j-1][i][k]
            - lhsY[4][3][AA][j][i][k]*lhsY[3][1][CC][j-1][i][k]
            - lhsY[4][4][AA][j][i][k]*lhsY[4][1][CC][j-1][i][k];
          lhsY[0][2][BB][j][i][k] = lhsY[0][2][BB][j][i][k] - lhsY[0][0][AA][j][i][k]*lhsY[0][2][CC][j-1][i][k]
            - lhsY[0][1][AA][j][i][k]*lhsY[1][2][CC][j-1][i][k]
            - lhsY[0][2][AA][j][i][k]*lhsY[2][2][CC][j-1][i][k]
            - lhsY[0][3][AA][j][i][k]*lhsY[3][2][CC][j-1][i][k]
            - lhsY[0][4][AA][j][i][k]*lhsY[4][2][CC][j-1][i][k];
          lhsY[1][2][BB][j][i][k] = lhsY[1][2][BB][j][i][k] - lhsY[1][0][AA][j][i][k]*lhsY[0][2][CC][j-1][i][k]
            - lhsY[1][1][AA][j][i][k]*lhsY[1][2][CC][j-1][i][k]
            - lhsY[1][2][AA][j][i][k]*lhsY[2][2][CC][j-1][i][k]
            - lhsY[1][3][AA][j][i][k]*lhsY[3][2][CC][j-1][i][k]
            - lhsY[1][4][AA][j][i][k]*lhsY[4][2][CC][j-1][i][k];
          lhsY[2][2][BB][j][i][k] = lhsY[2][2][BB][j][i][k] - lhsY[2][0][AA][j][i][k]*lhsY[0][2][CC][j-1][i][k]
            - lhsY[2][1][AA][j][i][k]*lhsY[1][2][CC][j-1][i][k]
            - lhsY[2][2][AA][j][i][k]*lhsY[2][2][CC][j-1][i][k]
            - lhsY[2][3][AA][j][i][k]*lhsY[3][2][CC][j-1][i][k]
            - lhsY[2][4][AA][j][i][k]*lhsY[4][2][CC][j-1][i][k];
          lhsY[3][2][BB][j][i][k] = lhsY[3][2][BB][j][i][k] - lhsY[3][0][AA][j][i][k]*lhsY[0][2][CC][j-1][i][k]
            - lhsY[3][1][AA][j][i][k]*lhsY[1][2][CC][j-1][i][k]
            - lhsY[3][2][AA][j][i][k]*lhsY[2][2][CC][j-1][i][k]
            - lhsY[3][3][AA][j][i][k]*lhsY[3][2][CC][j-1][i][k]
            - lhsY[3][4][AA][j][i][k]*lhsY[4][2][CC][j-1][i][k];
          lhsY[4][2][BB][j][i][k] = lhsY[4][2][BB][j][i][k] - lhsY[4][0][AA][j][i][k]*lhsY[0][2][CC][j-1][i][k]
            - lhsY[4][1][AA][j][i][k]*lhsY[1][2][CC][j-1][i][k]
            - lhsY[4][2][AA][j][i][k]*lhsY[2][2][CC][j-1][i][k]
            - lhsY[4][3][AA][j][i][k]*lhsY[3][2][CC][j-1][i][k]
            - lhsY[4][4][AA][j][i][k]*lhsY[4][2][CC][j-1][i][k];
          lhsY[0][3][BB][j][i][k] = lhsY[0][3][BB][j][i][k] - lhsY[0][0][AA][j][i][k]*lhsY[0][3][CC][j-1][i][k]
            - lhsY[0][1][AA][j][i][k]*lhsY[1][3][CC][j-1][i][k]
            - lhsY[0][2][AA][j][i][k]*lhsY[2][3][CC][j-1][i][k]
            - lhsY[0][3][AA][j][i][k]*lhsY[3][3][CC][j-1][i][k]
            - lhsY[0][4][AA][j][i][k]*lhsY[4][3][CC][j-1][i][k];
          lhsY[1][3][BB][j][i][k] = lhsY[1][3][BB][j][i][k] - lhsY[1][0][AA][j][i][k]*lhsY[0][3][CC][j-1][i][k]
            - lhsY[1][1][AA][j][i][k]*lhsY[1][3][CC][j-1][i][k]
            - lhsY[1][2][AA][j][i][k]*lhsY[2][3][CC][j-1][i][k]
            - lhsY[1][3][AA][j][i][k]*lhsY[3][3][CC][j-1][i][k]
            - lhsY[1][4][AA][j][i][k]*lhsY[4][3][CC][j-1][i][k];
          lhsY[2][3][BB][j][i][k] = lhsY[2][3][BB][j][i][k] - lhsY[2][0][AA][j][i][k]*lhsY[0][3][CC][j-1][i][k]
            - lhsY[2][1][AA][j][i][k]*lhsY[1][3][CC][j-1][i][k]
            - lhsY[2][2][AA][j][i][k]*lhsY[2][3][CC][j-1][i][k]
            - lhsY[2][3][AA][j][i][k]*lhsY[3][3][CC][j-1][i][k]
            - lhsY[2][4][AA][j][i][k]*lhsY[4][3][CC][j-1][i][k];
          lhsY[3][3][BB][j][i][k] = lhsY[3][3][BB][j][i][k] - lhsY[3][0][AA][j][i][k]*lhsY[0][3][CC][j-1][i][k]
            - lhsY[3][1][AA][j][i][k]*lhsY[1][3][CC][j-1][i][k]
            - lhsY[3][2][AA][j][i][k]*lhsY[2][3][CC][j-1][i][k]
            - lhsY[3][3][AA][j][i][k]*lhsY[3][3][CC][j-1][i][k]
            - lhsY[3][4][AA][j][i][k]*lhsY[4][3][CC][j-1][i][k];
          lhsY[4][3][BB][j][i][k] = lhsY[4][3][BB][j][i][k] - lhsY[4][0][AA][j][i][k]*lhsY[0][3][CC][j-1][i][k]
            - lhsY[4][1][AA][j][i][k]*lhsY[1][3][CC][j-1][i][k]
            - lhsY[4][2][AA][j][i][k]*lhsY[2][3][CC][j-1][i][k]
            - lhsY[4][3][AA][j][i][k]*lhsY[3][3][CC][j-1][i][k]
            - lhsY[4][4][AA][j][i][k]*lhsY[4][3][CC][j-1][i][k];
          lhsY[0][4][BB][j][i][k] = lhsY[0][4][BB][j][i][k] - lhsY[0][0][AA][j][i][k]*lhsY[0][4][CC][j-1][i][k]
            - lhsY[0][1][AA][j][i][k]*lhsY[1][4][CC][j-1][i][k]
            - lhsY[0][2][AA][j][i][k]*lhsY[2][4][CC][j-1][i][k]
            - lhsY[0][3][AA][j][i][k]*lhsY[3][4][CC][j-1][i][k]
            - lhsY[0][4][AA][j][i][k]*lhsY[4][4][CC][j-1][i][k];
          lhsY[1][4][BB][j][i][k] = lhsY[1][4][BB][j][i][k] - lhsY[1][0][AA][j][i][k]*lhsY[0][4][CC][j-1][i][k]
            - lhsY[1][1][AA][j][i][k]*lhsY[1][4][CC][j-1][i][k]
            - lhsY[1][2][AA][j][i][k]*lhsY[2][4][CC][j-1][i][k]
            - lhsY[1][3][AA][j][i][k]*lhsY[3][4][CC][j-1][i][k]
            - lhsY[1][4][AA][j][i][k]*lhsY[4][4][CC][j-1][i][k];
          lhsY[2][4][BB][j][i][k] = lhsY[2][4][BB][j][i][k] - lhsY[2][0][AA][j][i][k]*lhsY[0][4][CC][j-1][i][k]
            - lhsY[2][1][AA][j][i][k]*lhsY[1][4][CC][j-1][i][k]
            - lhsY[2][2][AA][j][i][k]*lhsY[2][4][CC][j-1][i][k]
            - lhsY[2][3][AA][j][i][k]*lhsY[3][4][CC][j-1][i][k]
            - lhsY[2][4][AA][j][i][k]*lhsY[4][4][CC][j-1][i][k];
          lhsY[3][4][BB][j][i][k] = lhsY[3][4][BB][j][i][k] - lhsY[3][0][AA][j][i][k]*lhsY[0][4][CC][j-1][i][k]
            - lhsY[3][1][AA][j][i][k]*lhsY[1][4][CC][j-1][i][k]
            - lhsY[3][2][AA][j][i][k]*lhsY[2][4][CC][j-1][i][k]
            - lhsY[3][3][AA][j][i][k]*lhsY[3][4][CC][j-1][i][k]
            - lhsY[3][4][AA][j][i][k]*lhsY[4][4][CC][j-1][i][k];
          lhsY[4][4][BB][j][i][k] = lhsY[4][4][BB][j][i][k] - lhsY[4][0][AA][j][i][k]*lhsY[0][4][CC][j-1][i][k]
            - lhsY[4][1][AA][j][i][k]*lhsY[1][4][CC][j-1][i][k]
            - lhsY[4][2][AA][j][i][k]*lhsY[2][4][CC][j-1][i][k]
            - lhsY[4][3][AA][j][i][k]*lhsY[3][4][CC][j-1][i][k]
            - lhsY[4][4][AA][j][i][k]*lhsY[4][4][CC][j-1][i][k];


          //-------------------------------------------------------------------
          // multiply c[k][j][i] by b_inverse and copy back to c
          // multiply rhs[k][0][i] by b_inverse[k][0][i] and copy to rhs
          //-------------------------------------------------------------------
          //binvcrhs( lhsY[j][i][BB], lhsY[k][j][i][k][CC], rhs[k][j][i] );
          /*
          for(m = 0; m < 5; m++){
            pivot = 1.00/lhsY[m][m][BB][j][i][k];
            for(n = m+1; n < 5; n++){
              lhsY[m][n][BB][j][i][k] = lhsY[m][n][BB][j][i][k]*pivot;
            }
            lhsY[m][0][CC][j][i][k] = lhsY[m][0][CC][j][i][k]*pivot;
            lhsY[m][1][CC][j][i][k] = lhsY[m][1][CC][j][i][k]*pivot;
            lhsY[m][2][CC][j][i][k] = lhsY[m][2][CC][j][i][k]*pivot;
            lhsY[m][3][CC][j][i][k] = lhsY[m][3][CC][j][i][k]*pivot;
            lhsY[m][4][CC][j][i][k] = lhsY[m][4][CC][j][i][k]*pivot;
            rhs[k][j][i][m] = rhs[k][j][i][m]*pivot;

            for(n = 0; n < 5; n++){
              if(n != m){
                coeff = lhsY[n][m][BB][j][i][k];
                for(z = m+1; z < 5; z++){
                  lhsY[n][z][BB][j][i][k] = lhsY[n][z][BB][j][i][k] - coeff*lhsY[m][z][BB][j][i][k];
                }
                lhsY[n][0][CC][j][i][k] = lhsY[n][0][CC][j][i][k] - coeff*lhsY[m][0][CC][j][i][k];
                lhsY[n][1][CC][j][i][k] = lhsY[n][1][CC][j][i][k] - coeff*lhsY[m][1][CC][j][i][k];
                lhsY[n][2][CC][j][i][k] = lhsY[n][2][CC][j][i][k] - coeff*lhsY[m][2][CC][j][i][k];
                lhsY[n][3][CC][j][i][k] = lhsY[n][3][CC][j][i][k] - coeff*lhsY[m][3][CC][j][i][k];
                lhsY[n][4][CC][j][i][k] = lhsY[n][4][CC][j][i][k] - coeff*lhsY[m][4][CC][j][i][k];
                rhs[k][j][i][n] = rhs[k][j][i][n] - coeff*rhs[k][j][i][m];
              }
            }
          }
          */
          pivot = 1.00/lhsY[0][0][BB][j][i][k];
          lhsY[0][1][BB][j][i][k] = lhsY[0][1][BB][j][i][k]*pivot;
          lhsY[0][2][BB][j][i][k] = lhsY[0][2][BB][j][i][k]*pivot;
          lhsY[0][3][BB][j][i][k] = lhsY[0][3][BB][j][i][k]*pivot;
          lhsY[0][4][BB][j][i][k] = lhsY[0][4][BB][j][i][k]*pivot;
          lhsY[0][0][CC][j][i][k] = lhsY[0][0][CC][j][i][k]*pivot;
          lhsY[0][1][CC][j][i][k] = lhsY[0][1][CC][j][i][k]*pivot;
          lhsY[0][2][CC][j][i][k] = lhsY[0][2][CC][j][i][k]*pivot;
          lhsY[0][3][CC][j][i][k] = lhsY[0][3][CC][j][i][k]*pivot;
          lhsY[0][4][CC][j][i][k] = lhsY[0][4][CC][j][i][k]*pivot;
          rhs[k][j][i][0]   = rhs[k][j][i][0]  *pivot;

          coeff = lhsY[1][0][BB][j][i][k];
          lhsY[1][1][BB][j][i][k]= lhsY[1][1][BB][j][i][k] - coeff*lhsY[0][1][BB][j][i][k];
          lhsY[1][2][BB][j][i][k]= lhsY[1][2][BB][j][i][k] - coeff*lhsY[0][2][BB][j][i][k];
          lhsY[1][3][BB][j][i][k]= lhsY[1][3][BB][j][i][k] - coeff*lhsY[0][3][BB][j][i][k];
          lhsY[1][4][BB][j][i][k]= lhsY[1][4][BB][j][i][k] - coeff*lhsY[0][4][BB][j][i][k];
          lhsY[1][0][CC][j][i][k] = lhsY[1][0][CC][j][i][k] - coeff*lhsY[0][0][CC][j][i][k];
          lhsY[1][1][CC][j][i][k] = lhsY[1][1][CC][j][i][k] - coeff*lhsY[0][1][CC][j][i][k];
          lhsY[1][2][CC][j][i][k] = lhsY[1][2][CC][j][i][k] - coeff*lhsY[0][2][CC][j][i][k];
          lhsY[1][3][CC][j][i][k] = lhsY[1][3][CC][j][i][k] - coeff*lhsY[0][3][CC][j][i][k];
          lhsY[1][4][CC][j][i][k] = lhsY[1][4][CC][j][i][k] - coeff*lhsY[0][4][CC][j][i][k];
          rhs[k][j][i][1]   = rhs[k][j][i][1]   - coeff*rhs[k][j][i][0];

          coeff = lhsY[2][0][BB][j][i][k];
          lhsY[2][1][BB][j][i][k]= lhsY[2][1][BB][j][i][k] - coeff*lhsY[0][1][BB][j][i][k];
          lhsY[2][2][BB][j][i][k]= lhsY[2][2][BB][j][i][k] - coeff*lhsY[0][2][BB][j][i][k];
          lhsY[2][3][BB][j][i][k]= lhsY[2][3][BB][j][i][k] - coeff*lhsY[0][3][BB][j][i][k];
          lhsY[2][4][BB][j][i][k]= lhsY[2][4][BB][j][i][k] - coeff*lhsY[0][4][BB][j][i][k];
          lhsY[2][0][CC][j][i][k] = lhsY[2][0][CC][j][i][k] - coeff*lhsY[0][0][CC][j][i][k];
          lhsY[2][1][CC][j][i][k] = lhsY[2][1][CC][j][i][k] - coeff*lhsY[0][1][CC][j][i][k];
          lhsY[2][2][CC][j][i][k] = lhsY[2][2][CC][j][i][k] - coeff*lhsY[0][2][CC][j][i][k];
          lhsY[2][3][CC][j][i][k] = lhsY[2][3][CC][j][i][k] - coeff*lhsY[0][3][CC][j][i][k];
          lhsY[2][4][CC][j][i][k] = lhsY[2][4][CC][j][i][k] - coeff*lhsY[0][4][CC][j][i][k];
          rhs[k][j][i][2]   = rhs[k][j][i][2]   - coeff*rhs[k][j][i][0];

          coeff = lhsY[3][0][BB][j][i][k];
          lhsY[3][1][BB][j][i][k]= lhsY[3][1][BB][j][i][k] - coeff*lhsY[0][1][BB][j][i][k];
          lhsY[3][2][BB][j][i][k]= lhsY[3][2][BB][j][i][k] - coeff*lhsY[0][2][BB][j][i][k];
          lhsY[3][3][BB][j][i][k]= lhsY[3][3][BB][j][i][k] - coeff*lhsY[0][3][BB][j][i][k];
          lhsY[3][4][BB][j][i][k]= lhsY[3][4][BB][j][i][k] - coeff*lhsY[0][4][BB][j][i][k];
          lhsY[3][0][CC][j][i][k] = lhsY[3][0][CC][j][i][k] - coeff*lhsY[0][0][CC][j][i][k];
          lhsY[3][1][CC][j][i][k] = lhsY[3][1][CC][j][i][k] - coeff*lhsY[0][1][CC][j][i][k];
          lhsY[3][2][CC][j][i][k] = lhsY[3][2][CC][j][i][k] - coeff*lhsY[0][2][CC][j][i][k];
          lhsY[3][3][CC][j][i][k] = lhsY[3][3][CC][j][i][k] - coeff*lhsY[0][3][CC][j][i][k];
          lhsY[3][4][CC][j][i][k] = lhsY[3][4][CC][j][i][k] - coeff*lhsY[0][4][CC][j][i][k];
          rhs[k][j][i][3]   = rhs[k][j][i][3]   - coeff*rhs[k][j][i][0];

          coeff = lhsY[4][0][BB][j][i][k];
          lhsY[4][1][BB][j][i][k]= lhsY[4][1][BB][j][i][k] - coeff*lhsY[0][1][BB][j][i][k];
          lhsY[4][2][BB][j][i][k]= lhsY[4][2][BB][j][i][k] - coeff*lhsY[0][2][BB][j][i][k];
          lhsY[4][3][BB][j][i][k]= lhsY[4][3][BB][j][i][k] - coeff*lhsY[0][3][BB][j][i][k];
          lhsY[4][4][BB][j][i][k]= lhsY[4][4][BB][j][i][k] - coeff*lhsY[0][4][BB][j][i][k];
          lhsY[4][0][CC][j][i][k] = lhsY[4][0][CC][j][i][k] - coeff*lhsY[0][0][CC][j][i][k];
          lhsY[4][1][CC][j][i][k] = lhsY[4][1][CC][j][i][k] - coeff*lhsY[0][1][CC][j][i][k];
          lhsY[4][2][CC][j][i][k] = lhsY[4][2][CC][j][i][k] - coeff*lhsY[0][2][CC][j][i][k];
          lhsY[4][3][CC][j][i][k] = lhsY[4][3][CC][j][i][k] - coeff*lhsY[0][3][CC][j][i][k];
          lhsY[4][4][CC][j][i][k] = lhsY[4][4][CC][j][i][k] - coeff*lhsY[0][4][CC][j][i][k];
          rhs[k][j][i][4]   = rhs[k][j][i][4]   - coeff*rhs[k][j][i][0];


          pivot = 1.00/lhsY[1][1][BB][j][i][k];
          lhsY[1][2][BB][j][i][k] = lhsY[1][2][BB][j][i][k]*pivot;
          lhsY[1][3][BB][j][i][k] = lhsY[1][3][BB][j][i][k]*pivot;
          lhsY[1][4][BB][j][i][k] = lhsY[1][4][BB][j][i][k]*pivot;
          lhsY[1][0][CC][j][i][k] = lhsY[1][0][CC][j][i][k]*pivot;
          lhsY[1][1][CC][j][i][k] = lhsY[1][1][CC][j][i][k]*pivot;
          lhsY[1][2][CC][j][i][k] = lhsY[1][2][CC][j][i][k]*pivot;
          lhsY[1][3][CC][j][i][k] = lhsY[1][3][CC][j][i][k]*pivot;
          lhsY[1][4][CC][j][i][k] = lhsY[1][4][CC][j][i][k]*pivot;
          rhs[k][j][i][1]   = rhs[k][j][i][1]  *pivot;

          coeff = lhsY[0][1][BB][j][i][k];
          lhsY[0][2][BB][j][i][k]= lhsY[0][2][BB][j][i][k] - coeff*lhsY[1][2][BB][j][i][k];
          lhsY[0][3][BB][j][i][k]= lhsY[0][3][BB][j][i][k] - coeff*lhsY[1][3][BB][j][i][k];
          lhsY[0][4][BB][j][i][k]= lhsY[0][4][BB][j][i][k] - coeff*lhsY[1][4][BB][j][i][k];
          lhsY[0][0][CC][j][i][k] = lhsY[0][0][CC][j][i][k] - coeff*lhsY[1][0][CC][j][i][k];
          lhsY[0][1][CC][j][i][k] = lhsY[0][1][CC][j][i][k] - coeff*lhsY[1][1][CC][j][i][k];
          lhsY[0][2][CC][j][i][k] = lhsY[0][2][CC][j][i][k] - coeff*lhsY[1][2][CC][j][i][k];
          lhsY[0][3][CC][j][i][k] = lhsY[0][3][CC][j][i][k] - coeff*lhsY[1][3][CC][j][i][k];
          lhsY[0][4][CC][j][i][k] = lhsY[0][4][CC][j][i][k] - coeff*lhsY[1][4][CC][j][i][k];
          rhs[k][j][i][0]   = rhs[k][j][i][0]   - coeff*rhs[k][j][i][1];

          coeff = lhsY[2][1][BB][j][i][k];
          lhsY[2][2][BB][j][i][k]= lhsY[2][2][BB][j][i][k] - coeff*lhsY[1][2][BB][j][i][k];
          lhsY[2][3][BB][j][i][k]= lhsY[2][3][BB][j][i][k] - coeff*lhsY[1][3][BB][j][i][k];
          lhsY[2][4][BB][j][i][k]= lhsY[2][4][BB][j][i][k] - coeff*lhsY[1][4][BB][j][i][k];
          lhsY[2][0][CC][j][i][k] = lhsY[2][0][CC][j][i][k] - coeff*lhsY[1][0][CC][j][i][k];
          lhsY[2][1][CC][j][i][k] = lhsY[2][1][CC][j][i][k] - coeff*lhsY[1][1][CC][j][i][k];
          lhsY[2][2][CC][j][i][k] = lhsY[2][2][CC][j][i][k] - coeff*lhsY[1][2][CC][j][i][k];
          lhsY[2][3][CC][j][i][k] = lhsY[2][3][CC][j][i][k] - coeff*lhsY[1][3][CC][j][i][k];
          lhsY[2][4][CC][j][i][k] = lhsY[2][4][CC][j][i][k] - coeff*lhsY[1][4][CC][j][i][k];
          rhs[k][j][i][2]   = rhs[k][j][i][2]   - coeff*rhs[k][j][i][1];

          coeff = lhsY[3][1][BB][j][i][k];
          lhsY[3][2][BB][j][i][k]= lhsY[3][2][BB][j][i][k] - coeff*lhsY[1][2][BB][j][i][k];
          lhsY[3][3][BB][j][i][k]= lhsY[3][3][BB][j][i][k] - coeff*lhsY[1][3][BB][j][i][k];
          lhsY[3][4][BB][j][i][k]= lhsY[3][4][BB][j][i][k] - coeff*lhsY[1][4][BB][j][i][k];
          lhsY[3][0][CC][j][i][k] = lhsY[3][0][CC][j][i][k] - coeff*lhsY[1][0][CC][j][i][k];
          lhsY[3][1][CC][j][i][k] = lhsY[3][1][CC][j][i][k] - coeff*lhsY[1][1][CC][j][i][k];
          lhsY[3][2][CC][j][i][k] = lhsY[3][2][CC][j][i][k] - coeff*lhsY[1][2][CC][j][i][k];
          lhsY[3][3][CC][j][i][k] = lhsY[3][3][CC][j][i][k] - coeff*lhsY[1][3][CC][j][i][k];
          lhsY[3][4][CC][j][i][k] = lhsY[3][4][CC][j][i][k] - coeff*lhsY[1][4][CC][j][i][k];
          rhs[k][j][i][3]   = rhs[k][j][i][3]   - coeff*rhs[k][j][i][1];

          coeff = lhsY[4][1][BB][j][i][k];
          lhsY[4][2][BB][j][i][k]= lhsY[4][2][BB][j][i][k] - coeff*lhsY[1][2][BB][j][i][k];
          lhsY[4][3][BB][j][i][k]= lhsY[4][3][BB][j][i][k] - coeff*lhsY[1][3][BB][j][i][k];
          lhsY[4][4][BB][j][i][k]= lhsY[4][4][BB][j][i][k] - coeff*lhsY[1][4][BB][j][i][k];
          lhsY[4][0][CC][j][i][k] = lhsY[4][0][CC][j][i][k] - coeff*lhsY[1][0][CC][j][i][k];
          lhsY[4][1][CC][j][i][k] = lhsY[4][1][CC][j][i][k] - coeff*lhsY[1][1][CC][j][i][k];
          lhsY[4][2][CC][j][i][k] = lhsY[4][2][CC][j][i][k] - coeff*lhsY[1][2][CC][j][i][k];
          lhsY[4][3][CC][j][i][k] = lhsY[4][3][CC][j][i][k] - coeff*lhsY[1][3][CC][j][i][k];
          lhsY[4][4][CC][j][i][k] = lhsY[4][4][CC][j][i][k] - coeff*lhsY[1][4][CC][j][i][k];
          rhs[k][j][i][4]   = rhs[k][j][i][4]   - coeff*rhs[k][j][i][1];


          pivot = 1.00/lhsY[2][2][BB][j][i][k];
          lhsY[2][3][BB][j][i][k] = lhsY[2][3][BB][j][i][k]*pivot;
          lhsY[2][4][BB][j][i][k] = lhsY[2][4][BB][j][i][k]*pivot;
          lhsY[2][0][CC][j][i][k] = lhsY[2][0][CC][j][i][k]*pivot;
          lhsY[2][1][CC][j][i][k] = lhsY[2][1][CC][j][i][k]*pivot;
          lhsY[2][2][CC][j][i][k] = lhsY[2][2][CC][j][i][k]*pivot;
          lhsY[2][3][CC][j][i][k] = lhsY[2][3][CC][j][i][k]*pivot;
          lhsY[2][4][CC][j][i][k] = lhsY[2][4][CC][j][i][k]*pivot;
          rhs[k][j][i][2]   = rhs[k][j][i][2]  *pivot;

          coeff = lhsY[0][2][BB][j][i][k];
          lhsY[0][3][BB][j][i][k]= lhsY[0][3][BB][j][i][k] - coeff*lhsY[2][3][BB][j][i][k];
          lhsY[0][4][BB][j][i][k]= lhsY[0][4][BB][j][i][k] - coeff*lhsY[2][4][BB][j][i][k];
          lhsY[0][0][CC][j][i][k] = lhsY[0][0][CC][j][i][k] - coeff*lhsY[2][0][CC][j][i][k];
          lhsY[0][1][CC][j][i][k] = lhsY[0][1][CC][j][i][k] - coeff*lhsY[2][1][CC][j][i][k];
          lhsY[0][2][CC][j][i][k] = lhsY[0][2][CC][j][i][k] - coeff*lhsY[2][2][CC][j][i][k];
          lhsY[0][3][CC][j][i][k] = lhsY[0][3][CC][j][i][k] - coeff*lhsY[2][3][CC][j][i][k];
          lhsY[0][4][CC][j][i][k] = lhsY[0][4][CC][j][i][k] - coeff*lhsY[2][4][CC][j][i][k];
          rhs[k][j][i][0]   = rhs[k][j][i][0]   - coeff*rhs[k][j][i][2];

          coeff = lhsY[1][2][BB][j][i][k];
          lhsY[1][3][BB][j][i][k]= lhsY[1][3][BB][j][i][k] - coeff*lhsY[2][3][BB][j][i][k];
          lhsY[1][4][BB][j][i][k]= lhsY[1][4][BB][j][i][k] - coeff*lhsY[2][4][BB][j][i][k];
          lhsY[1][0][CC][j][i][k] = lhsY[1][0][CC][j][i][k] - coeff*lhsY[2][0][CC][j][i][k];
          lhsY[1][1][CC][j][i][k] = lhsY[1][1][CC][j][i][k] - coeff*lhsY[2][1][CC][j][i][k];
          lhsY[1][2][CC][j][i][k] = lhsY[1][2][CC][j][i][k] - coeff*lhsY[2][2][CC][j][i][k];
          lhsY[1][3][CC][j][i][k] = lhsY[1][3][CC][j][i][k] - coeff*lhsY[2][3][CC][j][i][k];
          lhsY[1][4][CC][j][i][k] = lhsY[1][4][CC][j][i][k] - coeff*lhsY[2][4][CC][j][i][k];
          rhs[k][j][i][1]   = rhs[k][j][i][1]   - coeff*rhs[k][j][i][2];

          coeff = lhsY[3][2][BB][j][i][k];
          lhsY[3][3][BB][j][i][k]= lhsY[3][3][BB][j][i][k] - coeff*lhsY[2][3][BB][j][i][k];
          lhsY[3][4][BB][j][i][k]= lhsY[3][4][BB][j][i][k] - coeff*lhsY[2][4][BB][j][i][k];
          lhsY[3][0][CC][j][i][k] = lhsY[3][0][CC][j][i][k] - coeff*lhsY[2][0][CC][j][i][k];
          lhsY[3][1][CC][j][i][k] = lhsY[3][1][CC][j][i][k] - coeff*lhsY[2][1][CC][j][i][k];
          lhsY[3][2][CC][j][i][k] = lhsY[3][2][CC][j][i][k] - coeff*lhsY[2][2][CC][j][i][k];
          lhsY[3][3][CC][j][i][k] = lhsY[3][3][CC][j][i][k] - coeff*lhsY[2][3][CC][j][i][k];
          lhsY[3][4][CC][j][i][k] = lhsY[3][4][CC][j][i][k] - coeff*lhsY[2][4][CC][j][i][k];
          rhs[k][j][i][3]   = rhs[k][j][i][3]   - coeff*rhs[k][j][i][2];

          coeff = lhsY[4][2][BB][j][i][k];
          lhsY[4][3][BB][j][i][k]= lhsY[4][3][BB][j][i][k] - coeff*lhsY[2][3][BB][j][i][k];
          lhsY[4][4][BB][j][i][k]= lhsY[4][4][BB][j][i][k] - coeff*lhsY[2][4][BB][j][i][k];
          lhsY[4][0][CC][j][i][k] = lhsY[4][0][CC][j][i][k] - coeff*lhsY[2][0][CC][j][i][k];
          lhsY[4][1][CC][j][i][k] = lhsY[4][1][CC][j][i][k] - coeff*lhsY[2][1][CC][j][i][k];
          lhsY[4][2][CC][j][i][k] = lhsY[4][2][CC][j][i][k] - coeff*lhsY[2][2][CC][j][i][k];
          lhsY[4][3][CC][j][i][k] = lhsY[4][3][CC][j][i][k] - coeff*lhsY[2][3][CC][j][i][k];
          lhsY[4][4][CC][j][i][k] = lhsY[4][4][CC][j][i][k] - coeff*lhsY[2][4][CC][j][i][k];
          rhs[k][j][i][4]   = rhs[k][j][i][4]   - coeff*rhs[k][j][i][2];


          pivot = 1.00/lhsY[3][3][BB][j][i][k];
          lhsY[3][4][BB][j][i][k] = lhsY[3][4][BB][j][i][k]*pivot;
          lhsY[3][0][CC][j][i][k] = lhsY[3][0][CC][j][i][k]*pivot;
          lhsY[3][1][CC][j][i][k] = lhsY[3][1][CC][j][i][k]*pivot;
          lhsY[3][2][CC][j][i][k] = lhsY[3][2][CC][j][i][k]*pivot;
          lhsY[3][3][CC][j][i][k] = lhsY[3][3][CC][j][i][k]*pivot;
          lhsY[3][4][CC][j][i][k] = lhsY[3][4][CC][j][i][k]*pivot;
          rhs[k][j][i][3]   = rhs[k][j][i][3]  *pivot;

          coeff = lhsY[0][3][BB][j][i][k];
          lhsY[0][4][BB][j][i][k]= lhsY[0][4][BB][j][i][k] - coeff*lhsY[3][4][BB][j][i][k];
          lhsY[0][0][CC][j][i][k] = lhsY[0][0][CC][j][i][k] - coeff*lhsY[3][0][CC][j][i][k];
          lhsY[0][1][CC][j][i][k] = lhsY[0][1][CC][j][i][k] - coeff*lhsY[3][1][CC][j][i][k];
          lhsY[0][2][CC][j][i][k] = lhsY[0][2][CC][j][i][k] - coeff*lhsY[3][2][CC][j][i][k];
          lhsY[0][3][CC][j][i][k] = lhsY[0][3][CC][j][i][k] - coeff*lhsY[3][3][CC][j][i][k];
          lhsY[0][4][CC][j][i][k] = lhsY[0][4][CC][j][i][k] - coeff*lhsY[3][4][CC][j][i][k];
          rhs[k][j][i][0]   = rhs[k][j][i][0]   - coeff*rhs[k][j][i][3];

          coeff = lhsY[1][3][BB][j][i][k];
          lhsY[1][4][BB][j][i][k]= lhsY[1][4][BB][j][i][k] - coeff*lhsY[3][4][BB][j][i][k];
          lhsY[1][0][CC][j][i][k] = lhsY[1][0][CC][j][i][k] - coeff*lhsY[3][0][CC][j][i][k];
          lhsY[1][1][CC][j][i][k] = lhsY[1][1][CC][j][i][k] - coeff*lhsY[3][1][CC][j][i][k];
          lhsY[1][2][CC][j][i][k] = lhsY[1][2][CC][j][i][k] - coeff*lhsY[3][2][CC][j][i][k];
          lhsY[1][3][CC][j][i][k] = lhsY[1][3][CC][j][i][k] - coeff*lhsY[3][3][CC][j][i][k];
          lhsY[1][4][CC][j][i][k] = lhsY[1][4][CC][j][i][k] - coeff*lhsY[3][4][CC][j][i][k];
          rhs[k][j][i][1]   = rhs[k][j][i][1]   - coeff*rhs[k][j][i][3];

          coeff = lhsY[2][3][BB][j][i][k];
          lhsY[2][4][BB][j][i][k]= lhsY[2][4][BB][j][i][k] - coeff*lhsY[3][4][BB][j][i][k];
          lhsY[2][0][CC][j][i][k] = lhsY[2][0][CC][j][i][k] - coeff*lhsY[3][0][CC][j][i][k];
          lhsY[2][1][CC][j][i][k] = lhsY[2][1][CC][j][i][k] - coeff*lhsY[3][1][CC][j][i][k];
          lhsY[2][2][CC][j][i][k] = lhsY[2][2][CC][j][i][k] - coeff*lhsY[3][2][CC][j][i][k];
          lhsY[2][3][CC][j][i][k] = lhsY[2][3][CC][j][i][k] - coeff*lhsY[3][3][CC][j][i][k];
          lhsY[2][4][CC][j][i][k] = lhsY[2][4][CC][j][i][k] - coeff*lhsY[3][4][CC][j][i][k];
          rhs[k][j][i][2]   = rhs[k][j][i][2]   - coeff*rhs[k][j][i][3];

          coeff = lhsY[4][3][BB][j][i][k];
          lhsY[4][4][BB][j][i][k]= lhsY[4][4][BB][j][i][k] - coeff*lhsY[3][4][BB][j][i][k];
          lhsY[4][0][CC][j][i][k] = lhsY[4][0][CC][j][i][k] - coeff*lhsY[3][0][CC][j][i][k];
          lhsY[4][1][CC][j][i][k] = lhsY[4][1][CC][j][i][k] - coeff*lhsY[3][1][CC][j][i][k];
          lhsY[4][2][CC][j][i][k] = lhsY[4][2][CC][j][i][k] - coeff*lhsY[3][2][CC][j][i][k];
          lhsY[4][3][CC][j][i][k] = lhsY[4][3][CC][j][i][k] - coeff*lhsY[3][3][CC][j][i][k];
          lhsY[4][4][CC][j][i][k] = lhsY[4][4][CC][j][i][k] - coeff*lhsY[3][4][CC][j][i][k];
          rhs[k][j][i][4]   = rhs[k][j][i][4]   - coeff*rhs[k][j][i][3];


          pivot = 1.00/lhsY[4][4][BB][j][i][k];
          lhsY[4][0][CC][j][i][k] = lhsY[4][0][CC][j][i][k]*pivot;
          lhsY[4][1][CC][j][i][k] = lhsY[4][1][CC][j][i][k]*pivot;
          lhsY[4][2][CC][j][i][k] = lhsY[4][2][CC][j][i][k]*pivot;
          lhsY[4][3][CC][j][i][k] = lhsY[4][3][CC][j][i][k]*pivot;
          lhsY[4][4][CC][j][i][k] = lhsY[4][4][CC][j][i][k]*pivot;
          rhs[k][j][i][4]   = rhs[k][j][i][4]  *pivot;

          coeff = lhsY[0][4][BB][j][i][k];
          lhsY[0][0][CC][j][i][k] = lhsY[0][0][CC][j][i][k] - coeff*lhsY[4][0][CC][j][i][k];
          lhsY[0][1][CC][j][i][k] = lhsY[0][1][CC][j][i][k] - coeff*lhsY[4][1][CC][j][i][k];
          lhsY[0][2][CC][j][i][k] = lhsY[0][2][CC][j][i][k] - coeff*lhsY[4][2][CC][j][i][k];
          lhsY[0][3][CC][j][i][k] = lhsY[0][3][CC][j][i][k] - coeff*lhsY[4][3][CC][j][i][k];
          lhsY[0][4][CC][j][i][k] = lhsY[0][4][CC][j][i][k] - coeff*lhsY[4][4][CC][j][i][k];
          rhs[k][j][i][0]   = rhs[k][j][i][0]   - coeff*rhs[k][j][i][4];

          coeff = lhsY[1][4][BB][j][i][k];
          lhsY[1][0][CC][j][i][k] = lhsY[1][0][CC][j][i][k] - coeff*lhsY[4][0][CC][j][i][k];
          lhsY[1][1][CC][j][i][k] = lhsY[1][1][CC][j][i][k] - coeff*lhsY[4][1][CC][j][i][k];
          lhsY[1][2][CC][j][i][k] = lhsY[1][2][CC][j][i][k] - coeff*lhsY[4][2][CC][j][i][k];
          lhsY[1][3][CC][j][i][k] = lhsY[1][3][CC][j][i][k] - coeff*lhsY[4][3][CC][j][i][k];
          lhsY[1][4][CC][j][i][k] = lhsY[1][4][CC][j][i][k] - coeff*lhsY[4][4][CC][j][i][k];
          rhs[k][j][i][1]   = rhs[k][j][i][1]   - coeff*rhs[k][j][i][4];

          coeff = lhsY[2][4][BB][j][i][k];
          lhsY[2][0][CC][j][i][k] = lhsY[2][0][CC][j][i][k] - coeff*lhsY[4][0][CC][j][i][k];
          lhsY[2][1][CC][j][i][k] = lhsY[2][1][CC][j][i][k] - coeff*lhsY[4][1][CC][j][i][k];
          lhsY[2][2][CC][j][i][k] = lhsY[2][2][CC][j][i][k] - coeff*lhsY[4][2][CC][j][i][k];
          lhsY[2][3][CC][j][i][k] = lhsY[2][3][CC][j][i][k] - coeff*lhsY[4][3][CC][j][i][k];
          lhsY[2][4][CC][j][i][k] = lhsY[2][4][CC][j][i][k] - coeff*lhsY[4][4][CC][j][i][k];
          rhs[k][j][i][2]   = rhs[k][j][i][2]   - coeff*rhs[k][j][i][4];

          coeff = lhsY[3][4][BB][j][i][k];
          lhsY[3][0][CC][j][i][k] = lhsY[3][0][CC][j][i][k] - coeff*lhsY[4][0][CC][j][i][k];
          lhsY[3][1][CC][j][i][k] = lhsY[3][1][CC][j][i][k] - coeff*lhsY[4][1][CC][j][i][k];
          lhsY[3][2][CC][j][i][k] = lhsY[3][2][CC][j][i][k] - coeff*lhsY[4][2][CC][j][i][k];
          lhsY[3][3][CC][j][i][k] = lhsY[3][3][CC][j][i][k] - coeff*lhsY[4][3][CC][j][i][k];
          lhsY[3][4][CC][j][i][k] = lhsY[3][4][CC][j][i][k] - coeff*lhsY[4][4][CC][j][i][k];
          rhs[k][j][i][3]   = rhs[k][j][i][3]   - coeff*rhs[k][j][i][4];
        }/*end j*/
      }/*end i*/
    }/*end k*/
#endif

    //---------------------------------------------------------------------
    // rhs(jsize) = rhs(jsize) - A*rhs(jsize-1)
    //---------------------------------------------------------------------
    //matvec_sub(lhsY[i][jsize-1][AA], rhs[k][jsize][i][k], rhs[k][jsize][i]);
    size_t kernel_y_solve_6_off[2] = { 1, 1 };
    size_t kernel_y_solve_6_idx[2] = { gp22, gp02 };
    brisbane_kernel kernel_y_solve_6;
    brisbane_kernel_create("y_solve_6", &kernel_y_solve_6);
    brisbane_kernel_setmem(kernel_y_solve_6, 0, mem_lhsY, brisbane_rw);
    brisbane_kernel_setmem(kernel_y_solve_6, 1, mem_rhs, brisbane_rw);
    brisbane_kernel_setarg(kernel_y_solve_6, 2, sizeof(int), &jsize);

    brisbane_task task6;
    brisbane_task_create(&task6);
    brisbane_task_kernel(task6, kernel_y_solve_6, 2, kernel_y_solve_6_off, kernel_y_solve_6_idx);
    brisbane_task_submit(task6, brisbane_cpu, NULL, true);
#if 0
#ifdef SPEC_USE_INNER_SIMD
    #pragma omp target teams distribute parallel for private(i,k)
#else
    #pragma omp target teams distribute parallel for simd collapse(2)
#endif
    for (i = 1; i <= gp02; i++) {
#ifdef SPEC_USE_INNER_SIMD
      #pragma omp simd
#endif
      for (k = 1; k <= gp22; k++) {
        /*
        for(m = 0; m < 5; m++){
          rhs[k][jsize][i][m] = rhs[k][jsize][i][m] - lhsY[m][0][AA][jsize][i][k]*rhs[k][jsize-1][i][0]
            - lhsY[m][1][AA][jsize][i][k]*rhs[k][jsize-1][i][1]
            - lhsY[m][2][AA][jsize][i][k]*rhs[k][jsize-1][i][2]
            - lhsY[m][3][AA][jsize][i][k]*rhs[k][jsize-1][i][3]
            - lhsY[m][4][AA][jsize][i][k]*rhs[k][jsize-1][i][4];
        }
        */
        rhs[k][jsize][i][0] = rhs[k][jsize][i][0] - lhsY[0][0][AA][jsize][i][k]*rhs[k][jsize-1][i][0]
          - lhsY[0][1][AA][jsize][i][k]*rhs[k][jsize-1][i][1]
          - lhsY[0][2][AA][jsize][i][k]*rhs[k][jsize-1][i][2]
          - lhsY[0][3][AA][jsize][i][k]*rhs[k][jsize-1][i][3]
          - lhsY[0][4][AA][jsize][i][k]*rhs[k][jsize-1][i][4];
        rhs[k][jsize][i][1] = rhs[k][jsize][i][1] - lhsY[1][0][AA][jsize][i][k]*rhs[k][jsize-1][i][0]
          - lhsY[1][1][AA][jsize][i][k]*rhs[k][jsize-1][i][1]
          - lhsY[1][2][AA][jsize][i][k]*rhs[k][jsize-1][i][2]
          - lhsY[1][3][AA][jsize][i][k]*rhs[k][jsize-1][i][3]
          - lhsY[1][4][AA][jsize][i][k]*rhs[k][jsize-1][i][4];
        rhs[k][jsize][i][2] = rhs[k][jsize][i][2] - lhsY[2][0][AA][jsize][i][k]*rhs[k][jsize-1][i][0]
          - lhsY[2][1][AA][jsize][i][k]*rhs[k][jsize-1][i][1]
          - lhsY[2][2][AA][jsize][i][k]*rhs[k][jsize-1][i][2]
          - lhsY[2][3][AA][jsize][i][k]*rhs[k][jsize-1][i][3]
          - lhsY[2][4][AA][jsize][i][k]*rhs[k][jsize-1][i][4];
        rhs[k][jsize][i][3] = rhs[k][jsize][i][3] - lhsY[3][0][AA][jsize][i][k]*rhs[k][jsize-1][i][0]
          - lhsY[3][1][AA][jsize][i][k]*rhs[k][jsize-1][i][1]
          - lhsY[3][2][AA][jsize][i][k]*rhs[k][jsize-1][i][2]
          - lhsY[3][3][AA][jsize][i][k]*rhs[k][jsize-1][i][3]
          - lhsY[3][4][AA][jsize][i][k]*rhs[k][jsize-1][i][4];
        rhs[k][jsize][i][4] = rhs[k][jsize][i][4] - lhsY[4][0][AA][jsize][i][k]*rhs[k][jsize-1][i][0]
          - lhsY[4][1][AA][jsize][i][k]*rhs[k][jsize-1][i][1]
          - lhsY[4][2][AA][jsize][i][k]*rhs[k][jsize-1][i][2]
          - lhsY[4][3][AA][jsize][i][k]*rhs[k][jsize-1][i][3]
          - lhsY[4][4][AA][jsize][i][k]*rhs[k][jsize-1][i][4];
      }
    }
#endif 

    //---------------------------------------------------------------------
    // B(jsize) = B(jsize) - C(jsize-1)*A(jsize)
    // matmul_sub(AA,i,jsize,k,c,
    // $              CC,i,jsize-1,k,c,BB,i,jsize,k)
    //---------------------------------------------------------------------
    //matmul_sub(lhsY[jsize-1][i][AA], lhsY[k][jsize][i][k][CC], lhsY[k][i][jsize][BB]);
    size_t kernel_y_solve_7_off[2] = { 1, 1 };
    size_t kernel_y_solve_7_idx[2] = { gp22, gp02 };
    brisbane_kernel kernel_y_solve_7;
    brisbane_kernel_create("y_solve_7", &kernel_y_solve_7);
    brisbane_kernel_setmem(kernel_y_solve_7, 0, mem_lhsY, brisbane_rw);
    brisbane_kernel_setarg(kernel_y_solve_7, 1, sizeof(int), &jsize);

    brisbane_task task7;
    brisbane_task_create(&task7);
    brisbane_task_kernel(task7, kernel_y_solve_7, 2, kernel_y_solve_7_off, kernel_y_solve_7_idx);
    brisbane_task_submit(task7, brisbane_cpu, NULL, true);
#if 0
#ifdef SPEC_USE_INNER_SIMD
    #pragma omp target teams distribute parallel for private(i,k)
#else
    #pragma omp target teams distribute parallel for simd collapse(2)
#endif
    for (i = 1; i <= gp02; i++) {
#ifdef SPEC_USE_INNER_SIMD
      #pragma omp simd
#endif
      for (k = 1; k <= gp22; k++) {
        /*
        for(m = 0; m < 5; m++){
          for(n = 0; n < 5; n++){
            lhsY[n][m][BB][jsize][i][k] = lhsY[n][m][BB][jsize][i][k] - lhsY[n][0][AA][jsize][i][k]*lhsY[0][m][CC][jsize-1][i][k]
              - lhsY[n][1][AA][jsize][i][k]*lhsY[1][m][CC][jsize-1][i][k]
              - lhsY[n][2][AA][jsize][i][k]*lhsY[2][m][CC][jsize-1][i][k]
              - lhsY[n][3][AA][jsize][i][k]*lhsY[3][m][CC][jsize-1][i][k]
              - lhsY[n][4][AA][jsize][i][k]*lhsY[4][m][CC][jsize-1][i][k];
          }
        }
        */
        lhsY[0][0][BB][jsize][i][k] = lhsY[0][0][BB][jsize][i][k] - lhsY[0][0][AA][jsize][i][k]*lhsY[0][0][CC][jsize-1][i][k]
          - lhsY[0][1][AA][jsize][i][k]*lhsY[1][0][CC][jsize-1][i][k]
          - lhsY[0][2][AA][jsize][i][k]*lhsY[2][0][CC][jsize-1][i][k]
          - lhsY[0][3][AA][jsize][i][k]*lhsY[3][0][CC][jsize-1][i][k]
          - lhsY[0][4][AA][jsize][i][k]*lhsY[4][0][CC][jsize-1][i][k];
        lhsY[1][0][BB][jsize][i][k] = lhsY[1][0][BB][jsize][i][k] - lhsY[1][0][AA][jsize][i][k]*lhsY[0][0][CC][jsize-1][i][k]
          - lhsY[1][1][AA][jsize][i][k]*lhsY[1][0][CC][jsize-1][i][k]
          - lhsY[1][2][AA][jsize][i][k]*lhsY[2][0][CC][jsize-1][i][k]
          - lhsY[1][3][AA][jsize][i][k]*lhsY[3][0][CC][jsize-1][i][k]
          - lhsY[1][4][AA][jsize][i][k]*lhsY[4][0][CC][jsize-1][i][k];
        lhsY[2][0][BB][jsize][i][k] = lhsY[2][0][BB][jsize][i][k] - lhsY[2][0][AA][jsize][i][k]*lhsY[0][0][CC][jsize-1][i][k]
          - lhsY[2][1][AA][jsize][i][k]*lhsY[1][0][CC][jsize-1][i][k]
          - lhsY[2][2][AA][jsize][i][k]*lhsY[2][0][CC][jsize-1][i][k]
          - lhsY[2][3][AA][jsize][i][k]*lhsY[3][0][CC][jsize-1][i][k]
          - lhsY[2][4][AA][jsize][i][k]*lhsY[4][0][CC][jsize-1][i][k];
        lhsY[3][0][BB][jsize][i][k] = lhsY[3][0][BB][jsize][i][k] - lhsY[3][0][AA][jsize][i][k]*lhsY[0][0][CC][jsize-1][i][k]
          - lhsY[3][1][AA][jsize][i][k]*lhsY[1][0][CC][jsize-1][i][k]
          - lhsY[3][2][AA][jsize][i][k]*lhsY[2][0][CC][jsize-1][i][k]
          - lhsY[3][3][AA][jsize][i][k]*lhsY[3][0][CC][jsize-1][i][k]
          - lhsY[3][4][AA][jsize][i][k]*lhsY[4][0][CC][jsize-1][i][k];
        lhsY[4][0][BB][jsize][i][k] = lhsY[4][0][BB][jsize][i][k] - lhsY[4][0][AA][jsize][i][k]*lhsY[0][0][CC][jsize-1][i][k]
          - lhsY[4][1][AA][jsize][i][k]*lhsY[1][0][CC][jsize-1][i][k]
          - lhsY[4][2][AA][jsize][i][k]*lhsY[2][0][CC][jsize-1][i][k]
          - lhsY[4][3][AA][jsize][i][k]*lhsY[3][0][CC][jsize-1][i][k]
          - lhsY[4][4][AA][jsize][i][k]*lhsY[4][0][CC][jsize-1][i][k];
        lhsY[0][1][BB][jsize][i][k] = lhsY[0][1][BB][jsize][i][k] - lhsY[0][0][AA][jsize][i][k]*lhsY[0][1][CC][jsize-1][i][k]
          - lhsY[0][1][AA][jsize][i][k]*lhsY[1][1][CC][jsize-1][i][k]
          - lhsY[0][2][AA][jsize][i][k]*lhsY[2][1][CC][jsize-1][i][k]
          - lhsY[0][3][AA][jsize][i][k]*lhsY[3][1][CC][jsize-1][i][k]
          - lhsY[0][4][AA][jsize][i][k]*lhsY[4][1][CC][jsize-1][i][k];
        lhsY[1][1][BB][jsize][i][k] = lhsY[1][1][BB][jsize][i][k] - lhsY[1][0][AA][jsize][i][k]*lhsY[0][1][CC][jsize-1][i][k]
          - lhsY[1][1][AA][jsize][i][k]*lhsY[1][1][CC][jsize-1][i][k]
          - lhsY[1][2][AA][jsize][i][k]*lhsY[2][1][CC][jsize-1][i][k]
          - lhsY[1][3][AA][jsize][i][k]*lhsY[3][1][CC][jsize-1][i][k]
          - lhsY[1][4][AA][jsize][i][k]*lhsY[4][1][CC][jsize-1][i][k];
        lhsY[2][1][BB][jsize][i][k] = lhsY[2][1][BB][jsize][i][k] - lhsY[2][0][AA][jsize][i][k]*lhsY[0][1][CC][jsize-1][i][k]
          - lhsY[2][1][AA][jsize][i][k]*lhsY[1][1][CC][jsize-1][i][k]
          - lhsY[2][2][AA][jsize][i][k]*lhsY[2][1][CC][jsize-1][i][k]
          - lhsY[2][3][AA][jsize][i][k]*lhsY[3][1][CC][jsize-1][i][k]
          - lhsY[2][4][AA][jsize][i][k]*lhsY[4][1][CC][jsize-1][i][k];
        lhsY[3][1][BB][jsize][i][k] = lhsY[3][1][BB][jsize][i][k] - lhsY[3][0][AA][jsize][i][k]*lhsY[0][1][CC][jsize-1][i][k]
          - lhsY[3][1][AA][jsize][i][k]*lhsY[1][1][CC][jsize-1][i][k]
          - lhsY[3][2][AA][jsize][i][k]*lhsY[2][1][CC][jsize-1][i][k]
          - lhsY[3][3][AA][jsize][i][k]*lhsY[3][1][CC][jsize-1][i][k]
          - lhsY[3][4][AA][jsize][i][k]*lhsY[4][1][CC][jsize-1][i][k];
        lhsY[4][1][BB][jsize][i][k] = lhsY[4][1][BB][jsize][i][k] - lhsY[4][0][AA][jsize][i][k]*lhsY[0][1][CC][jsize-1][i][k]
          - lhsY[4][1][AA][jsize][i][k]*lhsY[1][1][CC][jsize-1][i][k]
          - lhsY[4][2][AA][jsize][i][k]*lhsY[2][1][CC][jsize-1][i][k]
          - lhsY[4][3][AA][jsize][i][k]*lhsY[3][1][CC][jsize-1][i][k]
          - lhsY[4][4][AA][jsize][i][k]*lhsY[4][1][CC][jsize-1][i][k];
        lhsY[0][2][BB][jsize][i][k] = lhsY[0][2][BB][jsize][i][k] - lhsY[0][0][AA][jsize][i][k]*lhsY[0][2][CC][jsize-1][i][k]
          - lhsY[0][1][AA][jsize][i][k]*lhsY[1][2][CC][jsize-1][i][k]
          - lhsY[0][2][AA][jsize][i][k]*lhsY[2][2][CC][jsize-1][i][k]
          - lhsY[0][3][AA][jsize][i][k]*lhsY[3][2][CC][jsize-1][i][k]
          - lhsY[0][4][AA][jsize][i][k]*lhsY[4][2][CC][jsize-1][i][k];
        lhsY[1][2][BB][jsize][i][k] = lhsY[1][2][BB][jsize][i][k] - lhsY[1][0][AA][jsize][i][k]*lhsY[0][2][CC][jsize-1][i][k]
          - lhsY[1][1][AA][jsize][i][k]*lhsY[1][2][CC][jsize-1][i][k]
          - lhsY[1][2][AA][jsize][i][k]*lhsY[2][2][CC][jsize-1][i][k]
          - lhsY[1][3][AA][jsize][i][k]*lhsY[3][2][CC][jsize-1][i][k]
          - lhsY[1][4][AA][jsize][i][k]*lhsY[4][2][CC][jsize-1][i][k];
        lhsY[2][2][BB][jsize][i][k] = lhsY[2][2][BB][jsize][i][k] - lhsY[2][0][AA][jsize][i][k]*lhsY[0][2][CC][jsize-1][i][k]
          - lhsY[2][1][AA][jsize][i][k]*lhsY[1][2][CC][jsize-1][i][k]
          - lhsY[2][2][AA][jsize][i][k]*lhsY[2][2][CC][jsize-1][i][k]
          - lhsY[2][3][AA][jsize][i][k]*lhsY[3][2][CC][jsize-1][i][k]
          - lhsY[2][4][AA][jsize][i][k]*lhsY[4][2][CC][jsize-1][i][k];
        lhsY[3][2][BB][jsize][i][k] = lhsY[3][2][BB][jsize][i][k] - lhsY[3][0][AA][jsize][i][k]*lhsY[0][2][CC][jsize-1][i][k]
          - lhsY[3][1][AA][jsize][i][k]*lhsY[1][2][CC][jsize-1][i][k]
          - lhsY[3][2][AA][jsize][i][k]*lhsY[2][2][CC][jsize-1][i][k]
          - lhsY[3][3][AA][jsize][i][k]*lhsY[3][2][CC][jsize-1][i][k]
          - lhsY[3][4][AA][jsize][i][k]*lhsY[4][2][CC][jsize-1][i][k];
        lhsY[4][2][BB][jsize][i][k] = lhsY[4][2][BB][jsize][i][k] - lhsY[4][0][AA][jsize][i][k]*lhsY[0][2][CC][jsize-1][i][k]
          - lhsY[4][1][AA][jsize][i][k]*lhsY[1][2][CC][jsize-1][i][k]
          - lhsY[4][2][AA][jsize][i][k]*lhsY[2][2][CC][jsize-1][i][k]
          - lhsY[4][3][AA][jsize][i][k]*lhsY[3][2][CC][jsize-1][i][k]
          - lhsY[4][4][AA][jsize][i][k]*lhsY[4][2][CC][jsize-1][i][k];
        lhsY[0][3][BB][jsize][i][k] = lhsY[0][3][BB][jsize][i][k] - lhsY[0][0][AA][jsize][i][k]*lhsY[0][3][CC][jsize-1][i][k]
          - lhsY[0][1][AA][jsize][i][k]*lhsY[1][3][CC][jsize-1][i][k]
          - lhsY[0][2][AA][jsize][i][k]*lhsY[2][3][CC][jsize-1][i][k]
          - lhsY[0][3][AA][jsize][i][k]*lhsY[3][3][CC][jsize-1][i][k]
          - lhsY[0][4][AA][jsize][i][k]*lhsY[4][3][CC][jsize-1][i][k];
        lhsY[1][3][BB][jsize][i][k] = lhsY[1][3][BB][jsize][i][k] - lhsY[1][0][AA][jsize][i][k]*lhsY[0][3][CC][jsize-1][i][k]
          - lhsY[1][1][AA][jsize][i][k]*lhsY[1][3][CC][jsize-1][i][k]
          - lhsY[1][2][AA][jsize][i][k]*lhsY[2][3][CC][jsize-1][i][k]
          - lhsY[1][3][AA][jsize][i][k]*lhsY[3][3][CC][jsize-1][i][k]
          - lhsY[1][4][AA][jsize][i][k]*lhsY[4][3][CC][jsize-1][i][k];
        lhsY[2][3][BB][jsize][i][k] = lhsY[2][3][BB][jsize][i][k] - lhsY[2][0][AA][jsize][i][k]*lhsY[0][3][CC][jsize-1][i][k]
          - lhsY[2][1][AA][jsize][i][k]*lhsY[1][3][CC][jsize-1][i][k]
          - lhsY[2][2][AA][jsize][i][k]*lhsY[2][3][CC][jsize-1][i][k]
          - lhsY[2][3][AA][jsize][i][k]*lhsY[3][3][CC][jsize-1][i][k]
          - lhsY[2][4][AA][jsize][i][k]*lhsY[4][3][CC][jsize-1][i][k];
        lhsY[3][3][BB][jsize][i][k] = lhsY[3][3][BB][jsize][i][k] - lhsY[3][0][AA][jsize][i][k]*lhsY[0][3][CC][jsize-1][i][k]
          - lhsY[3][1][AA][jsize][i][k]*lhsY[1][3][CC][jsize-1][i][k]
          - lhsY[3][2][AA][jsize][i][k]*lhsY[2][3][CC][jsize-1][i][k]
          - lhsY[3][3][AA][jsize][i][k]*lhsY[3][3][CC][jsize-1][i][k]
          - lhsY[3][4][AA][jsize][i][k]*lhsY[4][3][CC][jsize-1][i][k];
        lhsY[4][3][BB][jsize][i][k] = lhsY[4][3][BB][jsize][i][k] - lhsY[4][0][AA][jsize][i][k]*lhsY[0][3][CC][jsize-1][i][k]
          - lhsY[4][1][AA][jsize][i][k]*lhsY[1][3][CC][jsize-1][i][k]
          - lhsY[4][2][AA][jsize][i][k]*lhsY[2][3][CC][jsize-1][i][k]
          - lhsY[4][3][AA][jsize][i][k]*lhsY[3][3][CC][jsize-1][i][k]
          - lhsY[4][4][AA][jsize][i][k]*lhsY[4][3][CC][jsize-1][i][k];
        lhsY[0][4][BB][jsize][i][k] = lhsY[0][4][BB][jsize][i][k] - lhsY[0][0][AA][jsize][i][k]*lhsY[0][4][CC][jsize-1][i][k]
          - lhsY[0][1][AA][jsize][i][k]*lhsY[1][4][CC][jsize-1][i][k]
          - lhsY[0][2][AA][jsize][i][k]*lhsY[2][4][CC][jsize-1][i][k]
          - lhsY[0][3][AA][jsize][i][k]*lhsY[3][4][CC][jsize-1][i][k]
          - lhsY[0][4][AA][jsize][i][k]*lhsY[4][4][CC][jsize-1][i][k];
        lhsY[1][4][BB][jsize][i][k] = lhsY[1][4][BB][jsize][i][k] - lhsY[1][0][AA][jsize][i][k]*lhsY[0][4][CC][jsize-1][i][k]
          - lhsY[1][1][AA][jsize][i][k]*lhsY[1][4][CC][jsize-1][i][k]
          - lhsY[1][2][AA][jsize][i][k]*lhsY[2][4][CC][jsize-1][i][k]
          - lhsY[1][3][AA][jsize][i][k]*lhsY[3][4][CC][jsize-1][i][k]
          - lhsY[1][4][AA][jsize][i][k]*lhsY[4][4][CC][jsize-1][i][k];
        lhsY[2][4][BB][jsize][i][k] = lhsY[2][4][BB][jsize][i][k] - lhsY[2][0][AA][jsize][i][k]*lhsY[0][4][CC][jsize-1][i][k]
          - lhsY[2][1][AA][jsize][i][k]*lhsY[1][4][CC][jsize-1][i][k]
          - lhsY[2][2][AA][jsize][i][k]*lhsY[2][4][CC][jsize-1][i][k]
          - lhsY[2][3][AA][jsize][i][k]*lhsY[3][4][CC][jsize-1][i][k]
          - lhsY[2][4][AA][jsize][i][k]*lhsY[4][4][CC][jsize-1][i][k];
        lhsY[3][4][BB][jsize][i][k] = lhsY[3][4][BB][jsize][i][k] - lhsY[3][0][AA][jsize][i][k]*lhsY[0][4][CC][jsize-1][i][k]
          - lhsY[3][1][AA][jsize][i][k]*lhsY[1][4][CC][jsize-1][i][k]
          - lhsY[3][2][AA][jsize][i][k]*lhsY[2][4][CC][jsize-1][i][k]
          - lhsY[3][3][AA][jsize][i][k]*lhsY[3][4][CC][jsize-1][i][k]
          - lhsY[3][4][AA][jsize][i][k]*lhsY[4][4][CC][jsize-1][i][k];
        lhsY[4][4][BB][jsize][i][k] = lhsY[4][4][BB][jsize][i][k] - lhsY[4][0][AA][jsize][i][k]*lhsY[0][4][CC][jsize-1][i][k]
          - lhsY[4][1][AA][jsize][i][k]*lhsY[1][4][CC][jsize-1][i][k]
          - lhsY[4][2][AA][jsize][i][k]*lhsY[2][4][CC][jsize-1][i][k]
          - lhsY[4][3][AA][jsize][i][k]*lhsY[3][4][CC][jsize-1][i][k]
          - lhsY[4][4][AA][jsize][i][k]*lhsY[4][4][CC][jsize-1][i][k];


      }
    }
#endif

    //---------------------------------------------------------------------
    // multiply rhs(jsize) by b_inverse(jsize) and copy to rhs
    //---------------------------------------------------------------------
    //binvrhs( lhsY[i][jsize][BB], rhs[k][jsize][i][k] );
    size_t kernel_y_solve_8_off[2] = { 1, 1 };
    size_t kernel_y_solve_8_idx[2] = { gp22, gp02 };
    brisbane_kernel kernel_y_solve_8;
    brisbane_kernel_create("y_solve_8", &kernel_y_solve_8);
    brisbane_kernel_setmem(kernel_y_solve_8, 0, mem_lhsY, brisbane_rw);
    brisbane_kernel_setmem(kernel_y_solve_8, 1, mem_rhs, brisbane_rw);
    brisbane_kernel_setarg(kernel_y_solve_8, 2, sizeof(int), &jsize);

    brisbane_task task8;
    brisbane_task_create(&task8);
    brisbane_task_kernel(task8, kernel_y_solve_8, 2, kernel_y_solve_8_off, kernel_y_solve_8_idx);
    brisbane_task_submit(task8, brisbane_cpu, NULL, true);
#if 0
#ifdef SPEC_USE_INNER_SIMD
    #pragma omp target teams distribute parallel for private(i,k,pivot,coeff) 
#else
    #pragma omp target teams distribute parallel for simd private(pivot,coeff) collapse(2)
#endif
    for (i = 1; i <= gp02; i++) {
#ifdef SPEC_USE_INNER_SIMD
      #pragma omp simd private(pivot,coeff)
#endif
      for (k = 1; k <= gp22; k++) {
        /*
        for(m = 0; m < 5; m++){
          pivot = 1.00/lhsY[m][m][BB][jsize][i][k];
          for(n = m+1; n < 5; n++){
            lhsY[m][n][BB][jsize][i][k] = lhsY[m][n][BB][jsize][i][k]*pivot;
          }
          rhs[k][jsize][i][m] = rhs[k][jsize][i][m]*pivot;

          for(n = 0; n < 5; n++){
            if(n != m){
              coeff = lhsY[n][m][BB][jsize][i][k];
              for(z = m+1; z < 5; z++){
                lhsY[n][z][BB][jsize][i][k] = lhsY[n][z][BB][jsize][i][k] - coeff*lhsY[m][z][BB][jsize][i][k];
              }
              rhs[k][jsize][i][n] = rhs[k][jsize][i][n] - coeff*rhs[k][jsize][i][m];
            }
          }
        }
        */
        pivot = 1.00/lhsY[0][0][BB][jsize][i][k];
        lhsY[0][1][BB][jsize][i][k] = lhsY[0][1][BB][jsize][i][k]*pivot;
        lhsY[0][2][BB][jsize][i][k] = lhsY[0][2][BB][jsize][i][k]*pivot;
        lhsY[0][3][BB][jsize][i][k] = lhsY[0][3][BB][jsize][i][k]*pivot;
        lhsY[0][4][BB][jsize][i][k] = lhsY[0][4][BB][jsize][i][k]*pivot;
        rhs[k][jsize][i][0]   = rhs[k][jsize][i][0]  *pivot;

        coeff = lhsY[1][0][BB][jsize][i][k];
        lhsY[1][1][BB][jsize][i][k]= lhsY[1][1][BB][jsize][i][k] - coeff*lhsY[0][1][BB][jsize][i][k];
        lhsY[1][2][BB][jsize][i][k]= lhsY[1][2][BB][jsize][i][k] - coeff*lhsY[0][2][BB][jsize][i][k];
        lhsY[1][3][BB][jsize][i][k]= lhsY[1][3][BB][jsize][i][k] - coeff*lhsY[0][3][BB][jsize][i][k];
        lhsY[1][4][BB][jsize][i][k]= lhsY[1][4][BB][jsize][i][k] - coeff*lhsY[0][4][BB][jsize][i][k];
        rhs[k][jsize][i][1]   = rhs[k][jsize][i][1]   - coeff*rhs[k][jsize][i][0];

        coeff = lhsY[2][0][BB][jsize][i][k];
        lhsY[2][1][BB][jsize][i][k]= lhsY[2][1][BB][jsize][i][k] - coeff*lhsY[0][1][BB][jsize][i][k];
        lhsY[2][2][BB][jsize][i][k]= lhsY[2][2][BB][jsize][i][k] - coeff*lhsY[0][2][BB][jsize][i][k];
        lhsY[2][3][BB][jsize][i][k]= lhsY[2][3][BB][jsize][i][k] - coeff*lhsY[0][3][BB][jsize][i][k];
        lhsY[2][4][BB][jsize][i][k]= lhsY[2][4][BB][jsize][i][k] - coeff*lhsY[0][4][BB][jsize][i][k];
        rhs[k][jsize][i][2]   = rhs[k][jsize][i][2]   - coeff*rhs[k][jsize][i][0];

        coeff = lhsY[3][0][BB][jsize][i][k];
        lhsY[3][1][BB][jsize][i][k]= lhsY[3][1][BB][jsize][i][k] - coeff*lhsY[0][1][BB][jsize][i][k];
        lhsY[3][2][BB][jsize][i][k]= lhsY[3][2][BB][jsize][i][k] - coeff*lhsY[0][2][BB][jsize][i][k];
        lhsY[3][3][BB][jsize][i][k]= lhsY[3][3][BB][jsize][i][k] - coeff*lhsY[0][3][BB][jsize][i][k];
        lhsY[3][4][BB][jsize][i][k]= lhsY[3][4][BB][jsize][i][k] - coeff*lhsY[0][4][BB][jsize][i][k];
        rhs[k][jsize][i][3]   = rhs[k][jsize][i][3]   - coeff*rhs[k][jsize][i][0];

        coeff = lhsY[4][0][BB][jsize][i][k];
        lhsY[4][1][BB][jsize][i][k]= lhsY[4][1][BB][jsize][i][k] - coeff*lhsY[0][1][BB][jsize][i][k];
        lhsY[4][2][BB][jsize][i][k]= lhsY[4][2][BB][jsize][i][k] - coeff*lhsY[0][2][BB][jsize][i][k];
        lhsY[4][3][BB][jsize][i][k]= lhsY[4][3][BB][jsize][i][k] - coeff*lhsY[0][3][BB][jsize][i][k];
        lhsY[4][4][BB][jsize][i][k]= lhsY[4][4][BB][jsize][i][k] - coeff*lhsY[0][4][BB][jsize][i][k];
        rhs[k][jsize][i][4]   = rhs[k][jsize][i][4]   - coeff*rhs[k][jsize][i][0];


        pivot = 1.00/lhsY[1][1][BB][jsize][i][k];
        lhsY[1][2][BB][jsize][i][k] = lhsY[1][2][BB][jsize][i][k]*pivot;
        lhsY[1][3][BB][jsize][i][k] = lhsY[1][3][BB][jsize][i][k]*pivot;
        lhsY[1][4][BB][jsize][i][k] = lhsY[1][4][BB][jsize][i][k]*pivot;
        rhs[k][jsize][i][1]   = rhs[k][jsize][i][1]  *pivot;

        coeff = lhsY[0][1][BB][jsize][i][k];
        lhsY[0][2][BB][jsize][i][k]= lhsY[0][2][BB][jsize][i][k] - coeff*lhsY[1][2][BB][jsize][i][k];
        lhsY[0][3][BB][jsize][i][k]= lhsY[0][3][BB][jsize][i][k] - coeff*lhsY[1][3][BB][jsize][i][k];
        lhsY[0][4][BB][jsize][i][k]= lhsY[0][4][BB][jsize][i][k] - coeff*lhsY[1][4][BB][jsize][i][k];
        rhs[k][jsize][i][0]   = rhs[k][jsize][i][0]   - coeff*rhs[k][jsize][i][1];

        coeff = lhsY[2][1][BB][jsize][i][k];
        lhsY[2][2][BB][jsize][i][k]= lhsY[2][2][BB][jsize][i][k] - coeff*lhsY[1][2][BB][jsize][i][k];
        lhsY[2][3][BB][jsize][i][k]= lhsY[2][3][BB][jsize][i][k] - coeff*lhsY[1][3][BB][jsize][i][k];
        lhsY[2][4][BB][jsize][i][k]= lhsY[2][4][BB][jsize][i][k] - coeff*lhsY[1][4][BB][jsize][i][k];
        rhs[k][jsize][i][2]   = rhs[k][jsize][i][2]   - coeff*rhs[k][jsize][i][1];

        coeff = lhsY[3][1][BB][jsize][i][k];
        lhsY[3][2][BB][jsize][i][k]= lhsY[3][2][BB][jsize][i][k] - coeff*lhsY[1][2][BB][jsize][i][k];
        lhsY[3][3][BB][jsize][i][k]= lhsY[3][3][BB][jsize][i][k] - coeff*lhsY[1][3][BB][jsize][i][k];
        lhsY[3][4][BB][jsize][i][k]= lhsY[3][4][BB][jsize][i][k] - coeff*lhsY[1][4][BB][jsize][i][k];
        rhs[k][jsize][i][3]   = rhs[k][jsize][i][3]   - coeff*rhs[k][jsize][i][1];

        coeff = lhsY[4][1][BB][jsize][i][k];
        lhsY[4][2][BB][jsize][i][k]= lhsY[4][2][BB][jsize][i][k] - coeff*lhsY[1][2][BB][jsize][i][k];
        lhsY[4][3][BB][jsize][i][k]= lhsY[4][3][BB][jsize][i][k] - coeff*lhsY[1][3][BB][jsize][i][k];
        lhsY[4][4][BB][jsize][i][k]= lhsY[4][4][BB][jsize][i][k] - coeff*lhsY[1][4][BB][jsize][i][k];
        rhs[k][jsize][i][4]   = rhs[k][jsize][i][4]   - coeff*rhs[k][jsize][i][1];


        pivot = 1.00/lhsY[2][2][BB][jsize][i][k];
        lhsY[2][3][BB][jsize][i][k] = lhsY[2][3][BB][jsize][i][k]*pivot;
        lhsY[2][4][BB][jsize][i][k] = lhsY[2][4][BB][jsize][i][k]*pivot;
        rhs[k][jsize][i][2]   = rhs[k][jsize][i][2]  *pivot;

        coeff = lhsY[0][2][BB][jsize][i][k];
        lhsY[0][3][BB][jsize][i][k]= lhsY[0][3][BB][jsize][i][k] - coeff*lhsY[2][3][BB][jsize][i][k];
        lhsY[0][4][BB][jsize][i][k]= lhsY[0][4][BB][jsize][i][k] - coeff*lhsY[2][4][BB][jsize][i][k];
        rhs[k][jsize][i][0]   = rhs[k][jsize][i][0]   - coeff*rhs[k][jsize][i][2];

        coeff = lhsY[1][2][BB][jsize][i][k];
        lhsY[1][3][BB][jsize][i][k]= lhsY[1][3][BB][jsize][i][k] - coeff*lhsY[2][3][BB][jsize][i][k];
        lhsY[1][4][BB][jsize][i][k]= lhsY[1][4][BB][jsize][i][k] - coeff*lhsY[2][4][BB][jsize][i][k];
        rhs[k][jsize][i][1]   = rhs[k][jsize][i][1]   - coeff*rhs[k][jsize][i][2];

        coeff = lhsY[3][2][BB][jsize][i][k];
        lhsY[3][3][BB][jsize][i][k]= lhsY[3][3][BB][jsize][i][k] - coeff*lhsY[2][3][BB][jsize][i][k];
        lhsY[3][4][BB][jsize][i][k]= lhsY[3][4][BB][jsize][i][k] - coeff*lhsY[2][4][BB][jsize][i][k];
        rhs[k][jsize][i][3]   = rhs[k][jsize][i][3]   - coeff*rhs[k][jsize][i][2];

        coeff = lhsY[4][2][BB][jsize][i][k];
        lhsY[4][3][BB][jsize][i][k]= lhsY[4][3][BB][jsize][i][k] - coeff*lhsY[2][3][BB][jsize][i][k];
        lhsY[4][4][BB][jsize][i][k]= lhsY[4][4][BB][jsize][i][k] - coeff*lhsY[2][4][BB][jsize][i][k];
        rhs[k][jsize][i][4]   = rhs[k][jsize][i][4]   - coeff*rhs[k][jsize][i][2];


        pivot = 1.00/lhsY[3][3][BB][jsize][i][k];
        lhsY[3][4][BB][jsize][i][k] = lhsY[3][4][BB][jsize][i][k]*pivot;
        rhs[k][jsize][i][3]   = rhs[k][jsize][i][3]  *pivot;

        coeff = lhsY[0][3][BB][jsize][i][k];
        lhsY[0][4][BB][jsize][i][k]= lhsY[0][4][BB][jsize][i][k] - coeff*lhsY[3][4][BB][jsize][i][k];
        rhs[k][jsize][i][0]   = rhs[k][jsize][i][0]   - coeff*rhs[k][jsize][i][3];

        coeff = lhsY[1][3][BB][jsize][i][k];
        lhsY[1][4][BB][jsize][i][k]= lhsY[1][4][BB][jsize][i][k] - coeff*lhsY[3][4][BB][jsize][i][k];
        rhs[k][jsize][i][1]   = rhs[k][jsize][i][1]   - coeff*rhs[k][jsize][i][3];

        coeff = lhsY[2][3][BB][jsize][i][k];
        lhsY[2][4][BB][jsize][i][k]= lhsY[2][4][BB][jsize][i][k] - coeff*lhsY[3][4][BB][jsize][i][k];
        rhs[k][jsize][i][2]   = rhs[k][jsize][i][2]   - coeff*rhs[k][jsize][i][3];

        coeff = lhsY[4][3][BB][jsize][i][k];
        lhsY[4][4][BB][jsize][i][k]= lhsY[4][4][BB][jsize][i][k] - coeff*lhsY[3][4][BB][jsize][i][k];
        rhs[k][jsize][i][4]   = rhs[k][jsize][i][4]   - coeff*rhs[k][jsize][i][3];


        pivot = 1.00/lhsY[4][4][BB][jsize][i][k];
        rhs[k][jsize][i][4]   = rhs[k][jsize][i][4]  *pivot;

        coeff = lhsY[0][4][BB][jsize][i][k];
        rhs[k][jsize][i][0]   = rhs[k][jsize][i][0]   - coeff*rhs[k][jsize][i][4];

        coeff = lhsY[1][4][BB][jsize][i][k];
        rhs[k][jsize][i][1]   = rhs[k][jsize][i][1]   - coeff*rhs[k][jsize][i][4];

        coeff = lhsY[2][4][BB][jsize][i][k];
        rhs[k][jsize][i][2]   = rhs[k][jsize][i][2]   - coeff*rhs[k][jsize][i][4];

        coeff = lhsY[3][4][BB][jsize][i][k];
        rhs[k][jsize][i][3]   = rhs[k][jsize][i][3]   - coeff*rhs[k][jsize][i][4];
      }
    }
#endif

    //---------------------------------------------------------------------
    // back solve: if last cell, then generate U(jsize)=rhs(jsize)
    // else assume U(jsize) is loaded in un pack backsub_info
    // so just use it
    // after u(jstart) will be sent to next cell
    //---------------------------------------------------------------------
    size_t kernel_y_solve_9_off[2] = { 1, 1 };
    size_t kernel_y_solve_9_idx[2] = { gp02, gp22 };
    brisbane_kernel kernel_y_solve_9;
    brisbane_kernel_create("y_solve_9", &kernel_y_solve_9);
    brisbane_kernel_setmem(kernel_y_solve_9, 0, mem_lhsY, brisbane_r);
    brisbane_kernel_setmem(kernel_y_solve_9, 1, mem_rhs, brisbane_rw);
    brisbane_kernel_setarg(kernel_y_solve_9, 2, sizeof(int), &jsize);

    brisbane_task task9;
    brisbane_task_create(&task9);
    brisbane_task_kernel(task9, kernel_y_solve_9, 2, kernel_y_solve_9_off, kernel_y_solve_9_idx);
    //brisbane_task_submit(task9, brisbane_cpu, NULL, true);
#if 1
    brisbane_task task10;
    brisbane_task_create(&task10);
    brisbane_task_d2h_full(task10, mem_rhs, rhs);
    brisbane_task_d2h_full(task10, mem_lhsY, lhsY);
    brisbane_task_submit(task10, brisbane_cpu, NULL, true);
      #pragma omp target teams distribute parallel for collapse(2) private(i,k,m,n)
      for (k = 1; k <= gp22; k++) {
        for (i = 1; i <= gp02; i++) {
        for (j = jsize-1; j >= 0; j--) {
          for (m = 0; m < BLOCK_SIZE; m++) {
            for (n = 0; n < BLOCK_SIZE; n++) {
              rhs[k][j][i][m] = rhs[k][j][i][m]
                - lhsY[m][n][CC][j][i][k]*rhs[k][j+1][i][n];
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

  brisbane_mem_release(mem_fjacY);
  brisbane_mem_release(mem_njacY);
  brisbane_mem_release(mem_lhsY);
}
