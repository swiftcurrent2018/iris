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
// Performs line solves in Z direction by first factoring
// the block-tridiagonal matrix into an upper triangular matrix,
// and then performing back substitution to solve for the unknow
// vectors of each line.
//
// Make sure we treat elements zero to cell_size in the direction
// of the sweep.
//---------------------------------------------------------------------
void z_solve()
{
  int i, j, k, m, n, ksize, z;
  double pivot, coeff;
  int gp12, gp02;
  double fjacZ[5][5][PROBLEM_SIZE+1][IMAXP-1][JMAXP-1];
  double njacZ[5][5][PROBLEM_SIZE+1][IMAXP-1][JMAXP-1];
  double lhsZ[5][5][3][PROBLEM_SIZE][IMAXP-1][JMAXP-1];
  double temp1, temp2, temp3;

  gp12 = grid_points[1]-2;
  gp02 = grid_points[0]-2;

  //---------------------------------------------------------------------
  // This function computes the left hand side for the three z-factors
  //---------------------------------------------------------------------

  ksize = grid_points[2]-1;

  brisbane_mem mem_fjacZ;
  brisbane_mem mem_njacZ;
  brisbane_mem mem_lhsZ;
  brisbane_mem_create(sizeof(double) * 5 * 5 * (PROBLEM_SIZE + 1) * (IMAXP - 1) * (JMAXP - 1), &mem_fjacZ);
  brisbane_mem_create(sizeof(double) * 5 * 5 * (PROBLEM_SIZE + 1) * (IMAXP - 1) * (JMAXP - 1), &mem_njacZ);
  brisbane_mem_create(sizeof(double) * 5 * 5 * 3 * (PROBLEM_SIZE) * (IMAXP - 1) * (JMAXP - 1), &mem_lhsZ);

  //---------------------------------------------------------------------
  // Compute the indices for storing the block-diagonal matrix;
  // determine c (labeled f) and s jacobians
  //---------------------------------------------------------------------
  #pragma omp target data map(alloc:lhsZ[:][:][:][:][:][:],fjacZ[:][:][:][:][:],njacZ[:][:][:][:][:]) //present(rho_i,u,qs,rhs,square)
  {
    size_t kernel_z_solve_0_off[2] = { 1, 0 };
    size_t kernel_z_solve_0_idx[2] = { gp02, ksize + 1 };
    brisbane_kernel kernel_z_solve_0;
    brisbane_kernel_create("z_solve_0", &kernel_z_solve_0);
    brisbane_kernel_setmem(kernel_z_solve_0, 0, mem_u, brisbane_r);
    brisbane_kernel_setmem(kernel_z_solve_0, 1, mem_fjacZ, brisbane_w);
    brisbane_kernel_setmem(kernel_z_solve_0, 2, mem_njacZ, brisbane_w);
    brisbane_kernel_setmem(kernel_z_solve_0, 3, mem_qs, brisbane_r);
    brisbane_kernel_setmem(kernel_z_solve_0, 4, mem_square, brisbane_r);
    brisbane_kernel_setarg(kernel_z_solve_0, 5, sizeof(double), &c1);
    brisbane_kernel_setarg(kernel_z_solve_0, 6, sizeof(double), &c2);
    brisbane_kernel_setarg(kernel_z_solve_0, 7, sizeof(double), &c3c4);
    brisbane_kernel_setarg(kernel_z_solve_0, 8, sizeof(double), &c1345);
    brisbane_kernel_setarg(kernel_z_solve_0, 9, sizeof(double), &con43);
    brisbane_kernel_setarg(kernel_z_solve_0, 10, sizeof(int), &gp12);

    brisbane_task task0;
    brisbane_task_create(&task0);
    brisbane_task_kernel(task0, kernel_z_solve_0, 2, kernel_z_solve_0_off, kernel_z_solve_0_idx);
    brisbane_task_submit(task0, brisbane_cpu, NULL, true);
#if 0
    #pragma omp target teams distribute parallel for collapse(2) private(i,j,k,temp1,temp2,temp3) 
    for (k = 0; k <= ksize; k++) {
      for (i = 1; i <= gp02; i++) {
        for (j = 1; j <= gp12; j++) {
          temp1 = 1.0 / u[k][j][i][0];
          temp2 = temp1 * temp1;
          temp3 = temp1 * temp2;

          fjacZ[0][0][k][i][j] = 0.0;
          fjacZ[0][1][k][i][j] = 0.0;
          fjacZ[0][2][k][i][j] = 0.0;
          fjacZ[0][3][k][i][j] = 1.0;
          fjacZ[0][4][k][i][j] = 0.0;

          fjacZ[1][0][k][i][j] = - ( u[k][j][i][1]*u[k][j][i][3] ) * temp2;
          fjacZ[1][1][k][i][j] = u[k][j][i][3] * temp1;
          fjacZ[1][2][k][i][j] = 0.0;
          fjacZ[1][3][k][i][j] = u[k][j][i][1] * temp1;
          fjacZ[1][4][k][i][j] = 0.0;

          fjacZ[2][0][k][i][j] = - ( u[k][j][i][2]*u[k][j][i][3] ) * temp2;
          fjacZ[2][1][k][i][j] = 0.0;
          fjacZ[2][2][k][i][j] = u[k][j][i][3] * temp1;
          fjacZ[2][3][k][i][j] = u[k][j][i][2] * temp1;
          fjacZ[2][4][k][i][j] = 0.0;

          fjacZ[3][0][k][i][j] = - (u[k][j][i][3]*u[k][j][i][3] * temp2 )
            + c2 * qs[k][j][i];
          fjacZ[3][1][k][i][j] = - c2 *  u[k][j][i][1] * temp1;
          fjacZ[3][2][k][i][j] = - c2 *  u[k][j][i][2] * temp1;
          fjacZ[3][3][k][i][j] = ( 2.0 - c2 ) *  u[k][j][i][3] * temp1;
          fjacZ[3][4][k][i][j] = c2;

          fjacZ[4][0][k][i][j] = ( c2 * 2.0 * square[k][j][i] - c1 * u[k][j][i][4] )
            * u[k][j][i][3] * temp2;
          fjacZ[4][1][k][i][j] = - c2 * ( u[k][j][i][1]*u[k][j][i][3] ) * temp2;
          fjacZ[4][2][k][i][j] = - c2 * ( u[k][j][i][2]*u[k][j][i][3] ) * temp2;
          fjacZ[4][3][k][i][j] = c1 * ( u[k][j][i][4] * temp1 )
            - c2 * ( qs[k][j][i] + u[k][j][i][3]*u[k][j][i][3] * temp2 );
          fjacZ[4][4][k][i][j] = c1 * u[k][j][i][3] * temp1;

          njacZ[0][0][k][i][j] = 0.0;
          njacZ[0][1][k][i][j] = 0.0;
          njacZ[0][2][k][i][j] = 0.0;
          njacZ[0][3][k][i][j] = 0.0;
          njacZ[0][4][k][i][j] = 0.0;

          njacZ[1][0][k][i][j] = - c3c4 * temp2 * u[k][j][i][1];
          njacZ[1][1][k][i][j] =   c3c4 * temp1;
          njacZ[1][2][k][i][j] =   0.0;
          njacZ[1][3][k][i][j] =   0.0;
          njacZ[1][4][k][i][j] =   0.0;

          njacZ[2][0][k][i][j] = - c3c4 * temp2 * u[k][j][i][2];
          njacZ[2][1][k][i][j] =   0.0;
          njacZ[2][2][k][i][j] =   c3c4 * temp1;
          njacZ[2][3][k][i][j] =   0.0;
          njacZ[2][4][k][i][j] =   0.0;

          njacZ[3][0][k][i][j] = - con43 * c3c4 * temp2 * u[k][j][i][3];
          njacZ[3][1][k][i][j] =   0.0;
          njacZ[3][2][k][i][j] =   0.0;
          njacZ[3][3][k][i][j] =   con43 * c3c4 * temp1;
          njacZ[3][4][k][i][j] =   0.0;

          njacZ[4][0][k][i][j] = - (  c3c4
            - c1345 ) * temp3 * (u[k][j][i][1]*u[k][j][i][1])
            - ( c3c4 - c1345 ) * temp3 * (u[k][j][i][2]*u[k][j][i][2])
            - ( con43 * c3c4
            - c1345 ) * temp3 * (u[k][j][i][3]*u[k][j][i][3])
            - c1345 * temp2 * u[k][j][i][4];

          njacZ[4][1][k][i][j] = (  c3c4 - c1345 ) * temp2 * u[k][j][i][1];
          njacZ[4][2][k][i][j] = (  c3c4 - c1345 ) * temp2 * u[k][j][i][2];
          njacZ[4][3][k][i][j] = ( con43 * c3c4 - c1345 ) * temp2 * u[k][j][i][3];
          njacZ[4][4][k][i][j] = ( c1345 )* temp1;
        }
      }
    }
#endif

    //---------------------------------------------------------------------
    // now jacobians set, so form left hand side in z direction
    //---------------------------------------------------------------------
    //lhsZ[j][i]init(lhsZ[j][i], ksize);
    // zero the whole left hand side for starters
    size_t kernel_z_solve_1_off[3] = { 1, 0, 0 };
    size_t kernel_z_solve_1_idx[3] = { gp02, 5, 5 };
    brisbane_kernel kernel_z_solve_1;
    brisbane_kernel_create("z_solve_1", &kernel_z_solve_1);
    brisbane_kernel_setmem(kernel_z_solve_1, 0, mem_lhsZ, brisbane_w);
    brisbane_kernel_setarg(kernel_z_solve_1, 1, sizeof(int), &ksize);
    brisbane_kernel_setarg(kernel_z_solve_1, 2, sizeof(int), &gp12);

    brisbane_task task1;
    brisbane_task_create(&task1);
    brisbane_task_kernel(task1, kernel_z_solve_1, 3, kernel_z_solve_1_off, kernel_z_solve_1_idx);
    brisbane_task_submit(task1, brisbane_cpu, NULL, true);
#if 0
#ifdef SPEC_USE_INNER_SIMD
    #pragma omp target teams distribute parallel for private(i,j) collapse(3)
#else
    #pragma omp target teams distribute parallel for simd collapse(4)
#endif
    for (m = 0; m < 5; m++) {
      for (n = 0; n < 5; n++) {
        for (i = 1; i <= gp02; i++) {
#ifdef SPEC_USE_INNER_SIMD
          #pragma omp simd
#endif
          for (j = 1; j <= gp12; j++) {
            lhsZ[m][n][0][0][i][j] = 0.0;
            lhsZ[m][n][1][0][i][j] = 0.0;
            lhsZ[m][n][2][0][i][j] = 0.0;
            lhsZ[m][n][0][ksize][i][j] = 0.0;
            lhsZ[m][n][1][ksize][i][j] = 0.0;
            lhsZ[m][n][2][ksize][i][j] = 0.0;
          }
        }
      }
    }
#endif

    // next, set all diagonal values to 1. This is overkill, but convenient
    size_t kernel_z_solve_2_off[3] = { 1, 1, 0 };
    size_t kernel_z_solve_2_idx[3] = { gp12, gp02, 5 };
    brisbane_kernel kernel_z_solve_2;
    brisbane_kernel_create("z_solve_2", &kernel_z_solve_2);
    brisbane_kernel_setmem(kernel_z_solve_2, 0, mem_lhsZ, brisbane_w);
    brisbane_kernel_setarg(kernel_z_solve_2, 1, sizeof(int), &ksize);

    brisbane_task task2;
    brisbane_task_create(&task2);
    brisbane_task_kernel(task2, kernel_z_solve_2, 3, kernel_z_solve_2_off, kernel_z_solve_2_idx);
    brisbane_task_submit(task2, brisbane_cpu, NULL, true);
#if 0
#ifdef SPEC_USE_INNER_SIMD
    #pragma omp target teams distribute parallel for private(i,j) collapse(2)
#else
    #pragma omp target teams distribute parallel for simd collapse(3)
#endif
    for (m = 0; m < 5; m++){
      for (i = 1; i <= gp02; i++) {
#ifdef SPEC_USE_INNER_SIMD
        #pragma omp simd
#endif
        for (j = 1; j <= gp12; j++) {
          lhsZ[m][m][1][0][i][j] = 1.0;
          lhsZ[m][m][1][ksize][i][j] = 1.0;
        }
      }
    }
#endif

    size_t kernel_z_solve_3_off[3] = { 1, 1, 1 };
    size_t kernel_z_solve_3_idx[3] = { gp12, gp02, ksize - 1 };
    brisbane_kernel kernel_z_solve_3;
    brisbane_kernel_create("z_solve_3", &kernel_z_solve_3);
    brisbane_kernel_setmem(kernel_z_solve_3, 0, mem_lhsZ, brisbane_w);
    brisbane_kernel_setmem(kernel_z_solve_3, 1, mem_fjacZ, brisbane_r);
    brisbane_kernel_setmem(kernel_z_solve_3, 2, mem_njacZ, brisbane_r);
    brisbane_kernel_setarg(kernel_z_solve_3, 3, sizeof(double), &dttz1);
    brisbane_kernel_setarg(kernel_z_solve_3, 4, sizeof(double), &dttz2);
    brisbane_kernel_setarg(kernel_z_solve_3, 5, sizeof(double), &dz1);
    brisbane_kernel_setarg(kernel_z_solve_3, 6, sizeof(double), &dz2);
    brisbane_kernel_setarg(kernel_z_solve_3, 7, sizeof(double), &dz3);
    brisbane_kernel_setarg(kernel_z_solve_3, 8, sizeof(double), &dz4);
    brisbane_kernel_setarg(kernel_z_solve_3, 9, sizeof(double), &dz5);

    brisbane_task task3;
    brisbane_task_create(&task3);
    brisbane_task_kernel(task3, kernel_z_solve_3, 3, kernel_z_solve_3_off, kernel_z_solve_3_idx);
    brisbane_task_submit(task3, brisbane_cpu, NULL, true);
#if 0
#ifdef SPEC_USE_INNER_SIMD
    #pragma omp target teams distribute parallel for collapse(2) private(i,j,k)
#else
    #pragma omp target teams distribute parallel for simd collapse(3)
#endif
    for (k = 1; k <= ksize-1; k++) {
      for (i = 1; i <= gp02; i++) {
#ifdef SPEC_USE_INNER_SIMD
        #pragma omp simd
#endif
        for (j = 1; j <= gp12; j++) {
          lhsZ[0][0][AA][k][i][j] = - dttz2 * fjacZ[0][0][k-1][i][j]
            - dttz1 * njacZ[0][0][k-1][i][j]
            - dttz1 * dz1;
          lhsZ[0][1][AA][k][i][j] = - dttz2 * fjacZ[0][1][k-1][i][j]
            - dttz1 * njacZ[0][1][k-1][i][j];
          lhsZ[0][2][AA][k][i][j] = - dttz2 * fjacZ[0][2][k-1][i][j]
            - dttz1 * njacZ[0][2][k-1][i][j];
          lhsZ[0][3][AA][k][i][j] = - dttz2 * fjacZ[0][3][k-1][i][j]
            - dttz1 * njacZ[0][3][k-1][i][j];
          lhsZ[0][4][AA][k][i][j] = - dttz2 * fjacZ[0][4][k-1][i][j]
            - dttz1 * njacZ[0][4][k-1][i][j];

          lhsZ[1][0][AA][k][i][j] = - dttz2 * fjacZ[1][0][k-1][i][j]
            - dttz1 * njacZ[1][0][k-1][i][j];
          lhsZ[1][1][AA][k][i][j] = - dttz2 * fjacZ[1][1][k-1][i][j]
            - dttz1 * njacZ[1][1][k-1][i][j]
            - dttz1 * dz2;
          lhsZ[1][2][AA][k][i][j] = - dttz2 * fjacZ[1][2][k-1][i][j]
            - dttz1 * njacZ[1][2][k-1][i][j];
          lhsZ[1][3][AA][k][i][j] = - dttz2 * fjacZ[1][3][k-1][i][j]
            - dttz1 * njacZ[1][3][k-1][i][j];
          lhsZ[1][4][AA][k][i][j] = - dttz2 * fjacZ[1][4][k-1][i][j]
            - dttz1 * njacZ[1][4][k-1][i][j];

          lhsZ[2][0][AA][k][i][j] = - dttz2 * fjacZ[2][0][k-1][i][j]
            - dttz1 * njacZ[2][0][k-1][i][j];
          lhsZ[2][1][AA][k][i][j] = - dttz2 * fjacZ[2][1][k-1][i][j]
            - dttz1 * njacZ[2][1][k-1][i][j];
          lhsZ[2][2][AA][k][i][j] = - dttz2 * fjacZ[2][2][k-1][i][j]
            - dttz1 * njacZ[2][2][k-1][i][j]
            - dttz1 * dz3;
          lhsZ[2][3][AA][k][i][j] = - dttz2 * fjacZ[2][3][k-1][i][j]
            - dttz1 * njacZ[2][3][k-1][i][j];
          lhsZ[2][4][AA][k][i][j] = - dttz2 * fjacZ[2][4][k-1][i][j]
            - dttz1 * njacZ[2][4][k-1][i][j];

          lhsZ[3][0][AA][k][i][j] = - dttz2 * fjacZ[3][0][k-1][i][j]
            - dttz1 * njacZ[3][0][k-1][i][j];
          lhsZ[3][1][AA][k][i][j] = - dttz2 * fjacZ[3][1][k-1][i][j]
            - dttz1 * njacZ[3][1][k-1][i][j];
          lhsZ[3][2][AA][k][i][j] = - dttz2 * fjacZ[3][2][k-1][i][j]
            - dttz1 * njacZ[3][2][k-1][i][j];
          lhsZ[3][3][AA][k][i][j] = - dttz2 * fjacZ[3][3][k-1][i][j]
            - dttz1 * njacZ[3][3][k-1][i][j]
            - dttz1 * dz4;
          lhsZ[3][4][AA][k][i][j] = - dttz2 * fjacZ[3][4][k-1][i][j]
            - dttz1 * njacZ[3][4][k-1][i][j];

          lhsZ[4][0][AA][k][i][j] = - dttz2 * fjacZ[4][0][k-1][i][j]
            - dttz1 * njacZ[4][0][k-1][i][j];
          lhsZ[4][1][AA][k][i][j] = - dttz2 * fjacZ[4][1][k-1][i][j]
            - dttz1 * njacZ[4][1][k-1][i][j];
          lhsZ[4][2][AA][k][i][j] = - dttz2 * fjacZ[4][2][k-1][i][j]
            - dttz1 * njacZ[4][2][k-1][i][j];
          lhsZ[4][3][AA][k][i][j] = - dttz2 * fjacZ[4][3][k-1][i][j]
            - dttz1 * njacZ[4][3][k-1][i][j];
          lhsZ[4][4][AA][k][i][j] = - dttz2 * fjacZ[4][4][k-1][i][j]
            - dttz1 * njacZ[4][4][k-1][i][j]
            - dttz1 * dz5;

          lhsZ[0][0][BB][k][i][j] = 1.0
            + dttz1 * 2.0 * njacZ[0][0][k][i][j]
            + dttz1 * 2.0 * dz1;
          lhsZ[0][1][BB][k][i][j] = dttz1 * 2.0 * njacZ[0][1][k][i][j];
          lhsZ[0][2][BB][k][i][j] = dttz1 * 2.0 * njacZ[0][2][k][i][j];
          lhsZ[0][3][BB][k][i][j] = dttz1 * 2.0 * njacZ[0][3][k][i][j];
          lhsZ[0][4][BB][k][i][j] = dttz1 * 2.0 * njacZ[0][4][k][i][j];

          lhsZ[1][0][BB][k][i][j] = dttz1 * 2.0 * njacZ[1][0][k][i][j];
          lhsZ[1][1][BB][k][i][j] = 1.0
            + dttz1 * 2.0 * njacZ[1][1][k][i][j]
            + dttz1 * 2.0 * dz2;
          lhsZ[1][2][BB][k][i][j] = dttz1 * 2.0 * njacZ[1][2][k][i][j];
          lhsZ[1][3][BB][k][i][j] = dttz1 * 2.0 * njacZ[1][3][k][i][j];
          lhsZ[1][4][BB][k][i][j] = dttz1 * 2.0 * njacZ[1][4][k][i][j];

          lhsZ[2][0][BB][k][i][j] = dttz1 * 2.0 * njacZ[2][0][k][i][j];
          lhsZ[2][1][BB][k][i][j] = dttz1 * 2.0 * njacZ[2][1][k][i][j];
          lhsZ[2][2][BB][k][i][j] = 1.0
            + dttz1 * 2.0 * njacZ[2][2][k][i][j]
            + dttz1 * 2.0 * dz3;
          lhsZ[2][3][BB][k][i][j] = dttz1 * 2.0 * njacZ[2][3][k][i][j];
          lhsZ[2][4][BB][k][i][j] = dttz1 * 2.0 * njacZ[2][4][k][i][j];

          lhsZ[3][0][BB][k][i][j] = dttz1 * 2.0 * njacZ[3][0][k][i][j];
          lhsZ[3][1][BB][k][i][j] = dttz1 * 2.0 * njacZ[3][1][k][i][j];
          lhsZ[3][2][BB][k][i][j] = dttz1 * 2.0 * njacZ[3][2][k][i][j];
          lhsZ[3][3][BB][k][i][j] = 1.0
            + dttz1 * 2.0 * njacZ[3][3][k][i][j]
            + dttz1 * 2.0 * dz4;
          lhsZ[3][4][BB][k][i][j] = dttz1 * 2.0 * njacZ[3][4][k][i][j];

          lhsZ[4][0][BB][k][i][j] = dttz1 * 2.0 * njacZ[4][0][k][i][j];
          lhsZ[4][1][BB][k][i][j] = dttz1 * 2.0 * njacZ[4][1][k][i][j];
          lhsZ[4][2][BB][k][i][j] = dttz1 * 2.0 * njacZ[4][2][k][i][j];
          lhsZ[4][3][BB][k][i][j] = dttz1 * 2.0 * njacZ[4][3][k][i][j];
          lhsZ[4][4][BB][k][i][j] = 1.0
            + dttz1 * 2.0 * njacZ[4][4][k][i][j]
            + dttz1 * 2.0 * dz5;

          lhsZ[0][0][CC][k][i][j] =  dttz2 * fjacZ[0][0][k+1][i][j]
            - dttz1 * njacZ[0][0][k+1][i][j]
            - dttz1 * dz1;
          lhsZ[0][1][CC][k][i][j] =  dttz2 * fjacZ[0][1][k+1][i][j]
            - dttz1 * njacZ[0][1][k+1][i][j];
          lhsZ[0][2][CC][k][i][j] =  dttz2 * fjacZ[0][2][k+1][i][j]
            - dttz1 * njacZ[0][2][k+1][i][j];
          lhsZ[0][3][CC][k][i][j] =  dttz2 * fjacZ[0][3][k+1][i][j]
            - dttz1 * njacZ[0][3][k+1][i][j];
          lhsZ[0][4][CC][k][i][j] =  dttz2 * fjacZ[0][4][k+1][i][j]
            - dttz1 * njacZ[0][4][k+1][i][j];

          lhsZ[1][0][CC][k][i][j] =  dttz2 * fjacZ[1][0][k+1][i][j]
            - dttz1 * njacZ[1][0][k+1][i][j];
          lhsZ[1][1][CC][k][i][j] =  dttz2 * fjacZ[1][1][k+1][i][j]
            - dttz1 * njacZ[1][1][k+1][i][j]
            - dttz1 * dz2;
          lhsZ[1][2][CC][k][i][j] =  dttz2 * fjacZ[1][2][k+1][i][j]
            - dttz1 * njacZ[1][2][k+1][i][j];
          lhsZ[1][3][CC][k][i][j] =  dttz2 * fjacZ[1][3][k+1][i][j]
            - dttz1 * njacZ[1][3][k+1][i][j];
          lhsZ[1][4][CC][k][i][j] =  dttz2 * fjacZ[1][4][k+1][i][j]
            - dttz1 * njacZ[1][4][k+1][i][j];

          lhsZ[2][0][CC][k][i][j] =  dttz2 * fjacZ[2][0][k+1][i][j]
            - dttz1 * njacZ[2][0][k+1][i][j];
          lhsZ[2][1][CC][k][i][j] =  dttz2 * fjacZ[2][1][k+1][i][j]
            - dttz1 * njacZ[2][1][k+1][i][j];
          lhsZ[2][2][CC][k][i][j] =  dttz2 * fjacZ[2][2][k+1][i][j]
            - dttz1 * njacZ[2][2][k+1][i][j]
            - dttz1 * dz3;
          lhsZ[2][3][CC][k][i][j] =  dttz2 * fjacZ[2][3][k+1][i][j]
            - dttz1 * njacZ[2][3][k+1][i][j];
          lhsZ[2][4][CC][k][i][j] =  dttz2 * fjacZ[2][4][k+1][i][j]
            - dttz1 * njacZ[2][4][k+1][i][j];

          lhsZ[3][0][CC][k][i][j] =  dttz2 * fjacZ[3][0][k+1][i][j]
            - dttz1 * njacZ[3][0][k+1][i][j];
          lhsZ[3][1][CC][k][i][j] =  dttz2 * fjacZ[3][1][k+1][i][j]
            - dttz1 * njacZ[3][1][k+1][i][j];
          lhsZ[3][2][CC][k][i][j] =  dttz2 * fjacZ[3][2][k+1][i][j]
            - dttz1 * njacZ[3][2][k+1][i][j];
          lhsZ[3][3][CC][k][i][j] =  dttz2 * fjacZ[3][3][k+1][i][j]
            - dttz1 * njacZ[3][3][k+1][i][j]
            - dttz1 * dz4;
          lhsZ[3][4][CC][k][i][j] =  dttz2 * fjacZ[3][4][k+1][i][j]
            - dttz1 * njacZ[3][4][k+1][i][j];

          lhsZ[4][0][CC][k][i][j] =  dttz2 * fjacZ[4][0][k+1][i][j]
            - dttz1 * njacZ[4][0][k+1][i][j];
          lhsZ[4][1][CC][k][i][j] =  dttz2 * fjacZ[4][1][k+1][i][j]
            - dttz1 * njacZ[4][1][k+1][i][j];
          lhsZ[4][2][CC][k][i][j] =  dttz2 * fjacZ[4][2][k+1][i][j]
            - dttz1 * njacZ[4][2][k+1][i][j];
          lhsZ[4][3][CC][k][i][j] =  dttz2 * fjacZ[4][3][k+1][i][j]
            - dttz1 * njacZ[4][3][k+1][i][j];
          lhsZ[4][4][CC][k][i][j] =  dttz2 * fjacZ[4][4][k+1][i][j]
            - dttz1 * njacZ[4][4][k+1][i][j]
            - dttz1 * dz5;
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
    // c'(KMAX) and rhs'(KMAX) will be sent to next cell.
    //---------------------------------------------------------------------

    //---------------------------------------------------------------------
    // outer most do loops - sweeping in i direction
    //---------------------------------------------------------------------

    //---------------------------------------------------------------------
    // multiply c[0][j][i] by b_inverse and copy back to c
    // multiply rhs(0) by b_inverse(0) and copy to rhs
    //---------------------------------------------------------------------
    //binvcrhs( lhsZ[0][i][BB], lhsZ[j][0][i][j][CC], rhs[0][j][i] );
    size_t kernel_z_solve_4_off[2] = { 1, 1 };
    size_t kernel_z_solve_4_idx[2] = { gp12, gp02 };
    brisbane_kernel kernel_z_solve_4;
    brisbane_kernel_create("z_solve_4", &kernel_z_solve_4);
    brisbane_kernel_setmem(kernel_z_solve_4, 0, mem_lhsZ, brisbane_rw);
    brisbane_kernel_setmem(kernel_z_solve_4, 1, mem_rhs, brisbane_rw);

    brisbane_task task4;
    brisbane_task_create(&task4);
    brisbane_task_kernel(task4, kernel_z_solve_4, 2, kernel_z_solve_4_off, kernel_z_solve_4_idx);
    brisbane_task_submit(task4, brisbane_cpu, NULL, true);
#if 0
#ifdef SPEC_USE_INNER_SIMD
    #pragma omp target teams distribute parallel for private(i,j,pivot, coeff) 
#else
    #pragma omp target teams distribute parallel for simd private(pivot, coeff) collapse(2)
#endif
    for (i = 1; i <= gp02; i++) {
#ifdef SPEC_USE_INNER_SIMD
      #pragma omp simd private(pivot, coeff)
#endif
      for (j = 1; j <= gp12; j++) {
        /*
        for(m = 0; m < 5; m++){
          pivot = 1.00/lhsZ[m][m][BB][0][i][j];
          for(n = m+1; n < 5; n++){
            lhsZ[m][n][BB][0][i][j] = lhsZ[m][n][BB][0][i][j]*pivot;
          }
          lhsZ[m][0][CC][0][i][j] = lhsZ[m][0][CC][0][i][j]*pivot;
          lhsZ[m][1][CC][0][i][j] = lhsZ[m][1][CC][0][i][j]*pivot;
          lhsZ[m][2][CC][0][i][j] = lhsZ[m][2][CC][0][i][j]*pivot;
          lhsZ[m][3][CC][0][i][j] = lhsZ[m][3][CC][0][i][j]*pivot;
          lhsZ[m][4][CC][0][i][j] = lhsZ[m][4][CC][0][i][j]*pivot;
          rhs[0][j][i][m] = rhs[0][j][i][m]*pivot;

          for(n = 0; n < 5; n++){
            if(n != m){
              coeff = lhsZ[n][m][BB][0][i][j];
              for(z = m+1; z < 5; z++){
                lhsZ[n][z][BB][0][i][j] = lhsZ[n][z][BB][0][i][j] - coeff*lhsZ[m][z][BB][0][i][j];
              }
              lhsZ[n][0][CC][0][i][j] = lhsZ[n][0][CC][0][i][j] - coeff*lhsZ[m][0][CC][0][i][j];
              lhsZ[n][1][CC][0][i][j] = lhsZ[n][1][CC][0][i][j] - coeff*lhsZ[m][1][CC][0][i][j];
              lhsZ[n][2][CC][0][i][j] = lhsZ[n][2][CC][0][i][j] - coeff*lhsZ[m][2][CC][0][i][j];
              lhsZ[n][3][CC][0][i][j] = lhsZ[n][3][CC][0][i][j] - coeff*lhsZ[m][3][CC][0][i][j];
              lhsZ[n][4][CC][0][i][j] = lhsZ[n][4][CC][0][i][j] - coeff*lhsZ[m][4][CC][0][i][j];
              rhs[0][j][i][n] = rhs[0][j][i][n] - coeff*rhs[0][j][i][m];
            }
          }
        }
        */
        pivot = 1.00/lhsZ[0][0][BB][0][i][j];
        lhsZ[0][1][BB][0][i][j] = lhsZ[0][1][BB][0][i][j]*pivot;
        lhsZ[0][2][BB][0][i][j] = lhsZ[0][2][BB][0][i][j]*pivot;
        lhsZ[0][3][BB][0][i][j] = lhsZ[0][3][BB][0][i][j]*pivot;
        lhsZ[0][4][BB][0][i][j] = lhsZ[0][4][BB][0][i][j]*pivot;
        lhsZ[0][0][CC][0][i][j] = lhsZ[0][0][CC][0][i][j]*pivot;
        lhsZ[0][1][CC][0][i][j] = lhsZ[0][1][CC][0][i][j]*pivot;
        lhsZ[0][2][CC][0][i][j] = lhsZ[0][2][CC][0][i][j]*pivot;
        lhsZ[0][3][CC][0][i][j] = lhsZ[0][3][CC][0][i][j]*pivot;
        lhsZ[0][4][CC][0][i][j] = lhsZ[0][4][CC][0][i][j]*pivot;
        rhs[0][j][i][0]   = rhs[0][j][i][0]  *pivot;

        coeff = lhsZ[1][0][BB][0][i][j];
        lhsZ[1][1][BB][0][i][j]= lhsZ[1][1][BB][0][i][j] - coeff*lhsZ[0][1][BB][0][i][j];
        lhsZ[1][2][BB][0][i][j]= lhsZ[1][2][BB][0][i][j] - coeff*lhsZ[0][2][BB][0][i][j];
        lhsZ[1][3][BB][0][i][j]= lhsZ[1][3][BB][0][i][j] - coeff*lhsZ[0][3][BB][0][i][j];
        lhsZ[1][4][BB][0][i][j]= lhsZ[1][4][BB][0][i][j] - coeff*lhsZ[0][4][BB][0][i][j];
        lhsZ[1][0][CC][0][i][j] = lhsZ[1][0][CC][0][i][j] - coeff*lhsZ[0][0][CC][0][i][j];
        lhsZ[1][1][CC][0][i][j] = lhsZ[1][1][CC][0][i][j] - coeff*lhsZ[0][1][CC][0][i][j];
        lhsZ[1][2][CC][0][i][j] = lhsZ[1][2][CC][0][i][j] - coeff*lhsZ[0][2][CC][0][i][j];
        lhsZ[1][3][CC][0][i][j] = lhsZ[1][3][CC][0][i][j] - coeff*lhsZ[0][3][CC][0][i][j];
        lhsZ[1][4][CC][0][i][j] = lhsZ[1][4][CC][0][i][j] - coeff*lhsZ[0][4][CC][0][i][j];
        rhs[0][j][i][1]   = rhs[0][j][i][1]   - coeff*rhs[0][j][i][0];

        coeff = lhsZ[2][0][BB][0][i][j];
        lhsZ[2][1][BB][0][i][j]= lhsZ[2][1][BB][0][i][j] - coeff*lhsZ[0][1][BB][0][i][j];
        lhsZ[2][2][BB][0][i][j]= lhsZ[2][2][BB][0][i][j] - coeff*lhsZ[0][2][BB][0][i][j];
        lhsZ[2][3][BB][0][i][j]= lhsZ[2][3][BB][0][i][j] - coeff*lhsZ[0][3][BB][0][i][j];
        lhsZ[2][4][BB][0][i][j]= lhsZ[2][4][BB][0][i][j] - coeff*lhsZ[0][4][BB][0][i][j];
        lhsZ[2][0][CC][0][i][j] = lhsZ[2][0][CC][0][i][j] - coeff*lhsZ[0][0][CC][0][i][j];
        lhsZ[2][1][CC][0][i][j] = lhsZ[2][1][CC][0][i][j] - coeff*lhsZ[0][1][CC][0][i][j];
        lhsZ[2][2][CC][0][i][j] = lhsZ[2][2][CC][0][i][j] - coeff*lhsZ[0][2][CC][0][i][j];
        lhsZ[2][3][CC][0][i][j] = lhsZ[2][3][CC][0][i][j] - coeff*lhsZ[0][3][CC][0][i][j];
        lhsZ[2][4][CC][0][i][j] = lhsZ[2][4][CC][0][i][j] - coeff*lhsZ[0][4][CC][0][i][j];
        rhs[0][j][i][2]   = rhs[0][j][i][2]   - coeff*rhs[0][j][i][0];

        coeff = lhsZ[3][0][BB][0][i][j];
        lhsZ[3][1][BB][0][i][j]= lhsZ[3][1][BB][0][i][j] - coeff*lhsZ[0][1][BB][0][i][j];
        lhsZ[3][2][BB][0][i][j]= lhsZ[3][2][BB][0][i][j] - coeff*lhsZ[0][2][BB][0][i][j];
        lhsZ[3][3][BB][0][i][j]= lhsZ[3][3][BB][0][i][j] - coeff*lhsZ[0][3][BB][0][i][j];
        lhsZ[3][4][BB][0][i][j]= lhsZ[3][4][BB][0][i][j] - coeff*lhsZ[0][4][BB][0][i][j];
        lhsZ[3][0][CC][0][i][j] = lhsZ[3][0][CC][0][i][j] - coeff*lhsZ[0][0][CC][0][i][j];
        lhsZ[3][1][CC][0][i][j] = lhsZ[3][1][CC][0][i][j] - coeff*lhsZ[0][1][CC][0][i][j];
        lhsZ[3][2][CC][0][i][j] = lhsZ[3][2][CC][0][i][j] - coeff*lhsZ[0][2][CC][0][i][j];
        lhsZ[3][3][CC][0][i][j] = lhsZ[3][3][CC][0][i][j] - coeff*lhsZ[0][3][CC][0][i][j];
        lhsZ[3][4][CC][0][i][j] = lhsZ[3][4][CC][0][i][j] - coeff*lhsZ[0][4][CC][0][i][j];
        rhs[0][j][i][3]   = rhs[0][j][i][3]   - coeff*rhs[0][j][i][0];

        coeff = lhsZ[4][0][BB][0][i][j];
        lhsZ[4][1][BB][0][i][j]= lhsZ[4][1][BB][0][i][j] - coeff*lhsZ[0][1][BB][0][i][j];
        lhsZ[4][2][BB][0][i][j]= lhsZ[4][2][BB][0][i][j] - coeff*lhsZ[0][2][BB][0][i][j];
        lhsZ[4][3][BB][0][i][j]= lhsZ[4][3][BB][0][i][j] - coeff*lhsZ[0][3][BB][0][i][j];
        lhsZ[4][4][BB][0][i][j]= lhsZ[4][4][BB][0][i][j] - coeff*lhsZ[0][4][BB][0][i][j];
        lhsZ[4][0][CC][0][i][j] = lhsZ[4][0][CC][0][i][j] - coeff*lhsZ[0][0][CC][0][i][j];
        lhsZ[4][1][CC][0][i][j] = lhsZ[4][1][CC][0][i][j] - coeff*lhsZ[0][1][CC][0][i][j];
        lhsZ[4][2][CC][0][i][j] = lhsZ[4][2][CC][0][i][j] - coeff*lhsZ[0][2][CC][0][i][j];
        lhsZ[4][3][CC][0][i][j] = lhsZ[4][3][CC][0][i][j] - coeff*lhsZ[0][3][CC][0][i][j];
        lhsZ[4][4][CC][0][i][j] = lhsZ[4][4][CC][0][i][j] - coeff*lhsZ[0][4][CC][0][i][j];
        rhs[0][j][i][4]   = rhs[0][j][i][4]   - coeff*rhs[0][j][i][0];


        pivot = 1.00/lhsZ[1][1][BB][0][i][j];
        lhsZ[1][2][BB][0][i][j] = lhsZ[1][2][BB][0][i][j]*pivot;
        lhsZ[1][3][BB][0][i][j] = lhsZ[1][3][BB][0][i][j]*pivot;
        lhsZ[1][4][BB][0][i][j] = lhsZ[1][4][BB][0][i][j]*pivot;
        lhsZ[1][0][CC][0][i][j] = lhsZ[1][0][CC][0][i][j]*pivot;
        lhsZ[1][1][CC][0][i][j] = lhsZ[1][1][CC][0][i][j]*pivot;
        lhsZ[1][2][CC][0][i][j] = lhsZ[1][2][CC][0][i][j]*pivot;
        lhsZ[1][3][CC][0][i][j] = lhsZ[1][3][CC][0][i][j]*pivot;
        lhsZ[1][4][CC][0][i][j] = lhsZ[1][4][CC][0][i][j]*pivot;
        rhs[0][j][i][1]   = rhs[0][j][i][1]  *pivot;

        coeff = lhsZ[0][1][BB][0][i][j];
        lhsZ[0][2][BB][0][i][j]= lhsZ[0][2][BB][0][i][j] - coeff*lhsZ[1][2][BB][0][i][j];
        lhsZ[0][3][BB][0][i][j]= lhsZ[0][3][BB][0][i][j] - coeff*lhsZ[1][3][BB][0][i][j];
        lhsZ[0][4][BB][0][i][j]= lhsZ[0][4][BB][0][i][j] - coeff*lhsZ[1][4][BB][0][i][j];
        lhsZ[0][0][CC][0][i][j] = lhsZ[0][0][CC][0][i][j] - coeff*lhsZ[1][0][CC][0][i][j];
        lhsZ[0][1][CC][0][i][j] = lhsZ[0][1][CC][0][i][j] - coeff*lhsZ[1][1][CC][0][i][j];
        lhsZ[0][2][CC][0][i][j] = lhsZ[0][2][CC][0][i][j] - coeff*lhsZ[1][2][CC][0][i][j];
        lhsZ[0][3][CC][0][i][j] = lhsZ[0][3][CC][0][i][j] - coeff*lhsZ[1][3][CC][0][i][j];
        lhsZ[0][4][CC][0][i][j] = lhsZ[0][4][CC][0][i][j] - coeff*lhsZ[1][4][CC][0][i][j];
        rhs[0][j][i][0]   = rhs[0][j][i][0]   - coeff*rhs[0][j][i][1];

        coeff = lhsZ[2][1][BB][0][i][j];
        lhsZ[2][2][BB][0][i][j]= lhsZ[2][2][BB][0][i][j] - coeff*lhsZ[1][2][BB][0][i][j];
        lhsZ[2][3][BB][0][i][j]= lhsZ[2][3][BB][0][i][j] - coeff*lhsZ[1][3][BB][0][i][j];
        lhsZ[2][4][BB][0][i][j]= lhsZ[2][4][BB][0][i][j] - coeff*lhsZ[1][4][BB][0][i][j];
        lhsZ[2][0][CC][0][i][j] = lhsZ[2][0][CC][0][i][j] - coeff*lhsZ[1][0][CC][0][i][j];
        lhsZ[2][1][CC][0][i][j] = lhsZ[2][1][CC][0][i][j] - coeff*lhsZ[1][1][CC][0][i][j];
        lhsZ[2][2][CC][0][i][j] = lhsZ[2][2][CC][0][i][j] - coeff*lhsZ[1][2][CC][0][i][j];
        lhsZ[2][3][CC][0][i][j] = lhsZ[2][3][CC][0][i][j] - coeff*lhsZ[1][3][CC][0][i][j];
        lhsZ[2][4][CC][0][i][j] = lhsZ[2][4][CC][0][i][j] - coeff*lhsZ[1][4][CC][0][i][j];
        rhs[0][j][i][2]   = rhs[0][j][i][2]   - coeff*rhs[0][j][i][1];

        coeff = lhsZ[3][1][BB][0][i][j];
        lhsZ[3][2][BB][0][i][j]= lhsZ[3][2][BB][0][i][j] - coeff*lhsZ[1][2][BB][0][i][j];
        lhsZ[3][3][BB][0][i][j]= lhsZ[3][3][BB][0][i][j] - coeff*lhsZ[1][3][BB][0][i][j];
        lhsZ[3][4][BB][0][i][j]= lhsZ[3][4][BB][0][i][j] - coeff*lhsZ[1][4][BB][0][i][j];
        lhsZ[3][0][CC][0][i][j] = lhsZ[3][0][CC][0][i][j] - coeff*lhsZ[1][0][CC][0][i][j];
        lhsZ[3][1][CC][0][i][j] = lhsZ[3][1][CC][0][i][j] - coeff*lhsZ[1][1][CC][0][i][j];
        lhsZ[3][2][CC][0][i][j] = lhsZ[3][2][CC][0][i][j] - coeff*lhsZ[1][2][CC][0][i][j];
        lhsZ[3][3][CC][0][i][j] = lhsZ[3][3][CC][0][i][j] - coeff*lhsZ[1][3][CC][0][i][j];
        lhsZ[3][4][CC][0][i][j] = lhsZ[3][4][CC][0][i][j] - coeff*lhsZ[1][4][CC][0][i][j];
        rhs[0][j][i][3]   = rhs[0][j][i][3]   - coeff*rhs[0][j][i][1];

        coeff = lhsZ[4][1][BB][0][i][j];
        lhsZ[4][2][BB][0][i][j]= lhsZ[4][2][BB][0][i][j] - coeff*lhsZ[1][2][BB][0][i][j];
        lhsZ[4][3][BB][0][i][j]= lhsZ[4][3][BB][0][i][j] - coeff*lhsZ[1][3][BB][0][i][j];
        lhsZ[4][4][BB][0][i][j]= lhsZ[4][4][BB][0][i][j] - coeff*lhsZ[1][4][BB][0][i][j];
        lhsZ[4][0][CC][0][i][j] = lhsZ[4][0][CC][0][i][j] - coeff*lhsZ[1][0][CC][0][i][j];
        lhsZ[4][1][CC][0][i][j] = lhsZ[4][1][CC][0][i][j] - coeff*lhsZ[1][1][CC][0][i][j];
        lhsZ[4][2][CC][0][i][j] = lhsZ[4][2][CC][0][i][j] - coeff*lhsZ[1][2][CC][0][i][j];
        lhsZ[4][3][CC][0][i][j] = lhsZ[4][3][CC][0][i][j] - coeff*lhsZ[1][3][CC][0][i][j];
        lhsZ[4][4][CC][0][i][j] = lhsZ[4][4][CC][0][i][j] - coeff*lhsZ[1][4][CC][0][i][j];
        rhs[0][j][i][4]   = rhs[0][j][i][4]   - coeff*rhs[0][j][i][1];


        pivot = 1.00/lhsZ[2][2][BB][0][i][j];
        lhsZ[2][3][BB][0][i][j] = lhsZ[2][3][BB][0][i][j]*pivot;
        lhsZ[2][4][BB][0][i][j] = lhsZ[2][4][BB][0][i][j]*pivot;
        lhsZ[2][0][CC][0][i][j] = lhsZ[2][0][CC][0][i][j]*pivot;
        lhsZ[2][1][CC][0][i][j] = lhsZ[2][1][CC][0][i][j]*pivot;
        lhsZ[2][2][CC][0][i][j] = lhsZ[2][2][CC][0][i][j]*pivot;
        lhsZ[2][3][CC][0][i][j] = lhsZ[2][3][CC][0][i][j]*pivot;
        lhsZ[2][4][CC][0][i][j] = lhsZ[2][4][CC][0][i][j]*pivot;
        rhs[0][j][i][2]   = rhs[0][j][i][2]  *pivot;

        coeff = lhsZ[0][2][BB][0][i][j];
        lhsZ[0][3][BB][0][i][j]= lhsZ[0][3][BB][0][i][j] - coeff*lhsZ[2][3][BB][0][i][j];
        lhsZ[0][4][BB][0][i][j]= lhsZ[0][4][BB][0][i][j] - coeff*lhsZ[2][4][BB][0][i][j];
        lhsZ[0][0][CC][0][i][j] = lhsZ[0][0][CC][0][i][j] - coeff*lhsZ[2][0][CC][0][i][j];
        lhsZ[0][1][CC][0][i][j] = lhsZ[0][1][CC][0][i][j] - coeff*lhsZ[2][1][CC][0][i][j];
        lhsZ[0][2][CC][0][i][j] = lhsZ[0][2][CC][0][i][j] - coeff*lhsZ[2][2][CC][0][i][j];
        lhsZ[0][3][CC][0][i][j] = lhsZ[0][3][CC][0][i][j] - coeff*lhsZ[2][3][CC][0][i][j];
        lhsZ[0][4][CC][0][i][j] = lhsZ[0][4][CC][0][i][j] - coeff*lhsZ[2][4][CC][0][i][j];
        rhs[0][j][i][0]   = rhs[0][j][i][0]   - coeff*rhs[0][j][i][2];

        coeff = lhsZ[1][2][BB][0][i][j];
        lhsZ[1][3][BB][0][i][j]= lhsZ[1][3][BB][0][i][j] - coeff*lhsZ[2][3][BB][0][i][j];
        lhsZ[1][4][BB][0][i][j]= lhsZ[1][4][BB][0][i][j] - coeff*lhsZ[2][4][BB][0][i][j];
        lhsZ[1][0][CC][0][i][j] = lhsZ[1][0][CC][0][i][j] - coeff*lhsZ[2][0][CC][0][i][j];
        lhsZ[1][1][CC][0][i][j] = lhsZ[1][1][CC][0][i][j] - coeff*lhsZ[2][1][CC][0][i][j];
        lhsZ[1][2][CC][0][i][j] = lhsZ[1][2][CC][0][i][j] - coeff*lhsZ[2][2][CC][0][i][j];
        lhsZ[1][3][CC][0][i][j] = lhsZ[1][3][CC][0][i][j] - coeff*lhsZ[2][3][CC][0][i][j];
        lhsZ[1][4][CC][0][i][j] = lhsZ[1][4][CC][0][i][j] - coeff*lhsZ[2][4][CC][0][i][j];
        rhs[0][j][i][1]   = rhs[0][j][i][1]   - coeff*rhs[0][j][i][2];

        coeff = lhsZ[3][2][BB][0][i][j];
        lhsZ[3][3][BB][0][i][j]= lhsZ[3][3][BB][0][i][j] - coeff*lhsZ[2][3][BB][0][i][j];
        lhsZ[3][4][BB][0][i][j]= lhsZ[3][4][BB][0][i][j] - coeff*lhsZ[2][4][BB][0][i][j];
        lhsZ[3][0][CC][0][i][j] = lhsZ[3][0][CC][0][i][j] - coeff*lhsZ[2][0][CC][0][i][j];
        lhsZ[3][1][CC][0][i][j] = lhsZ[3][1][CC][0][i][j] - coeff*lhsZ[2][1][CC][0][i][j];
        lhsZ[3][2][CC][0][i][j] = lhsZ[3][2][CC][0][i][j] - coeff*lhsZ[2][2][CC][0][i][j];
        lhsZ[3][3][CC][0][i][j] = lhsZ[3][3][CC][0][i][j] - coeff*lhsZ[2][3][CC][0][i][j];
        lhsZ[3][4][CC][0][i][j] = lhsZ[3][4][CC][0][i][j] - coeff*lhsZ[2][4][CC][0][i][j];
        rhs[0][j][i][3]   = rhs[0][j][i][3]   - coeff*rhs[0][j][i][2];

        coeff = lhsZ[4][2][BB][0][i][j];
        lhsZ[4][3][BB][0][i][j]= lhsZ[4][3][BB][0][i][j] - coeff*lhsZ[2][3][BB][0][i][j];
        lhsZ[4][4][BB][0][i][j]= lhsZ[4][4][BB][0][i][j] - coeff*lhsZ[2][4][BB][0][i][j];
        lhsZ[4][0][CC][0][i][j] = lhsZ[4][0][CC][0][i][j] - coeff*lhsZ[2][0][CC][0][i][j];
        lhsZ[4][1][CC][0][i][j] = lhsZ[4][1][CC][0][i][j] - coeff*lhsZ[2][1][CC][0][i][j];
        lhsZ[4][2][CC][0][i][j] = lhsZ[4][2][CC][0][i][j] - coeff*lhsZ[2][2][CC][0][i][j];
        lhsZ[4][3][CC][0][i][j] = lhsZ[4][3][CC][0][i][j] - coeff*lhsZ[2][3][CC][0][i][j];
        lhsZ[4][4][CC][0][i][j] = lhsZ[4][4][CC][0][i][j] - coeff*lhsZ[2][4][CC][0][i][j];
        rhs[0][j][i][4]   = rhs[0][j][i][4]   - coeff*rhs[0][j][i][2];


        pivot = 1.00/lhsZ[3][3][BB][0][i][j];
        lhsZ[3][4][BB][0][i][j] = lhsZ[3][4][BB][0][i][j]*pivot;
        lhsZ[3][0][CC][0][i][j] = lhsZ[3][0][CC][0][i][j]*pivot;
        lhsZ[3][1][CC][0][i][j] = lhsZ[3][1][CC][0][i][j]*pivot;
        lhsZ[3][2][CC][0][i][j] = lhsZ[3][2][CC][0][i][j]*pivot;
        lhsZ[3][3][CC][0][i][j] = lhsZ[3][3][CC][0][i][j]*pivot;
        lhsZ[3][4][CC][0][i][j] = lhsZ[3][4][CC][0][i][j]*pivot;
        rhs[0][j][i][3]   = rhs[0][j][i][3]  *pivot;

        coeff = lhsZ[0][3][BB][0][i][j];
        lhsZ[0][4][BB][0][i][j]= lhsZ[0][4][BB][0][i][j] - coeff*lhsZ[3][4][BB][0][i][j];
        lhsZ[0][0][CC][0][i][j] = lhsZ[0][0][CC][0][i][j] - coeff*lhsZ[3][0][CC][0][i][j];
        lhsZ[0][1][CC][0][i][j] = lhsZ[0][1][CC][0][i][j] - coeff*lhsZ[3][1][CC][0][i][j];
        lhsZ[0][2][CC][0][i][j] = lhsZ[0][2][CC][0][i][j] - coeff*lhsZ[3][2][CC][0][i][j];
        lhsZ[0][3][CC][0][i][j] = lhsZ[0][3][CC][0][i][j] - coeff*lhsZ[3][3][CC][0][i][j];
        lhsZ[0][4][CC][0][i][j] = lhsZ[0][4][CC][0][i][j] - coeff*lhsZ[3][4][CC][0][i][j];
        rhs[0][j][i][0]   = rhs[0][j][i][0]   - coeff*rhs[0][j][i][3];

        coeff = lhsZ[1][3][BB][0][i][j];
        lhsZ[1][4][BB][0][i][j]= lhsZ[1][4][BB][0][i][j] - coeff*lhsZ[3][4][BB][0][i][j];
        lhsZ[1][0][CC][0][i][j] = lhsZ[1][0][CC][0][i][j] - coeff*lhsZ[3][0][CC][0][i][j];
        lhsZ[1][1][CC][0][i][j] = lhsZ[1][1][CC][0][i][j] - coeff*lhsZ[3][1][CC][0][i][j];
        lhsZ[1][2][CC][0][i][j] = lhsZ[1][2][CC][0][i][j] - coeff*lhsZ[3][2][CC][0][i][j];
        lhsZ[1][3][CC][0][i][j] = lhsZ[1][3][CC][0][i][j] - coeff*lhsZ[3][3][CC][0][i][j];
        lhsZ[1][4][CC][0][i][j] = lhsZ[1][4][CC][0][i][j] - coeff*lhsZ[3][4][CC][0][i][j];
        rhs[0][j][i][1]   = rhs[0][j][i][1]   - coeff*rhs[0][j][i][3];

        coeff = lhsZ[2][3][BB][0][i][j];
        lhsZ[2][4][BB][0][i][j]= lhsZ[2][4][BB][0][i][j] - coeff*lhsZ[3][4][BB][0][i][j];
        lhsZ[2][0][CC][0][i][j] = lhsZ[2][0][CC][0][i][j] - coeff*lhsZ[3][0][CC][0][i][j];
        lhsZ[2][1][CC][0][i][j] = lhsZ[2][1][CC][0][i][j] - coeff*lhsZ[3][1][CC][0][i][j];
        lhsZ[2][2][CC][0][i][j] = lhsZ[2][2][CC][0][i][j] - coeff*lhsZ[3][2][CC][0][i][j];
        lhsZ[2][3][CC][0][i][j] = lhsZ[2][3][CC][0][i][j] - coeff*lhsZ[3][3][CC][0][i][j];
        lhsZ[2][4][CC][0][i][j] = lhsZ[2][4][CC][0][i][j] - coeff*lhsZ[3][4][CC][0][i][j];
        rhs[0][j][i][2]   = rhs[0][j][i][2]   - coeff*rhs[0][j][i][3];

        coeff = lhsZ[4][3][BB][0][i][j];
        lhsZ[4][4][BB][0][i][j]= lhsZ[4][4][BB][0][i][j] - coeff*lhsZ[3][4][BB][0][i][j];
        lhsZ[4][0][CC][0][i][j] = lhsZ[4][0][CC][0][i][j] - coeff*lhsZ[3][0][CC][0][i][j];
        lhsZ[4][1][CC][0][i][j] = lhsZ[4][1][CC][0][i][j] - coeff*lhsZ[3][1][CC][0][i][j];
        lhsZ[4][2][CC][0][i][j] = lhsZ[4][2][CC][0][i][j] - coeff*lhsZ[3][2][CC][0][i][j];
        lhsZ[4][3][CC][0][i][j] = lhsZ[4][3][CC][0][i][j] - coeff*lhsZ[3][3][CC][0][i][j];
        lhsZ[4][4][CC][0][i][j] = lhsZ[4][4][CC][0][i][j] - coeff*lhsZ[3][4][CC][0][i][j];
        rhs[0][j][i][4]   = rhs[0][j][i][4]   - coeff*rhs[0][j][i][3];


        pivot = 1.00/lhsZ[4][4][BB][0][i][j];
        lhsZ[4][0][CC][0][i][j] = lhsZ[4][0][CC][0][i][j]*pivot;
        lhsZ[4][1][CC][0][i][j] = lhsZ[4][1][CC][0][i][j]*pivot;
        lhsZ[4][2][CC][0][i][j] = lhsZ[4][2][CC][0][i][j]*pivot;
        lhsZ[4][3][CC][0][i][j] = lhsZ[4][3][CC][0][i][j]*pivot;
        lhsZ[4][4][CC][0][i][j] = lhsZ[4][4][CC][0][i][j]*pivot;
        rhs[0][j][i][4]   = rhs[0][j][i][4]  *pivot;

        coeff = lhsZ[0][4][BB][0][i][j];
        lhsZ[0][0][CC][0][i][j] = lhsZ[0][0][CC][0][i][j] - coeff*lhsZ[4][0][CC][0][i][j];
        lhsZ[0][1][CC][0][i][j] = lhsZ[0][1][CC][0][i][j] - coeff*lhsZ[4][1][CC][0][i][j];
        lhsZ[0][2][CC][0][i][j] = lhsZ[0][2][CC][0][i][j] - coeff*lhsZ[4][2][CC][0][i][j];
        lhsZ[0][3][CC][0][i][j] = lhsZ[0][3][CC][0][i][j] - coeff*lhsZ[4][3][CC][0][i][j];
        lhsZ[0][4][CC][0][i][j] = lhsZ[0][4][CC][0][i][j] - coeff*lhsZ[4][4][CC][0][i][j];
        rhs[0][j][i][0]   = rhs[0][j][i][0]   - coeff*rhs[0][j][i][4];

        coeff = lhsZ[1][4][BB][0][i][j];
        lhsZ[1][0][CC][0][i][j] = lhsZ[1][0][CC][0][i][j] - coeff*lhsZ[4][0][CC][0][i][j];
        lhsZ[1][1][CC][0][i][j] = lhsZ[1][1][CC][0][i][j] - coeff*lhsZ[4][1][CC][0][i][j];
        lhsZ[1][2][CC][0][i][j] = lhsZ[1][2][CC][0][i][j] - coeff*lhsZ[4][2][CC][0][i][j];
        lhsZ[1][3][CC][0][i][j] = lhsZ[1][3][CC][0][i][j] - coeff*lhsZ[4][3][CC][0][i][j];
        lhsZ[1][4][CC][0][i][j] = lhsZ[1][4][CC][0][i][j] - coeff*lhsZ[4][4][CC][0][i][j];
        rhs[0][j][i][1]   = rhs[0][j][i][1]   - coeff*rhs[0][j][i][4];

        coeff = lhsZ[2][4][BB][0][i][j];
        lhsZ[2][0][CC][0][i][j] = lhsZ[2][0][CC][0][i][j] - coeff*lhsZ[4][0][CC][0][i][j];
        lhsZ[2][1][CC][0][i][j] = lhsZ[2][1][CC][0][i][j] - coeff*lhsZ[4][1][CC][0][i][j];
        lhsZ[2][2][CC][0][i][j] = lhsZ[2][2][CC][0][i][j] - coeff*lhsZ[4][2][CC][0][i][j];
        lhsZ[2][3][CC][0][i][j] = lhsZ[2][3][CC][0][i][j] - coeff*lhsZ[4][3][CC][0][i][j];
        lhsZ[2][4][CC][0][i][j] = lhsZ[2][4][CC][0][i][j] - coeff*lhsZ[4][4][CC][0][i][j];
        rhs[0][j][i][2]   = rhs[0][j][i][2]   - coeff*rhs[0][j][i][4];

        coeff = lhsZ[3][4][BB][0][i][j];
        lhsZ[3][0][CC][0][i][j] = lhsZ[3][0][CC][0][i][j] - coeff*lhsZ[4][0][CC][0][i][j];
        lhsZ[3][1][CC][0][i][j] = lhsZ[3][1][CC][0][i][j] - coeff*lhsZ[4][1][CC][0][i][j];
        lhsZ[3][2][CC][0][i][j] = lhsZ[3][2][CC][0][i][j] - coeff*lhsZ[4][2][CC][0][i][j];
        lhsZ[3][3][CC][0][i][j] = lhsZ[3][3][CC][0][i][j] - coeff*lhsZ[4][3][CC][0][i][j];
        lhsZ[3][4][CC][0][i][j] = lhsZ[3][4][CC][0][i][j] - coeff*lhsZ[4][4][CC][0][i][j];
        rhs[0][j][i][3]   = rhs[0][j][i][3]   - coeff*rhs[0][j][i][4];

      }
    }
#endif

    //---------------------------------------------------------------------
    // begin inner most do loop
    // do all the elements of the cell unless last
    //---------------------------------------------------------------------
    size_t kernel_z_solve_5_off[1] = { 1 };
    size_t kernel_z_solve_5_idx[1] = { gp02 };
    brisbane_kernel kernel_z_solve_5;
    brisbane_kernel_create("z_solve_5", &kernel_z_solve_5);
    brisbane_kernel_setmem(kernel_z_solve_5, 0, mem_lhsZ, brisbane_rw);
    brisbane_kernel_setmem(kernel_z_solve_5, 1, mem_rhs, brisbane_rw);
    brisbane_kernel_setarg(kernel_z_solve_5, 2, sizeof(int), &ksize);
    brisbane_kernel_setarg(kernel_z_solve_5, 3, sizeof(int), &gp12);

    brisbane_task task5;
    brisbane_task_create(&task5);
    brisbane_task_kernel(task5, kernel_z_solve_5, 1, kernel_z_solve_5_off, kernel_z_solve_5_idx);
    brisbane_task_submit(task5, brisbane_cpu, NULL, true);
#if 0
    #pragma omp target teams distribute parallel for private(k,j) 
    for (i = 1; i <= gp02; i++) {
      for (k = 1; k <= ksize-1; k++) {
        #pragma omp simd private(pivot,coeff) 
        for (j = 1; j <= gp12; j++) {
          //-------------------------------------------------------------------
          // subtract A*lhsZ[j][i]_vector(k-1) from lhsZ[j][i]_vector(k)
          //
          // rhs(k) = rhs(k) - A*rhs(k-1)
          //-------------------------------------------------------------------
          //matvec_sub(lhsZ[i][j][AA], rhs[k-1][k][i][j], rhs[k][j][i]);
          /*
          for(m = 0; m < 5; m++){
            rhs[k][j][i][m] = rhs[k][j][i][m] - lhsZ[m][0][AA][k][i][j]*rhs[k-1][j][i][0]
              - lhsZ[m][1][AA][k][i][j]*rhs[k-1][j][i][1]
              - lhsZ[m][2][AA][k][i][j]*rhs[k-1][j][i][2]
              - lhsZ[m][3][AA][k][i][j]*rhs[k-1][j][i][3]
              - lhsZ[m][4][AA][k][i][j]*rhs[k-1][j][i][4];
          }
          */
          rhs[k][j][i][0] = rhs[k][j][i][0] - lhsZ[0][0][AA][k][i][j]*rhs[k-1][j][i][0]
            - lhsZ[0][1][AA][k][i][j]*rhs[k-1][j][i][1]
            - lhsZ[0][2][AA][k][i][j]*rhs[k-1][j][i][2]
            - lhsZ[0][3][AA][k][i][j]*rhs[k-1][j][i][3]
            - lhsZ[0][4][AA][k][i][j]*rhs[k-1][j][i][4];
          rhs[k][j][i][1] = rhs[k][j][i][1] - lhsZ[1][0][AA][k][i][j]*rhs[k-1][j][i][0]
            - lhsZ[1][1][AA][k][i][j]*rhs[k-1][j][i][1]
            - lhsZ[1][2][AA][k][i][j]*rhs[k-1][j][i][2]
            - lhsZ[1][3][AA][k][i][j]*rhs[k-1][j][i][3]
            - lhsZ[1][4][AA][k][i][j]*rhs[k-1][j][i][4];
          rhs[k][j][i][2] = rhs[k][j][i][2] - lhsZ[2][0][AA][k][i][j]*rhs[k-1][j][i][0]
            - lhsZ[2][1][AA][k][i][j]*rhs[k-1][j][i][1]
            - lhsZ[2][2][AA][k][i][j]*rhs[k-1][j][i][2]
            - lhsZ[2][3][AA][k][i][j]*rhs[k-1][j][i][3]
            - lhsZ[2][4][AA][k][i][j]*rhs[k-1][j][i][4];
          rhs[k][j][i][3] = rhs[k][j][i][3] - lhsZ[3][0][AA][k][i][j]*rhs[k-1][j][i][0]
            - lhsZ[3][1][AA][k][i][j]*rhs[k-1][j][i][1]
            - lhsZ[3][2][AA][k][i][j]*rhs[k-1][j][i][2]
            - lhsZ[3][3][AA][k][i][j]*rhs[k-1][j][i][3]
            - lhsZ[3][4][AA][k][i][j]*rhs[k-1][j][i][4];
          rhs[k][j][i][4] = rhs[k][j][i][4] - lhsZ[4][0][AA][k][i][j]*rhs[k-1][j][i][0]
            - lhsZ[4][1][AA][k][i][j]*rhs[k-1][j][i][1]
            - lhsZ[4][2][AA][k][i][j]*rhs[k-1][j][i][2]
            - lhsZ[4][3][AA][k][i][j]*rhs[k-1][j][i][3]
            - lhsZ[4][4][AA][k][i][j]*rhs[k-1][j][i][4];



          //-------------------------------------------------------------------
          // B(k) = B(k) - C(k-1)*A(k)
          // matmul_sub(AA,i,j,k,c,CC,i,j,k-1,c,BB,i,j,k)
          //-------------------------------------------------------------------
          //matmul_sub(lhsZ[k-1][i][AA], lhsZ[j][k][i][j][CC], lhsZ[j][i][k][BB]);
          /*
          for(m = 0; m < 5; m++){
            for(n = 0; n < 5; n++){
              lhsZ[n][m][BB][k][i][j] = lhsZ[n][m][BB][k][i][j] - lhsZ[n][0][AA][k][i][j]*lhsZ[0][m][CC][k-1][i][j]
                - lhsZ[n][1][AA][k][i][j]*lhsZ[1][m][CC][k-1][i][j]
                - lhsZ[n][2][AA][k][i][j]*lhsZ[2][m][CC][k-1][i][j]
                - lhsZ[n][3][AA][k][i][j]*lhsZ[3][m][CC][k-1][i][j]
                - lhsZ[n][4][AA][k][i][j]*lhsZ[4][m][CC][k-1][i][j];
            }
          }
          */
          lhsZ[0][0][BB][k][i][j] = lhsZ[0][0][BB][k][i][j] - lhsZ[0][0][AA][k][i][j]*lhsZ[0][0][CC][k-1][i][j]
            - lhsZ[0][1][AA][k][i][j]*lhsZ[1][0][CC][k-1][i][j]
            - lhsZ[0][2][AA][k][i][j]*lhsZ[2][0][CC][k-1][i][j]
            - lhsZ[0][3][AA][k][i][j]*lhsZ[3][0][CC][k-1][i][j]
            - lhsZ[0][4][AA][k][i][j]*lhsZ[4][0][CC][k-1][i][j];
          lhsZ[1][0][BB][k][i][j] = lhsZ[1][0][BB][k][i][j] - lhsZ[1][0][AA][k][i][j]*lhsZ[0][0][CC][k-1][i][j]
            - lhsZ[1][1][AA][k][i][j]*lhsZ[1][0][CC][k-1][i][j]
            - lhsZ[1][2][AA][k][i][j]*lhsZ[2][0][CC][k-1][i][j]
            - lhsZ[1][3][AA][k][i][j]*lhsZ[3][0][CC][k-1][i][j]
            - lhsZ[1][4][AA][k][i][j]*lhsZ[4][0][CC][k-1][i][j];
          lhsZ[2][0][BB][k][i][j] = lhsZ[2][0][BB][k][i][j] - lhsZ[2][0][AA][k][i][j]*lhsZ[0][0][CC][k-1][i][j]
            - lhsZ[2][1][AA][k][i][j]*lhsZ[1][0][CC][k-1][i][j]
            - lhsZ[2][2][AA][k][i][j]*lhsZ[2][0][CC][k-1][i][j]
            - lhsZ[2][3][AA][k][i][j]*lhsZ[3][0][CC][k-1][i][j]
            - lhsZ[2][4][AA][k][i][j]*lhsZ[4][0][CC][k-1][i][j];
          lhsZ[3][0][BB][k][i][j] = lhsZ[3][0][BB][k][i][j] - lhsZ[3][0][AA][k][i][j]*lhsZ[0][0][CC][k-1][i][j]
            - lhsZ[3][1][AA][k][i][j]*lhsZ[1][0][CC][k-1][i][j]
            - lhsZ[3][2][AA][k][i][j]*lhsZ[2][0][CC][k-1][i][j]
            - lhsZ[3][3][AA][k][i][j]*lhsZ[3][0][CC][k-1][i][j]
            - lhsZ[3][4][AA][k][i][j]*lhsZ[4][0][CC][k-1][i][j];
          lhsZ[4][0][BB][k][i][j] = lhsZ[4][0][BB][k][i][j] - lhsZ[4][0][AA][k][i][j]*lhsZ[0][0][CC][k-1][i][j]
            - lhsZ[4][1][AA][k][i][j]*lhsZ[1][0][CC][k-1][i][j]
            - lhsZ[4][2][AA][k][i][j]*lhsZ[2][0][CC][k-1][i][j]
            - lhsZ[4][3][AA][k][i][j]*lhsZ[3][0][CC][k-1][i][j]
            - lhsZ[4][4][AA][k][i][j]*lhsZ[4][0][CC][k-1][i][j];
          lhsZ[0][1][BB][k][i][j] = lhsZ[0][1][BB][k][i][j] - lhsZ[0][0][AA][k][i][j]*lhsZ[0][1][CC][k-1][i][j]
            - lhsZ[0][1][AA][k][i][j]*lhsZ[1][1][CC][k-1][i][j]
            - lhsZ[0][2][AA][k][i][j]*lhsZ[2][1][CC][k-1][i][j]
            - lhsZ[0][3][AA][k][i][j]*lhsZ[3][1][CC][k-1][i][j]
            - lhsZ[0][4][AA][k][i][j]*lhsZ[4][1][CC][k-1][i][j];
          lhsZ[1][1][BB][k][i][j] = lhsZ[1][1][BB][k][i][j] - lhsZ[1][0][AA][k][i][j]*lhsZ[0][1][CC][k-1][i][j]
            - lhsZ[1][1][AA][k][i][j]*lhsZ[1][1][CC][k-1][i][j]
            - lhsZ[1][2][AA][k][i][j]*lhsZ[2][1][CC][k-1][i][j]
            - lhsZ[1][3][AA][k][i][j]*lhsZ[3][1][CC][k-1][i][j]
            - lhsZ[1][4][AA][k][i][j]*lhsZ[4][1][CC][k-1][i][j];
          lhsZ[2][1][BB][k][i][j] = lhsZ[2][1][BB][k][i][j] - lhsZ[2][0][AA][k][i][j]*lhsZ[0][1][CC][k-1][i][j]
            - lhsZ[2][1][AA][k][i][j]*lhsZ[1][1][CC][k-1][i][j]
            - lhsZ[2][2][AA][k][i][j]*lhsZ[2][1][CC][k-1][i][j]
            - lhsZ[2][3][AA][k][i][j]*lhsZ[3][1][CC][k-1][i][j]
            - lhsZ[2][4][AA][k][i][j]*lhsZ[4][1][CC][k-1][i][j];
          lhsZ[3][1][BB][k][i][j] = lhsZ[3][1][BB][k][i][j] - lhsZ[3][0][AA][k][i][j]*lhsZ[0][1][CC][k-1][i][j]
            - lhsZ[3][1][AA][k][i][j]*lhsZ[1][1][CC][k-1][i][j]
            - lhsZ[3][2][AA][k][i][j]*lhsZ[2][1][CC][k-1][i][j]
            - lhsZ[3][3][AA][k][i][j]*lhsZ[3][1][CC][k-1][i][j]
            - lhsZ[3][4][AA][k][i][j]*lhsZ[4][1][CC][k-1][i][j];
          lhsZ[4][1][BB][k][i][j] = lhsZ[4][1][BB][k][i][j] - lhsZ[4][0][AA][k][i][j]*lhsZ[0][1][CC][k-1][i][j]
            - lhsZ[4][1][AA][k][i][j]*lhsZ[1][1][CC][k-1][i][j]
            - lhsZ[4][2][AA][k][i][j]*lhsZ[2][1][CC][k-1][i][j]
            - lhsZ[4][3][AA][k][i][j]*lhsZ[3][1][CC][k-1][i][j]
            - lhsZ[4][4][AA][k][i][j]*lhsZ[4][1][CC][k-1][i][j];
          lhsZ[0][2][BB][k][i][j] = lhsZ[0][2][BB][k][i][j] - lhsZ[0][0][AA][k][i][j]*lhsZ[0][2][CC][k-1][i][j]
            - lhsZ[0][1][AA][k][i][j]*lhsZ[1][2][CC][k-1][i][j]
            - lhsZ[0][2][AA][k][i][j]*lhsZ[2][2][CC][k-1][i][j]
            - lhsZ[0][3][AA][k][i][j]*lhsZ[3][2][CC][k-1][i][j]
            - lhsZ[0][4][AA][k][i][j]*lhsZ[4][2][CC][k-1][i][j];
          lhsZ[1][2][BB][k][i][j] = lhsZ[1][2][BB][k][i][j] - lhsZ[1][0][AA][k][i][j]*lhsZ[0][2][CC][k-1][i][j]
            - lhsZ[1][1][AA][k][i][j]*lhsZ[1][2][CC][k-1][i][j]
            - lhsZ[1][2][AA][k][i][j]*lhsZ[2][2][CC][k-1][i][j]
            - lhsZ[1][3][AA][k][i][j]*lhsZ[3][2][CC][k-1][i][j]
            - lhsZ[1][4][AA][k][i][j]*lhsZ[4][2][CC][k-1][i][j];
          lhsZ[2][2][BB][k][i][j] = lhsZ[2][2][BB][k][i][j] - lhsZ[2][0][AA][k][i][j]*lhsZ[0][2][CC][k-1][i][j]
            - lhsZ[2][1][AA][k][i][j]*lhsZ[1][2][CC][k-1][i][j]
            - lhsZ[2][2][AA][k][i][j]*lhsZ[2][2][CC][k-1][i][j]
            - lhsZ[2][3][AA][k][i][j]*lhsZ[3][2][CC][k-1][i][j]
            - lhsZ[2][4][AA][k][i][j]*lhsZ[4][2][CC][k-1][i][j];
          lhsZ[3][2][BB][k][i][j] = lhsZ[3][2][BB][k][i][j] - lhsZ[3][0][AA][k][i][j]*lhsZ[0][2][CC][k-1][i][j]
            - lhsZ[3][1][AA][k][i][j]*lhsZ[1][2][CC][k-1][i][j]
            - lhsZ[3][2][AA][k][i][j]*lhsZ[2][2][CC][k-1][i][j]
            - lhsZ[3][3][AA][k][i][j]*lhsZ[3][2][CC][k-1][i][j]
            - lhsZ[3][4][AA][k][i][j]*lhsZ[4][2][CC][k-1][i][j];
          lhsZ[4][2][BB][k][i][j] = lhsZ[4][2][BB][k][i][j] - lhsZ[4][0][AA][k][i][j]*lhsZ[0][2][CC][k-1][i][j]
            - lhsZ[4][1][AA][k][i][j]*lhsZ[1][2][CC][k-1][i][j]
            - lhsZ[4][2][AA][k][i][j]*lhsZ[2][2][CC][k-1][i][j]
            - lhsZ[4][3][AA][k][i][j]*lhsZ[3][2][CC][k-1][i][j]
            - lhsZ[4][4][AA][k][i][j]*lhsZ[4][2][CC][k-1][i][j];
          lhsZ[0][3][BB][k][i][j] = lhsZ[0][3][BB][k][i][j] - lhsZ[0][0][AA][k][i][j]*lhsZ[0][3][CC][k-1][i][j]
            - lhsZ[0][1][AA][k][i][j]*lhsZ[1][3][CC][k-1][i][j]
            - lhsZ[0][2][AA][k][i][j]*lhsZ[2][3][CC][k-1][i][j]
            - lhsZ[0][3][AA][k][i][j]*lhsZ[3][3][CC][k-1][i][j]
            - lhsZ[0][4][AA][k][i][j]*lhsZ[4][3][CC][k-1][i][j];
          lhsZ[1][3][BB][k][i][j] = lhsZ[1][3][BB][k][i][j] - lhsZ[1][0][AA][k][i][j]*lhsZ[0][3][CC][k-1][i][j]
            - lhsZ[1][1][AA][k][i][j]*lhsZ[1][3][CC][k-1][i][j]
            - lhsZ[1][2][AA][k][i][j]*lhsZ[2][3][CC][k-1][i][j]
            - lhsZ[1][3][AA][k][i][j]*lhsZ[3][3][CC][k-1][i][j]
            - lhsZ[1][4][AA][k][i][j]*lhsZ[4][3][CC][k-1][i][j];
          lhsZ[2][3][BB][k][i][j] = lhsZ[2][3][BB][k][i][j] - lhsZ[2][0][AA][k][i][j]*lhsZ[0][3][CC][k-1][i][j]
            - lhsZ[2][1][AA][k][i][j]*lhsZ[1][3][CC][k-1][i][j]
            - lhsZ[2][2][AA][k][i][j]*lhsZ[2][3][CC][k-1][i][j]
            - lhsZ[2][3][AA][k][i][j]*lhsZ[3][3][CC][k-1][i][j]
            - lhsZ[2][4][AA][k][i][j]*lhsZ[4][3][CC][k-1][i][j];
          lhsZ[3][3][BB][k][i][j] = lhsZ[3][3][BB][k][i][j] - lhsZ[3][0][AA][k][i][j]*lhsZ[0][3][CC][k-1][i][j]
            - lhsZ[3][1][AA][k][i][j]*lhsZ[1][3][CC][k-1][i][j]
            - lhsZ[3][2][AA][k][i][j]*lhsZ[2][3][CC][k-1][i][j]
            - lhsZ[3][3][AA][k][i][j]*lhsZ[3][3][CC][k-1][i][j]
            - lhsZ[3][4][AA][k][i][j]*lhsZ[4][3][CC][k-1][i][j];
          lhsZ[4][3][BB][k][i][j] = lhsZ[4][3][BB][k][i][j] - lhsZ[4][0][AA][k][i][j]*lhsZ[0][3][CC][k-1][i][j]
            - lhsZ[4][1][AA][k][i][j]*lhsZ[1][3][CC][k-1][i][j]
            - lhsZ[4][2][AA][k][i][j]*lhsZ[2][3][CC][k-1][i][j]
            - lhsZ[4][3][AA][k][i][j]*lhsZ[3][3][CC][k-1][i][j]
            - lhsZ[4][4][AA][k][i][j]*lhsZ[4][3][CC][k-1][i][j];
          lhsZ[0][4][BB][k][i][j] = lhsZ[0][4][BB][k][i][j] - lhsZ[0][0][AA][k][i][j]*lhsZ[0][4][CC][k-1][i][j]
            - lhsZ[0][1][AA][k][i][j]*lhsZ[1][4][CC][k-1][i][j]
            - lhsZ[0][2][AA][k][i][j]*lhsZ[2][4][CC][k-1][i][j]
            - lhsZ[0][3][AA][k][i][j]*lhsZ[3][4][CC][k-1][i][j]
            - lhsZ[0][4][AA][k][i][j]*lhsZ[4][4][CC][k-1][i][j];
          lhsZ[1][4][BB][k][i][j] = lhsZ[1][4][BB][k][i][j] - lhsZ[1][0][AA][k][i][j]*lhsZ[0][4][CC][k-1][i][j]
            - lhsZ[1][1][AA][k][i][j]*lhsZ[1][4][CC][k-1][i][j]
            - lhsZ[1][2][AA][k][i][j]*lhsZ[2][4][CC][k-1][i][j]
            - lhsZ[1][3][AA][k][i][j]*lhsZ[3][4][CC][k-1][i][j]
            - lhsZ[1][4][AA][k][i][j]*lhsZ[4][4][CC][k-1][i][j];
          lhsZ[2][4][BB][k][i][j] = lhsZ[2][4][BB][k][i][j] - lhsZ[2][0][AA][k][i][j]*lhsZ[0][4][CC][k-1][i][j]
            - lhsZ[2][1][AA][k][i][j]*lhsZ[1][4][CC][k-1][i][j]
            - lhsZ[2][2][AA][k][i][j]*lhsZ[2][4][CC][k-1][i][j]
            - lhsZ[2][3][AA][k][i][j]*lhsZ[3][4][CC][k-1][i][j]
            - lhsZ[2][4][AA][k][i][j]*lhsZ[4][4][CC][k-1][i][j];
          lhsZ[3][4][BB][k][i][j] = lhsZ[3][4][BB][k][i][j] - lhsZ[3][0][AA][k][i][j]*lhsZ[0][4][CC][k-1][i][j]
            - lhsZ[3][1][AA][k][i][j]*lhsZ[1][4][CC][k-1][i][j]
            - lhsZ[3][2][AA][k][i][j]*lhsZ[2][4][CC][k-1][i][j]
            - lhsZ[3][3][AA][k][i][j]*lhsZ[3][4][CC][k-1][i][j]
            - lhsZ[3][4][AA][k][i][j]*lhsZ[4][4][CC][k-1][i][j];
          lhsZ[4][4][BB][k][i][j] = lhsZ[4][4][BB][k][i][j] - lhsZ[4][0][AA][k][i][j]*lhsZ[0][4][CC][k-1][i][j]
            - lhsZ[4][1][AA][k][i][j]*lhsZ[1][4][CC][k-1][i][j]
            - lhsZ[4][2][AA][k][i][j]*lhsZ[2][4][CC][k-1][i][j]
            - lhsZ[4][3][AA][k][i][j]*lhsZ[3][4][CC][k-1][i][j]
            - lhsZ[4][4][AA][k][i][j]*lhsZ[4][4][CC][k-1][i][j];

          //-------------------------------------------------------------------
          // multiply c[k][j][i] by b_inverse and copy back to c
          // multiply rhs[0][j][i] by b_inverse[0][j][i] and copy to rhs
          //-------------------------------------------------------------------
          //binvcrhs( lhsZ[k][i][BB], lhsZ[j][k][i][j][CC], rhs[k][j][i] );
          /*
          for(m = 0; m < 5; m++){
            pivot = 1.00/lhsZ[m][m][BB][k][i][j];
            for(n = m+1; n < 5; n++){
              lhsZ[m][n][BB][k][i][j] = lhsZ[m][n][BB][k][i][j]*pivot;
            }
            lhsZ[m][0][CC][k][i][j] = lhsZ[m][0][CC][k][i][j]*pivot;
            lhsZ[m][1][CC][k][i][j] = lhsZ[m][1][CC][k][i][j]*pivot;
            lhsZ[m][2][CC][k][i][j] = lhsZ[m][2][CC][k][i][j]*pivot;
            lhsZ[m][3][CC][k][i][j] = lhsZ[m][3][CC][k][i][j]*pivot;
            lhsZ[m][4][CC][k][i][j] = lhsZ[m][4][CC][k][i][j]*pivot;
            rhs[k][j][i][m] = rhs[k][j][i][m]*pivot;

            for(n = 0; n < 5; n++){
              if(n != m){
                coeff = lhsZ[n][m][BB][k][i][j];
                for(z = m+1; z < 5; z++){
                  lhsZ[n][z][BB][k][i][j] = lhsZ[n][z][BB][k][i][j] - coeff*lhsZ[m][z][BB][k][i][j];
                }
                lhsZ[n][0][CC][k][i][j] = lhsZ[n][0][CC][k][i][j] - coeff*lhsZ[m][0][CC][k][i][j];
                lhsZ[n][1][CC][k][i][j] = lhsZ[n][1][CC][k][i][j] - coeff*lhsZ[m][1][CC][k][i][j];
                lhsZ[n][2][CC][k][i][j] = lhsZ[n][2][CC][k][i][j] - coeff*lhsZ[m][2][CC][k][i][j];
                lhsZ[n][3][CC][k][i][j] = lhsZ[n][3][CC][k][i][j] - coeff*lhsZ[m][3][CC][k][i][j];
                lhsZ[n][4][CC][k][i][j] = lhsZ[n][4][CC][k][i][j] - coeff*lhsZ[m][4][CC][k][i][j];
                rhs[k][j][i][n] = rhs[k][j][i][n] - coeff*rhs[k][j][i][m];
              }
            }
          }
          */
          pivot = 1.00/lhsZ[0][0][BB][k][i][j];
          lhsZ[0][1][BB][k][i][j] = lhsZ[0][1][BB][k][i][j]*pivot;
          lhsZ[0][2][BB][k][i][j] = lhsZ[0][2][BB][k][i][j]*pivot;
          lhsZ[0][3][BB][k][i][j] = lhsZ[0][3][BB][k][i][j]*pivot;
          lhsZ[0][4][BB][k][i][j] = lhsZ[0][4][BB][k][i][j]*pivot;
          lhsZ[0][0][CC][k][i][j] = lhsZ[0][0][CC][k][i][j]*pivot;
          lhsZ[0][1][CC][k][i][j] = lhsZ[0][1][CC][k][i][j]*pivot;
          lhsZ[0][2][CC][k][i][j] = lhsZ[0][2][CC][k][i][j]*pivot;
          lhsZ[0][3][CC][k][i][j] = lhsZ[0][3][CC][k][i][j]*pivot;
          lhsZ[0][4][CC][k][i][j] = lhsZ[0][4][CC][k][i][j]*pivot;
          rhs[k][j][i][0]   = rhs[k][j][i][0]  *pivot;

          coeff = lhsZ[1][0][BB][k][i][j];
          lhsZ[1][1][BB][k][i][j]= lhsZ[1][1][BB][k][i][j] - coeff*lhsZ[0][1][BB][k][i][j];
          lhsZ[1][2][BB][k][i][j]= lhsZ[1][2][BB][k][i][j] - coeff*lhsZ[0][2][BB][k][i][j];
          lhsZ[1][3][BB][k][i][j]= lhsZ[1][3][BB][k][i][j] - coeff*lhsZ[0][3][BB][k][i][j];
          lhsZ[1][4][BB][k][i][j]= lhsZ[1][4][BB][k][i][j] - coeff*lhsZ[0][4][BB][k][i][j];
          lhsZ[1][0][CC][k][i][j] = lhsZ[1][0][CC][k][i][j] - coeff*lhsZ[0][0][CC][k][i][j];
          lhsZ[1][1][CC][k][i][j] = lhsZ[1][1][CC][k][i][j] - coeff*lhsZ[0][1][CC][k][i][j];
          lhsZ[1][2][CC][k][i][j] = lhsZ[1][2][CC][k][i][j] - coeff*lhsZ[0][2][CC][k][i][j];
          lhsZ[1][3][CC][k][i][j] = lhsZ[1][3][CC][k][i][j] - coeff*lhsZ[0][3][CC][k][i][j];
          lhsZ[1][4][CC][k][i][j] = lhsZ[1][4][CC][k][i][j] - coeff*lhsZ[0][4][CC][k][i][j];
          rhs[k][j][i][1]   = rhs[k][j][i][1]   - coeff*rhs[k][j][i][0];

          coeff = lhsZ[2][0][BB][k][i][j];
          lhsZ[2][1][BB][k][i][j]= lhsZ[2][1][BB][k][i][j] - coeff*lhsZ[0][1][BB][k][i][j];
          lhsZ[2][2][BB][k][i][j]= lhsZ[2][2][BB][k][i][j] - coeff*lhsZ[0][2][BB][k][i][j];
          lhsZ[2][3][BB][k][i][j]= lhsZ[2][3][BB][k][i][j] - coeff*lhsZ[0][3][BB][k][i][j];
          lhsZ[2][4][BB][k][i][j]= lhsZ[2][4][BB][k][i][j] - coeff*lhsZ[0][4][BB][k][i][j];
          lhsZ[2][0][CC][k][i][j] = lhsZ[2][0][CC][k][i][j] - coeff*lhsZ[0][0][CC][k][i][j];
          lhsZ[2][1][CC][k][i][j] = lhsZ[2][1][CC][k][i][j] - coeff*lhsZ[0][1][CC][k][i][j];
          lhsZ[2][2][CC][k][i][j] = lhsZ[2][2][CC][k][i][j] - coeff*lhsZ[0][2][CC][k][i][j];
          lhsZ[2][3][CC][k][i][j] = lhsZ[2][3][CC][k][i][j] - coeff*lhsZ[0][3][CC][k][i][j];
          lhsZ[2][4][CC][k][i][j] = lhsZ[2][4][CC][k][i][j] - coeff*lhsZ[0][4][CC][k][i][j];
          rhs[k][j][i][2]   = rhs[k][j][i][2]   - coeff*rhs[k][j][i][0];

          coeff = lhsZ[3][0][BB][k][i][j];
          lhsZ[3][1][BB][k][i][j]= lhsZ[3][1][BB][k][i][j] - coeff*lhsZ[0][1][BB][k][i][j];
          lhsZ[3][2][BB][k][i][j]= lhsZ[3][2][BB][k][i][j] - coeff*lhsZ[0][2][BB][k][i][j];
          lhsZ[3][3][BB][k][i][j]= lhsZ[3][3][BB][k][i][j] - coeff*lhsZ[0][3][BB][k][i][j];
          lhsZ[3][4][BB][k][i][j]= lhsZ[3][4][BB][k][i][j] - coeff*lhsZ[0][4][BB][k][i][j];
          lhsZ[3][0][CC][k][i][j] = lhsZ[3][0][CC][k][i][j] - coeff*lhsZ[0][0][CC][k][i][j];
          lhsZ[3][1][CC][k][i][j] = lhsZ[3][1][CC][k][i][j] - coeff*lhsZ[0][1][CC][k][i][j];
          lhsZ[3][2][CC][k][i][j] = lhsZ[3][2][CC][k][i][j] - coeff*lhsZ[0][2][CC][k][i][j];
          lhsZ[3][3][CC][k][i][j] = lhsZ[3][3][CC][k][i][j] - coeff*lhsZ[0][3][CC][k][i][j];
          lhsZ[3][4][CC][k][i][j] = lhsZ[3][4][CC][k][i][j] - coeff*lhsZ[0][4][CC][k][i][j];
          rhs[k][j][i][3]   = rhs[k][j][i][3]   - coeff*rhs[k][j][i][0];

          coeff = lhsZ[4][0][BB][k][i][j];
          lhsZ[4][1][BB][k][i][j]= lhsZ[4][1][BB][k][i][j] - coeff*lhsZ[0][1][BB][k][i][j];
          lhsZ[4][2][BB][k][i][j]= lhsZ[4][2][BB][k][i][j] - coeff*lhsZ[0][2][BB][k][i][j];
          lhsZ[4][3][BB][k][i][j]= lhsZ[4][3][BB][k][i][j] - coeff*lhsZ[0][3][BB][k][i][j];
          lhsZ[4][4][BB][k][i][j]= lhsZ[4][4][BB][k][i][j] - coeff*lhsZ[0][4][BB][k][i][j];
          lhsZ[4][0][CC][k][i][j] = lhsZ[4][0][CC][k][i][j] - coeff*lhsZ[0][0][CC][k][i][j];
          lhsZ[4][1][CC][k][i][j] = lhsZ[4][1][CC][k][i][j] - coeff*lhsZ[0][1][CC][k][i][j];
          lhsZ[4][2][CC][k][i][j] = lhsZ[4][2][CC][k][i][j] - coeff*lhsZ[0][2][CC][k][i][j];
          lhsZ[4][3][CC][k][i][j] = lhsZ[4][3][CC][k][i][j] - coeff*lhsZ[0][3][CC][k][i][j];
          lhsZ[4][4][CC][k][i][j] = lhsZ[4][4][CC][k][i][j] - coeff*lhsZ[0][4][CC][k][i][j];
          rhs[k][j][i][4]   = rhs[k][j][i][4]   - coeff*rhs[k][j][i][0];


          pivot = 1.00/lhsZ[1][1][BB][k][i][j];
          lhsZ[1][2][BB][k][i][j] = lhsZ[1][2][BB][k][i][j]*pivot;
          lhsZ[1][3][BB][k][i][j] = lhsZ[1][3][BB][k][i][j]*pivot;
          lhsZ[1][4][BB][k][i][j] = lhsZ[1][4][BB][k][i][j]*pivot;
          lhsZ[1][0][CC][k][i][j] = lhsZ[1][0][CC][k][i][j]*pivot;
          lhsZ[1][1][CC][k][i][j] = lhsZ[1][1][CC][k][i][j]*pivot;
          lhsZ[1][2][CC][k][i][j] = lhsZ[1][2][CC][k][i][j]*pivot;
          lhsZ[1][3][CC][k][i][j] = lhsZ[1][3][CC][k][i][j]*pivot;
          lhsZ[1][4][CC][k][i][j] = lhsZ[1][4][CC][k][i][j]*pivot;
          rhs[k][j][i][1]   = rhs[k][j][i][1]  *pivot;

          coeff = lhsZ[0][1][BB][k][i][j];
          lhsZ[0][2][BB][k][i][j]= lhsZ[0][2][BB][k][i][j] - coeff*lhsZ[1][2][BB][k][i][j];
          lhsZ[0][3][BB][k][i][j]= lhsZ[0][3][BB][k][i][j] - coeff*lhsZ[1][3][BB][k][i][j];
          lhsZ[0][4][BB][k][i][j]= lhsZ[0][4][BB][k][i][j] - coeff*lhsZ[1][4][BB][k][i][j];
          lhsZ[0][0][CC][k][i][j] = lhsZ[0][0][CC][k][i][j] - coeff*lhsZ[1][0][CC][k][i][j];
          lhsZ[0][1][CC][k][i][j] = lhsZ[0][1][CC][k][i][j] - coeff*lhsZ[1][1][CC][k][i][j];
          lhsZ[0][2][CC][k][i][j] = lhsZ[0][2][CC][k][i][j] - coeff*lhsZ[1][2][CC][k][i][j];
          lhsZ[0][3][CC][k][i][j] = lhsZ[0][3][CC][k][i][j] - coeff*lhsZ[1][3][CC][k][i][j];
          lhsZ[0][4][CC][k][i][j] = lhsZ[0][4][CC][k][i][j] - coeff*lhsZ[1][4][CC][k][i][j];
          rhs[k][j][i][0]   = rhs[k][j][i][0]   - coeff*rhs[k][j][i][1];

          coeff = lhsZ[2][1][BB][k][i][j];
          lhsZ[2][2][BB][k][i][j]= lhsZ[2][2][BB][k][i][j] - coeff*lhsZ[1][2][BB][k][i][j];
          lhsZ[2][3][BB][k][i][j]= lhsZ[2][3][BB][k][i][j] - coeff*lhsZ[1][3][BB][k][i][j];
          lhsZ[2][4][BB][k][i][j]= lhsZ[2][4][BB][k][i][j] - coeff*lhsZ[1][4][BB][k][i][j];
          lhsZ[2][0][CC][k][i][j] = lhsZ[2][0][CC][k][i][j] - coeff*lhsZ[1][0][CC][k][i][j];
          lhsZ[2][1][CC][k][i][j] = lhsZ[2][1][CC][k][i][j] - coeff*lhsZ[1][1][CC][k][i][j];
          lhsZ[2][2][CC][k][i][j] = lhsZ[2][2][CC][k][i][j] - coeff*lhsZ[1][2][CC][k][i][j];
          lhsZ[2][3][CC][k][i][j] = lhsZ[2][3][CC][k][i][j] - coeff*lhsZ[1][3][CC][k][i][j];
          lhsZ[2][4][CC][k][i][j] = lhsZ[2][4][CC][k][i][j] - coeff*lhsZ[1][4][CC][k][i][j];
          rhs[k][j][i][2]   = rhs[k][j][i][2]   - coeff*rhs[k][j][i][1];

          coeff = lhsZ[3][1][BB][k][i][j];
          lhsZ[3][2][BB][k][i][j]= lhsZ[3][2][BB][k][i][j] - coeff*lhsZ[1][2][BB][k][i][j];
          lhsZ[3][3][BB][k][i][j]= lhsZ[3][3][BB][k][i][j] - coeff*lhsZ[1][3][BB][k][i][j];
          lhsZ[3][4][BB][k][i][j]= lhsZ[3][4][BB][k][i][j] - coeff*lhsZ[1][4][BB][k][i][j];
          lhsZ[3][0][CC][k][i][j] = lhsZ[3][0][CC][k][i][j] - coeff*lhsZ[1][0][CC][k][i][j];
          lhsZ[3][1][CC][k][i][j] = lhsZ[3][1][CC][k][i][j] - coeff*lhsZ[1][1][CC][k][i][j];
          lhsZ[3][2][CC][k][i][j] = lhsZ[3][2][CC][k][i][j] - coeff*lhsZ[1][2][CC][k][i][j];
          lhsZ[3][3][CC][k][i][j] = lhsZ[3][3][CC][k][i][j] - coeff*lhsZ[1][3][CC][k][i][j];
          lhsZ[3][4][CC][k][i][j] = lhsZ[3][4][CC][k][i][j] - coeff*lhsZ[1][4][CC][k][i][j];
          rhs[k][j][i][3]   = rhs[k][j][i][3]   - coeff*rhs[k][j][i][1];

          coeff = lhsZ[4][1][BB][k][i][j];
          lhsZ[4][2][BB][k][i][j]= lhsZ[4][2][BB][k][i][j] - coeff*lhsZ[1][2][BB][k][i][j];
          lhsZ[4][3][BB][k][i][j]= lhsZ[4][3][BB][k][i][j] - coeff*lhsZ[1][3][BB][k][i][j];
          lhsZ[4][4][BB][k][i][j]= lhsZ[4][4][BB][k][i][j] - coeff*lhsZ[1][4][BB][k][i][j];
          lhsZ[4][0][CC][k][i][j] = lhsZ[4][0][CC][k][i][j] - coeff*lhsZ[1][0][CC][k][i][j];
          lhsZ[4][1][CC][k][i][j] = lhsZ[4][1][CC][k][i][j] - coeff*lhsZ[1][1][CC][k][i][j];
          lhsZ[4][2][CC][k][i][j] = lhsZ[4][2][CC][k][i][j] - coeff*lhsZ[1][2][CC][k][i][j];
          lhsZ[4][3][CC][k][i][j] = lhsZ[4][3][CC][k][i][j] - coeff*lhsZ[1][3][CC][k][i][j];
          lhsZ[4][4][CC][k][i][j] = lhsZ[4][4][CC][k][i][j] - coeff*lhsZ[1][4][CC][k][i][j];
          rhs[k][j][i][4]   = rhs[k][j][i][4]   - coeff*rhs[k][j][i][1];


          pivot = 1.00/lhsZ[2][2][BB][k][i][j];
          lhsZ[2][3][BB][k][i][j] = lhsZ[2][3][BB][k][i][j]*pivot;
          lhsZ[2][4][BB][k][i][j] = lhsZ[2][4][BB][k][i][j]*pivot;
          lhsZ[2][0][CC][k][i][j] = lhsZ[2][0][CC][k][i][j]*pivot;
          lhsZ[2][1][CC][k][i][j] = lhsZ[2][1][CC][k][i][j]*pivot;
          lhsZ[2][2][CC][k][i][j] = lhsZ[2][2][CC][k][i][j]*pivot;
          lhsZ[2][3][CC][k][i][j] = lhsZ[2][3][CC][k][i][j]*pivot;
          lhsZ[2][4][CC][k][i][j] = lhsZ[2][4][CC][k][i][j]*pivot;
          rhs[k][j][i][2]   = rhs[k][j][i][2]  *pivot;

          coeff = lhsZ[0][2][BB][k][i][j];
          lhsZ[0][3][BB][k][i][j]= lhsZ[0][3][BB][k][i][j] - coeff*lhsZ[2][3][BB][k][i][j];
          lhsZ[0][4][BB][k][i][j]= lhsZ[0][4][BB][k][i][j] - coeff*lhsZ[2][4][BB][k][i][j];
          lhsZ[0][0][CC][k][i][j] = lhsZ[0][0][CC][k][i][j] - coeff*lhsZ[2][0][CC][k][i][j];
          lhsZ[0][1][CC][k][i][j] = lhsZ[0][1][CC][k][i][j] - coeff*lhsZ[2][1][CC][k][i][j];
          lhsZ[0][2][CC][k][i][j] = lhsZ[0][2][CC][k][i][j] - coeff*lhsZ[2][2][CC][k][i][j];
          lhsZ[0][3][CC][k][i][j] = lhsZ[0][3][CC][k][i][j] - coeff*lhsZ[2][3][CC][k][i][j];
          lhsZ[0][4][CC][k][i][j] = lhsZ[0][4][CC][k][i][j] - coeff*lhsZ[2][4][CC][k][i][j];
          rhs[k][j][i][0]   = rhs[k][j][i][0]   - coeff*rhs[k][j][i][2];

          coeff = lhsZ[1][2][BB][k][i][j];
          lhsZ[1][3][BB][k][i][j]= lhsZ[1][3][BB][k][i][j] - coeff*lhsZ[2][3][BB][k][i][j];
          lhsZ[1][4][BB][k][i][j]= lhsZ[1][4][BB][k][i][j] - coeff*lhsZ[2][4][BB][k][i][j];
          lhsZ[1][0][CC][k][i][j] = lhsZ[1][0][CC][k][i][j] - coeff*lhsZ[2][0][CC][k][i][j];
          lhsZ[1][1][CC][k][i][j] = lhsZ[1][1][CC][k][i][j] - coeff*lhsZ[2][1][CC][k][i][j];
          lhsZ[1][2][CC][k][i][j] = lhsZ[1][2][CC][k][i][j] - coeff*lhsZ[2][2][CC][k][i][j];
          lhsZ[1][3][CC][k][i][j] = lhsZ[1][3][CC][k][i][j] - coeff*lhsZ[2][3][CC][k][i][j];
          lhsZ[1][4][CC][k][i][j] = lhsZ[1][4][CC][k][i][j] - coeff*lhsZ[2][4][CC][k][i][j];
          rhs[k][j][i][1]   = rhs[k][j][i][1]   - coeff*rhs[k][j][i][2];

          coeff = lhsZ[3][2][BB][k][i][j];
          lhsZ[3][3][BB][k][i][j]= lhsZ[3][3][BB][k][i][j] - coeff*lhsZ[2][3][BB][k][i][j];
          lhsZ[3][4][BB][k][i][j]= lhsZ[3][4][BB][k][i][j] - coeff*lhsZ[2][4][BB][k][i][j];
          lhsZ[3][0][CC][k][i][j] = lhsZ[3][0][CC][k][i][j] - coeff*lhsZ[2][0][CC][k][i][j];
          lhsZ[3][1][CC][k][i][j] = lhsZ[3][1][CC][k][i][j] - coeff*lhsZ[2][1][CC][k][i][j];
          lhsZ[3][2][CC][k][i][j] = lhsZ[3][2][CC][k][i][j] - coeff*lhsZ[2][2][CC][k][i][j];
          lhsZ[3][3][CC][k][i][j] = lhsZ[3][3][CC][k][i][j] - coeff*lhsZ[2][3][CC][k][i][j];
          lhsZ[3][4][CC][k][i][j] = lhsZ[3][4][CC][k][i][j] - coeff*lhsZ[2][4][CC][k][i][j];
          rhs[k][j][i][3]   = rhs[k][j][i][3]   - coeff*rhs[k][j][i][2];

          coeff = lhsZ[4][2][BB][k][i][j];
          lhsZ[4][3][BB][k][i][j]= lhsZ[4][3][BB][k][i][j] - coeff*lhsZ[2][3][BB][k][i][j];
          lhsZ[4][4][BB][k][i][j]= lhsZ[4][4][BB][k][i][j] - coeff*lhsZ[2][4][BB][k][i][j];
          lhsZ[4][0][CC][k][i][j] = lhsZ[4][0][CC][k][i][j] - coeff*lhsZ[2][0][CC][k][i][j];
          lhsZ[4][1][CC][k][i][j] = lhsZ[4][1][CC][k][i][j] - coeff*lhsZ[2][1][CC][k][i][j];
          lhsZ[4][2][CC][k][i][j] = lhsZ[4][2][CC][k][i][j] - coeff*lhsZ[2][2][CC][k][i][j];
          lhsZ[4][3][CC][k][i][j] = lhsZ[4][3][CC][k][i][j] - coeff*lhsZ[2][3][CC][k][i][j];
          lhsZ[4][4][CC][k][i][j] = lhsZ[4][4][CC][k][i][j] - coeff*lhsZ[2][4][CC][k][i][j];
          rhs[k][j][i][4]   = rhs[k][j][i][4]   - coeff*rhs[k][j][i][2];


          pivot = 1.00/lhsZ[3][3][BB][k][i][j];
          lhsZ[3][4][BB][k][i][j] = lhsZ[3][4][BB][k][i][j]*pivot;
          lhsZ[3][0][CC][k][i][j] = lhsZ[3][0][CC][k][i][j]*pivot;
          lhsZ[3][1][CC][k][i][j] = lhsZ[3][1][CC][k][i][j]*pivot;
          lhsZ[3][2][CC][k][i][j] = lhsZ[3][2][CC][k][i][j]*pivot;
          lhsZ[3][3][CC][k][i][j] = lhsZ[3][3][CC][k][i][j]*pivot;
          lhsZ[3][4][CC][k][i][j] = lhsZ[3][4][CC][k][i][j]*pivot;
          rhs[k][j][i][3]   = rhs[k][j][i][3]  *pivot;

          coeff = lhsZ[0][3][BB][k][i][j];
          lhsZ[0][4][BB][k][i][j]= lhsZ[0][4][BB][k][i][j] - coeff*lhsZ[3][4][BB][k][i][j];
          lhsZ[0][0][CC][k][i][j] = lhsZ[0][0][CC][k][i][j] - coeff*lhsZ[3][0][CC][k][i][j];
          lhsZ[0][1][CC][k][i][j] = lhsZ[0][1][CC][k][i][j] - coeff*lhsZ[3][1][CC][k][i][j];
          lhsZ[0][2][CC][k][i][j] = lhsZ[0][2][CC][k][i][j] - coeff*lhsZ[3][2][CC][k][i][j];
          lhsZ[0][3][CC][k][i][j] = lhsZ[0][3][CC][k][i][j] - coeff*lhsZ[3][3][CC][k][i][j];
          lhsZ[0][4][CC][k][i][j] = lhsZ[0][4][CC][k][i][j] - coeff*lhsZ[3][4][CC][k][i][j];
          rhs[k][j][i][0]   = rhs[k][j][i][0]   - coeff*rhs[k][j][i][3];

          coeff = lhsZ[1][3][BB][k][i][j];
          lhsZ[1][4][BB][k][i][j]= lhsZ[1][4][BB][k][i][j] - coeff*lhsZ[3][4][BB][k][i][j];
          lhsZ[1][0][CC][k][i][j] = lhsZ[1][0][CC][k][i][j] - coeff*lhsZ[3][0][CC][k][i][j];
          lhsZ[1][1][CC][k][i][j] = lhsZ[1][1][CC][k][i][j] - coeff*lhsZ[3][1][CC][k][i][j];
          lhsZ[1][2][CC][k][i][j] = lhsZ[1][2][CC][k][i][j] - coeff*lhsZ[3][2][CC][k][i][j];
          lhsZ[1][3][CC][k][i][j] = lhsZ[1][3][CC][k][i][j] - coeff*lhsZ[3][3][CC][k][i][j];
          lhsZ[1][4][CC][k][i][j] = lhsZ[1][4][CC][k][i][j] - coeff*lhsZ[3][4][CC][k][i][j];
          rhs[k][j][i][1]   = rhs[k][j][i][1]   - coeff*rhs[k][j][i][3];

          coeff = lhsZ[2][3][BB][k][i][j];
          lhsZ[2][4][BB][k][i][j]= lhsZ[2][4][BB][k][i][j] - coeff*lhsZ[3][4][BB][k][i][j];
          lhsZ[2][0][CC][k][i][j] = lhsZ[2][0][CC][k][i][j] - coeff*lhsZ[3][0][CC][k][i][j];
          lhsZ[2][1][CC][k][i][j] = lhsZ[2][1][CC][k][i][j] - coeff*lhsZ[3][1][CC][k][i][j];
          lhsZ[2][2][CC][k][i][j] = lhsZ[2][2][CC][k][i][j] - coeff*lhsZ[3][2][CC][k][i][j];
          lhsZ[2][3][CC][k][i][j] = lhsZ[2][3][CC][k][i][j] - coeff*lhsZ[3][3][CC][k][i][j];
          lhsZ[2][4][CC][k][i][j] = lhsZ[2][4][CC][k][i][j] - coeff*lhsZ[3][4][CC][k][i][j];
          rhs[k][j][i][2]   = rhs[k][j][i][2]   - coeff*rhs[k][j][i][3];

          coeff = lhsZ[4][3][BB][k][i][j];
          lhsZ[4][4][BB][k][i][j]= lhsZ[4][4][BB][k][i][j] - coeff*lhsZ[3][4][BB][k][i][j];
          lhsZ[4][0][CC][k][i][j] = lhsZ[4][0][CC][k][i][j] - coeff*lhsZ[3][0][CC][k][i][j];
          lhsZ[4][1][CC][k][i][j] = lhsZ[4][1][CC][k][i][j] - coeff*lhsZ[3][1][CC][k][i][j];
          lhsZ[4][2][CC][k][i][j] = lhsZ[4][2][CC][k][i][j] - coeff*lhsZ[3][2][CC][k][i][j];
          lhsZ[4][3][CC][k][i][j] = lhsZ[4][3][CC][k][i][j] - coeff*lhsZ[3][3][CC][k][i][j];
          lhsZ[4][4][CC][k][i][j] = lhsZ[4][4][CC][k][i][j] - coeff*lhsZ[3][4][CC][k][i][j];
          rhs[k][j][i][4]   = rhs[k][j][i][4]   - coeff*rhs[k][j][i][3];


          pivot = 1.00/lhsZ[4][4][BB][k][i][j];
          lhsZ[4][0][CC][k][i][j] = lhsZ[4][0][CC][k][i][j]*pivot;
          lhsZ[4][1][CC][k][i][j] = lhsZ[4][1][CC][k][i][j]*pivot;
          lhsZ[4][2][CC][k][i][j] = lhsZ[4][2][CC][k][i][j]*pivot;
          lhsZ[4][3][CC][k][i][j] = lhsZ[4][3][CC][k][i][j]*pivot;
          lhsZ[4][4][CC][k][i][j] = lhsZ[4][4][CC][k][i][j]*pivot;
          rhs[k][j][i][4]   = rhs[k][j][i][4]  *pivot;

          coeff = lhsZ[0][4][BB][k][i][j];
          lhsZ[0][0][CC][k][i][j] = lhsZ[0][0][CC][k][i][j] - coeff*lhsZ[4][0][CC][k][i][j];
          lhsZ[0][1][CC][k][i][j] = lhsZ[0][1][CC][k][i][j] - coeff*lhsZ[4][1][CC][k][i][j];
          lhsZ[0][2][CC][k][i][j] = lhsZ[0][2][CC][k][i][j] - coeff*lhsZ[4][2][CC][k][i][j];
          lhsZ[0][3][CC][k][i][j] = lhsZ[0][3][CC][k][i][j] - coeff*lhsZ[4][3][CC][k][i][j];
          lhsZ[0][4][CC][k][i][j] = lhsZ[0][4][CC][k][i][j] - coeff*lhsZ[4][4][CC][k][i][j];
          rhs[k][j][i][0]   = rhs[k][j][i][0]   - coeff*rhs[k][j][i][4];

          coeff = lhsZ[1][4][BB][k][i][j];
          lhsZ[1][0][CC][k][i][j] = lhsZ[1][0][CC][k][i][j] - coeff*lhsZ[4][0][CC][k][i][j];
          lhsZ[1][1][CC][k][i][j] = lhsZ[1][1][CC][k][i][j] - coeff*lhsZ[4][1][CC][k][i][j];
          lhsZ[1][2][CC][k][i][j] = lhsZ[1][2][CC][k][i][j] - coeff*lhsZ[4][2][CC][k][i][j];
          lhsZ[1][3][CC][k][i][j] = lhsZ[1][3][CC][k][i][j] - coeff*lhsZ[4][3][CC][k][i][j];
          lhsZ[1][4][CC][k][i][j] = lhsZ[1][4][CC][k][i][j] - coeff*lhsZ[4][4][CC][k][i][j];
          rhs[k][j][i][1]   = rhs[k][j][i][1]   - coeff*rhs[k][j][i][4];

          coeff = lhsZ[2][4][BB][k][i][j];
          lhsZ[2][0][CC][k][i][j] = lhsZ[2][0][CC][k][i][j] - coeff*lhsZ[4][0][CC][k][i][j];
          lhsZ[2][1][CC][k][i][j] = lhsZ[2][1][CC][k][i][j] - coeff*lhsZ[4][1][CC][k][i][j];
          lhsZ[2][2][CC][k][i][j] = lhsZ[2][2][CC][k][i][j] - coeff*lhsZ[4][2][CC][k][i][j];
          lhsZ[2][3][CC][k][i][j] = lhsZ[2][3][CC][k][i][j] - coeff*lhsZ[4][3][CC][k][i][j];
          lhsZ[2][4][CC][k][i][j] = lhsZ[2][4][CC][k][i][j] - coeff*lhsZ[4][4][CC][k][i][j];
          rhs[k][j][i][2]   = rhs[k][j][i][2]   - coeff*rhs[k][j][i][4];

          coeff = lhsZ[3][4][BB][k][i][j];
          lhsZ[3][0][CC][k][i][j] = lhsZ[3][0][CC][k][i][j] - coeff*lhsZ[4][0][CC][k][i][j];
          lhsZ[3][1][CC][k][i][j] = lhsZ[3][1][CC][k][i][j] - coeff*lhsZ[4][1][CC][k][i][j];
          lhsZ[3][2][CC][k][i][j] = lhsZ[3][2][CC][k][i][j] - coeff*lhsZ[4][2][CC][k][i][j];
          lhsZ[3][3][CC][k][i][j] = lhsZ[3][3][CC][k][i][j] - coeff*lhsZ[4][3][CC][k][i][j];
          lhsZ[3][4][CC][k][i][j] = lhsZ[3][4][CC][k][i][j] - coeff*lhsZ[4][4][CC][k][i][j];
          rhs[k][j][i][3]   = rhs[k][j][i][3]   - coeff*rhs[k][j][i][4];



        }/*end loop k*/
      }/*end loop i*/
    }/*end loop j*/
#endif

    //---------------------------------------------------------------------
    // Now finish up special cases for last cell
    //---------------------------------------------------------------------

    //---------------------------------------------------------------------
    // rhs(ksize) = rhs(ksize) - A*rhs(ksize-1)
    //---------------------------------------------------------------------
    //matvec_sub(lhsZ[i][j][AA], rhs[ksize-1][ksize][i][j], rhs[ksize][j][i]);
    size_t kernel_z_solve_6_off[2] = { 1, 1 };
    size_t kernel_z_solve_6_idx[2] = { gp12, gp02 };
    brisbane_kernel kernel_z_solve_6;
    brisbane_kernel_create("z_solve_6", &kernel_z_solve_6);
    brisbane_kernel_setmem(kernel_z_solve_6, 0, mem_lhsZ, brisbane_rw);
    brisbane_kernel_setmem(kernel_z_solve_6, 1, mem_rhs, brisbane_rw);
    brisbane_kernel_setarg(kernel_z_solve_6, 2, sizeof(int), &ksize);

    brisbane_task task6;
    brisbane_task_create(&task6);
    brisbane_task_kernel(task6, kernel_z_solve_6, 2, kernel_z_solve_6_off, kernel_z_solve_6_idx);
    brisbane_task_submit(task6, brisbane_cpu, NULL, true);
#if 0
#ifdef SPEC_USE_INNER_SIMD
    #pragma omp target teams distribute parallel for private(i,j)
#else
    #pragma omp target teams distribute parallel for simd collapse(2)
#endif
    for (i = 1; i <= gp02; i++) {
#ifdef SPEC_USE_INNER_SIMD
      #pragma omp simd
#endif
      for (j = 1; j <= gp12; j++) {
        /*
        for(m = 0; m < 5; m++){
          rhs[ksize][j][i][m] = rhs[ksize][j][i][m] - lhsZ[m][0][AA][ksize][i][j]*rhs[ksize-1][j][i][0]
            - lhsZ[m][1][AA][ksize][i][j]*rhs[ksize-1][j][i][1]
            - lhsZ[m][2][AA][ksize][i][j]*rhs[ksize-1][j][i][2]
            - lhsZ[m][3][AA][ksize][i][j]*rhs[ksize-1][j][i][3]
            - lhsZ[m][4][AA][ksize][i][j]*rhs[ksize-1][j][i][4];
        }
        */
        rhs[ksize][j][i][0] = rhs[ksize][j][i][0] - lhsZ[0][0][AA][ksize][i][j]*rhs[ksize-1][j][i][0]
          - lhsZ[0][1][AA][ksize][i][j]*rhs[ksize-1][j][i][1]
          - lhsZ[0][2][AA][ksize][i][j]*rhs[ksize-1][j][i][2]
          - lhsZ[0][3][AA][ksize][i][j]*rhs[ksize-1][j][i][3]
          - lhsZ[0][4][AA][ksize][i][j]*rhs[ksize-1][j][i][4];
        rhs[ksize][j][i][1] = rhs[ksize][j][i][1] - lhsZ[1][0][AA][ksize][i][j]*rhs[ksize-1][j][i][0]
          - lhsZ[1][1][AA][ksize][i][j]*rhs[ksize-1][j][i][1]
          - lhsZ[1][2][AA][ksize][i][j]*rhs[ksize-1][j][i][2]
          - lhsZ[1][3][AA][ksize][i][j]*rhs[ksize-1][j][i][3]
          - lhsZ[1][4][AA][ksize][i][j]*rhs[ksize-1][j][i][4];
        rhs[ksize][j][i][2] = rhs[ksize][j][i][2] - lhsZ[2][0][AA][ksize][i][j]*rhs[ksize-1][j][i][0]
          - lhsZ[2][1][AA][ksize][i][j]*rhs[ksize-1][j][i][1]
          - lhsZ[2][2][AA][ksize][i][j]*rhs[ksize-1][j][i][2]
          - lhsZ[2][3][AA][ksize][i][j]*rhs[ksize-1][j][i][3]
          - lhsZ[2][4][AA][ksize][i][j]*rhs[ksize-1][j][i][4];
        rhs[ksize][j][i][3] = rhs[ksize][j][i][3] - lhsZ[3][0][AA][ksize][i][j]*rhs[ksize-1][j][i][0]
          - lhsZ[3][1][AA][ksize][i][j]*rhs[ksize-1][j][i][1]
          - lhsZ[3][2][AA][ksize][i][j]*rhs[ksize-1][j][i][2]
          - lhsZ[3][3][AA][ksize][i][j]*rhs[ksize-1][j][i][3]
          - lhsZ[3][4][AA][ksize][i][j]*rhs[ksize-1][j][i][4];
        rhs[ksize][j][i][4] = rhs[ksize][j][i][4] - lhsZ[4][0][AA][ksize][i][j]*rhs[ksize-1][j][i][0]
          - lhsZ[4][1][AA][ksize][i][j]*rhs[ksize-1][j][i][1]
          - lhsZ[4][2][AA][ksize][i][j]*rhs[ksize-1][j][i][2]
          - lhsZ[4][3][AA][ksize][i][j]*rhs[ksize-1][j][i][3]
          - lhsZ[4][4][AA][ksize][i][j]*rhs[ksize-1][j][i][4];
      }
    }
#endif

    //---------------------------------------------------------------------
    // B(ksize) = B(ksize) - C(ksize-1)*A(ksize)
    // matmul_sub(AA,i,j,ksize,c,
    // $              CC,i,j,ksize-1,c,BB,i,j,ksize)
    //---------------------------------------------------------------------
    size_t kernel_z_solve_7_off[2] = { 1, 1 };
    size_t kernel_z_solve_7_idx[2] = { gp12, gp02 };
    brisbane_kernel kernel_z_solve_7;
    brisbane_kernel_create("z_solve_7", &kernel_z_solve_7);
    brisbane_kernel_setmem(kernel_z_solve_7, 0, mem_lhsZ, brisbane_rw);
    brisbane_kernel_setarg(kernel_z_solve_7, 1, sizeof(int), &ksize);

    brisbane_task task7;
    brisbane_task_create(&task7);
    brisbane_task_kernel(task7, kernel_z_solve_7, 2, kernel_z_solve_7_off, kernel_z_solve_7_idx);
    brisbane_task_submit(task7, brisbane_cpu, NULL, true);
#if 0
    //matmul_sub(lhsZ[ksize-1][i][AA], lhsZ[j][ksize][i][j][CC], lhsZ[j][i][ksize][BB]);
#ifdef SPEC_USE_INNER_SIMD
    #pragma omp target teams distribute parallel for private(i,j)
#else
    #pragma omp target teams distribute parallel for simd collapse(2)
#endif
    for (i = 1; i <= gp02; i++) {
#ifdef SPEC_USE_INNER_SIMD
      #pragma omp simd
#endif
      for (j = 1; j <= gp12; j++) {
        /*
        for(m = 0; m < 5; m++){
          for(n = 0; n < 5; n++){
            lhsZ[n][m][BB][ksize][i][j] = lhsZ[n][m][BB][ksize][i][j] - lhsZ[n][0][AA][ksize][i][j]*lhsZ[0][m][CC][ksize-1][i][j]
              - lhsZ[n][1][AA][ksize][i][j]*lhsZ[1][m][CC][ksize-1][i][j]
              - lhsZ[n][2][AA][ksize][i][j]*lhsZ[2][m][CC][ksize-1][i][j]
              - lhsZ[n][3][AA][ksize][i][j]*lhsZ[3][m][CC][ksize-1][i][j]
              - lhsZ[n][4][AA][ksize][i][j]*lhsZ[4][m][CC][ksize-1][i][j];
          }
        }
        */
        lhsZ[0][0][BB][ksize][i][j] = lhsZ[0][0][BB][ksize][i][j] - lhsZ[0][0][AA][ksize][i][j]*lhsZ[0][0][CC][ksize-1][i][j]
          - lhsZ[0][1][AA][ksize][i][j]*lhsZ[1][0][CC][ksize-1][i][j]
          - lhsZ[0][2][AA][ksize][i][j]*lhsZ[2][0][CC][ksize-1][i][j]
          - lhsZ[0][3][AA][ksize][i][j]*lhsZ[3][0][CC][ksize-1][i][j]
          - lhsZ[0][4][AA][ksize][i][j]*lhsZ[4][0][CC][ksize-1][i][j];
        lhsZ[1][0][BB][ksize][i][j] = lhsZ[1][0][BB][ksize][i][j] - lhsZ[1][0][AA][ksize][i][j]*lhsZ[0][0][CC][ksize-1][i][j]
          - lhsZ[1][1][AA][ksize][i][j]*lhsZ[1][0][CC][ksize-1][i][j]
          - lhsZ[1][2][AA][ksize][i][j]*lhsZ[2][0][CC][ksize-1][i][j]
          - lhsZ[1][3][AA][ksize][i][j]*lhsZ[3][0][CC][ksize-1][i][j]
          - lhsZ[1][4][AA][ksize][i][j]*lhsZ[4][0][CC][ksize-1][i][j];
        lhsZ[2][0][BB][ksize][i][j] = lhsZ[2][0][BB][ksize][i][j] - lhsZ[2][0][AA][ksize][i][j]*lhsZ[0][0][CC][ksize-1][i][j]
          - lhsZ[2][1][AA][ksize][i][j]*lhsZ[1][0][CC][ksize-1][i][j]
          - lhsZ[2][2][AA][ksize][i][j]*lhsZ[2][0][CC][ksize-1][i][j]
          - lhsZ[2][3][AA][ksize][i][j]*lhsZ[3][0][CC][ksize-1][i][j]
          - lhsZ[2][4][AA][ksize][i][j]*lhsZ[4][0][CC][ksize-1][i][j];
        lhsZ[3][0][BB][ksize][i][j] = lhsZ[3][0][BB][ksize][i][j] - lhsZ[3][0][AA][ksize][i][j]*lhsZ[0][0][CC][ksize-1][i][j]
          - lhsZ[3][1][AA][ksize][i][j]*lhsZ[1][0][CC][ksize-1][i][j]
          - lhsZ[3][2][AA][ksize][i][j]*lhsZ[2][0][CC][ksize-1][i][j]
          - lhsZ[3][3][AA][ksize][i][j]*lhsZ[3][0][CC][ksize-1][i][j]
          - lhsZ[3][4][AA][ksize][i][j]*lhsZ[4][0][CC][ksize-1][i][j];
        lhsZ[4][0][BB][ksize][i][j] = lhsZ[4][0][BB][ksize][i][j] - lhsZ[4][0][AA][ksize][i][j]*lhsZ[0][0][CC][ksize-1][i][j]
          - lhsZ[4][1][AA][ksize][i][j]*lhsZ[1][0][CC][ksize-1][i][j]
          - lhsZ[4][2][AA][ksize][i][j]*lhsZ[2][0][CC][ksize-1][i][j]
          - lhsZ[4][3][AA][ksize][i][j]*lhsZ[3][0][CC][ksize-1][i][j]
          - lhsZ[4][4][AA][ksize][i][j]*lhsZ[4][0][CC][ksize-1][i][j];
        lhsZ[0][1][BB][ksize][i][j] = lhsZ[0][1][BB][ksize][i][j] - lhsZ[0][0][AA][ksize][i][j]*lhsZ[0][1][CC][ksize-1][i][j]
          - lhsZ[0][1][AA][ksize][i][j]*lhsZ[1][1][CC][ksize-1][i][j]
          - lhsZ[0][2][AA][ksize][i][j]*lhsZ[2][1][CC][ksize-1][i][j]
          - lhsZ[0][3][AA][ksize][i][j]*lhsZ[3][1][CC][ksize-1][i][j]
          - lhsZ[0][4][AA][ksize][i][j]*lhsZ[4][1][CC][ksize-1][i][j];
        lhsZ[1][1][BB][ksize][i][j] = lhsZ[1][1][BB][ksize][i][j] - lhsZ[1][0][AA][ksize][i][j]*lhsZ[0][1][CC][ksize-1][i][j]
          - lhsZ[1][1][AA][ksize][i][j]*lhsZ[1][1][CC][ksize-1][i][j]
          - lhsZ[1][2][AA][ksize][i][j]*lhsZ[2][1][CC][ksize-1][i][j]
          - lhsZ[1][3][AA][ksize][i][j]*lhsZ[3][1][CC][ksize-1][i][j]
          - lhsZ[1][4][AA][ksize][i][j]*lhsZ[4][1][CC][ksize-1][i][j];
        lhsZ[2][1][BB][ksize][i][j] = lhsZ[2][1][BB][ksize][i][j] - lhsZ[2][0][AA][ksize][i][j]*lhsZ[0][1][CC][ksize-1][i][j]
          - lhsZ[2][1][AA][ksize][i][j]*lhsZ[1][1][CC][ksize-1][i][j]
          - lhsZ[2][2][AA][ksize][i][j]*lhsZ[2][1][CC][ksize-1][i][j]
          - lhsZ[2][3][AA][ksize][i][j]*lhsZ[3][1][CC][ksize-1][i][j]
          - lhsZ[2][4][AA][ksize][i][j]*lhsZ[4][1][CC][ksize-1][i][j];
        lhsZ[3][1][BB][ksize][i][j] = lhsZ[3][1][BB][ksize][i][j] - lhsZ[3][0][AA][ksize][i][j]*lhsZ[0][1][CC][ksize-1][i][j]
          - lhsZ[3][1][AA][ksize][i][j]*lhsZ[1][1][CC][ksize-1][i][j]
          - lhsZ[3][2][AA][ksize][i][j]*lhsZ[2][1][CC][ksize-1][i][j]
          - lhsZ[3][3][AA][ksize][i][j]*lhsZ[3][1][CC][ksize-1][i][j]
          - lhsZ[3][4][AA][ksize][i][j]*lhsZ[4][1][CC][ksize-1][i][j];
        lhsZ[4][1][BB][ksize][i][j] = lhsZ[4][1][BB][ksize][i][j] - lhsZ[4][0][AA][ksize][i][j]*lhsZ[0][1][CC][ksize-1][i][j]
          - lhsZ[4][1][AA][ksize][i][j]*lhsZ[1][1][CC][ksize-1][i][j]
          - lhsZ[4][2][AA][ksize][i][j]*lhsZ[2][1][CC][ksize-1][i][j]
          - lhsZ[4][3][AA][ksize][i][j]*lhsZ[3][1][CC][ksize-1][i][j]
          - lhsZ[4][4][AA][ksize][i][j]*lhsZ[4][1][CC][ksize-1][i][j];
        lhsZ[0][2][BB][ksize][i][j] = lhsZ[0][2][BB][ksize][i][j] - lhsZ[0][0][AA][ksize][i][j]*lhsZ[0][2][CC][ksize-1][i][j]
          - lhsZ[0][1][AA][ksize][i][j]*lhsZ[1][2][CC][ksize-1][i][j]
          - lhsZ[0][2][AA][ksize][i][j]*lhsZ[2][2][CC][ksize-1][i][j]
          - lhsZ[0][3][AA][ksize][i][j]*lhsZ[3][2][CC][ksize-1][i][j]
          - lhsZ[0][4][AA][ksize][i][j]*lhsZ[4][2][CC][ksize-1][i][j];
        lhsZ[1][2][BB][ksize][i][j] = lhsZ[1][2][BB][ksize][i][j] - lhsZ[1][0][AA][ksize][i][j]*lhsZ[0][2][CC][ksize-1][i][j]
          - lhsZ[1][1][AA][ksize][i][j]*lhsZ[1][2][CC][ksize-1][i][j]
          - lhsZ[1][2][AA][ksize][i][j]*lhsZ[2][2][CC][ksize-1][i][j]
          - lhsZ[1][3][AA][ksize][i][j]*lhsZ[3][2][CC][ksize-1][i][j]
          - lhsZ[1][4][AA][ksize][i][j]*lhsZ[4][2][CC][ksize-1][i][j];
        lhsZ[2][2][BB][ksize][i][j] = lhsZ[2][2][BB][ksize][i][j] - lhsZ[2][0][AA][ksize][i][j]*lhsZ[0][2][CC][ksize-1][i][j]
          - lhsZ[2][1][AA][ksize][i][j]*lhsZ[1][2][CC][ksize-1][i][j]
          - lhsZ[2][2][AA][ksize][i][j]*lhsZ[2][2][CC][ksize-1][i][j]
          - lhsZ[2][3][AA][ksize][i][j]*lhsZ[3][2][CC][ksize-1][i][j]
          - lhsZ[2][4][AA][ksize][i][j]*lhsZ[4][2][CC][ksize-1][i][j];
        lhsZ[3][2][BB][ksize][i][j] = lhsZ[3][2][BB][ksize][i][j] - lhsZ[3][0][AA][ksize][i][j]*lhsZ[0][2][CC][ksize-1][i][j]
          - lhsZ[3][1][AA][ksize][i][j]*lhsZ[1][2][CC][ksize-1][i][j]
          - lhsZ[3][2][AA][ksize][i][j]*lhsZ[2][2][CC][ksize-1][i][j]
          - lhsZ[3][3][AA][ksize][i][j]*lhsZ[3][2][CC][ksize-1][i][j]
          - lhsZ[3][4][AA][ksize][i][j]*lhsZ[4][2][CC][ksize-1][i][j];
        lhsZ[4][2][BB][ksize][i][j] = lhsZ[4][2][BB][ksize][i][j] - lhsZ[4][0][AA][ksize][i][j]*lhsZ[0][2][CC][ksize-1][i][j]
          - lhsZ[4][1][AA][ksize][i][j]*lhsZ[1][2][CC][ksize-1][i][j]
          - lhsZ[4][2][AA][ksize][i][j]*lhsZ[2][2][CC][ksize-1][i][j]
          - lhsZ[4][3][AA][ksize][i][j]*lhsZ[3][2][CC][ksize-1][i][j]
          - lhsZ[4][4][AA][ksize][i][j]*lhsZ[4][2][CC][ksize-1][i][j];
        lhsZ[0][3][BB][ksize][i][j] = lhsZ[0][3][BB][ksize][i][j] - lhsZ[0][0][AA][ksize][i][j]*lhsZ[0][3][CC][ksize-1][i][j]
          - lhsZ[0][1][AA][ksize][i][j]*lhsZ[1][3][CC][ksize-1][i][j]
          - lhsZ[0][2][AA][ksize][i][j]*lhsZ[2][3][CC][ksize-1][i][j]
          - lhsZ[0][3][AA][ksize][i][j]*lhsZ[3][3][CC][ksize-1][i][j]
          - lhsZ[0][4][AA][ksize][i][j]*lhsZ[4][3][CC][ksize-1][i][j];
        lhsZ[1][3][BB][ksize][i][j] = lhsZ[1][3][BB][ksize][i][j] - lhsZ[1][0][AA][ksize][i][j]*lhsZ[0][3][CC][ksize-1][i][j]
          - lhsZ[1][1][AA][ksize][i][j]*lhsZ[1][3][CC][ksize-1][i][j]
          - lhsZ[1][2][AA][ksize][i][j]*lhsZ[2][3][CC][ksize-1][i][j]
          - lhsZ[1][3][AA][ksize][i][j]*lhsZ[3][3][CC][ksize-1][i][j]
          - lhsZ[1][4][AA][ksize][i][j]*lhsZ[4][3][CC][ksize-1][i][j];
        lhsZ[2][3][BB][ksize][i][j] = lhsZ[2][3][BB][ksize][i][j] - lhsZ[2][0][AA][ksize][i][j]*lhsZ[0][3][CC][ksize-1][i][j]
          - lhsZ[2][1][AA][ksize][i][j]*lhsZ[1][3][CC][ksize-1][i][j]
          - lhsZ[2][2][AA][ksize][i][j]*lhsZ[2][3][CC][ksize-1][i][j]
          - lhsZ[2][3][AA][ksize][i][j]*lhsZ[3][3][CC][ksize-1][i][j]
          - lhsZ[2][4][AA][ksize][i][j]*lhsZ[4][3][CC][ksize-1][i][j];
        lhsZ[3][3][BB][ksize][i][j] = lhsZ[3][3][BB][ksize][i][j] - lhsZ[3][0][AA][ksize][i][j]*lhsZ[0][3][CC][ksize-1][i][j]
          - lhsZ[3][1][AA][ksize][i][j]*lhsZ[1][3][CC][ksize-1][i][j]
          - lhsZ[3][2][AA][ksize][i][j]*lhsZ[2][3][CC][ksize-1][i][j]
          - lhsZ[3][3][AA][ksize][i][j]*lhsZ[3][3][CC][ksize-1][i][j]
          - lhsZ[3][4][AA][ksize][i][j]*lhsZ[4][3][CC][ksize-1][i][j];
        lhsZ[4][3][BB][ksize][i][j] = lhsZ[4][3][BB][ksize][i][j] - lhsZ[4][0][AA][ksize][i][j]*lhsZ[0][3][CC][ksize-1][i][j]
          - lhsZ[4][1][AA][ksize][i][j]*lhsZ[1][3][CC][ksize-1][i][j]
          - lhsZ[4][2][AA][ksize][i][j]*lhsZ[2][3][CC][ksize-1][i][j]
          - lhsZ[4][3][AA][ksize][i][j]*lhsZ[3][3][CC][ksize-1][i][j]
          - lhsZ[4][4][AA][ksize][i][j]*lhsZ[4][3][CC][ksize-1][i][j];
        lhsZ[0][4][BB][ksize][i][j] = lhsZ[0][4][BB][ksize][i][j] - lhsZ[0][0][AA][ksize][i][j]*lhsZ[0][4][CC][ksize-1][i][j]
          - lhsZ[0][1][AA][ksize][i][j]*lhsZ[1][4][CC][ksize-1][i][j]
          - lhsZ[0][2][AA][ksize][i][j]*lhsZ[2][4][CC][ksize-1][i][j]
          - lhsZ[0][3][AA][ksize][i][j]*lhsZ[3][4][CC][ksize-1][i][j]
          - lhsZ[0][4][AA][ksize][i][j]*lhsZ[4][4][CC][ksize-1][i][j];
        lhsZ[1][4][BB][ksize][i][j] = lhsZ[1][4][BB][ksize][i][j] - lhsZ[1][0][AA][ksize][i][j]*lhsZ[0][4][CC][ksize-1][i][j]
          - lhsZ[1][1][AA][ksize][i][j]*lhsZ[1][4][CC][ksize-1][i][j]
          - lhsZ[1][2][AA][ksize][i][j]*lhsZ[2][4][CC][ksize-1][i][j]
          - lhsZ[1][3][AA][ksize][i][j]*lhsZ[3][4][CC][ksize-1][i][j]
          - lhsZ[1][4][AA][ksize][i][j]*lhsZ[4][4][CC][ksize-1][i][j];
        lhsZ[2][4][BB][ksize][i][j] = lhsZ[2][4][BB][ksize][i][j] - lhsZ[2][0][AA][ksize][i][j]*lhsZ[0][4][CC][ksize-1][i][j]
          - lhsZ[2][1][AA][ksize][i][j]*lhsZ[1][4][CC][ksize-1][i][j]
          - lhsZ[2][2][AA][ksize][i][j]*lhsZ[2][4][CC][ksize-1][i][j]
          - lhsZ[2][3][AA][ksize][i][j]*lhsZ[3][4][CC][ksize-1][i][j]
          - lhsZ[2][4][AA][ksize][i][j]*lhsZ[4][4][CC][ksize-1][i][j];
        lhsZ[3][4][BB][ksize][i][j] = lhsZ[3][4][BB][ksize][i][j] - lhsZ[3][0][AA][ksize][i][j]*lhsZ[0][4][CC][ksize-1][i][j]
          - lhsZ[3][1][AA][ksize][i][j]*lhsZ[1][4][CC][ksize-1][i][j]
          - lhsZ[3][2][AA][ksize][i][j]*lhsZ[2][4][CC][ksize-1][i][j]
          - lhsZ[3][3][AA][ksize][i][j]*lhsZ[3][4][CC][ksize-1][i][j]
          - lhsZ[3][4][AA][ksize][i][j]*lhsZ[4][4][CC][ksize-1][i][j];
        lhsZ[4][4][BB][ksize][i][j] = lhsZ[4][4][BB][ksize][i][j] - lhsZ[4][0][AA][ksize][i][j]*lhsZ[0][4][CC][ksize-1][i][j]
          - lhsZ[4][1][AA][ksize][i][j]*lhsZ[1][4][CC][ksize-1][i][j]
          - lhsZ[4][2][AA][ksize][i][j]*lhsZ[2][4][CC][ksize-1][i][j]
          - lhsZ[4][3][AA][ksize][i][j]*lhsZ[3][4][CC][ksize-1][i][j]
          - lhsZ[4][4][AA][ksize][i][j]*lhsZ[4][4][CC][ksize-1][i][j];

      }
    }
#endif

    //---------------------------------------------------------------------
    // multiply rhs(ksize) by b_inverse(ksize) and copy to rhs
    //---------------------------------------------------------------------
    //binvrhs( lhsZ[i][j][BB], rhs[ksize][ksize][i][j] );
    size_t kernel_z_solve_8_off[2] = { 1, 1 };
    size_t kernel_z_solve_8_idx[2] = { gp12, gp02 };
    brisbane_kernel kernel_z_solve_8;
    brisbane_kernel_create("z_solve_8", &kernel_z_solve_8);
    brisbane_kernel_setmem(kernel_z_solve_8, 0, mem_lhsZ, brisbane_rw);
    brisbane_kernel_setmem(kernel_z_solve_8, 1, mem_rhs, brisbane_rw);
    brisbane_kernel_setarg(kernel_z_solve_8, 2, sizeof(int), &ksize);

    brisbane_task task8;
    brisbane_task_create(&task8);
    brisbane_task_kernel(task8, kernel_z_solve_8, 2, kernel_z_solve_8_off, kernel_z_solve_8_idx);
    brisbane_task_submit(task8, brisbane_cpu, NULL, true);
#if 0
#ifdef SPEC_USE_INNER_SIMD
    #pragma omp target teams distribute parallel for private(i,j,pivot,coeff) 
#else
    #pragma omp target teams distribute parallel for simd private(pivot,coeff) collapse(2)
#endif
    for (i = 1; i <= gp02; i++) {
#ifdef SPEC_USE_INNER_SIMD
      #pragma omp simd private(pivot,coeff)
#endif
      for (j = 1; j <= gp12; j++) {
        /*
        for(m = 0; m < 5; m++){
          pivot = 1.00/lhsZ[m][m][BB][ksize][i][j];
          for(n = m+1; n < 5; n++){
            lhsZ[m][n][BB][ksize][i][j] = lhsZ[m][n][BB][ksize][i][j]*pivot;
          }
          rhs[ksize][j][i][m] = rhs[ksize][j][i][m]*pivot;

          for(n = 0; n < 5; n++){
            if(n != m){
              coeff = lhsZ[n][m][BB][ksize][i][j];
              for(z = m+1; z < 5; z++){
                lhsZ[n][z][BB][ksize][i][j] = lhsZ[n][z][BB][ksize][i][j] - coeff*lhsZ[m][z][BB][ksize][i][j];
              }
              rhs[ksize][j][i][n] = rhs[ksize][j][i][n] - coeff*rhs[ksize][j][i][m];
            }
          }
        }
        */

        pivot = 1.00/lhsZ[0][0][BB][ksize][i][j];
        lhsZ[0][1][BB][ksize][i][j] = lhsZ[0][1][BB][ksize][i][j]*pivot;
        lhsZ[0][2][BB][ksize][i][j] = lhsZ[0][2][BB][ksize][i][j]*pivot;
        lhsZ[0][3][BB][ksize][i][j] = lhsZ[0][3][BB][ksize][i][j]*pivot;
        lhsZ[0][4][BB][ksize][i][j] = lhsZ[0][4][BB][ksize][i][j]*pivot;
        rhs[ksize][j][i][0]   = rhs[ksize][j][i][0]  *pivot;

        coeff = lhsZ[1][0][BB][ksize][i][j];
        lhsZ[1][1][BB][ksize][i][j]= lhsZ[1][1][BB][ksize][i][j] - coeff*lhsZ[0][1][BB][ksize][i][j];
        lhsZ[1][2][BB][ksize][i][j]= lhsZ[1][2][BB][ksize][i][j] - coeff*lhsZ[0][2][BB][ksize][i][j];
        lhsZ[1][3][BB][ksize][i][j]= lhsZ[1][3][BB][ksize][i][j] - coeff*lhsZ[0][3][BB][ksize][i][j];
        lhsZ[1][4][BB][ksize][i][j]= lhsZ[1][4][BB][ksize][i][j] - coeff*lhsZ[0][4][BB][ksize][i][j];
        rhs[ksize][j][i][1]   = rhs[ksize][j][i][1]   - coeff*rhs[ksize][j][i][0];

        coeff = lhsZ[2][0][BB][ksize][i][j];
        lhsZ[2][1][BB][ksize][i][j]= lhsZ[2][1][BB][ksize][i][j] - coeff*lhsZ[0][1][BB][ksize][i][j];
        lhsZ[2][2][BB][ksize][i][j]= lhsZ[2][2][BB][ksize][i][j] - coeff*lhsZ[0][2][BB][ksize][i][j];
        lhsZ[2][3][BB][ksize][i][j]= lhsZ[2][3][BB][ksize][i][j] - coeff*lhsZ[0][3][BB][ksize][i][j];
        lhsZ[2][4][BB][ksize][i][j]= lhsZ[2][4][BB][ksize][i][j] - coeff*lhsZ[0][4][BB][ksize][i][j];
        rhs[ksize][j][i][2]   = rhs[ksize][j][i][2]   - coeff*rhs[ksize][j][i][0];

        coeff = lhsZ[3][0][BB][ksize][i][j];
        lhsZ[3][1][BB][ksize][i][j]= lhsZ[3][1][BB][ksize][i][j] - coeff*lhsZ[0][1][BB][ksize][i][j];
        lhsZ[3][2][BB][ksize][i][j]= lhsZ[3][2][BB][ksize][i][j] - coeff*lhsZ[0][2][BB][ksize][i][j];
        lhsZ[3][3][BB][ksize][i][j]= lhsZ[3][3][BB][ksize][i][j] - coeff*lhsZ[0][3][BB][ksize][i][j];
        lhsZ[3][4][BB][ksize][i][j]= lhsZ[3][4][BB][ksize][i][j] - coeff*lhsZ[0][4][BB][ksize][i][j];
        rhs[ksize][j][i][3]   = rhs[ksize][j][i][3]   - coeff*rhs[ksize][j][i][0];

        coeff = lhsZ[4][0][BB][ksize][i][j];
        lhsZ[4][1][BB][ksize][i][j]= lhsZ[4][1][BB][ksize][i][j] - coeff*lhsZ[0][1][BB][ksize][i][j];
        lhsZ[4][2][BB][ksize][i][j]= lhsZ[4][2][BB][ksize][i][j] - coeff*lhsZ[0][2][BB][ksize][i][j];
        lhsZ[4][3][BB][ksize][i][j]= lhsZ[4][3][BB][ksize][i][j] - coeff*lhsZ[0][3][BB][ksize][i][j];
        lhsZ[4][4][BB][ksize][i][j]= lhsZ[4][4][BB][ksize][i][j] - coeff*lhsZ[0][4][BB][ksize][i][j];
        rhs[ksize][j][i][4]   = rhs[ksize][j][i][4]   - coeff*rhs[ksize][j][i][0];


        pivot = 1.00/lhsZ[1][1][BB][ksize][i][j];
        lhsZ[1][2][BB][ksize][i][j] = lhsZ[1][2][BB][ksize][i][j]*pivot;
        lhsZ[1][3][BB][ksize][i][j] = lhsZ[1][3][BB][ksize][i][j]*pivot;
        lhsZ[1][4][BB][ksize][i][j] = lhsZ[1][4][BB][ksize][i][j]*pivot;
        rhs[ksize][j][i][1]   = rhs[ksize][j][i][1]  *pivot;

        coeff = lhsZ[0][1][BB][ksize][i][j];
        lhsZ[0][2][BB][ksize][i][j]= lhsZ[0][2][BB][ksize][i][j] - coeff*lhsZ[1][2][BB][ksize][i][j];
        lhsZ[0][3][BB][ksize][i][j]= lhsZ[0][3][BB][ksize][i][j] - coeff*lhsZ[1][3][BB][ksize][i][j];
        lhsZ[0][4][BB][ksize][i][j]= lhsZ[0][4][BB][ksize][i][j] - coeff*lhsZ[1][4][BB][ksize][i][j];
        rhs[ksize][j][i][0]   = rhs[ksize][j][i][0]   - coeff*rhs[ksize][j][i][1];

        coeff = lhsZ[2][1][BB][ksize][i][j];
        lhsZ[2][2][BB][ksize][i][j]= lhsZ[2][2][BB][ksize][i][j] - coeff*lhsZ[1][2][BB][ksize][i][j];
        lhsZ[2][3][BB][ksize][i][j]= lhsZ[2][3][BB][ksize][i][j] - coeff*lhsZ[1][3][BB][ksize][i][j];
        lhsZ[2][4][BB][ksize][i][j]= lhsZ[2][4][BB][ksize][i][j] - coeff*lhsZ[1][4][BB][ksize][i][j];
        rhs[ksize][j][i][2]   = rhs[ksize][j][i][2]   - coeff*rhs[ksize][j][i][1];

        coeff = lhsZ[3][1][BB][ksize][i][j];
        lhsZ[3][2][BB][ksize][i][j]= lhsZ[3][2][BB][ksize][i][j] - coeff*lhsZ[1][2][BB][ksize][i][j];
        lhsZ[3][3][BB][ksize][i][j]= lhsZ[3][3][BB][ksize][i][j] - coeff*lhsZ[1][3][BB][ksize][i][j];
        lhsZ[3][4][BB][ksize][i][j]= lhsZ[3][4][BB][ksize][i][j] - coeff*lhsZ[1][4][BB][ksize][i][j];
        rhs[ksize][j][i][3]   = rhs[ksize][j][i][3]   - coeff*rhs[ksize][j][i][1];

        coeff = lhsZ[4][1][BB][ksize][i][j];
        lhsZ[4][2][BB][ksize][i][j]= lhsZ[4][2][BB][ksize][i][j] - coeff*lhsZ[1][2][BB][ksize][i][j];
        lhsZ[4][3][BB][ksize][i][j]= lhsZ[4][3][BB][ksize][i][j] - coeff*lhsZ[1][3][BB][ksize][i][j];
        lhsZ[4][4][BB][ksize][i][j]= lhsZ[4][4][BB][ksize][i][j] - coeff*lhsZ[1][4][BB][ksize][i][j];
        rhs[ksize][j][i][4]   = rhs[ksize][j][i][4]   - coeff*rhs[ksize][j][i][1];


        pivot = 1.00/lhsZ[2][2][BB][ksize][i][j];
        lhsZ[2][3][BB][ksize][i][j] = lhsZ[2][3][BB][ksize][i][j]*pivot;
        lhsZ[2][4][BB][ksize][i][j] = lhsZ[2][4][BB][ksize][i][j]*pivot;
        rhs[ksize][j][i][2]   = rhs[ksize][j][i][2]  *pivot;

        coeff = lhsZ[0][2][BB][ksize][i][j];
        lhsZ[0][3][BB][ksize][i][j]= lhsZ[0][3][BB][ksize][i][j] - coeff*lhsZ[2][3][BB][ksize][i][j];
        lhsZ[0][4][BB][ksize][i][j]= lhsZ[0][4][BB][ksize][i][j] - coeff*lhsZ[2][4][BB][ksize][i][j];
        rhs[ksize][j][i][0]   = rhs[ksize][j][i][0]   - coeff*rhs[ksize][j][i][2];

        coeff = lhsZ[1][2][BB][ksize][i][j];
        lhsZ[1][3][BB][ksize][i][j]= lhsZ[1][3][BB][ksize][i][j] - coeff*lhsZ[2][3][BB][ksize][i][j];
        lhsZ[1][4][BB][ksize][i][j]= lhsZ[1][4][BB][ksize][i][j] - coeff*lhsZ[2][4][BB][ksize][i][j];
        rhs[ksize][j][i][1]   = rhs[ksize][j][i][1]   - coeff*rhs[ksize][j][i][2];

        coeff = lhsZ[3][2][BB][ksize][i][j];
        lhsZ[3][3][BB][ksize][i][j]= lhsZ[3][3][BB][ksize][i][j] - coeff*lhsZ[2][3][BB][ksize][i][j];
        lhsZ[3][4][BB][ksize][i][j]= lhsZ[3][4][BB][ksize][i][j] - coeff*lhsZ[2][4][BB][ksize][i][j];
        rhs[ksize][j][i][3]   = rhs[ksize][j][i][3]   - coeff*rhs[ksize][j][i][2];

        coeff = lhsZ[4][2][BB][ksize][i][j];
        lhsZ[4][3][BB][ksize][i][j]= lhsZ[4][3][BB][ksize][i][j] - coeff*lhsZ[2][3][BB][ksize][i][j];
        lhsZ[4][4][BB][ksize][i][j]= lhsZ[4][4][BB][ksize][i][j] - coeff*lhsZ[2][4][BB][ksize][i][j];
        rhs[ksize][j][i][4]   = rhs[ksize][j][i][4]   - coeff*rhs[ksize][j][i][2];


        pivot = 1.00/lhsZ[3][3][BB][ksize][i][j];
        lhsZ[3][4][BB][ksize][i][j] = lhsZ[3][4][BB][ksize][i][j]*pivot;
        rhs[ksize][j][i][3]   = rhs[ksize][j][i][3]  *pivot;

        coeff = lhsZ[0][3][BB][ksize][i][j];
        lhsZ[0][4][BB][ksize][i][j]= lhsZ[0][4][BB][ksize][i][j] - coeff*lhsZ[3][4][BB][ksize][i][j];
        rhs[ksize][j][i][0]   = rhs[ksize][j][i][0]   - coeff*rhs[ksize][j][i][3];

        coeff = lhsZ[1][3][BB][ksize][i][j];
        lhsZ[1][4][BB][ksize][i][j]= lhsZ[1][4][BB][ksize][i][j] - coeff*lhsZ[3][4][BB][ksize][i][j];
        rhs[ksize][j][i][1]   = rhs[ksize][j][i][1]   - coeff*rhs[ksize][j][i][3];

        coeff = lhsZ[2][3][BB][ksize][i][j];
        lhsZ[2][4][BB][ksize][i][j]= lhsZ[2][4][BB][ksize][i][j] - coeff*lhsZ[3][4][BB][ksize][i][j];
        rhs[ksize][j][i][2]   = rhs[ksize][j][i][2]   - coeff*rhs[ksize][j][i][3];

        coeff = lhsZ[4][3][BB][ksize][i][j];
        lhsZ[4][4][BB][ksize][i][j]= lhsZ[4][4][BB][ksize][i][j] - coeff*lhsZ[3][4][BB][ksize][i][j];
        rhs[ksize][j][i][4]   = rhs[ksize][j][i][4]   - coeff*rhs[ksize][j][i][3];


        pivot = 1.00/lhsZ[4][4][BB][ksize][i][j];
        rhs[ksize][j][i][4]   = rhs[ksize][j][i][4]  *pivot;

        coeff = lhsZ[0][4][BB][ksize][i][j];
        rhs[ksize][j][i][0]   = rhs[ksize][j][i][0]   - coeff*rhs[ksize][j][i][4];

        coeff = lhsZ[1][4][BB][ksize][i][j];
        rhs[ksize][j][i][1]   = rhs[ksize][j][i][1]   - coeff*rhs[ksize][j][i][4];

        coeff = lhsZ[2][4][BB][ksize][i][j];
        rhs[ksize][j][i][2]   = rhs[ksize][j][i][2]   - coeff*rhs[ksize][j][i][4];

        coeff = lhsZ[3][4][BB][ksize][i][j];
        rhs[ksize][j][i][3]   = rhs[ksize][j][i][3]   - coeff*rhs[ksize][j][i][4];
      }
    }
#endif
    //---------------------------------------------------------------------
    //---------------------------------------------------------------------

    //---------------------------------------------------------------------
    // back solve: if last cell, then generate U(ksize)=rhs(ksize)
    // else assume U(ksize) is loaded in un pack backsub_info
    // so just use it
    // after u(kstart) will be sent to next cell
    //---------------------------------------------------------------------

    size_t kernel_z_solve_9_off[2] = { 1, 1 };
    size_t kernel_z_solve_9_idx[2] = { gp02, gp12 };
    brisbane_kernel kernel_z_solve_9;
    brisbane_kernel_create("z_solve_9", &kernel_z_solve_9);
    brisbane_kernel_setmem(kernel_z_solve_9, 0, mem_lhsZ, brisbane_r);
    brisbane_kernel_setmem(kernel_z_solve_9, 1, mem_rhs, brisbane_rw);
    brisbane_kernel_setarg(kernel_z_solve_9, 2, sizeof(int), &ksize);

    brisbane_task task9;
    brisbane_task_create(&task9);
    brisbane_task_kernel(task9, kernel_z_solve_9, 2, kernel_z_solve_9_off, kernel_z_solve_9_idx);
    //brisbane_task_submit(task9, brisbane_cpu, NULL, true);
#if 1
    brisbane_task task10;
    brisbane_task_create(&task10);
    brisbane_task_d2h_full(task10, mem_rhs, rhs);
    brisbane_task_d2h_full(task10, mem_lhsZ, lhsZ);
    brisbane_task_submit(task10, brisbane_cpu, NULL, true);
      #pragma omp target teams distribute parallel for collapse(2) private(i,j,m,n)
      for (j = 1; j <= gp12; j++) {
        for (i = 1; i <= gp02; i++) {
        for (k = ksize-1; k >= 0; k--) {
          for (m = 0; m < BLOCK_SIZE; m++) {
            for (n = 0; n < BLOCK_SIZE; n++) {
              rhs[k][j][i][m] = rhs[k][j][i][m]
                - lhsZ[m][n][CC][k][i][j]*rhs[k+1][j][i][n];
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

  brisbane_mem_release(mem_fjacZ);
  brisbane_mem_release(mem_njacZ);
  brisbane_mem_release(mem_lhsZ);
}
