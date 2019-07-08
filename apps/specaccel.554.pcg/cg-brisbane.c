//-------------------------------------------------------------------------//
//                                                                         //
//  This benchmark is a serial C version of the NPB CG code. This C        //
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

//---------------------------------------------------------------------
// NPB CG OPENMP version      
//---------------------------------------------------------------------

#include <brisbane/brisbane.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "globals.h"
#include "randdp.h"
#include "timers.h"
#include "print_results.h"

//---------------------------------------------------------------------
/* common / main_int_mem / */
#pragma omp declare target
static brisbane_mem mem_colidx;
static brisbane_mem mem_rowstr;
static int colidx[NZ];
static int rowstr[NA+1];
#pragma omp end declare target
static int iv[NA];
static int arow[NA];
static int acol[NAZ];

/* common / main_flt_mem / */
static double aelt[NAZ];
#pragma omp declare target
static brisbane_mem mem_a;
static brisbane_mem mem_x;
static brisbane_mem mem_z;
static brisbane_mem mem_p;
static brisbane_mem mem_q;
static brisbane_mem mem_r;
static double a[NZ];
static double x[NA+2];
static double z[NA+2];
static double p[NA+2];
static double q[NA+2];
static double r[NA+2];

#pragma omp end declare target

/* common / partit_size / */
static int nzz;
#pragma omp declare target
static int naa;
static int firstrow;
static int lastrow;
static int firstcol;
static int lastcol;
#pragma omp end declare target

/* common /urando/ */
static double amult;
static double tran;

/* common /timers/ */
static int timeron;
//---------------------------------------------------------------------


//---------------------------------------------------------------------
static void conj_grad(int colidx[],
                      int rowstr[],
                      double x[],
                      double z[],
                      double a[],
                      double p[],
                      double q[],
                      double r[],
                      double *rnorm);
static void makea(int n,
                  int nz,
                  double a[],
                  int colidx[],
                  int rowstr[],
                  int firstrow,
                  int lastrow,
                  int firstcol,
                  int lastcol,
                  int arow[],
                  int acol[][NONZER+1],
                  double aelt[][NONZER+1],
                  int iv[]);
static void sparse(double a[],
                   int colidx[],
                   int rowstr[],
                   int n,
                   int nz,
                   int nozer,
                   int arow[],
                   int acol[][NONZER+1],
                   double aelt[][NONZER+1],
                   int firstrow,
                   int lastrow,
                   int nzloc[],
                   double rcond,
                   double shift);
static void sprnvc(int n, int nz, int nn1, double v[], int iv[]);
static int icnvrt(double x, int ipwr2);
static void vecset(int n, double v[], int iv[], int *nzv, int i, double val);
//---------------------------------------------------------------------


int main(int argc, char *argv[])
{
  brisbane_init(&argc, &argv);
  int i, j, k, it, nit, set_zeta;
  int end;

  double zeta;
  double rnorm;
  double norm_temp1, norm_temp2;

  double t, mflops, tmax;
  char Class;
  int verified;
  double zeta_verify_value, epsilon, err;

  char *t_names[T_last];

  for (i = 0; i < T_last; i++) {
    timer_clear(i);
  }

  FILE *fp;
  if ((fp = fopen("timer.flag", "r")) != NULL) {
    timeron = true;
    t_names[T_init] = "init";
    t_names[T_bench] = "benchmk";
    t_names[T_conj_grad] = "conjgd";
    fclose(fp);
  } else {
    timeron = false;
  }

  if ((fp = fopen("cg.input", "r")) != NULL) {
    int result;
    printf(" Reading from input file cg.input\n");
    result = fscanf(fp, "%d", &nit);
    while (fgetc(fp) != '\n');
    result = fscanf(fp, "%lf", &zeta_verify_value);
    while (fgetc(fp) != '\n');
    fclose(fp);
    set_zeta = false;
  } else {
    printf(" No input file. Using compiled defaults \n");
    nit = NITER;
    set_zeta = true;
  }

  timer_start(T_init);

  firstrow = 0;
  lastrow  = NA-1;
  firstcol = 0;
  lastcol  = NA-1;

  if (NA == 1400 && NONZER == 7 && nit == 15 && SHIFT == 10) {
    Class = 'S';
    if (set_zeta) zeta_verify_value = 8.5971775078648;
  } else if (NA == 7000 && NONZER == 8 && nit == 15 && SHIFT == 12) {
    Class = 'W';
    if (set_zeta) zeta_verify_value = 10.362595087124;
  } else if (NA == 14000 && NONZER == 11 && nit == 15 && SHIFT == 20) {
    Class = 'A';
    if (set_zeta) zeta_verify_value = 17.130235054029;
  } else if (NA == 75000 && NONZER == 13 && nit == 75 && SHIFT == 60) {
    Class = 'B';
    if (set_zeta) zeta_verify_value = 22.712745482631;
  } else if (NA == 150000 && NONZER == 15 && nit == 75 && SHIFT == 110) {
    Class = 'C';
    if (set_zeta) zeta_verify_value = 28.973605592845;
  } else if (NA == 1500000 && NONZER == 21 && nit == 100 && SHIFT == 500) {
    Class = 'D';
    if (set_zeta) zeta_verify_value = 52.514532105794;
  } else if (NA == 9000000 && NONZER == 26 && nit == 100 && SHIFT == 1500) {
    Class = 'E';
    if (set_zeta) zeta_verify_value = 77.522164599383;
  } else if (NA == 400000 && NONZER == 15 && nit == 2 && SHIFT == 110) {
    Class = 'T';
    if (set_zeta) zeta_verify_value = 28.5;
  } else if (NA == 400000 && NONZER == 15 && nit == 10 && SHIFT == 110) {
    Class = 'N';
    if (set_zeta) zeta_verify_value = 28.5;
  } else if (NA == 400000 && NONZER == 15 && nit == 100 && SHIFT == 110) {
    Class = 'R';
    if (set_zeta) zeta_verify_value = 28.5;
  } else {
    Class = 'U';
  }

  printf("\n\n NAS Parallel Benchmarks (NPB3.3-OPENMP-C) - CG Benchmark\n\n");
  printf(" Size: %11d\n", NA);
  printf(" Iterations: %5d\n", nit);
  printf("\n");

  naa = NA;
  nzz = NZ;

  #pragma omp target update to(naa,firstrow,lastrow,firstcol,lastcol)

  //---------------------------------------------------------------------
  // Inialize random number generator
  //---------------------------------------------------------------------
  tran    = 314159265.0;
  amult   = 1220703125.0;
  zeta    = randlc(&tran, amult);

  //---------------------------------------------------------------------
  //  
  //---------------------------------------------------------------------
  makea(naa, nzz, a, colidx, rowstr, 
        firstrow, lastrow, firstcol, lastcol, 
        arow, 
        (int (*)[NONZER+1])(void*)acol, 
        (double (*)[NONZER+1])(void*)aelt,
        iv);

  //---------------------------------------------------------------------
  // Note: as a result of the above call to makea:
  //      values of j used in indexing rowstr go from 0 --> lastrow-firstrow
  //      values of colidx which are col indexes go from firstcol --> lastcol
  //      So:
  //      Shift the col index vals from actual (firstcol --> lastcol ) 
  //      to local, i.e., (0 --> lastcol-firstcol)
  //---------------------------------------------------------------------
  for (j = 0; j < lastrow - firstrow + 1; j++) {
    for (k = rowstr[j]; k < rowstr[j+1]; k++) {
      colidx[k] = colidx[k] - firstcol;
    }
  }

  end = lastcol - firstcol + 1;

  //map(to:colidx[0:NZ],a[0:NZ], rowstr[0:NA+1]) map(alloc:x[0:NA+2],z[0:NA+2], p[0:NA+2],q[0:NA+2], r[0:NA+2])
  //Mapping of global arrays has no effect (but it has on pointers) so use target update
  brisbane_mem_create(NZ * sizeof(int), &mem_colidx);
  brisbane_mem_create(NZ * sizeof(double), &mem_a);
  brisbane_mem_create((NA + 1) * sizeof(int), &mem_rowstr);
  brisbane_mem_create((NA + 2) * sizeof(double), &mem_x);
  brisbane_mem_create((NA + 2) * sizeof(double), &mem_z);
  brisbane_mem_create((NA + 2) * sizeof(double), &mem_p);
  brisbane_mem_create((NA + 2) * sizeof(double), &mem_q);
  brisbane_mem_create((NA + 2) * sizeof(double), &mem_r);

  brisbane_task task16;
  brisbane_task_create(&task16);
  brisbane_task_h2d(task16, mem_colidx, 0, NZ * sizeof(int), colidx);
  brisbane_task_h2d(task16, mem_a, 0, NZ * sizeof(double), a);
  brisbane_task_h2d(task16, mem_rowstr, 0, (NA + 1) * sizeof(int), rowstr);
  brisbane_task_submit(task16, brisbane_gpu, NULL, true);

  #pragma omp target update to(colidx,a,rowstr)
    //---------------------------------------------------------------------
    // set starting vector to (1, 1, .... 1)
    //---------------------------------------------------------------------
      

  size_t kernel_loop0_off[1] = { 0 };
  size_t kernel_loop0_idx[1] = { NA + 1 };
  brisbane_kernel kernel_loop0;
  brisbane_kernel_create("loop0", &kernel_loop0);
  brisbane_kernel_setmem(kernel_loop0, 0, mem_x, brisbane_wr);

  brisbane_task task0;
  brisbane_task_create(&task0);
  brisbane_task_kernel(task0, kernel_loop0, 1, kernel_loop0_off, kernel_loop0_idx);
  brisbane_task_submit(task0, brisbane_gpu, NULL, true);

        /*
        #pragma omp target
        #pragma omp teams distribute parallel for simd 
        for (i = 0; i < NA+1; i++) {
          x[i] = 1.0;
        }
        */

  size_t kernel_loop1_off[1] = { 0 };
  size_t kernel_loop1_idx[1] = { NA + 1 };
  brisbane_kernel kernel_loop1;
  brisbane_kernel_create("loop1", &kernel_loop1);
  brisbane_kernel_setmem(kernel_loop1, 0, mem_q, brisbane_wr);
  brisbane_kernel_setmem(kernel_loop1, 1, mem_z, brisbane_wr);
  brisbane_kernel_setmem(kernel_loop1, 2, mem_r, brisbane_wr);
  brisbane_kernel_setmem(kernel_loop1, 3, mem_p, brisbane_wr);

  brisbane_task task1;
  brisbane_task_create(&task1);
  brisbane_task_kernel(task1, kernel_loop1, 1, kernel_loop1_off, kernel_loop1_idx);
  brisbane_task_submit(task1, brisbane_gpu, NULL, true);
  /*
        #pragma omp target
        #pragma omp teams distribute parallel for simd 
        for (j = 0; j < end; j++) {
          q[j] = 0.0;
          z[j] = 0.0;
          r[j] = 0.0;
          p[j] = 0.0;
        }
        */
      
    zeta = 0.0;

    //---------------------------------------------------------------------
    //---->
    // Do one iteration untimed to init all code and data page tables
    //---->                    (then reinit, start timing, to niter its)
    //---------------------------------------------------------------------
    for (it = 1; it <= 1; it++) {
      //---------------------------------------------------------------------
      // The call to the conjugate gradient routine:
      //---------------------------------------------------------------------
      conj_grad(colidx, rowstr, x, z, a, p, q, r, &rnorm);

      //---------------------------------------------------------------------
      // zeta = shift + 1/(x.z)
      // So, first: (x.z)
      // Also, find norm of z
      // So, first: (z.z)
      //---------------------------------------------------------------------
      
        norm_temp1 = 0.0;
        norm_temp2 = 0.0;

        brisbane_mem mem_norm_temp1;
        brisbane_mem mem_norm_temp2;
        brisbane_mem_create(sizeof(double), &mem_norm_temp1);
        brisbane_mem_create(sizeof(double), &mem_norm_temp2);
        brisbane_mem_reduce(mem_norm_temp1, brisbane_sum, brisbane_double);
        brisbane_mem_reduce(mem_norm_temp2, brisbane_sum, brisbane_double);

        size_t kernel_loop12_off[1] = { 0 };
        size_t kernel_loop12_idx[1] = { end };
        brisbane_kernel kernel_loop12;
        brisbane_kernel_create("loop12", &kernel_loop12);
        brisbane_kernel_setmem(kernel_loop1, 0, mem_x, brisbane_rd);
        brisbane_kernel_setmem(kernel_loop1, 1, mem_z, brisbane_rd);
        brisbane_kernel_setmem(kernel_loop1, 2, mem_norm_temp1, brisbane_rdwr);
        brisbane_kernel_setmem(kernel_loop1, 4, mem_norm_temp2, brisbane_rdwr);

        brisbane_task task12;
        brisbane_task_create(&task12);
        brisbane_task_kernel(task12, kernel_loop12, 1, kernel_loop12_off, kernel_loop12_idx);
        brisbane_task_d2h(task12, mem_norm_temp1, 0, sizeof(double), &norm_temp1);
        brisbane_task_d2h(task12, mem_norm_temp2, 0, sizeof(double), &norm_temp2);
        brisbane_task_submit(task12, brisbane_gpu, NULL, true);
        /*
        #pragma omp target map(tofrom: norm_temp1,norm_temp2)
        #pragma omp teams distribute parallel for simd reduction(+:norm_temp1,norm_temp2)
        for (j = 0; j < end; j++) {
          norm_temp1 = norm_temp1 + x[j] * z[j];
          norm_temp2 = norm_temp2 + z[j] * z[j];
        }
        */

        norm_temp2 = 1.0 / sqrt(norm_temp2);

        //---------------------------------------------------------------------
        // Normalize z to obtain x
        //---------------------------------------------------------------------
        size_t kernel_loop13_off[1] = { 0 };
        size_t kernel_loop13_idx[1] = { end };
        brisbane_kernel kernel_loop13;
        brisbane_kernel_create("loop13", &kernel_loop13);
        brisbane_kernel_setmem(kernel_loop13, 0, mem_x, brisbane_wr);
        brisbane_kernel_setmem(kernel_loop13, 1, mem_z, brisbane_rd);
        brisbane_kernel_setarg(kernel_loop13, 2, sizeof(double), &norm_temp2);

        brisbane_task task13;
        brisbane_task_create(&task13);
        brisbane_task_kernel(task13, kernel_loop13, 1, kernel_loop13_off, kernel_loop13_idx);
        brisbane_task_submit(task13, brisbane_gpu, NULL, true);
        /*
        #pragma omp target 
        #pragma omp teams distribute parallel for simd
        for (j = 0; j < end; j++) {     
          x[j] = norm_temp2 * z[j];
        }
        */
      
    } // end of do one iteration untimed

    //---------------------------------------------------------------------
    // set starting vector to (1, 1, .... 1)
    //---------------------------------------------------------------------
    size_t kernel_loop14_off[1] = { 0 };
    size_t kernel_loop14_idx[1] = { NA + 1 };
    brisbane_kernel kernel_loop14;
    brisbane_kernel_create("loop14", &kernel_loop14);
    brisbane_kernel_setmem(kernel_loop14, 0, mem_x, brisbane_wr);

    brisbane_task task14;
    brisbane_task_create(&task14);
    brisbane_task_kernel(task14, kernel_loop14, 1, kernel_loop14_off, kernel_loop14_idx);
    brisbane_task_submit(task14, brisbane_gpu, NULL, true);
    /*
    #pragma omp target
    #pragma omp teams distribute parallel for simd
    for (i = 0; i < NA+1; i++) {
      x[i] = 1.0;
    }
    */

    zeta = 0.0;

    timer_stop(T_init);

  #ifndef SPEC
   printf(" Initialization time = %15.3f seconds\n", timer_read(T_init));
  #endif

    timer_start(T_bench);

    //---------------------------------------------------------------------
    //---->
    // Main Iteration for inverse power method
    //---->
    //---------------------------------------------------------------------
    for (it = 1; it <= nit; it++) {
      //---------------------------------------------------------------------
      // The call to the conjugate gradient routine:
      //---------------------------------------------------------------------
      conj_grad(colidx, rowstr, x, z, a, p, q, r, &rnorm);

      //---------------------------------------------------------------------
      // zeta = shift + 1/(x.z)
      // So, first: (x.z)
      // Also, find norm of z
      // So, first: (z.z)
      //---------------------------------------------------------------------
      norm_temp1 = 0.0;
      norm_temp2 = 0.0;

      brisbane_mem mem_norm_temp1;
      brisbane_mem mem_norm_temp2;
      brisbane_mem_create(sizeof(double), &mem_norm_temp1);
      brisbane_mem_create(sizeof(double), &mem_norm_temp2);
      brisbane_mem_reduce(mem_norm_temp1, brisbane_sum, brisbane_double);
      brisbane_mem_reduce(mem_norm_temp2, brisbane_sum, brisbane_double);

      size_t kernel_loop2_off[1] = { 0 };
      size_t kernel_loop2_idx[1] = { end };
      brisbane_kernel kernel_loop2;
      brisbane_kernel_create("loop2", &kernel_loop2);
      brisbane_kernel_setmem(kernel_loop2, 0, mem_x, brisbane_rd);
      brisbane_kernel_setmem(kernel_loop2, 1, mem_z, brisbane_rd);
      brisbane_kernel_setmem(kernel_loop2, 2, mem_norm_temp1, brisbane_rdwr);
      brisbane_kernel_setmem(kernel_loop2, 4, mem_norm_temp2, brisbane_rdwr);

      brisbane_task task2;
      brisbane_task_create(&task2);
      brisbane_task_kernel(task2, kernel_loop2, 1, kernel_loop2_off, kernel_loop2_idx);
      brisbane_task_d2h(task2, mem_norm_temp1, 0, sizeof(double), &norm_temp1);
      brisbane_task_d2h(task2, mem_norm_temp2, 0, sizeof(double), &norm_temp2);
      brisbane_task_submit(task2, brisbane_gpu, NULL, true);
      /*
      #pragma omp target map(tofrom: norm_temp1,norm_temp2)
      #pragma omp teams distribute parallel for simd reduction(+:norm_temp1,norm_temp2)
      for (j = 0; j < end; j++) {
        norm_temp1 = norm_temp1 + x[j]*z[j];
        norm_temp2 = norm_temp2 + z[j]*z[j];
      }
      */

      norm_temp2 = 1.0 / sqrt(norm_temp2);
      zeta = SHIFT + 1.0 / norm_temp1;

      if (it == 1) 
        printf("\n   iteration           ||r||                 zeta\n");
      printf("    %5d       %20.14E%20.13f\n", it, rnorm, zeta);

      //---------------------------------------------------------------------
      // Normalize z to obtain x
      //---------------------------------------------------------------------
      size_t kernel_loop3_off[1] = { 0 };
      size_t kernel_loop3_idx[1] = { end };
      brisbane_kernel kernel_loop3;
      brisbane_kernel_create("loop3", &kernel_loop3);
      brisbane_kernel_setmem(kernel_loop3, 0, mem_x, brisbane_wr);
      brisbane_kernel_setmem(kernel_loop3, 1, mem_z, brisbane_rd);
      brisbane_kernel_setarg(kernel_loop3, 2, sizeof(double), &norm_temp2);

      brisbane_task task3;
      brisbane_task_create(&task3);
      brisbane_task_kernel(task3, kernel_loop3, 1, kernel_loop3_off, kernel_loop3_idx);
      brisbane_task_submit(task3, brisbane_gpu, NULL, true);
      /*
      #pragma omp target
      #pragma omp teams distribute parallel for simd 
      for (j = 0; j < end; j++) {
        x[j] = norm_temp2 * z[j];
      }
      */
    } // end of main iter inv pow meth

    timer_stop(T_bench);

  //---------------------------------------------------------------------
  // End of timed section
  //---------------------------------------------------------------------

  t = timer_read(T_bench);

  printf(" Benchmark completed\n");

  epsilon = 1.0e-10;
  if (Class != 'U') {
    err = fabs(zeta - zeta_verify_value) / zeta_verify_value;
    if (err <= epsilon) {
      verified = true;
      printf(" VERIFICATION SUCCESSFUL\n");
      printf(" Zeta is    %22.16E\n", zeta);
      printf(" Error is   %22.16E\n", err);
    } else {
      verified = false;
      printf(" VERIFICATION FAILED\n");
      printf(" Zeta                %22.16E\n", zeta);
      printf(" The correct zeta is %22.16E\n", zeta_verify_value);
    }
  } else {
    verified = false;
    printf(" Problem size unknown\n");
    printf(" NO VERIFICATION PERFORMED\n");
  }

  if (t != 0.0) {
    mflops = (double)(2*nit*NA)
                   * (3.0+(double)(NONZER*(NONZER+1))
                     + 25.0*(5.0+(double)(NONZER*(NONZER+1)))
                     + 3.0) / t / 1000000.0;
  } else {
    mflops = 0.0;
  }

  print_results("CG", Class, NA, 0, 0,
                nit, t,
                mflops, "          floating point", 
                verified, NPBVERSION, COMPILETIME,
                CS1, CS2, CS3, CS4, CS5, CS6, CS7);

  //---------------------------------------------------------------------
  // More timers
  //---------------------------------------------------------------------
  if (timeron) {
    tmax = timer_read(T_bench);
    if (tmax == 0.0) tmax = 1.0;
    printf("  SECTION   Time (secs)\n");
    for (i = 0; i < T_last; i++) {
      t = timer_read(i);
      if (i == T_init) {
        printf("  %8s:%9.3f\n", t_names[i], t);
      } else {
        printf("  %8s:%9.3f  (%6.2f%%)\n", t_names[i], t, t*100.0/tmax);
        if (i == T_conj_grad) {
          t = tmax - t;
          printf("    --> %8s:%9.3f  (%6.2f%%)\n", "rest", t, t*100.0/tmax);
        }
      }
    }
  }

  brisbane_finalize();
  return 0;
}


//---------------------------------------------------------------------
// Floaging point arrays here are named as in NPB1 spec discussion of 
// CG algorithm
//---------------------------------------------------------------------
static void conj_grad(int colidx[],
                      int rowstr[],
                      double x[],
                      double z[],
                      double a[],
                      double p[],
                      double q[],
                      double r[],
                      double *rnorm)
{
  double sum;
  //pcopyin(colidx[0:NZ],rowstr[0:NA+1],x[0:NA+2],a[0:NZ]), pcopyout(z[0:NA+2],p[0:NA+2],q[0:NA+2],r[0:NA+2])
  #pragma omp target data map(alloc:colidx[0:NZ],rowstr[0:NA+1],x[0:NA+2],a[0:NZ],z[0:NA+2],p[0:NA+2],q[0:NA+2],r[0:NA+2])
  {
    int j, k,tmp1,tmp2,tmp3;
    int end;
    int cgit, cgitmax = 25;
    double d, rho, rho0, alpha, beta;
    double sum_array[NA+2];

    brisbane_mem mem_d;
    brisbane_mem_create(sizeof(double), &mem_d);
    brisbane_mem_reduce(mem_d, brisbane_sum, brisbane_double);

    brisbane_mem mem_rho;
    brisbane_mem_create(sizeof(double), &mem_rho);
    brisbane_mem_reduce(mem_rho, brisbane_sum, brisbane_double);

    rho = 0.0;
    //---------------------------------------------------------------------
    // Initialize the CG algorithm:
    //---------------------------------------------------------------------
      size_t kernel_loop4_off[1] = { 0 };
      size_t kernel_loop4_idx[1] = { naa + 1 };
      brisbane_kernel kernel_loop4;
      brisbane_kernel_create("loop4", &kernel_loop4);
      brisbane_kernel_setmem(kernel_loop4, 0, mem_x, brisbane_rd);
      brisbane_kernel_setmem(kernel_loop4, 1, mem_q, brisbane_wr);
      brisbane_kernel_setmem(kernel_loop4, 2, mem_z, brisbane_wr);
      brisbane_kernel_setmem(kernel_loop4, 3, mem_r, brisbane_rdwr);
      brisbane_kernel_setmem(kernel_loop4, 4, mem_p, brisbane_wr);

      brisbane_task task4;
      brisbane_task_create(&task4);
      brisbane_task_kernel(task4, kernel_loop4, 1, kernel_loop4_off, kernel_loop4_idx);
      brisbane_task_submit(task4, brisbane_gpu, NULL, true);
      /*
      #pragma omp target map(q[:0],z[:0],r[:0],x[:0],p[:0])
      #pragma omp teams distribute parallel for simd
      for (j = 0; j < naa+1; j++) {
        q[j] = 0.0;
        z[j] = 0.0;
        r[j] = x[j];
        p[j] = r[j];
      }
      */

      //---------------------------------------------------------------------
      // rho = r.r
      // Now, obtain the norm of r: First, sum squares of r elements locally...
      //---------------------------------------------------------------------

      size_t kernel_loop5_off[1] = { 0 };
      size_t kernel_loop5_idx[1] = { lastcol - firstcol + 1 };
      brisbane_kernel kernel_loop5;
      brisbane_kernel_create("loop5", &kernel_loop5);
      brisbane_kernel_setmem(kernel_loop5, 0, mem_r, brisbane_rd);
      brisbane_kernel_setmem(kernel_loop5, 1, mem_rho, brisbane_rdwr);

      brisbane_task task5;
      brisbane_task_create(&task5);
      brisbane_task_kernel(task5, kernel_loop5, 1, kernel_loop5_off, kernel_loop5_idx);
      brisbane_task_d2h(task5, mem_rho, 0, sizeof(double), &rho);
      brisbane_task_submit(task5, brisbane_gpu, NULL, true);
      /*
      #pragma omp target map(tofrom: rho) map(r[:0])
      #pragma omp teams distribute parallel for simd reduction(+:rho) 
      for (j = 0; j < lastcol - firstcol + 1; j++) {
        rho = rho + r[j]*r[j];
      }
      */

    //---------------------------------------------------------------------
    //---->
    // The conj grad iteration loop
    //---->
    //---------------------------------------------------------------------
    for (cgit = 1; cgit <= cgitmax; cgit++) {
      //---------------------------------------------------------------------
      // q = A.p
      // The partition submatrix-vector multiply: use workspace w
      //---------------------------------------------------------------------
      //
      // NOTE: this version of the multiply is actually (slightly: maybe %5) 
      //       faster on the sp2 on 16 nodes than is the unrolled-by-2 version 
      //       below.   On the Cray t3d, the reverse is true, i.e., the 
      //       unrolled-by-two version is some 10% faster.  
      //       The unrolled-by-8 version below is significantly faster
      //       on the Cray t3d - overall speed of code is 1.5 times faster.
      end = lastrow - firstrow + 1;

      size_t kernel_loop6_off[1] = { 0 };
      size_t kernel_loop6_idx[1] = { end };
      brisbane_kernel kernel_loop6;
      brisbane_kernel_create("loop6", &kernel_loop6);
      brisbane_kernel_setmem(kernel_loop6, 0, mem_rowstr, brisbane_rd);
      brisbane_kernel_setmem(kernel_loop6, 1, mem_colidx, brisbane_rd);
      brisbane_kernel_setmem(kernel_loop6, 2, mem_a, brisbane_rd);
      brisbane_kernel_setmem(kernel_loop6, 3, mem_p, brisbane_rd);
      brisbane_kernel_setmem(kernel_loop6, 4, mem_q, brisbane_wr);

      brisbane_task task6;
      brisbane_task_create(&task6);
      brisbane_task_kernel(task6, kernel_loop6, 1, kernel_loop6_off, kernel_loop6_idx);
      brisbane_task_submit(task6, brisbane_gpu, NULL, true);
      /*
      #pragma omp target map(rowstr[:0],colidx[:0],a[:0],p[:0],q[:0])
      #pragma omp teams distribute parallel for private(tmp1,tmp2,sum)
      for (j = 0; j < end; j++) {
         tmp1 = rowstr[j];
         tmp2 = rowstr[j+1];
         sum = 0.0;
         #pragma omp simd reduction(+:sum) private(tmp3)
         for (k = tmp1; k < tmp2; k++) {
            tmp3 = colidx[k];
            sum = sum + a[k]*p[tmp3];
         }
         q[j] = sum;
      }
      */

      //---------------------------------------------------------------------
      // Obtain p.q
      //---------------------------------------------------------------------
      d = 0.0;
      end = lastcol - firstcol + 1;

      size_t kernel_loop7_off[1] = { 0 };
      size_t kernel_loop7_idx[1] = { end };
      brisbane_kernel kernel_loop7;
      brisbane_kernel_create("loop7", &kernel_loop7);
      brisbane_kernel_setmem(kernel_loop7, 0, mem_p, brisbane_rd);
      brisbane_kernel_setmem(kernel_loop7, 1, mem_q, brisbane_rd);
      brisbane_kernel_setmem(kernel_loop7, 2, mem_d, brisbane_rdwr);

      brisbane_task task7;
      brisbane_task_create(&task7);
      brisbane_task_kernel(task7, kernel_loop7, 1, kernel_loop7_off, kernel_loop7_idx);
      brisbane_task_d2h(task7, mem_d, 0, sizeof(double), &d);
      brisbane_task_submit(task7, brisbane_gpu, NULL, true);
      /*
      #pragma omp target map(tofrom: d) map(p[:0],q[:0])
      #pragma omp teams distribute parallel for simd reduction(+:d)
      for (j = 0; j < end; j++) {
        d = d + p[j]*q[j];
      }
      */

      //---------------------------------------------------------------------
      // Obtain alpha = rho / (p.q)
      //---------------------------------------------------------------------
      alpha = rho / d;

      //---------------------------------------------------------------------
      // Save a temporary of rho
      //---------------------------------------------------------------------
      rho0 = rho;

      //---------------------------------------------------------------------
      // Obtain z = z + alpha*p
      // and    r = r - alpha*q
      //---------------------------------------------------------------------
      rho = 0.0;
      size_t kernel_loop8_off[1] = { 0 };
      size_t kernel_loop8_idx[1] = { end };
      brisbane_kernel kernel_loop8;
      brisbane_kernel_create("loop8", &kernel_loop8);
      brisbane_kernel_setmem(kernel_loop8, 0, mem_z, brisbane_rdwr);
      brisbane_kernel_setmem(kernel_loop8, 1, mem_p, brisbane_rd);
      brisbane_kernel_setmem(kernel_loop8, 2, mem_r, brisbane_rdwr);
      brisbane_kernel_setmem(kernel_loop8, 3, mem_q, brisbane_rd);
      brisbane_kernel_setarg(kernel_loop8, 4, sizeof(double), &alpha);

      brisbane_task task8;
      brisbane_task_create(&task8);
      brisbane_task_kernel(task8, kernel_loop8, 1, kernel_loop8_off, kernel_loop8_idx);
      brisbane_task_submit(task8, brisbane_gpu, NULL, true);
      /*
      #pragma omp target map(z[:0],p[:0],r[:0],q[:0])
      #pragma omp teams distribute parallel for simd
      for (j = 0; j < end; j++) {
        z[j] = z[j] + alpha*p[j];
        r[j] = r[j] - alpha*q[j];
      }
      */
              
      //---------------------------------------------------------------------
      // rho = r.r
      // Now, obtain the norm of r: First, sum squares of r elements locally...
      //---------------------------------------------------------------------
      size_t kernel_loop9_off[1] = { 0 };
      size_t kernel_loop9_idx[1] = { end };
      brisbane_kernel kernel_loop9;
      brisbane_kernel_create("loop9", &kernel_loop9);
      brisbane_kernel_setmem(kernel_loop9, 0, mem_r, brisbane_rd);
      brisbane_kernel_setmem(kernel_loop9, 1, mem_rho, brisbane_rdwr);

      brisbane_task task9;
      brisbane_task_create(&task9);
      brisbane_task_kernel(task9, kernel_loop9, 1, kernel_loop9_off, kernel_loop9_idx);
      brisbane_task_d2h(task9, mem_rho, 0, sizeof(double), &rho);
      brisbane_task_submit(task9, brisbane_gpu, NULL, true);
      /*
      #pragma omp target map(tofrom: rho) map(r[:0])
      #pragma omp teams distribute parallel for simd reduction(+:rho)
      for (j = 0; j < end; j++) {
        rho = rho + r[j]*r[j];
      }
      */

      //---------------------------------------------------------------------
      // Obtain beta:
      //---------------------------------------------------------------------
      beta = rho / rho0;

      //---------------------------------------------------------------------
      // p = r + beta*p
      //---------------------------------------------------------------------
      size_t kernel_loop10_off[1] = { 0 };
      size_t kernel_loop10_idx[1] = { end };
      brisbane_kernel kernel_loop10;
      brisbane_kernel_create("loop10", &kernel_loop10);
      brisbane_kernel_setmem(kernel_loop10, 0, mem_p, brisbane_rdwr);
      brisbane_kernel_setmem(kernel_loop10, 1, mem_r, brisbane_rd);
      brisbane_kernel_setarg(kernel_loop10, 2, sizeof(double), &beta);

      brisbane_task task10;
      brisbane_task_create(&task10);
      brisbane_task_kernel(task10, kernel_loop10, 1, kernel_loop10_off, kernel_loop10_idx);
      brisbane_task_submit(task10, brisbane_gpu, NULL, true);
      /*
      #pragma omp target map(p[:0],r[:0])
      #pragma omp teams distribute parallel for simd 
      for (j = 0; j < end; j++) {
        p[j] = r[j] + beta*p[j];
      }
      */
    } // end of do cgit=1,cgitmax

    //---------------------------------------------------------------------
    // Compute residual norm explicitly:  ||r|| = ||x - A.z||
    // First, form A.z
    // The partition submatrix-vector multiply
    //---------------------------------------------------------------------

    end = lastrow - firstrow + 1;
    size_t kernel_loop11_off[1] = { 0 };
    size_t kernel_loop11_idx[1] = { end };
    brisbane_kernel kernel_loop11;
    brisbane_kernel_create("loop11", &kernel_loop11);
    brisbane_kernel_setmem(kernel_loop11, 0, mem_rowstr, brisbane_rd);
    brisbane_kernel_setmem(kernel_loop11, 1, mem_colidx, brisbane_rd);
    brisbane_kernel_setmem(kernel_loop11, 2, mem_a, brisbane_rd);
    brisbane_kernel_setmem(kernel_loop11, 3, mem_z, brisbane_rd);
    brisbane_kernel_setmem(kernel_loop11, 4, mem_r, brisbane_wr);

    brisbane_task task11;
    brisbane_task_create(&task11);
    brisbane_task_kernel(task11, kernel_loop11, 1, kernel_loop11_off, kernel_loop11_idx);
    brisbane_task_submit(task11, brisbane_gpu, NULL, true);
    /*
    #pragma omp target map(rowstr[:0],colidx[:0],a[:0],z[:0],r[:0])
    #pragma omp teams distribute parallel for private(tmp1,tmp2,d)
    for (j = 0; j < end; j++) {
      tmp1=rowstr[j];
      tmp2=rowstr[j+1];
      d = 0.0;
      #pragma omp simd reduction(+:d) private(tmp3)
      for (k = tmp1; k < tmp2; k++) {
          tmp3=colidx[k];
          d = d + a[k]*z[tmp3];
      }
      r[j] = d;
    }
    */

    //---------------------------------------------------------------------
    // At this point, r contains A.z
    //---------------------------------------------------------------------
    sum = 0.0;

    brisbane_mem mem_sum;
    brisbane_mem_create(sizeof(double), &mem_sum);
    brisbane_mem_reduce(mem_sum, brisbane_sum, brisbane_double);

    size_t kernel_loop15_off[1] = { 0 };
    size_t kernel_loop15_idx[1] = { lastcol - firstcol + 1};
    brisbane_kernel kernel_loop15;
    brisbane_kernel_create("loop15", &kernel_loop15);
    brisbane_kernel_setmem(kernel_loop15, 0, mem_x, brisbane_rd);
    brisbane_kernel_setmem(kernel_loop15, 1, mem_r, brisbane_rd);
    brisbane_kernel_setmem(kernel_loop15, 2, mem_sum, brisbane_rdwr);

    brisbane_task task15;
    brisbane_task_create(&task15);
    brisbane_task_kernel(task15, kernel_loop15, 1, kernel_loop15_off, kernel_loop15_idx);
    brisbane_task_d2h(task15, mem_sum, 0, sizeof(double), &sum);
    brisbane_task_submit(task15, brisbane_gpu, NULL, true);
    /*
    #pragma omp target map(tofrom: sum) map(x[:0],r[:0])
    #pragma omp teams distribute parallel for simd reduction(+:sum) private(d)
    for (j = 0; j < lastcol-firstcol+1; j++) {
      d   = x[j] - r[j];
      sum = sum + d*d;
    }
    */

  }//end omp target data
  *rnorm = sqrt(sum);
}


//---------------------------------------------------------------------
// generate the test problem for benchmark 6
// makea generates a sparse matrix with a
// prescribed sparsity distribution
//
// parameter    type        usage
//
// input
//
// n            i           number of cols/rows of matrix
// nz           i           nonzeros as declared array size
// rcond        r*8         condition number
// shift        r*8         main diagonal shift
//
// output
//
// a            r*8         array for nonzeros
// colidx       i           col indices
// rowstr       i           row pointers
//
// workspace
//
// iv, arow, acol i
// aelt           r*8
//---------------------------------------------------------------------
static void makea(int n,
                  int nz,
                  double a[],
                  int colidx[],
                  int rowstr[],
                  int firstrow,
                  int lastrow,
                  int firstcol,
                  int lastcol,
                  int arow[],
                  int acol[][NONZER+1],
                  double aelt[][NONZER+1],
                  int iv[])
{
  int iouter, ivelt, nzv, nn1;
  int ivc[NONZER+1];
  double vc[NONZER+1];

  //---------------------------------------------------------------------
  // nonzer is approximately  (int(sqrt(nnza /n)));
  //---------------------------------------------------------------------

  //---------------------------------------------------------------------
  // nn1 is the smallest power of two not less than n
  //---------------------------------------------------------------------
  nn1 = 1;
  do {
    nn1 = 2 * nn1;
  } while (nn1 < n);

  //---------------------------------------------------------------------
  // Generate nonzero positions and save for the use in sparse.
  //---------------------------------------------------------------------
  for (iouter = 0; iouter < n; iouter++) {
    nzv = NONZER;
    sprnvc(n, nzv, nn1, vc, ivc);
    vecset(n, vc, ivc, &nzv, iouter+1, 0.5);
    arow[iouter] = nzv;
    
    for (ivelt = 0; ivelt < nzv; ivelt++) {
      acol[iouter][ivelt] = ivc[ivelt] - 1;
      aelt[iouter][ivelt] = vc[ivelt];
    }
  }

  //---------------------------------------------------------------------
  // ... make the sparse matrix from list of elements with duplicates
  //     (iv is used as  workspace)
  //---------------------------------------------------------------------
  sparse(a, colidx, rowstr, n, nz, NONZER, arow, acol, 
         aelt, firstrow, lastrow,
         iv, RCOND, SHIFT);
}


//---------------------------------------------------------------------
// rows range from firstrow to lastrow
// the rowstr pointers are defined for nrows = lastrow-firstrow+1 values
//---------------------------------------------------------------------
static void sparse(double a[],
                   int colidx[],
                   int rowstr[],
                   int n,
                   int nz,
                   int nozer,
                   int arow[],
                   int acol[][NONZER+1],
                   double aelt[][NONZER+1],
                   int firstrow,
                   int lastrow,
                   int nzloc[],
                   double rcond,
                   double shift)
{
  int nrows;

  //---------------------------------------------------
  // generate a sparse matrix from a list of
  // [col, row, element] tri
  //---------------------------------------------------
  int i, j, j1, j2, nza, k, kk, nzrow, jcol;
  double size, scale, ratio, va;
  int cont40;

  //---------------------------------------------------------------------
  // how many rows of result
  //---------------------------------------------------------------------
  nrows = lastrow - firstrow + 1;

  //---------------------------------------------------------------------
  // ...count the number of triples in each row
  //---------------------------------------------------------------------
  for (j = 0; j < nrows+1; j++) {
    rowstr[j] = 0;
  }

  for (i = 0; i < n; i++) {
    for (nza = 0; nza < arow[i]; nza++) {
      j = acol[i][nza] + 1;
      rowstr[j] = rowstr[j] + arow[i];
    }
  }

  rowstr[0] = 0;
  for (j = 1; j < nrows+1; j++) {
    rowstr[j] = rowstr[j] + rowstr[j-1];
  }
  nza = rowstr[nrows] - 1;

  //---------------------------------------------------------------------
  // ... rowstr(j) now is the location of the first nonzero
  //     of row j of a
  //---------------------------------------------------------------------
  if (nza > nz) {
    printf("Space for matrix elements exceeded in sparse\n");
    printf("nza, nzmax = %d, %d\n", nza, nz);
    exit(EXIT_FAILURE);
  }

  //---------------------------------------------------------------------
  // ... preload data pages
  //---------------------------------------------------------------------
  for (j = 0; j < nrows; j++) {
    for (k = rowstr[j]; k < rowstr[j+1]; k++) {
      a[k] = 0.0;
      colidx[k] = -1;
    }
    nzloc[j] = 0;
  }

  //---------------------------------------------------------------------
  // ... generate actual values by summing duplicates
  //---------------------------------------------------------------------
  size = 1.0;
  ratio = pow(rcond, (1.0 / (double)(n)));

  for (i = 0; i < n; i++) {
    for (nza = 0; nza < arow[i]; nza++) {
      j = acol[i][nza];

      scale = size * aelt[i][nza];
      for (nzrow = 0; nzrow < arow[i]; nzrow++) {
        jcol = acol[i][nzrow];
        va = aelt[i][nzrow] * scale;

        //--------------------------------------------------------------------
        // ... add the identity * rcond to the generated matrix to bound
        //     the smallest eigenvalue from below by rcond
        //--------------------------------------------------------------------
        if (jcol == j && j == i) {
          va = va + rcond - shift;
        }

        cont40 = 0;
        for (k = rowstr[j]; k < rowstr[j+1]; k++) {
          if (colidx[k] > jcol) {
            //----------------------------------------------------------------
            // ... insert colidx here orderly
            //----------------------------------------------------------------
            for (kk = rowstr[j+1]-2; kk >= k; kk--) {
              if (colidx[kk] > -1) {
                a[kk+1]  = a[kk];
                colidx[kk+1] = colidx[kk];
              }
            }
            colidx[k] = jcol;
            a[k]  = 0.0;
            cont40 = 1;
            break;
          } else if (colidx[k] == -1) {
            colidx[k] = jcol;
            cont40 = 1;
            break;
          } else if (colidx[k] == jcol) {
            //--------------------------------------------------------------
            // ... mark the duplicated entry
            //--------------------------------------------------------------
            nzloc[j] = nzloc[j] + 1;
            cont40 = 1;
            break;
          }
        }
        if (cont40 == 0) {
          printf("internal error in sparse: i=%d\n", i);
          exit(EXIT_FAILURE);
        }
        a[k] = a[k] + va;
      }
    }
    size = size * ratio;
  }

  //---------------------------------------------------------------------
  // ... remove empty entries and generate final results
  //---------------------------------------------------------------------
  for (j = 1; j < nrows; j++) {
    nzloc[j] = nzloc[j] + nzloc[j-1];
  }

  for (j = 0; j < nrows; j++) {
    if (j > 0) {
      j1 = rowstr[j] - nzloc[j-1];
    } else {
      j1 = 0;
    }
    j2 = rowstr[j+1] - nzloc[j];
    nza = rowstr[j];
    for (k = j1; k < j2; k++) {
      a[k] = a[nza];
      colidx[k] = colidx[nza];
      nza = nza + 1;
    }
  }
  for (j = 1; j < nrows+1; j++) {
    rowstr[j] = rowstr[j] - nzloc[j-1];
  }
  nza = rowstr[nrows] - 1;
}


//---------------------------------------------------------------------
// generate a sparse n-vector (v, iv)
// having nzv nonzeros
//
// mark(i) is set to 1 if position i is nonzero.
// mark is all zero on entry and is reset to all zero before exit
// this corrects a performance bug found by John G. Lewis, caused by
// reinitialization of mark on every one of the n calls to sprnvc
//---------------------------------------------------------------------
static void sprnvc(int n, int nz, int nn1, double v[], int iv[])
{
  int nzv, ii, i;
  double vecelt, vecloc;

  nzv = 0;

  while (nzv < nz) {
    vecelt = randlc(&tran, amult);

    //---------------------------------------------------------------------
    // generate an integer between 1 and n in a portable manner
    //---------------------------------------------------------------------
    vecloc = randlc(&tran, amult);
    i = icnvrt(vecloc, nn1) + 1;
    if (i > n) continue;

    //---------------------------------------------------------------------
    // was this integer generated already?
    //---------------------------------------------------------------------
    int was_gen = 0;
    for (ii = 0; ii < nzv; ii++) {
      if (iv[ii] == i) {
        was_gen = 1;
        break;
      }
    }
    if (was_gen) continue;
    v[nzv] = vecelt;
    iv[nzv] = i;
    nzv = nzv + 1;
  }
}


//---------------------------------------------------------------------
// scale a double precision number x in (0,1) by a power of 2 and chop it
//---------------------------------------------------------------------
static int icnvrt(double x, int ipwr2)
{
  return (int)(ipwr2 * x);
}


//---------------------------------------------------------------------
// set ith element of sparse vector (v, iv) with
// nzv nonzeros to val
//---------------------------------------------------------------------
static void vecset(int n, double v[], int iv[], int *nzv, int i, double val)
{
  int k;
  int set;

  set = 0;
  for (k = 0; k < *nzv; k++) {
    if (iv[k] == i) {
      v[k] = val;
      set  = 1;
    }
  }
  if (set == 0) {
    v[*nzv]  = val;
    iv[*nzv] = i;
    *nzv     = *nzv + 1;
  }
}

