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

#include "header.h"
#include <math.h>

//---------------------------------------------------------------------
// this function computes the norm of the difference between the
// computed solution and the exact solution
//---------------------------------------------------------------------
void error_norm(double rms[5])
{
  int i, j, k, m, d;
  double xi, eta, zeta, u_exact[5], add;

  for (m = 0; m < 5; m++) {
    rms[m] = 0.0;
  }

  for (k = 0; k <= grid_points[2]-1; k++) {
    zeta = (double)k * dnzm1;
    for (j = 0; j <= grid_points[1]-1; j++) {
      eta = (double)j * dnym1;
      for (i = 0; i <= grid_points[0]-1; i++) {
        xi = (double)i * dnxm1;
        exact_solution(xi, eta, zeta, u_exact);

        for (m = 0; m < 5; m++) {
          add = u[m][k][j][i]-u_exact[m];
          rms[m] = rms[m] + add*add;
        }
      }
    }
  }

  for (m = 0; m < 5; m++) {
    for (d = 0; d < 3; d++) {
      rms[m] = rms[m] / (double)(grid_points[d]-2);
    }
    rms[m] = sqrt(rms[m]);
  }
}


void rhs_norm(double rms[5])
{
  int i, j, k, d, m;
  double add;
  double rms0, rms1,rms2,rms3,rms4;

  rms0=0.0;
  rms1=0.0;
  rms2=0.0;
  rms3=0.0;
  rms4=0.0;

#pragma omp target map(tofrom: rms0,rms1,rms2,rms3,rms4) //present(rhs)
#ifdef SPEC_USE_INNER_SIMD
#pragma omp teams distribute parallel for collapse(2) reduction(+:rms0,rms1,rms2,rms3,rms4)
#else
#pragma omp teams distribute parallel for simd collapse(3) reduction(+:rms0,rms1,rms2,rms3,rms4) private(add)
#endif
  for (k = 1; k <= nz2; k++) {
    for (j = 1; j <= ny2; j++) {
#ifdef SPEC_USE_INNER_SIMD
      #pragma omp simd reduction(+:rms0,rms1,rms2,rms3,rms4) private(add)
#endif
      for (i = 1; i <= nx2; i++) {
        add = rhs[0][k][j][i];
        rms0 = rms0 + add*add;
        add = rhs[1][k][j][i];
        rms1 = rms1 + add*add;
        add = rhs[2][k][j][i];
        rms2 = rms2 + add*add;
        add = rhs[3][k][j][i];
        rms3 = rms3 + add*add;
        add = rhs[4][k][j][i];
        rms4 = rms4 + add*add;
      } 
    } 
  }

  rms[0]=rms0;
  rms[1]=rms1;
  rms[2]=rms2;
  rms[3]=rms3;
  rms[4]=rms4;
  for (m = 0; m < 5; m++) {
    for (d = 0; d < 3; d++) {
      rms[m] = rms[m] / (double)(grid_points[d]-2);
    }
    rms[m] = sqrt(rms[m]);
  }

  

}

