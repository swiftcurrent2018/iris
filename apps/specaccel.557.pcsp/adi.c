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

}

void pinvr()
{
  int i, j, k;
  double r1, r2, r3, r4, r5, t1, t2;

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

}

void tzetar()
{
  int i, j, k;
  double t1, t2, t3, ac, xvel, yvel, zvel, r1, r2, r3, r4, r5;
  double btuz, ac2u, uzik1;

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


  #pragma omp target data map(alloc:lhsX[:][:][:][:],lhspX[:][:][:][:], lhsmX[:][:][:][:],rhonX[:][:][:],rhsX[:][:][:][:]) //present(rho_i,us,speed,rhs)
  {

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

    //---------------------------------------------------------------------
    // Computes the left hand side for the three x-factors  
    //---------------------------------------------------------------------
  
    //---------------------------------------------------------------------
    // first fill the lhs for the u-eigenvalue                   
    //---------------------------------------------------------------------
  #pragma omp target teams distribute parallel for collapse(2) private(i,j,k,ru1)
  for (k = 1; k <= nz2; k++) {
    for (j = 1; j <= ny2; j++) {
      #pragma omp simd private(ru1)
      for (i = 0; i <= gp01; i++) {
        ru1 = c3c4*rho_i[k][j][i];
        //cv[i] = us[k][j][i];
        rhonX[k][j][i] = max(max(dx2+con43*ru1,dx5+c1c5*ru1), max(dxmax+ru1,dx1));
      }
    #pragma omp simd
    for (i = 1; i <= nx2; i++) {
        lhsX[0][k][i][j] =  0.0;
      //      lhsX[1][k][i][j] = -dttx2 * cv[i-1] - dttx1 * rhon[i-1];
        lhsX[1][k][i][j] = -dttx2 * us[k][j][i-1] - dttx1 * rhonX[k][j][i-1];
        lhsX[2][k][i][j] =  1.0 + c2dttx1 * rhonX[k][j][i];
      //       lhsX[3][k][i][j] =  dttx2 * cv[i+1] - dttx1 * rhon[i+1];
        lhsX[3][k][i][j] =  dttx2 * us[k][j][i+1] - dttx1 * rhonX[k][j][i+1];
        lhsX[4][k][i][j] =  0.0;
      }
    }
  }

    //---------------------------------------------------------------------
    // add fourth order dissipation                             
    //---------------------------------------------------------------------
  i = 1;
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

  i = gp03;
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

    //---------------------------------------------------------------------
    // subsequently, fill the other factors (u+c), (u-c) by adding to 
    // the first  
    //---------------------------------------------------------------------
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

    //---------------------------------------------------------------------
    // FORWARD ELIMINATION  
    //---------------------------------------------------------------------

    //---------------------------------------------------------------------
    // perform the Thomas algorithm; first, FORWARD ELIMINATION     
    //---------------------------------------------------------------------
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

    //---------------------------------------------------------------------
    // The last two rows in this grid block are a bit different, 
    // since they for (not have two more rows available for the
    // elimination of off-diagonal entries
    //---------------------------------------------------------------------
  i  = gp02;
  i1 = gp01;
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

    //---------------------------------------------------------------------
    // for (the u+c and the u-c factors                 
    //---------------------------------------------------------------------
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

    //---------------------------------------------------------------------
    // And again the last two rows separately
    //---------------------------------------------------------------------
  i  = gp02;
  i1 = gp01;
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

    //---------------------------------------------------------------------
    // BACKSUBSTITUTION 
    //---------------------------------------------------------------------
  i  = gp02;
  i1 = gp01;
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

    //---------------------------------------------------------------------
    // The first three factors
    //---------------------------------------------------------------------
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

#pragma omp target data map(alloc:lhsY[:][:][:][:],lhspY[:][:][:][:],lhsmY[:][:][:][:],rhoqY[:][:][:]) //present(rho_i,vs,speed,rhs)
{
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

    //---------------------------------------------------------------------
    // Computes the left hand side for the three y-factors   
    //---------------------------------------------------------------------

    //---------------------------------------------------------------------
    // first fill the lhs for the u-eigenvalue         
    //---------------------------------------------------------------------
  #pragma omp target teams distribute parallel for collapse(2) private(i,j,k,ru1)
  for (k = 1; k <= nz2; k++) {
    for (i = 1; i <= gp0-2; i++) {
      #pragma omp simd private(ru1)
      for (j = 0; j <= gp1-1; j++) {
        ru1 = c3c4*rho_i[k][j][i];
   //     cv[j] = vs[k][j][i];
        rhoqY[k][j][i] = max(max(dy3+con43*ru1, dy5+c1c5*ru1), max(dymax+ru1, dy1));
      }
      #pragma omp simd
      for (j = 1; j <= gp1-2; j++) {
        lhsY[0][k][j][i] =  0.0;
       // lhsY[1][k][j][i] = -dtty2 * cv[j-1] - dtty1 * rhoqY[j-1];
        lhsY[1][k][j][i] = -dtty2 * vs[k][j-1][i] - dtty1 * rhoqY[k][j-1][i];
        lhsY[2][k][j][i] =  1.0 + c2dtty1 * rhoqY[k][j][i];
       // lhsY[3][k][j][i] =  dtty2 * cv[j+1] - dtty1 * rhoq[j+1];
        lhsY[3][k][j][i] =  dtty2 * vs[k][j+1][i] - dtty1 * rhoqY[k][j+1][i];
        lhsY[4][k][j][i] =  0.0;
      }
    }
  }

    //---------------------------------------------------------------------
    // add fourth order dissipation                             
    //---------------------------------------------------------------------
  j = 1;
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

  j = gp1-3;
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

    //---------------------------------------------------------------------
    // subsequently, for (the other two factors                    
    //---------------------------------------------------------------------
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


    //---------------------------------------------------------------------
    // FORWARD ELIMINATION  
    //---------------------------------------------------------------------
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

    //---------------------------------------------------------------------
    // The last two rows in this grid block are a bit different, 
    // since they for (not have two more rows available for the
    // elimination of off-diagonal entries
    //---------------------------------------------------------------------
  j  = gp1-2;
  j1 = gp1-1;
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

    //---------------------------------------------------------------------
    // for (the u+c and the u-c factors                 
    //---------------------------------------------------------------------
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

    //---------------------------------------------------------------------
    // And again the last two rows separately
    //---------------------------------------------------------------------
  j  = gp1-2;
  j1 = gp1-1;
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


    //---------------------------------------------------------------------
    // BACKSUBSTITUTION 
    //---------------------------------------------------------------------
  j  = gp1-2;
  j1 = gp1-1;
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

    //---------------------------------------------------------------------
    // The first three factors
    //---------------------------------------------------------------------
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

}/* end omp target data */
  
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
  
#pragma omp target data map(alloc:lhsZ[:][:][:][:],lhspZ[:][:][:][:],lhsmZ[:][:][:][:],rhosZ[:][:][:]) //present(rho_i,ws,speed,rhs)
{
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

    //---------------------------------------------------------------------
    // Computes the left hand side for the three z-factors   
    //---------------------------------------------------------------------

    //---------------------------------------------------------------------
    // first fill the lhs for the u-eigenvalue                          
    //---------------------------------------------------------------------
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
     //   cv[k] = ws[k][j][i];
        rhosZ[j][i][k] = max(max(dz4+con43*ru1, dz5+c1c5*ru1), max(dzmax+ru1, dz1));
      }
    }
  }

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
      //  lhs[k][i][1] = -dttz2 * cv[k-1] - dttz1 * rhos[k-1];
        lhsZ[1][j][k][i] = -dttz2 * ws[k-1][j][i] - dttz1 * rhosZ[j][i][k-1];
        lhsZ[2][j][k][i] =  1.0 + c2dttz1 * rhosZ[j][i][k];
      //  lhs[k][i][3] =  dttz2 * cv[k+1] - dttz1 * rhos[k+1];
        lhsZ[3][j][k][i] =  dttz2 * ws[k+1][j][i] - dttz1 * rhosZ[j][i][k+1];
        lhsZ[4][j][k][i] =  0.0;
      }
    }
  }

    //---------------------------------------------------------------------
    // add fourth order dissipation                                  
    //---------------------------------------------------------------------
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

    //---------------------------------------------------------------------
    // subsequently, fill the other factors (u+c), (u-c) 
    //---------------------------------------------------------------------
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


    //---------------------------------------------------------------------
    // FORWARD ELIMINATION  
    //---------------------------------------------------------------------
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

    //---------------------------------------------------------------------
    // The last two rows in this grid block are a bit different, 
    // since they for (not have two more rows available for the
    // elimination of off-diagonal entries
    //---------------------------------------------------------------------
    k  = gp22;
    k1 = gp21;
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

    //---------------------------------------------------------------------
    // for (the u+c and the u-c factors               
    //---------------------------------------------------------------------
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

    //---------------------------------------------------------------------
    // And again the last two rows separately
    //---------------------------------------------------------------------
    k  = gp22;
    k1 = gp21;
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


    //---------------------------------------------------------------------
    // BACKSUBSTITUTION 
    //---------------------------------------------------------------------
    k  = gp22;
    k1 = gp21;
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

    //---------------------------------------------------------------------
    // Whether or not this is the last processor, we always have
    // to complete the back-substitution 
    //---------------------------------------------------------------------

    //---------------------------------------------------------------------
    // The first three factors
    //---------------------------------------------------------------------
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

}/* end omp target data */
  
  tzetar();
}
