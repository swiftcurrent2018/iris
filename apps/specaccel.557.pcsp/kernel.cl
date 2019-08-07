#pragma OPENCL EXTENSION cl_khr_fp64 : enable

#define PROBLEM_SIZE   162
#define IMAX    PROBLEM_SIZE
#define JMAX    PROBLEM_SIZE
#define KMAX    PROBLEM_SIZE
#define IMAXP   (IMAX/2*2)
#define JMAXP   (JMAX/2*2)

#define nx2     10
#define ny2     10
#define nz2     10

__kernel void add_0(__global double* restrict u_, __global double* restrict rhs_) {
    __global double (*u)[KMAX][JMAXP+1][IMAXP+1] = (__global double (*)[KMAX][JMAXP+1][IMAXP+1]) u_;
    __global double (*rhs)[KMAX][JMAXP+1][IMAXP+1] = (__global double (*)[KMAX][JMAXP+1][IMAXP+1]) rhs_;
    int k = get_global_id(2);
    int j = get_global_id(1);
    int i = get_global_id(0);

    for (int m = 0; m < 5; m++) {
        u[m][k][j][i] = u[m][k][j][i] + rhs[m][k][j][i];
    }
}

__kernel void ninvr_0(__global double* restrict rhs_, double bt) {
    __global double (*rhs)[KMAX][JMAXP+1][IMAXP+1] = (__global double (*)[KMAX][JMAXP+1][IMAXP+1]) rhs_;
    int k = get_global_id(2);
    int j = get_global_id(1);
    int i = get_global_id(0);

    double r1, r2, r3, r4, r5, t1, t2;

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

__kernel void pinvr_0(__global double* restrict rhs_, double bt) {
    __global double (*rhs)[KMAX][JMAXP+1][IMAXP+1] = (__global double (*)[KMAX][JMAXP+1][IMAXP+1]) rhs_;
    int k = get_global_id(2);
    int j = get_global_id(1);
    int i = get_global_id(0);

    double r1, r2, r3, r4, r5, t1, t2;

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

__kernel void tzetar_0(__global double* restrict us_, __global double* restrict vs_, __global double* restrict ws_, __global double* restrict qs_, __global double* restrict u_, __global double* restrict speed_, __global double* restrict rhs_, double bt, double c2iv) {
    __global double (*u)[KMAX][JMAXP+1][IMAXP+1] = (__global double (*)[KMAX][JMAXP+1][IMAXP+1]) u_;
    __global double (*us)[JMAXP+1][IMAXP+1] = (__global double (*)[JMAXP+1][IMAXP+1]) us_;
    __global double (*vs)[JMAXP+1][IMAXP+1] = (__global double (*)[JMAXP+1][IMAXP+1]) vs_;
    __global double (*ws)[JMAXP+1][IMAXP+1] = (__global double (*)[JMAXP+1][IMAXP+1]) ws_;
    __global double (*qs)[JMAXP+1][IMAXP+1] = (__global double (*)[JMAXP+1][IMAXP+1]) qs_;
    __global double (*speed)[JMAXP+1][IMAXP+1] = (__global double (*)[JMAXP+1][IMAXP+1]) speed_;
    __global double (*rhs)[KMAX][JMAXP+1][IMAXP+1] = (__global double (*)[KMAX][JMAXP+1][IMAXP+1]) rhs_;

    int k = get_global_id(2);
    int j = get_global_id(1);
    int i = get_global_id(0);

    double t1, t2, t3, ac, xvel, yvel, zvel, r1, r2, r3, r4, r5;
    double btuz, ac2u, uzik1;

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
    rhs[4][k][j][i] =  uzik1*(-xvel*r2 + yvel*r1) + qs[k][j][i]*t2 + c2iv*ac2u*t1 + zvel*t3;
}

__kernel void x_solve_0(__global double* restrict rhsX_, __global double* restrict rhs_) {
    __global double (*rhsX)[nz2+1][IMAXP+1][JMAXP+1] = (__global double (*)[nz2+1][IMAXP+1][JMAXP+1]) rhsX_;
    __global double (*rhs)[KMAX][JMAXP+1][IMAXP+1] = (__global double (*)[KMAX][JMAXP+1][IMAXP+1]) rhs_;

    int k = get_global_id(2);
    int j = get_global_id(1);
    int i = get_global_id(0);

    rhsX[0][k][i][j] = rhs[0][k][j][i];
    rhsX[1][k][i][j] = rhs[1][k][j][i];
    rhsX[2][k][i][j] = rhs[2][k][j][i];
    rhsX[3][k][i][j] = rhs[3][k][j][i];
    rhsX[4][k][i][j] = rhs[4][k][j][i];
}

__kernel void x_solve_1(__global double* restrict lhsX_, __global double* restrict lhspX_, __global double* restrict lhsmX_, int ni) {
    __global double (*lhsX)[nz2+1][IMAXP+1][IMAXP+1] = (__global double (*)[nz2+1][IMAXP+1][IMAXP+1]) lhsX_;
    __global double (*lhspX)[nz2+1][IMAXP+1][IMAXP+1] = (__global double (*)[nz2+1][IMAXP+1][IMAXP+1]) lhspX_;
    __global double (*lhsmX)[nz2+1][IMAXP+1][IMAXP+1] = (__global double (*)[nz2+1][IMAXP+1][IMAXP+1]) lhsmX_;

    int k = get_global_id(1);
    int j = get_global_id(0);

    for (int m = 0; m < 5; m++) {
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

__kernel void x_solve_2(__global double* restrict rho_i_, __global double* restrict rhonX_, __global double* restrict lhsX_, __global double* restrict us_, int gp01, double dx1, double dx2, double dx5, double dxmax, double c1c5, double c3c4, double dttx1, double dttx2, double c2dttx1, double con43) {
    __global double (*rho_i)[JMAXP+1][IMAXP+1] = (__global double (*)[JMAXP+1][IMAXP+1]) rho_i_;
    __global double (*rhonX)[IMAXP+1][PROBLEM_SIZE] = (__global double (*)[IMAXP+1][PROBLEM_SIZE]) rhonX_;
    __global double (*lhsX)[nz2+1][IMAXP+1][IMAXP+1] = (__global double (*)[nz2+1][IMAXP+1][IMAXP+1]) lhsX_;
    __global double (*us)[JMAXP+1][IMAXP+1] = (__global double (*)[JMAXP+1][IMAXP+1]) us_;

    int k = get_global_id(1);
    int j = get_global_id(0);

    double ru1, fac1, fac2;

    for (int i = 0; i <= gp01; i++) {
        ru1 = c3c4*rho_i[k][j][i];
        rhonX[k][j][i] = max(max(dx2+con43*ru1,dx5+c1c5*ru1), max(dxmax+ru1,dx1));
    }
    for (int i = 1; i <= nx2; i++) {
        lhsX[0][k][i][j] =  0.0;
        lhsX[1][k][i][j] = -dttx2 * us[k][j][i-1] - dttx1 * rhonX[k][j][i-1];
        lhsX[2][k][i][j] =  1.0 + c2dttx1 * rhonX[k][j][i];
        lhsX[3][k][i][j] =  dttx2 * us[k][j][i+1] - dttx1 * rhonX[k][j][i+1];
        lhsX[4][k][i][j] =  0.0;
    }
}

__kernel void x_solve_3(__global double* restrict lhsX_, int i, double comz1, double comz4, double comz5, double comz6) {
    __global double (*lhsX)[nz2+1][IMAXP+1][IMAXP+1] = (__global double (*)[nz2+1][IMAXP+1][IMAXP+1]) lhsX_;

    int k = get_global_id(1);
    int j = get_global_id(0);

    lhsX[2][k][i][j] = lhsX[2][k][i][j] + comz5;
    lhsX[3][k][i][j] = lhsX[3][k][i][j] - comz4;
    lhsX[4][k][i][j] = lhsX[4][k][i][j] + comz1;

    lhsX[1][k][i+1][j] = lhsX[1][k][i+1][j] - comz4;
    lhsX[2][k][i+1][j] = lhsX[2][k][i+1][j] + comz6;
    lhsX[3][k][i+1][j] = lhsX[3][k][i+1][j] - comz4;
    lhsX[4][k][i+1][j] = lhsX[4][k][i+1][j] + comz1;
}

__kernel void x_solve_4(__global double* restrict lhsX_, double comz1, double comz4, double comz6) {
    __global double (*lhsX)[nz2+1][IMAXP+1][IMAXP+1] = (__global double (*)[nz2+1][IMAXP+1][IMAXP+1]) lhsX_;

    int k = get_global_id(2);
    int j = get_global_id(1);
    int i = get_global_id(0);

    lhsX[0][k][i][j] = lhsX[0][k][i][j] + comz1;
    lhsX[1][k][i][j] = lhsX[1][k][i][j] - comz4;
    lhsX[2][k][i][j] = lhsX[2][k][i][j] + comz6;
    lhsX[3][k][i][j] = lhsX[3][k][i][j] - comz4;
    lhsX[4][k][i][j] = lhsX[4][k][i][j] + comz1;
}

__kernel void x_solve_5(__global double* restrict lhsX_, int i, double comz1, double comz4, double comz5, double comz6) {
    __global double (*lhsX)[nz2+1][IMAXP+1][IMAXP+1] = (__global double (*)[nz2+1][IMAXP+1][IMAXP+1]) lhsX_;

    int k = get_global_id(1);
    int j = get_global_id(0);

    lhsX[0][k][i][j] = lhsX[0][k][i][j] + comz1;
    lhsX[1][k][i][j] = lhsX[1][k][i][j] - comz4;
    lhsX[2][k][i][j] = lhsX[2][k][i][j] + comz6;
    lhsX[3][k][i][j] = lhsX[3][k][i][j] - comz4;

    lhsX[0][k][i+1][j] = lhsX[0][k][i+1][j] + comz1;
    lhsX[1][k][i+1][j] = lhsX[1][k][i+1][j] - comz4;
    lhsX[2][k][i+1][j] = lhsX[2][k][i+1][j] + comz5;
}

__kernel void x_solve_6(__global double* restrict lhspX_, __global double* restrict lhsmX_, __global double* restrict lhsX_, __global double* restrict speed_, double dttx2) {
    __global double (*lhsX)[nz2+1][IMAXP+1][IMAXP+1] = (__global double (*)[nz2+1][IMAXP+1][IMAXP+1]) lhsX_;
    __global double (*lhspX)[nz2+1][IMAXP+1][IMAXP+1] = (__global double (*)[nz2+1][IMAXP+1][IMAXP+1]) lhspX_;
    __global double (*lhsmX)[nz2+1][IMAXP+1][IMAXP+1] = (__global double (*)[nz2+1][IMAXP+1][IMAXP+1]) lhsmX_;
    __global double (*speed)[JMAXP+1][IMAXP+1] = (__global double (*)[JMAXP+1][IMAXP+1]) speed_;

    int k = get_global_id(2);
    int j = get_global_id(1);
    int i = get_global_id(0);

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

__kernel void x_solve_7(__global double* restrict lhsX_, __global double* restrict rhsX_, int gp03) {
    __global double (*lhsX)[nz2+1][IMAXP+1][IMAXP+1] = (__global double (*)[nz2+1][IMAXP+1][IMAXP+1]) lhsX_;
    __global double (*rhsX)[nz2+1][IMAXP+1][JMAXP+1] = (__global double (*)[nz2+1][IMAXP+1][JMAXP+1]) rhsX_;

    int k = get_global_id(1);
    int j = get_global_id(0);

    int m, i, i1, i2;
    double fac1;

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

__kernel void x_solve_8(__global double* restrict lhsX_, __global double* restrict rhsX_, int i, int i1) {
    __global double (*lhsX)[nz2+1][IMAXP+1][IMAXP+1] = (__global double (*)[nz2+1][IMAXP+1][IMAXP+1]) lhsX_;
    __global double (*rhsX)[nz2+1][IMAXP+1][JMAXP+1] = (__global double (*)[nz2+1][IMAXP+1][JMAXP+1]) rhsX_;

    int k = get_global_id(1);
    int j = get_global_id(0);

    int m;
    double fac1, fac2;

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

    fac2 = 1.0/lhsX[2][k][i1][j];
    for (m = 0; m < 3; m++) {
        rhsX[m][k][i1][j] = fac2*rhsX[m][k][i1][j];
    }
}

__kernel void x_solve_9(__global double* restrict lhspX_, __global double* restrict lhsmX_, __global double* restrict rhsX_, int gp03) {
    __global double (*lhspX)[nz2+1][IMAXP+1][IMAXP+1] = (__global double (*)[nz2+1][IMAXP+1][IMAXP+1]) lhspX_;
    __global double (*lhsmX)[nz2+1][IMAXP+1][IMAXP+1] = (__global double (*)[nz2+1][IMAXP+1][IMAXP+1]) lhsmX_;
    __global double (*rhsX)[nz2+1][IMAXP+1][JMAXP+1] = (__global double (*)[nz2+1][IMAXP+1][JMAXP+1]) rhsX_;

    int k = get_global_id(1);
    int j = get_global_id(0);

    int m, i, i1, i2;
    double fac1;

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

__kernel void x_solve_10(__global double* restrict lhspX_, __global double* restrict lhsmX_, __global double* restrict rhsX_, int i, int i1) {
    __global double (*lhspX)[nz2+1][IMAXP+1][IMAXP+1] = (__global double (*)[nz2+1][IMAXP+1][IMAXP+1]) lhspX_;
    __global double (*lhsmX)[nz2+1][IMAXP+1][IMAXP+1] = (__global double (*)[nz2+1][IMAXP+1][IMAXP+1]) lhsmX_;
    __global double (*rhsX)[nz2+1][IMAXP+1][JMAXP+1] = (__global double (*)[nz2+1][IMAXP+1][JMAXP+1]) rhsX_;

    int k = get_global_id(1);
    int j = get_global_id(0);

    int m;
    double fac1;

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

    rhsX[3][k][i1][j] = rhsX[3][k][i1][j]/lhspX[2][k][i1][j];
    rhsX[4][k][i1][j] = rhsX[4][k][i1][j]/lhsmX[2][k][i1][j];
}

__kernel void x_solve_11(__global double* restrict rhsX_, __global double* restrict lhsX_, __global double* restrict lhspX_, __global double* restrict lhsmX_, int i, int i1) {
    __global double (*rhsX)[nz2+1][IMAXP+1][JMAXP+1] = (__global double (*)[nz2+1][IMAXP+1][JMAXP+1]) rhsX_;
    __global double (*lhsX)[nz2+1][IMAXP+1][IMAXP+1] = (__global double (*)[nz2+1][IMAXP+1][IMAXP+1]) lhsX_;
    __global double (*lhspX)[nz2+1][IMAXP+1][IMAXP+1] = (__global double (*)[nz2+1][IMAXP+1][IMAXP+1]) lhspX_;
    __global double (*lhsmX)[nz2+1][IMAXP+1][IMAXP+1] = (__global double (*)[nz2+1][IMAXP+1][IMAXP+1]) lhsmX_;

    int k = get_global_id(1);
    int j = get_global_id(0);

    int m;

    for (m = 0; m < 3; m++) {
        rhsX[m][k][i][j] = rhsX[m][k][i][j] - lhsX[3][k][i][j]*rhsX[m][k][i1][j];
    }

    rhsX[3][k][i][j] = rhsX[3][k][i][j] - lhspX[3][k][i][j]*rhsX[3][k][i1][j];
    rhsX[4][k][i][j] = rhsX[4][k][i][j] - lhsmX[3][k][i][j]*rhsX[4][k][i1][j];
}

__kernel void x_solve_12(__global double* restrict rhsX_, __global double* restrict lhsX_, __global double* restrict lhspX_, __global double* restrict lhsmX_, int gp03) {
    __global double (*rhsX)[nz2+1][IMAXP+1][JMAXP+1] = (__global double (*)[nz2+1][IMAXP+1][JMAXP+1]) rhsX_;
    __global double (*lhsX)[nz2+1][IMAXP+1][IMAXP+1] = (__global double (*)[nz2+1][IMAXP+1][IMAXP+1]) lhsX_;
    __global double (*lhspX)[nz2+1][IMAXP+1][IMAXP+1] = (__global double (*)[nz2+1][IMAXP+1][IMAXP+1]) lhspX_;
    __global double (*lhsmX)[nz2+1][IMAXP+1][IMAXP+1] = (__global double (*)[nz2+1][IMAXP+1][IMAXP+1]) lhsmX_;

    int k = get_global_id(1);
    int j = get_global_id(0);

    int i, i1, i2, m;

    for (i = gp03; i >= 0; i--) {
        i1 = i + 1;
        i2 = i + 2;
        for (m = 0; m < 3; m++) {
            rhsX[m][k][i][j] = rhsX[m][k][i][j] - 
                lhsX[3][k][i][j]*rhsX[m][k][i1][j] -
                lhsX[4][k][i][j]*rhsX[m][k][i2][j];
        }

        rhsX[3][k][i][j] = rhsX[3][k][i][j] - 
            lhspX[3][k][i][j]*rhsX[3][k][i1][j] -
            lhspX[4][k][i][j]*rhsX[3][k][i2][j];
        rhsX[4][k][i][j] = rhsX[4][k][i][j] - 
            lhsmX[3][k][i][j]*rhsX[4][k][i1][j] -
            lhsmX[4][k][i][j]*rhsX[4][k][i2][j];
    }
}

__kernel void x_solve_13(__global double* restrict rhs_, __global double* restrict rhsX_) {
    __global double (*rhs)[KMAX][JMAXP+1][IMAXP+1] = (__global double (*)[KMAX][JMAXP+1][IMAXP+1]) rhs_;
    __global double (*rhsX)[nz2+1][IMAXP+1][JMAXP+1] = (__global double (*)[nz2+1][IMAXP+1][JMAXP+1]) rhsX_;

    int k = get_global_id(2);
    int j = get_global_id(1);
    int i = get_global_id(0);

    rhs[0][k][j][i] = rhsX[0][k][i][j];
    rhs[1][k][j][i] = rhsX[1][k][i][j];
    rhs[2][k][j][i] = rhsX[2][k][i][j];
    rhs[3][k][j][i] = rhsX[3][k][i][j];
    rhs[4][k][j][i] = rhsX[4][k][i][j];
}

/*
  double lhsY[5][nz2+1][IMAXP+1][IMAXP+1];
  double lhspY[5][nz2+1][IMAXP+1][IMAXP+1];
  double lhsmY[5][nz2+1][IMAXP+1][IMAXP+1];
  double rhoqY[nz2+1][IMAXP+1][PROBLEM_SIZE];
  */
__kernel void y_solve_0(__global double* restrict lhsY_, __global double* restrict lhspY_, __global double* restrict lhsmY_, int ni) {
    __global double (*lhsY)[nz2+1][IMAXP+1][IMAXP+1] = (__global double (*)[nz2+1][IMAXP+1][IMAXP+1]) lhsY_;
    __global double (*lhspY)[nz2+1][IMAXP+1][IMAXP+1] = (__global double (*)[nz2+1][IMAXP+1][IMAXP+1]) lhspY_;
    __global double (*lhsmY)[nz2+1][IMAXP+1][IMAXP+1] = (__global double (*)[nz2+1][IMAXP+1][IMAXP+1]) lhsmY_;

    int k = get_global_id(1);
    int i = get_global_id(0);

    int m;

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

__kernel void y_solve_1(__global double* restrict rho_i_, __global double* restrict rhoqY_, __global double* restrict lhsY_, __global double* restrict vs_, int gp1, double dy1, double dy3, double dy5, double dymax, double c1c5, double c3c4, double dtty1, double dtty2, double c2dtty1, double con43) {
    __global double (*rho_i)[JMAXP+1][IMAXP+1] = (__global double (*)[JMAXP+1][IMAXP+1]) rho_i_;
    __global double (*rhoqY)[IMAXP+1][PROBLEM_SIZE] = (__global double (*)[IMAXP+1][PROBLEM_SIZE]) rhoqY_;
    __global double (*lhsY)[nz2+1][IMAXP+1][IMAXP+1] = (__global double (*)[nz2+1][IMAXP+1][IMAXP+1]) lhsY_;
    __global double (*vs)[JMAXP+1][IMAXP+1] = (__global double (*)[JMAXP+1][IMAXP+1]) vs_;

    int k = get_global_id(1);
    int i = get_global_id(0);

    int j;
    double ru1;

    for (j = 0; j <= gp1-1; j++) {
        ru1 = c3c4*rho_i[k][j][i];
        rhoqY[k][j][i] = max(max(dy3+con43*ru1, dy5+c1c5*ru1), max(dymax+ru1, dy1));
    }
    for (j = 1; j <= gp1-2; j++) {
        lhsY[0][k][j][i] =  0.0;
        lhsY[1][k][j][i] = -dtty2 * vs[k][j-1][i] - dtty1 * rhoqY[k][j-1][i];
        lhsY[2][k][j][i] =  1.0 + c2dtty1 * rhoqY[k][j][i];
        lhsY[3][k][j][i] =  dtty2 * vs[k][j+1][i] - dtty1 * rhoqY[k][j+1][i];
        lhsY[4][k][j][i] =  0.0;
    }
}

__kernel void y_solve_2(__global double* restrict lhsY_, double comz1, double comz4, double comz5, double comz6, int j) {
    __global double (*lhsY)[nz2+1][IMAXP+1][IMAXP+1] = (__global double (*)[nz2+1][IMAXP+1][IMAXP+1]) lhsY_;

    int k = get_global_id(1);
    int i = get_global_id(0);

    lhsY[2][k][j][i] = lhsY[2][k][j][i] + comz5;
    lhsY[3][k][j][i] = lhsY[3][k][j][i] - comz4;
    lhsY[4][k][j][i] = lhsY[4][k][j][i] + comz1;

    lhsY[1][k][j+1][i] = lhsY[1][k][j+1][i] - comz4;
    lhsY[2][k][j+1][i] = lhsY[2][k][j+1][i] + comz6;
    lhsY[3][k][j+1][i] = lhsY[3][k][j+1][i] - comz4;
    lhsY[4][k][j+1][i] = lhsY[4][k][j+1][i] + comz1;
}

__kernel void y_solve_3(__global double* restrict lhsY_, double comz1, double comz4, double comz6) {
    __global double (*lhsY)[nz2+1][IMAXP+1][IMAXP+1] = (__global double (*)[nz2+1][IMAXP+1][IMAXP+1]) lhsY_;

    int k = get_global_id(2);
    int j = get_global_id(1);
    int i = get_global_id(0);

    lhsY[0][k][j][i] = lhsY[0][k][j][i] + comz1;
    lhsY[1][k][j][i] = lhsY[1][k][j][i] - comz4;
    lhsY[2][k][j][i] = lhsY[2][k][j][i] + comz6;
    lhsY[3][k][j][i] = lhsY[3][k][j][i] - comz4;
    lhsY[4][k][j][i] = lhsY[4][k][j][i] + comz1;
}

__kernel void y_solve_4(__global double* restrict lhsY_, double comz1, double comz4, double comz5, double comz6, int j) {
    __global double (*lhsY)[nz2+1][IMAXP+1][IMAXP+1] = (__global double (*)[nz2+1][IMAXP+1][IMAXP+1]) lhsY_;

    int k = get_global_id(1);
    int i = get_global_id(0);

    lhsY[0][k][j][i] = lhsY[0][k][j][i] + comz1;
    lhsY[1][k][j][i] = lhsY[1][k][j][i] - comz4;
    lhsY[2][k][j][i] = lhsY[2][k][j][i] + comz6;
    lhsY[3][k][j][i] = lhsY[3][k][j][i] - comz4;

    lhsY[0][k][j+1][i] = lhsY[0][k][j+1][i] + comz1;
    lhsY[1][k][j+1][i] = lhsY[1][k][j+1][i] - comz4;
    lhsY[2][k][j+1][i] = lhsY[2][k][j+1][i] + comz5;
}

__kernel void y_solve_5(__global double* restrict lhspY_, __global double* restrict lhsmY_, __global double* restrict lhsY_, __global double* restrict speed_, double dtty2) {
    __global double (*lhspY)[nz2+1][IMAXP+1][IMAXP+1] = (__global double (*)[nz2+1][IMAXP+1][IMAXP+1]) lhspY_;
    __global double (*lhsmY)[nz2+1][IMAXP+1][IMAXP+1] = (__global double (*)[nz2+1][IMAXP+1][IMAXP+1]) lhsmY_;
    __global double (*lhsY)[nz2+1][IMAXP+1][IMAXP+1] = (__global double (*)[nz2+1][IMAXP+1][IMAXP+1]) lhsY_;
    __global double (*speed)[JMAXP+1][IMAXP+1] = (__global double (*)[JMAXP+1][IMAXP+1]) speed_;

    int k = get_global_id(2);
    int j = get_global_id(1);
    int i = get_global_id(0);

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

__kernel void y_solve_6(__global double* restrict lhsY_, __global double* restrict rhs_, int gp0, int gp1) {
    __global double (*lhsY)[nz2+1][IMAXP+1][IMAXP+1] = (__global double (*)[nz2+1][IMAXP+1][IMAXP+1]) lhsY_;
    __global double (*rhs)[KMAX][JMAXP+1][IMAXP+1] = (__global double (*)[KMAX][JMAXP+1][IMAXP+1]) rhs_;

    int k = get_global_id(0);

    int i, j, j1, j2, m;
    double fac1;

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

__kernel void y_solve_7(__global double* restrict lhsY_, __global double* restrict rhs_, int j, int j1) {
    __global double (*lhsY)[nz2+1][IMAXP+1][IMAXP+1] = (__global double (*)[nz2+1][IMAXP+1][IMAXP+1]) lhsY_;
    __global double (*rhs)[KMAX][JMAXP+1][IMAXP+1] = (__global double (*)[KMAX][JMAXP+1][IMAXP+1]) rhs_;

    int k = get_global_id(1);
    int i = get_global_id(0);

    int m;
    double fac1, fac2;

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
    fac2 = 1.0/lhsY[2][k][j1][i];
    for (m = 0; m < 3; m++) {
        rhs[m][k][j1][i] = fac2*rhs[m][k][j1][i];
    }
}

__kernel void y_solve_8(__global double* restrict lhspY_, __global double* restrict lhsmY_, __global double* restrict rhs_, int gp1) {
    __global double (*lhspY)[nz2+1][IMAXP+1][IMAXP+1] = (__global double (*)[nz2+1][IMAXP+1][IMAXP+1]) lhspY_;
    __global double (*lhsmY)[nz2+1][IMAXP+1][IMAXP+1] = (__global double (*)[nz2+1][IMAXP+1][IMAXP+1]) lhsmY_;
    __global double (*rhs)[KMAX][JMAXP+1][IMAXP+1] = (__global double (*)[KMAX][JMAXP+1][IMAXP+1]) rhs_;

    int k = get_global_id(1);
    int i = get_global_id(0);

    int j, j1, j2, m;
    double fac1;

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


__kernel void y_solve_9(__global double* restrict lhspY_, __global double* restrict lhsmY_, __global double* restrict rhs_, int j, int j1) {
    __global double (*lhspY)[nz2+1][IMAXP+1][IMAXP+1] = (__global double (*)[nz2+1][IMAXP+1][IMAXP+1]) lhspY_;
    __global double (*lhsmY)[nz2+1][IMAXP+1][IMAXP+1] = (__global double (*)[nz2+1][IMAXP+1][IMAXP+1]) lhsmY_;
    __global double (*rhs)[KMAX][JMAXP+1][IMAXP+1] = (__global double (*)[KMAX][JMAXP+1][IMAXP+1]) rhs_;

    int k = get_global_id(1);
    int i = get_global_id(0);

    int m;
    double fac1;

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

    rhs[3][k][j1][i]   = rhs[3][k][j1][i]/lhspY[2][k][j1][i];
    rhs[4][k][j1][i]   = rhs[4][k][j1][i]/lhsmY[2][k][j1][i];
}

__kernel void y_solve_10(__global double* restrict rhs_, __global double* restrict lhsY_, __global double* restrict lhspY_, __global double* restrict lhsmY_, int j, int j1) {
    __global double (*rhs)[KMAX][JMAXP+1][IMAXP+1] = (__global double (*)[KMAX][JMAXP+1][IMAXP+1]) rhs_;
    __global double (*lhsY)[nz2+1][IMAXP+1][IMAXP+1] = (__global double (*)[nz2+1][IMAXP+1][IMAXP+1]) lhsY_;
    __global double (*lhspY)[nz2+1][IMAXP+1][IMAXP+1] = (__global double (*)[nz2+1][IMAXP+1][IMAXP+1]) lhspY_;
    __global double (*lhsmY)[nz2+1][IMAXP+1][IMAXP+1] = (__global double (*)[nz2+1][IMAXP+1][IMAXP+1]) lhsmY_;

    int k = get_global_id(1);
    int i = get_global_id(0);

    int m;

    for (m = 0; m < 3; m++) {
        rhs[m][k][j][i] = rhs[m][k][j][i] - lhsY[3][k][j][i]*rhs[m][k][j1][i];
    }

    rhs[3][k][j][i] = rhs[3][k][j][i] - lhspY[3][k][j][i]*rhs[3][k][j1][i];
    rhs[4][k][j][i] = rhs[4][k][j][i] - lhsmY[3][k][j][i]*rhs[4][k][j1][i];
}

__kernel void y_solve_11(__global double* restrict rhs_, __global double* restrict lhsY_, __global double* restrict lhspY_, __global double* restrict lhsmY_, int gp1) {
    __global double (*rhs)[KMAX][JMAXP+1][IMAXP+1] = (__global double (*)[KMAX][JMAXP+1][IMAXP+1]) rhs_;
    __global double (*lhsY)[nz2+1][IMAXP+1][IMAXP+1] = (__global double (*)[nz2+1][IMAXP+1][IMAXP+1]) lhsY_;
    __global double (*lhspY)[nz2+1][IMAXP+1][IMAXP+1] = (__global double (*)[nz2+1][IMAXP+1][IMAXP+1]) lhspY_;
    __global double (*lhsmY)[nz2+1][IMAXP+1][IMAXP+1] = (__global double (*)[nz2+1][IMAXP+1][IMAXP+1]) lhsmY_;

    int k = get_global_id(1);
    int i = get_global_id(0);

    int j, j1, j2, m;

    for (j = gp1-3; j >= 0; j--) {
        j1 = j + 1;
        j2 = j + 2;
        for (m = 0; m < 3; m++) {
            rhs[m][k][j][i] = rhs[m][k][j][i] - 
                lhsY[3][k][j][i]*rhs[m][k][j1][i] -
                lhsY[4][k][j][i]*rhs[m][k][j2][i];
        }

        rhs[3][k][j][i] = rhs[3][k][j][i] - 
            lhspY[3][k][j][i]*rhs[3][k][j1][i] -
            lhspY[4][k][j][i]*rhs[3][k][j2][i];
        rhs[4][k][j][i] = rhs[4][k][j][i] - 
            lhsmY[3][k][j][i]*rhs[4][k][j1][i] -
            lhsmY[4][k][j][i]*rhs[4][k][j2][i];
    }
}

/*
  double lhsZ[5][ny2+1][IMAXP+1][IMAXP+1];
  double lhspZ[5][ny2+1][IMAXP+1][IMAXP+1];
  double lhsmZ[5][ny2+1][IMAXP+1][IMAXP+1];
  double rhosZ[ny2+1][IMAXP+1][PROBLEM_SIZE];
  */

__kernel void z_solve_0(__global double* restrict lhsZ_, __global double* restrict lhspZ_, __global double* restrict lhsmZ_, int ni) {
    __global double (*lhsZ)[ny2+1][IMAXP+1][IMAXP+1] = (__global double (*)[ny2+1][IMAXP+1][IMAXP+1]) lhsZ_;
    __global double (*lhspZ)[ny2+1][IMAXP+1][IMAXP+1] = (__global double (*)[ny2+1][IMAXP+1][IMAXP+1]) lhspZ_;
    __global double (*lhsmZ)[ny2+1][IMAXP+1][IMAXP+1] = (__global double (*)[ny2+1][IMAXP+1][IMAXP+1]) lhsmZ_;

    int j = get_global_id(1);
    int i = get_global_id(0);

    int m;

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

__kernel void z_solve_1(__global double* restrict rho_i_, __global double* restrict rhosZ_, double dz1, double dz4, double dz5, double dzmax, double c1c5, double c3c4, double con43) {
    __global double (*rho_i)[JMAXP+1][IMAXP+1] = (__global double (*)[JMAXP+1][IMAXP+1]) rho_i_;
    __global double (*rhosZ)[IMAXP+1][PROBLEM_SIZE] = (__global double (*)[IMAXP+1][PROBLEM_SIZE]) rhosZ_;

    int j = get_global_id(2);
    int i = get_global_id(1);
    int k = get_global_id(0);

    double ru1;

    ru1 = c3c4*rho_i[k][j][i];
    rhosZ[j][i][k] = max(max(dz4+con43*ru1, dz5+c1c5*ru1), max(dzmax+ru1, dz1));
}

__kernel void z_solve_2(__global double* restrict lhsZ_, __global double* restrict ws_, __global double* restrict rhosZ_, double dttz1, double dttz2, double c2dttz1) {
    __global double (*lhsZ)[ny2+1][IMAXP+1][IMAXP+1] = (__global double (*)[ny2+1][IMAXP+1][IMAXP+1]) lhsZ_;
    __global double (*ws)[JMAXP+1][IMAXP+1] = (__global double (*)[JMAXP+1][IMAXP+1]) ws_;
    __global double (*rhosZ)[IMAXP+1][PROBLEM_SIZE] = (__global double (*)[IMAXP+1][PROBLEM_SIZE]) rhosZ_;

    int j = get_global_id(2);
    int i = get_global_id(1);
    int k = get_global_id(0);

    lhsZ[0][j][k][i] =  0.0;
    lhsZ[1][j][k][i] = -dttz2 * ws[k-1][j][i] - dttz1 * rhosZ[j][i][k-1];
    lhsZ[2][j][k][i] =  1.0 + c2dttz1 * rhosZ[j][i][k];
    lhsZ[3][j][k][i] =  dttz2 * ws[k+1][j][i] - dttz1 * rhosZ[j][i][k+1];
    lhsZ[4][j][k][i] =  0.0;
}

__kernel void z_solve_3(__global double* restrict lhsZ_, double comz1, double comz4, double comz5, double comz6) {
    __global double (*lhsZ)[ny2+1][IMAXP+1][IMAXP+1] = (__global double (*)[ny2+1][IMAXP+1][IMAXP+1]) lhsZ_;

    int j = get_global_id(1);
    int i = get_global_id(0);

    int k;

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

__kernel void z_solve_4(__global double* restrict lhsZ_, double comz1, double comz4, double comz6) {
    __global double (*lhsZ)[ny2+1][IMAXP+1][IMAXP+1] = (__global double (*)[ny2+1][IMAXP+1][IMAXP+1]) lhsZ_;

    int j = get_global_id(2);
    int k = get_global_id(1);
    int i = get_global_id(0);

    lhsZ[0][j][k][i] = lhsZ[0][j][k][i] + comz1;
    lhsZ[1][j][k][i] = lhsZ[1][j][k][i] - comz4;
    lhsZ[2][j][k][i] = lhsZ[2][j][k][i] + comz6;
    lhsZ[3][j][k][i] = lhsZ[3][j][k][i] - comz4;
    lhsZ[4][j][k][i] = lhsZ[4][j][k][i] + comz1;
}

__kernel void z_solve_5(__global double* restrict lhsZ_, double comz1, double comz4, double comz5, double comz6) {
    __global double (*lhsZ)[ny2+1][IMAXP+1][IMAXP+1] = (__global double (*)[ny2+1][IMAXP+1][IMAXP+1]) lhsZ_;

    int j = get_global_id(1);
    int i = get_global_id(0);

    int k;

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

__kernel void z_solve_6(__global double* restrict lhsZ_, __global double* restrict lhspZ_, __global double* restrict lhsmZ_, __global double* restrict speed_, double dttz2) {
    __global double (*lhsZ)[ny2+1][IMAXP+1][IMAXP+1] = (__global double (*)[ny2+1][IMAXP+1][IMAXP+1]) lhsZ_;
    __global double (*lhspZ)[ny2+1][IMAXP+1][IMAXP+1] = (__global double (*)[ny2+1][IMAXP+1][IMAXP+1]) lhspZ_;
    __global double (*lhsmZ)[ny2+1][IMAXP+1][IMAXP+1] = (__global double (*)[ny2+1][IMAXP+1][IMAXP+1]) lhsmZ_;
    __global double (*speed)[JMAXP+1][IMAXP+1] = (__global double (*)[JMAXP+1][IMAXP+1]) speed_;

    int j = get_global_id(2);
    int k = get_global_id(1);
    int i = get_global_id(0);

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

__kernel void z_solve_7(__global double* restrict lhsZ_, __global double* restrict rhs_, int gp23) {
    __global double (*lhsZ)[ny2+1][IMAXP+1][IMAXP+1] = (__global double (*)[ny2+1][IMAXP+1][IMAXP+1]) lhsZ_;
    __global double (*rhs)[KMAX][JMAXP+1][IMAXP+1] = (__global double (*)[KMAX][JMAXP+1][IMAXP+1]) rhs_;

    int j = get_global_id(1);
    int i = get_global_id(0);

    int m, k, k1, k2;
    double fac1;

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

__kernel void z_solve_8(__global double* restrict lhsZ_, __global double* restrict rhs_, int k, int k1) {
    __global double (*lhsZ)[ny2+1][IMAXP+1][IMAXP+1] = (__global double (*)[ny2+1][IMAXP+1][IMAXP+1]) lhsZ_;
    __global double (*rhs)[KMAX][JMAXP+1][IMAXP+1] = (__global double (*)[KMAX][JMAXP+1][IMAXP+1]) rhs_;

    int j = get_global_id(1);
    int i = get_global_id(0);

    int m;
    double fac1, fac2;

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

    fac2 = 1.0/lhsZ[2][j][k1][i];
    for (m = 0; m < 3; m++) {
        rhs[m][k1][j][i] = fac2*rhs[m][k1][j][i];
    }
}

__kernel void z_solve_9(__global double* restrict lhspZ_, __global double* restrict lhsmZ_, __global double* restrict rhs_, int gp23) {
    __global double (*lhspZ)[ny2+1][IMAXP+1][IMAXP+1] = (__global double (*)[ny2+1][IMAXP+1][IMAXP+1]) lhspZ_;
    __global double (*lhsmZ)[ny2+1][IMAXP+1][IMAXP+1] = (__global double (*)[ny2+1][IMAXP+1][IMAXP+1]) lhsmZ_;
    __global double (*rhs)[KMAX][JMAXP+1][IMAXP+1] = (__global double (*)[KMAX][JMAXP+1][IMAXP+1]) rhs_;

    int j = get_global_id(1);
    int i = get_global_id(0);

    int m, k, k1, k2;
    double fac1;

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

__kernel void z_solve_10(__global double* restrict lhspZ_, __global double* restrict lhsmZ_, __global double* restrict rhs_, int k, int k1) {
    __global double (*lhspZ)[ny2+1][IMAXP+1][IMAXP+1] = (__global double (*)[ny2+1][IMAXP+1][IMAXP+1]) lhspZ_;
    __global double (*lhsmZ)[ny2+1][IMAXP+1][IMAXP+1] = (__global double (*)[ny2+1][IMAXP+1][IMAXP+1]) lhsmZ_;
    __global double (*rhs)[KMAX][JMAXP+1][IMAXP+1] = (__global double (*)[KMAX][JMAXP+1][IMAXP+1]) rhs_;

    int j = get_global_id(1);
    int i = get_global_id(0);

    int m;
    double fac1;

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

    rhs[3][k1][j][i] = rhs[3][k1][j][i]/lhspZ[2][j][k1][i];
    rhs[4][k1][j][i] = rhs[4][k1][j][i]/lhsmZ[2][j][k1][i];
}

__kernel void z_solve_11(__global double* restrict lhsZ_, __global double* restrict lhspZ_, __global double* restrict lhsmZ_, __global double* restrict rhs_, int k, int k1) {
    __global double (*lhsZ)[ny2+1][IMAXP+1][IMAXP+1] = (__global double (*)[ny2+1][IMAXP+1][IMAXP+1]) lhsZ_;
    __global double (*lhspZ)[ny2+1][IMAXP+1][IMAXP+1] = (__global double (*)[ny2+1][IMAXP+1][IMAXP+1]) lhspZ_;
    __global double (*lhsmZ)[ny2+1][IMAXP+1][IMAXP+1] = (__global double (*)[ny2+1][IMAXP+1][IMAXP+1]) lhsmZ_;
    __global double (*rhs)[KMAX][JMAXP+1][IMAXP+1] = (__global double (*)[KMAX][JMAXP+1][IMAXP+1]) rhs_;

    int j = get_global_id(1);
    int i = get_global_id(0);

    int m;

    for (m = 0; m < 3; m++) {
        rhs[m][k][j][i] = rhs[m][k][j][i] - lhsZ[3][j][k][i]*rhs[m][k1][j][i];
    }

    rhs[3][k][j][i] = rhs[3][k][j][i] - lhspZ[3][j][k][i]*rhs[3][k1][j][i];
    rhs[4][k][j][i] = rhs[4][k][j][i] - lhsmZ[3][j][k][i]*rhs[4][k1][j][i];
}

__kernel void z_solve_12(__global double* restrict lhsZ_, __global double* restrict lhspZ_, __global double* restrict lhsmZ_, __global double* restrict rhs_, int gp23) {
    __global double (*lhsZ)[ny2+1][IMAXP+1][IMAXP+1] = (__global double (*)[ny2+1][IMAXP+1][IMAXP+1]) lhsZ_;
    __global double (*lhspZ)[ny2+1][IMAXP+1][IMAXP+1] = (__global double (*)[ny2+1][IMAXP+1][IMAXP+1]) lhspZ_;
    __global double (*lhsmZ)[ny2+1][IMAXP+1][IMAXP+1] = (__global double (*)[ny2+1][IMAXP+1][IMAXP+1]) lhsmZ_;
    __global double (*rhs)[KMAX][JMAXP+1][IMAXP+1] = (__global double (*)[KMAX][JMAXP+1][IMAXP+1]) rhs_;

    int j = get_global_id(1);
    int i = get_global_id(0);

    int m, k, k1, k2;

    for (k = gp23; k >= 0; k--) {
        k1 = k + 1;
        k2 = k + 2;
        for (m = 0; m < 3; m++) {
            rhs[m][k][j][i] = rhs[m][k][j][i] - 
                lhsZ[3][j][k][i]*rhs[m][k1][j][i] -
                lhsZ[4][j][k][i]*rhs[m][k2][j][i];
        }

        rhs[3][k][j][i] = rhs[3][k][j][i] - 
            lhspZ[3][j][k][i]*rhs[3][k1][j][i] -
            lhspZ[4][j][k][i]*rhs[3][k2][j][i];
        rhs[4][k][j][i] = rhs[4][k][j][i] - 
            lhsmZ[3][j][k][i]*rhs[4][k1][j][i] -
            lhsmZ[4][j][k][i]*rhs[4][k2][j][i];
    }
}

__kernel void txinvr_0(__global double* restrict rho_i_, __global double* restrict us_, __global double* restrict vs_, __global double* restrict ws_, __global double* restrict rhs_, __global double* restrict speed_, __global double* restrict qs_, double c2, double bt) {
    __global double (*rho_i)[JMAXP+1][IMAXP+1] = (__global double (*)[JMAXP+1][IMAXP+1]) rho_i_;
    __global double (*us)[JMAXP+1][IMAXP+1] = (__global double (*)[JMAXP+1][IMAXP+1]) us_;
    __global double (*vs)[JMAXP+1][IMAXP+1] = (__global double (*)[JMAXP+1][IMAXP+1]) vs_;
    __global double (*ws)[JMAXP+1][IMAXP+1] = (__global double (*)[JMAXP+1][IMAXP+1]) ws_;
    __global double (*rhs)[KMAX][JMAXP+1][IMAXP+1] = (__global double (*)[KMAX][JMAXP+1][IMAXP+1]) rhs_;
    __global double (*speed)[JMAXP+1][IMAXP+1] = (__global double (*)[JMAXP+1][IMAXP+1]) speed_;
    __global double (*qs)[JMAXP+1][IMAXP+1] = (__global double (*)[JMAXP+1][IMAXP+1]) qs_;

    int k = get_global_id(2);
    int j = get_global_id(1);
    int i = get_global_id(0);

    double ru1, uu, vv, ww, ac, ac2inv, r1, r2, r3, r4, r5, t1, t2, t3;

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

__kernel void compute_rhs_0(__global double* restrict u_, __global double* restrict rho_i_, __global double* restrict us_, __global double* restrict vs_, __global double* restrict ws_, __global double* restrict square_, __global double* restrict qs_, __global double* restrict speed_, double c1c2) {
    __global double (*u)[KMAX][JMAXP+1][IMAXP+1] = (__global double (*)[KMAX][JMAXP+1][IMAXP+1]) u_;
    __global double (*rho_i)[JMAXP+1][IMAXP+1] = (__global double (*)[JMAXP+1][IMAXP+1]) rho_i_;
    __global double (*us)[JMAXP+1][IMAXP+1] = (__global double (*)[JMAXP+1][IMAXP+1]) us_;
    __global double (*vs)[JMAXP+1][IMAXP+1] = (__global double (*)[JMAXP+1][IMAXP+1]) vs_;
    __global double (*ws)[JMAXP+1][IMAXP+1] = (__global double (*)[JMAXP+1][IMAXP+1]) ws_;
    __global double (*square)[JMAXP+1][IMAXP+1] = (__global double (*)[JMAXP+1][IMAXP+1]) square_;
    __global double (*qs)[JMAXP+1][IMAXP+1] = (__global double (*)[JMAXP+1][IMAXP+1]) qs_;
    __global double (*speed)[JMAXP+1][IMAXP+1] = (__global double (*)[JMAXP+1][IMAXP+1]) speed_;

    int k = get_global_id(2);
    int j = get_global_id(1);
    int i = get_global_id(0);

    double rho_inv, aux;

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
    aux = c1c2*rho_inv* (u[4][k][j][i] - square[k][j][i]);
    speed[k][j][i] = sqrt(aux);
}

__kernel void compute_rhs_1(__global double* restrict rhs_, __global double* restrict forcing_, int gp0) {
    __global double (*rhs)[KMAX][JMAXP+1][IMAXP+1] = (__global double (*)[KMAX][JMAXP+1][IMAXP+1]) rhs_;
    __global double (*forcing)[KMAX][JMAXP+1][IMAXP+1] = (__global double (*)[KMAX][JMAXP+1][IMAXP+1]) forcing_;

    int k = get_global_id(1);
    int j = get_global_id(0);

    int i, m;

    for (i = 0; i <= gp0-1; i++) {
        for (m = 0; m < 5; m++) {
            rhs[m][k][j][i] = forcing[m][k][j][i];
        }
    }
}

__kernel void compute_rhs_2(__global double* restrict us_, __global double* restrict rhs_, __global double* restrict u_, __global double* restrict square_, __global double* restrict vs_, __global double* restrict ws_, __global double* restrict qs_, __global double* restrict rho_i_, double dx1tx1, double dx2tx1, double dx3tx1, double dx4tx1, double dx5tx1, double tx2, double xxcon2, double xxcon3, double xxcon4, double xxcon5, double con43, double c1, double c2) {
    __global double (*us)[JMAXP+1][IMAXP+1] = (__global double (*)[JMAXP+1][IMAXP+1]) us_;
    __global double (*rhs)[KMAX][JMAXP+1][IMAXP+1] = (__global double (*)[KMAX][JMAXP+1][IMAXP+1]) rhs_;
    __global double (*u)[KMAX][JMAXP+1][IMAXP+1] = (__global double (*)[KMAX][JMAXP+1][IMAXP+1]) u_;
    __global double (*square)[JMAXP+1][IMAXP+1] = (__global double (*)[JMAXP+1][IMAXP+1]) square_;
    __global double (*vs)[JMAXP+1][IMAXP+1] = (__global double (*)[JMAXP+1][IMAXP+1]) vs_;
    __global double (*ws)[JMAXP+1][IMAXP+1] = (__global double (*)[JMAXP+1][IMAXP+1]) ws_;
    __global double (*qs)[JMAXP+1][IMAXP+1] = (__global double (*)[JMAXP+1][IMAXP+1]) qs_;
    __global double (*rho_i)[JMAXP+1][IMAXP+1] = (__global double (*)[JMAXP+1][IMAXP+1]) rho_i_;

    int k = get_global_id(2);
    int j = get_global_id(1);
    int i = get_global_id(0);

    double uijk, up1, um1;

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

__kernel void compute_rhs_3(__global double* restrict rhs_, __global double* restrict u_, double dssp, int i) {
    __global double (*rhs)[KMAX][JMAXP+1][IMAXP+1] = (__global double (*)[KMAX][JMAXP+1][IMAXP+1]) rhs_;
    __global double (*u)[KMAX][JMAXP+1][IMAXP+1] = (__global double (*)[KMAX][JMAXP+1][IMAXP+1]) u_;

    int k = get_global_id(2);
    int j = get_global_id(1);
    int m = get_global_id(0);

    rhs[m][k][j][i] = rhs[m][k][j][i]- dssp * 
        (5.0*u[m][k][j][i] - 4.0*u[m][k][j][i+1] + u[m][k][j][i+2]);
}

__kernel void compute_rhs_4(__global double* restrict rhs_, __global double* restrict u_, double dssp, int i) {
    __global double (*rhs)[KMAX][JMAXP+1][IMAXP+1] = (__global double (*)[KMAX][JMAXP+1][IMAXP+1]) rhs_;
    __global double (*u)[KMAX][JMAXP+1][IMAXP+1] = (__global double (*)[KMAX][JMAXP+1][IMAXP+1]) u_;

    int m = get_global_id(2);
    int k = get_global_id(1);
    int j = get_global_id(0);

    rhs[m][k][j][i] = rhs[m][k][j][i] - dssp * 
        (-4.0*u[m][k][j][i-1] + 6.0*u[m][k][j][i] -
         4.0*u[m][k][j][i+1] + u[m][k][j][i+2]);
}

__kernel void compute_rhs_5(__global double* restrict rhs_, __global double* restrict u_, double dssp) {
    __global double (*rhs)[KMAX][JMAXP+1][IMAXP+1] = (__global double (*)[KMAX][JMAXP+1][IMAXP+1]) rhs_;
    __global double (*u)[KMAX][JMAXP+1][IMAXP+1] = (__global double (*)[KMAX][JMAXP+1][IMAXP+1]) u_;

    int k = get_global_id(2);
    int j = get_global_id(1);
    int i = get_global_id(0);

    int m;

    for (m = 0; m < 5; m++) {
        rhs[m][k][j][i] = rhs[m][k][j][i] - dssp * 
            ( u[m][k][j][i-2] - 4.0*u[m][k][j][i-1] + 
              6.0*u[m][k][j][i] - 4.0*u[m][k][j][i+1] + 
              u[m][k][j][i+2] );
    }
}

__kernel void compute_rhs_6(__global double* restrict rhs_, __global double* restrict u_, double dssp, int i) {
    __global double (*rhs)[KMAX][JMAXP+1][IMAXP+1] = (__global double (*)[KMAX][JMAXP+1][IMAXP+1]) rhs_;
    __global double (*u)[KMAX][JMAXP+1][IMAXP+1] = (__global double (*)[KMAX][JMAXP+1][IMAXP+1]) u_;

    int k = get_global_id(0);

    int j, m;

    for (j = 1; j <= ny2; j++) {
        for (m = 0; m < 5; m++) {
            rhs[m][k][j][i] = rhs[m][k][j][i] - dssp *
                ( u[m][k][j][i-2] - 4.0*u[m][k][j][i-1] + 
                  6.0*u[m][k][j][i] - 4.0*u[m][k][j][i+1] );
        }
    }
}

__kernel void compute_rhs_7(__global double* restrict rhs_, __global double* restrict u_, double dssp, int i) {
    __global double (*rhs)[KMAX][JMAXP+1][IMAXP+1] = (__global double (*)[KMAX][JMAXP+1][IMAXP+1]) rhs_;
    __global double (*u)[KMAX][JMAXP+1][IMAXP+1] = (__global double (*)[KMAX][JMAXP+1][IMAXP+1]) u_;

    int k = get_global_id(0);

    int j, m;
    for (j = 1; j <= ny2; j++) {
        for (m = 0; m < 5; m++) {
            rhs[m][k][j][i] = rhs[m][k][j][i] - dssp *
                ( u[m][k][j][i-2] - 4.0*u[m][k][j][i-1] + 5.0*u[m][k][j][i] );
        }
    }
}

__kernel void compute_rhs_8(__global double* restrict vs_, __global double* restrict rhs_, __global double* restrict u_, __global double* restrict us_, __global double* restrict square_, __global double* restrict ws_, __global double* restrict qs_, __global double* restrict rho_i_, double dy1ty1, double dy2ty1, double dy3ty1, double dy4ty1, double dy5ty1, double ty2, double yycon2, double yycon3, double yycon4, double yycon5, double con43, double c1, double c2) {
    __global double (*vs)[JMAXP+1][IMAXP+1] = (__global double (*)[JMAXP+1][IMAXP+1]) vs_;
    __global double (*rhs)[KMAX][JMAXP+1][IMAXP+1] = (__global double (*)[KMAX][JMAXP+1][IMAXP+1]) rhs_;
    __global double (*u)[KMAX][JMAXP+1][IMAXP+1] = (__global double (*)[KMAX][JMAXP+1][IMAXP+1]) u_;
    __global double (*us)[JMAXP+1][IMAXP+1] = (__global double (*)[JMAXP+1][IMAXP+1]) us_;
    __global double (*square)[JMAXP+1][IMAXP+1] = (__global double (*)[JMAXP+1][IMAXP+1]) square_;
    __global double (*ws)[JMAXP+1][IMAXP+1] = (__global double (*)[JMAXP+1][IMAXP+1]) ws_;
    __global double (*qs)[JMAXP+1][IMAXP+1] = (__global double (*)[JMAXP+1][IMAXP+1]) qs_;
    __global double (*rho_i)[JMAXP+1][IMAXP+1] = (__global double (*)[JMAXP+1][IMAXP+1]) rho_i_;

    int k = get_global_id(2);
    int j = get_global_id(1);
    int i = get_global_id(0);

    double vijk, vp1, vm1;

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

__kernel void compute_rhs_9(__global double* restrict rhs_, __global double* restrict u_, double dssp, int j) {
    __global double (*rhs)[KMAX][JMAXP+1][IMAXP+1] = (__global double (*)[KMAX][JMAXP+1][IMAXP+1]) rhs_;
    __global double (*u)[KMAX][JMAXP+1][IMAXP+1] = (__global double (*)[KMAX][JMAXP+1][IMAXP+1]) u_;

    int k = get_global_id(2);
    int i = get_global_id(1);
    int m = get_global_id(0);

    rhs[m][k][j][i] = rhs[m][k][j][i]- dssp * 
        ( 5.0*u[m][k][j][i] - 4.0*u[m][k][j+1][i] + u[m][k][j+2][i]);
}

__kernel void compute_rhs_10(__global double* restrict rhs_, __global double* restrict u_, double dssp, int j) {
    __global double (*rhs)[KMAX][JMAXP+1][IMAXP+1] = (__global double (*)[KMAX][JMAXP+1][IMAXP+1]) rhs_;
    __global double (*u)[KMAX][JMAXP+1][IMAXP+1] = (__global double (*)[KMAX][JMAXP+1][IMAXP+1]) u_;

    int k = get_global_id(2);
    int i = get_global_id(1);
    int m = get_global_id(0);

    rhs[m][k][j][i] = rhs[m][k][j][i] - dssp * 
        (-4.0*u[m][k][j-1][i] + 6.0*u[m][k][j][i] -
         4.0*u[m][k][j+1][i] + u[m][k][j+2][i]);
}

__kernel void compute_rhs_11(__global double* restrict rhs_, __global double* restrict u_, double dssp) {
    __global double (*rhs)[KMAX][JMAXP+1][IMAXP+1] = (__global double (*)[KMAX][JMAXP+1][IMAXP+1]) rhs_;
    __global double (*u)[KMAX][JMAXP+1][IMAXP+1] = (__global double (*)[KMAX][JMAXP+1][IMAXP+1]) u_;

    int k = get_global_id(2);
    int j = get_global_id(1);
    int i = get_global_id(0);

    int m;

    for (m = 0; m < 5; m++) {
        rhs[m][k][j][i] = rhs[m][k][j][i] - dssp * 
            ( u[m][k][j-2][i] - 4.0*u[m][k][j-1][i] + 
              6.0*u[m][k][j][i] - 4.0*u[m][k][j+1][i] + 
              u[m][k][j+2][i] );
    }
}

__kernel void compute_rhs_12(__global double* restrict rhs_, __global double* restrict u_, double dssp, int j) {
    __global double (*rhs)[KMAX][JMAXP+1][IMAXP+1] = (__global double (*)[KMAX][JMAXP+1][IMAXP+1]) rhs_;
    __global double (*u)[KMAX][JMAXP+1][IMAXP+1] = (__global double (*)[KMAX][JMAXP+1][IMAXP+1]) u_;

    int k = get_global_id(2);
    int i = get_global_id(1);
    int m = get_global_id(0);

    rhs[m][k][j][i] = rhs[m][k][j][i] - dssp *
        ( u[m][k][j-2][i] - 4.0*u[m][k][j-1][i] + 
          6.0*u[m][k][j][i] - 4.0*u[m][k][j+1][i] );
}

__kernel void compute_rhs_13(__global double* restrict rhs_, __global double* restrict u_, double dssp, int j) {
    __global double (*rhs)[KMAX][JMAXP+1][IMAXP+1] = (__global double (*)[KMAX][JMAXP+1][IMAXP+1]) rhs_;
    __global double (*u)[KMAX][JMAXP+1][IMAXP+1] = (__global double (*)[KMAX][JMAXP+1][IMAXP+1]) u_;

    int k = get_global_id(2);
    int i = get_global_id(1);
    int m = get_global_id(0);

    rhs[m][k][j][i] = rhs[m][k][j][i] - dssp *
        ( u[m][k][j-2][i] - 4.0*u[m][k][j-1][i] + 5.0*u[m][k][j][i] );
}

__kernel void compute_rhs_14(__global double* restrict ws_, __global double* restrict rhs_, __global double* restrict u_, __global double* restrict us_, __global double* restrict vs_, __global double* restrict square_, __global double* restrict qs_, __global double* restrict rho_i_, double dz1tz1, double dz2tz1, double dz3tz1, double dz4tz1, double dz5tz1, double tz2, double zzcon2, double zzcon3, double zzcon4, double zzcon5, double con43, double c1, double c2) {
    __global double (*ws)[JMAXP+1][IMAXP+1] = (__global double (*)[JMAXP+1][IMAXP+1]) ws_;
    __global double (*rhs)[KMAX][JMAXP+1][IMAXP+1] = (__global double (*)[KMAX][JMAXP+1][IMAXP+1]) rhs_;
    __global double (*u)[KMAX][JMAXP+1][IMAXP+1] = (__global double (*)[KMAX][JMAXP+1][IMAXP+1]) u_;
    __global double (*us)[JMAXP+1][IMAXP+1] = (__global double (*)[JMAXP+1][IMAXP+1]) us_;
    __global double (*vs)[JMAXP+1][IMAXP+1] = (__global double (*)[JMAXP+1][IMAXP+1]) vs_;
    __global double (*square)[JMAXP+1][IMAXP+1] = (__global double (*)[JMAXP+1][IMAXP+1]) square_;
    __global double (*qs)[JMAXP+1][IMAXP+1] = (__global double (*)[JMAXP+1][IMAXP+1]) qs_;
    __global double (*rho_i)[JMAXP+1][IMAXP+1] = (__global double (*)[JMAXP+1][IMAXP+1]) rho_i_;

    int k = get_global_id(2);
    int j = get_global_id(1);
    int i = get_global_id(0);

    double wijk, wp1, wm1;

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

__kernel void compute_rhs_15(__global double* restrict rhs_, __global double* restrict u_, double dssp, int k) {
    __global double (*rhs)[KMAX][JMAXP+1][IMAXP+1] = (__global double (*)[KMAX][JMAXP+1][IMAXP+1]) rhs_;
    __global double (*u)[KMAX][JMAXP+1][IMAXP+1] = (__global double (*)[KMAX][JMAXP+1][IMAXP+1]) u_;

    int j = get_global_id(2);
    int i = get_global_id(1);
    int m = get_global_id(0);

    rhs[m][k][j][i] = rhs[m][k][j][i]- dssp * 
        (5.0*u[m][k][j][i] - 4.0*u[m][k+1][j][i] + u[m][k+2][j][i]);
}

__kernel void compute_rhs_16(__global double* restrict rhs_, __global double* restrict u_, double dssp, int k) {
    __global double (*rhs)[KMAX][JMAXP+1][IMAXP+1] = (__global double (*)[KMAX][JMAXP+1][IMAXP+1]) rhs_;
    __global double (*u)[KMAX][JMAXP+1][IMAXP+1] = (__global double (*)[KMAX][JMAXP+1][IMAXP+1]) u_;

    int j = get_global_id(2);
    int i = get_global_id(1);
    int m = get_global_id(0);

    rhs[m][k][j][i] = rhs[m][k][j][i] - dssp * 
        (-4.0*u[m][k-1][j][i] + 6.0*u[m][k][j][i] -
         4.0*u[m][k+1][j][i] + u[m][k+2][j][i]);
}

__kernel void compute_rhs_17(__global double* restrict rhs_, __global double* restrict u_, double dssp) {
    __global double (*rhs)[KMAX][JMAXP+1][IMAXP+1] = (__global double (*)[KMAX][JMAXP+1][IMAXP+1]) rhs_;
    __global double (*u)[KMAX][JMAXP+1][IMAXP+1] = (__global double (*)[KMAX][JMAXP+1][IMAXP+1]) u_;

    int k = get_global_id(2);
    int j = get_global_id(1);
    int i = get_global_id(0);

    int m;

    for (m = 0; m < 5; m++) {
        rhs[m][k][j][i] = rhs[m][k][j][i] - dssp * 
            ( u[m][k-2][j][i] - 4.0*u[m][k-1][j][i] + 
              6.0*u[m][k][j][i] - 4.0*u[m][k+1][j][i] + 
              u[m][k+2][j][i] );
    }
}

__kernel void compute_rhs_18(__global double* restrict rhs_, __global double* restrict u_, double dssp, int k) {
    __global double (*rhs)[KMAX][JMAXP+1][IMAXP+1] = (__global double (*)[KMAX][JMAXP+1][IMAXP+1]) rhs_;
    __global double (*u)[KMAX][JMAXP+1][IMAXP+1] = (__global double (*)[KMAX][JMAXP+1][IMAXP+1]) u_;

    int j = get_global_id(2);
    int i = get_global_id(1);
    int m = get_global_id(0);

    rhs[m][k][j][i] = rhs[m][k][j][i] - dssp *
        ( u[m][k-2][j][i] - 4.0*u[m][k-1][j][i] + 
          6.0*u[m][k][j][i] - 4.0*u[m][k+1][j][i] );
}

__kernel void compute_rhs_19(__global double* restrict rhs_, __global double* restrict u_, double dssp, int k) {
    __global double (*rhs)[KMAX][JMAXP+1][IMAXP+1] = (__global double (*)[KMAX][JMAXP+1][IMAXP+1]) rhs_;
    __global double (*u)[KMAX][JMAXP+1][IMAXP+1] = (__global double (*)[KMAX][JMAXP+1][IMAXP+1]) u_;

    int j = get_global_id(2);
    int i = get_global_id(1);
    int m = get_global_id(0);

    rhs[m][k][j][i] = rhs[m][k][j][i] - dssp *
        ( u[m][k-2][j][i] - 4.0*u[m][k-1][j][i] + 5.0*u[m][k][j][i] );
}

__kernel void compute_rhs_20(__global double* restrict rhs_, double dt) {
    __global double (*rhs)[KMAX][JMAXP+1][IMAXP+1] = (__global double (*)[KMAX][JMAXP+1][IMAXP+1]) rhs_;

    int k = get_global_id(2);
    int j = get_global_id(1);
    int i = get_global_id(0);

    int m;

    for (m = 0; m < 5; m++) {
        rhs[m][k][j][i] = rhs[m][k][j][i] * dt;
    }
}
