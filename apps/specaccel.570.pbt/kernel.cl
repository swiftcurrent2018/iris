#pragma OPENCL EXTENSION cl_khr_fp64 : enable

#define PROBLEM_SIZE 102
#define IMAX      PROBLEM_SIZE
#define JMAX      PROBLEM_SIZE
#define KMAX      PROBLEM_SIZE
#define IMAXP     IMAX/2*2
#define JMAXP     JMAX/2*2

#define AA            0
#define BB            1
#define CC            2
#define BLOCK_SIZE    5

__kernel void add_0(__global double* restrict u_, __global double* restrict rhs_) {
    __global double (*u)[JMAXP+1][IMAXP+1][5] = (__global double (*)[JMAXP+1][IMAXP+1][5]) u_;
    __global double (*rhs)[JMAXP+1][IMAXP+1][5] = (__global double (*)[JMAXP+1][IMAXP+1][5]) rhs_;

    int k = get_global_id(2);
    int j = get_global_id(1);
    int i = get_global_id(0);

    u[k][j][i][0] = u[k][j][i][0] + rhs[k][j][i][0];
    u[k][j][i][1] = u[k][j][i][1] + rhs[k][j][i][1];
    u[k][j][i][2] = u[k][j][i][2] + rhs[k][j][i][2];
    u[k][j][i][3] = u[k][j][i][3] + rhs[k][j][i][3];
    u[k][j][i][4] = u[k][j][i][4] + rhs[k][j][i][4];
}

__kernel void compute_rhs_0(__global double* restrict u_, __global double* restrict rho_i_, __global double* restrict us_, __global double* restrict vs_, __global double* restrict ws_, __global double* restrict square_, __global double* restrict qs_) {
    __global double (*u)[JMAXP+1][IMAXP+1][5] = (__global double (*)[JMAXP+1][IMAXP+1][5]) u_;
    __global double (*rho_i)[JMAXP+1][IMAXP+1] = (__global double (*)[JMAXP+1][IMAXP+1]) rho_i_;
    __global double (*us)[JMAXP+1][IMAXP+1] = (__global double (*)[JMAXP+1][IMAXP+1]) us_;
    __global double (*vs)[JMAXP+1][IMAXP+1] = (__global double (*)[JMAXP+1][IMAXP+1]) vs_;
    __global double (*ws)[JMAXP+1][IMAXP+1] = (__global double (*)[JMAXP+1][IMAXP+1]) ws_;
    __global double (*square)[JMAXP+1][IMAXP+1] = (__global double (*)[JMAXP+1][IMAXP+1]) square_;
    __global double (*qs)[JMAXP+1][IMAXP+1] = (__global double (*)[JMAXP+1][IMAXP+1]) qs_;

    int k = get_global_id(2);
    int j = get_global_id(1);
    int i = get_global_id(0);

    double rho_inv;

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

__kernel void compute_rhs_1(__global double* restrict rhs_, __global double* restrict forcing_, int gp01) {
    __global double (*rhs)[JMAXP+1][IMAXP+1][5] = (__global double (*)[JMAXP+1][IMAXP+1][5]) rhs_;
    __global double (*forcing)[JMAXP+1][IMAXP+1][5] = (__global double (*)[JMAXP+1][IMAXP+1][5]) forcing_;

    int k = get_global_id(1);
    int j = get_global_id(0);

    int i;

    for (i = 0; i <= gp01; i++) {
        rhs[k][j][i][0] = forcing[k][j][i][0];
        rhs[k][j][i][1] = forcing[k][j][i][1];
        rhs[k][j][i][2] = forcing[k][j][i][2];
        rhs[k][j][i][3] = forcing[k][j][i][3];
        rhs[k][j][i][4] = forcing[k][j][i][4];
    }
}

__kernel void compute_rhs_2(__global double* restrict us_, __global double* restrict rhs_, __global double* restrict u_, __global double* restrict square_, __global double* restrict vs_, __global double* restrict ws_, __global double* restrict qs_, __global double* restrict rho_i_, double dx1tx1, double dx2tx1, double dx3tx1, double dx4tx1, double dx5tx1, double tx2, double xxcon2, double xxcon3, double xxcon4, double xxcon5, double con43, double c1, double c2, double dssp, int gp0, int gp02, int gp12) {
    __global double (*us)[JMAXP+1][IMAXP+1] = (__global double (*)[JMAXP+1][IMAXP+1]) us_;
    __global double (*rhs)[JMAXP+1][IMAXP+1][5] = (__global double (*)[JMAXP+1][IMAXP+1][5]) rhs_;
    __global double (*u)[JMAXP+1][IMAXP+1][5] = (__global double (*)[JMAXP+1][IMAXP+1][5]) u_;
    __global double (*square)[JMAXP+1][IMAXP+1] = (__global double (*)[JMAXP+1][IMAXP+1]) square_;
    __global double (*vs)[JMAXP+1][IMAXP+1] = (__global double (*)[JMAXP+1][IMAXP+1]) vs_;
    __global double (*ws)[JMAXP+1][IMAXP+1] = (__global double (*)[JMAXP+1][IMAXP+1]) ws_;
    __global double (*qs)[JMAXP+1][IMAXP+1] = (__global double (*)[JMAXP+1][IMAXP+1]) qs_;
    __global double (*rho_i)[JMAXP+1][IMAXP+1] = (__global double (*)[JMAXP+1][IMAXP+1]) rho_i_;

    int k = get_global_id(0);
    int j, i;
    double uijk, up1, um1;

    for (j = 1; j <= gp12; j++) {
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

__kernel void compute_rhs_3(__global double* restrict vs_, __global double* restrict rhs_, __global double* restrict u_, __global double* restrict us_, __global double* restrict square_, __global double* restrict ws_, __global double* restrict qs_, __global double* restrict rho_i_, double dy1ty1, double dy2ty1, double dy3ty1, double dy4ty1, double dy5ty1, double ty2, double yycon2, double yycon3, double yycon4, double yycon5, double con43, double c1, double c2, double dssp, int gp1, int gp02, int gp12) {
    __global double (*vs)[JMAXP+1][IMAXP+1] = (__global double (*)[JMAXP+1][IMAXP+1]) vs_;
    __global double (*rhs)[JMAXP+1][IMAXP+1][5] = (__global double (*)[JMAXP+1][IMAXP+1][5]) rhs_;
    __global double (*u)[JMAXP+1][IMAXP+1][5] = (__global double (*)[JMAXP+1][IMAXP+1][5]) u_;
    __global double (*us)[JMAXP+1][IMAXP+1] = (__global double (*)[JMAXP+1][IMAXP+1]) us_;
    __global double (*square)[JMAXP+1][IMAXP+1] = (__global double (*)[JMAXP+1][IMAXP+1]) square_;
    __global double (*ws)[JMAXP+1][IMAXP+1] = (__global double (*)[JMAXP+1][IMAXP+1]) ws_;
    __global double (*qs)[JMAXP+1][IMAXP+1] = (__global double (*)[JMAXP+1][IMAXP+1]) qs_;
    __global double (*rho_i)[JMAXP+1][IMAXP+1] = (__global double (*)[JMAXP+1][IMAXP+1]) rho_i_;

    int k = get_global_id(0);
    int j, i;
    double vijk, vp1, vm1;

    for (j = 1; j <= gp12; j++) {
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

    j = 1;
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

__kernel void compute_rhs_4(__global double* restrict ws_, __global double* restrict rhs_, __global double* restrict u_, __global double* restrict us_, __global double* restrict vs_, __global double* restrict square_, __global double* restrict qs_, __global double* restrict rho_i_, double dz1tz1, double dz2tz1, double dz3tz1, double dz4tz1, double dz5tz1, double tz2, double zzcon2, double zzcon3, double zzcon4, double zzcon5, double con43, double c1, double c2, int gp02) {
    __global double (*ws)[JMAXP+1][IMAXP+1] = (__global double (*)[JMAXP+1][IMAXP+1]) ws_;
    __global double (*rhs)[JMAXP+1][IMAXP+1][5] = (__global double (*)[JMAXP+1][IMAXP+1][5]) rhs_;
    __global double (*u)[JMAXP+1][IMAXP+1][5] = (__global double (*)[JMAXP+1][IMAXP+1][5]) u_;
    __global double (*us)[JMAXP+1][IMAXP+1] = (__global double (*)[JMAXP+1][IMAXP+1]) us_;
    __global double (*vs)[JMAXP+1][IMAXP+1] = (__global double (*)[JMAXP+1][IMAXP+1]) vs_;
    __global double (*square)[JMAXP+1][IMAXP+1] = (__global double (*)[JMAXP+1][IMAXP+1]) square_;
    __global double (*qs)[JMAXP+1][IMAXP+1] = (__global double (*)[JMAXP+1][IMAXP+1]) qs_;
    __global double (*rho_i)[JMAXP+1][IMAXP+1] = (__global double (*)[JMAXP+1][IMAXP+1]) rho_i_;

    int k = get_global_id(0);
    int j = get_global_id(1);

    int i;
    double wijk, wp1, wm1;

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

__kernel void compute_rhs_5(__global double* restrict rhs_, __global double* restrict u_, double dssp, int k) {
    __global double (*rhs)[JMAXP+1][IMAXP+1][5] = (__global double (*)[JMAXP+1][IMAXP+1][5]) rhs_;
    __global double (*u)[JMAXP+1][IMAXP+1][5] = (__global double (*)[JMAXP+1][IMAXP+1][5]) u_;

    int j = get_global_id(1);
    int i = get_global_id(0);

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

__kernel void compute_rhs_6(__global double* restrict rhs_, __global double* restrict u_, double dssp, int k) {
    __global double (*rhs)[JMAXP+1][IMAXP+1][5] = (__global double (*)[JMAXP+1][IMAXP+1][5]) rhs_;
    __global double (*u)[JMAXP+1][IMAXP+1][5] = (__global double (*)[JMAXP+1][IMAXP+1][5]) u_;

    int j = get_global_id(1);
    int i = get_global_id(0);

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

__kernel void compute_rhs_7(__global double* restrict rhs_, __global double* restrict u_, double dssp) {
    __global double (*rhs)[JMAXP+1][IMAXP+1][5] = (__global double (*)[JMAXP+1][IMAXP+1][5]) rhs_;
    __global double (*u)[JMAXP+1][IMAXP+1][5] = (__global double (*)[JMAXP+1][IMAXP+1][5]) u_;

    int k = get_global_id(2);
    int j = get_global_id(1);
    int i = get_global_id(0);

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

__kernel void compute_rhs_8(__global double* restrict rhs_, __global double* restrict u_, double dssp, int k) {
    __global double (*rhs)[JMAXP+1][IMAXP+1][5] = (__global double (*)[JMAXP+1][IMAXP+1][5]) rhs_;
    __global double (*u)[JMAXP+1][IMAXP+1][5] = (__global double (*)[JMAXP+1][IMAXP+1][5]) u_;

    int j = get_global_id(1);
    int i = get_global_id(0);

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

__kernel void compute_rhs_9(__global double* restrict rhs_, __global double* restrict u_, double dssp, int k) {
    __global double (*rhs)[JMAXP+1][IMAXP+1][5] = (__global double (*)[JMAXP+1][IMAXP+1][5]) rhs_;
    __global double (*u)[JMAXP+1][IMAXP+1][5] = (__global double (*)[JMAXP+1][IMAXP+1][5]) u_;

    int j = get_global_id(1);
    int i = get_global_id(0);

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

__kernel void compute_rhs_10(__global double* restrict rhs_, double dt) {
    __global double (*rhs)[JMAXP+1][IMAXP+1][5] = (__global double (*)[JMAXP+1][IMAXP+1][5]) rhs_;

    int k = get_global_id(2);
    int j = get_global_id(1);
    int i = get_global_id(0);

    rhs[k][j][i][0] = rhs[k][j][i][0] * dt;
    rhs[k][j][i][1] = rhs[k][j][i][1] * dt;
    rhs[k][j][i][2] = rhs[k][j][i][2] * dt;
    rhs[k][j][i][3] = rhs[k][j][i][3] * dt;
    rhs[k][j][i][4] = rhs[k][j][i][4] * dt;
}

__kernel void x_solve_0(__global double* restrict rho_i_, __global double* restrict fjacX_, __global double* restrict njacX_, __global double* restrict u_, __global double* restrict qs_, __global double* restrict square_, double c1, double c2, double c3c4, double c1345, double con43, int gp22) {
    __global double (*rho_i)[JMAXP+1][IMAXP+1] = (__global double (*)[JMAXP+1][IMAXP+1]) rho_i_;
    __global double (*fjacX)[5][PROBLEM_SIZE+1][JMAXP-1][KMAX-1] = (__global double (*)[5][PROBLEM_SIZE+1][JMAXP-1][KMAX-1]) fjacX_;
    __global double (*njacX)[5][PROBLEM_SIZE+1][JMAXP-1][KMAX-1] = (__global double (*)[5][PROBLEM_SIZE+1][JMAXP-1][KMAX-1]) njacX_;
    __global double (*u)[JMAXP+1][IMAXP+1][5] = (__global double (*)[JMAXP+1][IMAXP+1][5]) u_;
    __global double (*qs)[JMAXP+1][IMAXP+1] = (__global double (*)[JMAXP+1][IMAXP+1]) qs_;
    __global double (*square)[JMAXP+1][IMAXP+1] = (__global double (*)[JMAXP+1][IMAXP+1]) square_;

    int i = get_global_id(1);
    int j = get_global_id(0);
    int k;

    double temp1, temp2, temp3;

    for (k = 1; k <= gp22; k++) {

        temp1 = rho_i[k][j][i];
        temp2 = temp1 * temp1;
        temp3 = temp1 * temp2;

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

__kernel void x_solve_1(__global double* restrict lhsX_, int isize) {
    __global double (*lhsX)[5][3][PROBLEM_SIZE][JMAXP-1][KMAX-1] = (__global double (*)[5][3][PROBLEM_SIZE][JMAXP-1][KMAX-1]) lhsX_;
    int k = get_global_id(2);
    int j = get_global_id(1);
    int n = get_global_id(0);

    int m;

    for (m = 0; m < 5; m++){
        lhsX[m][n][0][0][j][k] = 0.0;
        lhsX[m][n][1][0][j][k] = 0.0;
        lhsX[m][n][2][0][j][k] = 0.0;
        lhsX[m][n][0][isize][j][k] = 0.0;
        lhsX[m][n][1][isize][j][k] = 0.0;
        lhsX[m][n][2][isize][j][k] = 0.0;
    }
}

__kernel void x_solve_2(__global double* restrict lhsX_, int isize) {
    __global double (*lhsX)[5][3][PROBLEM_SIZE][JMAXP-1][KMAX-1] = (__global double (*)[5][3][PROBLEM_SIZE][JMAXP-1][KMAX-1]) lhsX_;
    int k = get_global_id(1);
    int j = get_global_id(0);

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

__kernel void x_solve_3(__global double* restrict lhsX_, __global double* restrict fjacX_, __global double* restrict njacX_, double dt, double tx1, double tx2, double dx1, double dx2, double dx3, double dx4, double dx5) {
    __global double (*lhsX)[5][3][PROBLEM_SIZE][JMAXP-1][KMAX-1] = (__global double (*)[5][3][PROBLEM_SIZE][JMAXP-1][KMAX-1]) lhsX_;
    __global double (*fjacX)[5][PROBLEM_SIZE+1][JMAXP-1][KMAX-1] = (__global double (*)[5][PROBLEM_SIZE+1][JMAXP-1][KMAX-1]) fjacX_;
    __global double (*njacX)[5][PROBLEM_SIZE+1][JMAXP-1][KMAX-1] = (__global double (*)[5][PROBLEM_SIZE+1][JMAXP-1][KMAX-1]) njacX_;

    int i = get_global_id(2);
    int j = get_global_id(1);
    int k = get_global_id(0);

    double temp1, temp2;

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

__kernel void x_solve_4(__global double* restrict lhsX_, __global double* restrict rhs_) {
    __global double (*lhsX)[5][3][PROBLEM_SIZE][JMAXP-1][KMAX-1] = (__global double (*)[5][3][PROBLEM_SIZE][JMAXP-1][KMAX-1]) lhsX_;
    __global double (*rhs)[JMAXP+1][IMAXP+1][5] = (__global double (*)[JMAXP+1][IMAXP+1][5]) rhs_;

    int j = get_global_id(1);
    int k = get_global_id(0);

    double pivot, coeff;

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
}

__kernel void x_solve_5(__global double* restrict lhsX_, __global double* restrict rhs_, int isize, int gp22) {
    __global double (*lhsX)[5][3][PROBLEM_SIZE][JMAXP-1][KMAX-1] = (__global double (*)[5][3][PROBLEM_SIZE][JMAXP-1][KMAX-1]) lhsX_;
    __global double (*rhs)[JMAXP+1][IMAXP+1][5] = (__global double (*)[JMAXP+1][IMAXP+1][5]) rhs_;

    int j = get_global_id(0);
    int i, k;

    double pivot, coeff;

    for (i = 1; i <= isize-1; i++) {
        for (k = 1; k <= gp22; k++) {
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

        }
    }
}

__kernel void x_solve_6(__global double* restrict lhsX_, __global double* restrict rhs_, int isize) {
    __global double (*lhsX)[5][3][PROBLEM_SIZE][JMAXP-1][KMAX-1] = (__global double (*)[5][3][PROBLEM_SIZE][JMAXP-1][KMAX-1]) lhsX_;
    __global double (*rhs)[JMAXP+1][IMAXP+1][5] = (__global double (*)[JMAXP+1][IMAXP+1][5]) rhs_;

    int k = get_global_id(1);
    int j = get_global_id(0);

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

__kernel void x_solve_7(__global double* restrict lhsX_, int isize) {
    __global double (*lhsX)[5][3][PROBLEM_SIZE][JMAXP-1][KMAX-1] = (__global double (*)[5][3][PROBLEM_SIZE][JMAXP-1][KMAX-1]) lhsX_;

    int k = get_global_id(1);
    int j = get_global_id(0);

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

__kernel void x_solve_8(__global double* restrict lhsX_, __global double* restrict rhs_, int isize) {
    __global double (*lhsX)[5][3][PROBLEM_SIZE][JMAXP-1][KMAX-1] = (__global double (*)[5][3][PROBLEM_SIZE][JMAXP-1][KMAX-1]) lhsX_;
    __global double (*rhs)[JMAXP+1][IMAXP+1][5] = (__global double (*)[JMAXP+1][IMAXP+1][5]) rhs_;

    int k = get_global_id(1);
    int j = get_global_id(0);

    double pivot, coeff;

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

__kernel void x_solve_9(__global double* restrict lhsX_, __global double* restrict rhs_, int isize) {
    __global double (*lhsX)[5][3][PROBLEM_SIZE][JMAXP-1][KMAX-1] = (__global double (*)[5][3][PROBLEM_SIZE][JMAXP-1][KMAX-1]) lhsX_;
    __global double (*rhs)[JMAXP+1][IMAXP+1][5] = (__global double (*)[JMAXP+1][IMAXP+1][5]) rhs_;

    int k = get_global_id(1);
    int j = get_global_id(0);
    int i, m, n;

    for (i = isize-1; i >=0; i--) {
        for (m = 0; m < BLOCK_SIZE; m++) {
            for (n = 0; n < BLOCK_SIZE; n++) {
                rhs[k][j][i][m] = rhs[k][j][i][m]
                    - lhsX[m][n][CC][i][j][k]*rhs[k][j][i+1][n];
            }
        }
    }
}

__kernel void y_solve_0(__global double* restrict rho_i_, __global double* restrict fjacY_, __global double* restrict njacY_, __global double* restrict u_, __global double* restrict qs_, __global double* restrict square_, double c1, double c2, double c3c4, double c1345, double con43, int gp22) {
    __global double (*rho_i)[JMAXP+1][IMAXP+1] = (__global double (*)[JMAXP+1][IMAXP+1]) rho_i_;
    __global double (*fjacY)[5][PROBLEM_SIZE+1][IMAXP-1][KMAX-1] = (__global double (*)[5][PROBLEM_SIZE+1][IMAXP-1][KMAX-1]) fjacY_;
    __global double (*njacY)[5][PROBLEM_SIZE+1][IMAXP-1][KMAX-1] = (__global double (*)[5][PROBLEM_SIZE+1][IMAXP-1][KMAX-1]) njacY_;
    __global double (*u)[JMAXP+1][IMAXP+1][5] = (__global double (*)[JMAXP+1][IMAXP+1][5]) u_;
    __global double (*qs)[JMAXP+1][IMAXP+1] = (__global double (*)[JMAXP+1][IMAXP+1]) qs_;
    __global double (*square)[JMAXP+1][IMAXP+1] = (__global double (*)[JMAXP+1][IMAXP+1]) square_;

    int j = get_global_id(1);
    int i = get_global_id(0);
    int k;

    double temp1, temp2, temp3;

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

__kernel void y_solve_1(__global double* restrict lhsY_, int jsize, int gp22) {
    __global double (*lhsY)[5][3][PROBLEM_SIZE][IMAXP-1][KMAX-1] = (__global double (*)[5][3][PROBLEM_SIZE][IMAXP-1][KMAX-1]) lhsY_;

    int m = get_global_id(2);
    int n = get_global_id(1);
    int i = get_global_id(0);
    int k;

    for (k = 1; k <= gp22; k++) {
        lhsY[m][n][0][0][i][k] = 0.0;
        lhsY[m][n][1][0][i][k] = 0.0;
        lhsY[m][n][2][0][i][k] = 0.0;
        lhsY[m][n][0][jsize][i][k] = 0.0;
        lhsY[m][n][1][jsize][i][k] = 0.0;
        lhsY[m][n][2][jsize][i][k] = 0.0;
    }

}

__kernel void y_solve_2(__global double* restrict lhsY_, int jsize) {
    __global double (*lhsY)[5][3][PROBLEM_SIZE][IMAXP-1][KMAX-1] = (__global double (*)[5][3][PROBLEM_SIZE][IMAXP-1][KMAX-1]) lhsY_;

    int m = get_global_id(2);
    int i = get_global_id(1);
    int k = get_global_id(0);

    lhsY[m][m][1][0][i][k] = 1.0;
    lhsY[m][m][1][jsize][i][k] = 1.0;
}

__kernel void y_solve_3(__global double* restrict lhsY_, __global double* restrict fjacY_, __global double* restrict njacY_, double dtty1, double dtty2, double dy1, double dy2, double dy3, double dy4, double dy5) {
    __global double (*lhsY)[5][3][PROBLEM_SIZE][IMAXP-1][KMAX-1] = (__global double (*)[5][3][PROBLEM_SIZE][IMAXP-1][KMAX-1]) lhsY_;
    __global double (*fjacY)[5][PROBLEM_SIZE+1][IMAXP-1][KMAX-1] = (__global double (*)[5][PROBLEM_SIZE+1][IMAXP-1][KMAX-1]) fjacY_;
    __global double (*njacY)[5][PROBLEM_SIZE+1][IMAXP-1][KMAX-1] = (__global double (*)[5][PROBLEM_SIZE+1][IMAXP-1][KMAX-1]) njacY_;

    int j = get_global_id(2);
    int i = get_global_id(1);
    int k = get_global_id(0);

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

__kernel void y_solve_4(__global double* restrict lhsY_, __global double* restrict rhs_) {
    __global double (*lhsY)[5][3][PROBLEM_SIZE][IMAXP-1][KMAX-1] = (__global double (*)[5][3][PROBLEM_SIZE][IMAXP-1][KMAX-1]) lhsY_;
    __global double (*rhs)[JMAXP+1][IMAXP+1][5] = (__global double (*)[JMAXP+1][IMAXP+1][5]) rhs_;

    int i = get_global_id(1);
    int k = get_global_id(0);

    double pivot, coeff;

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
}

__kernel void y_solve_5(__global double* restrict lhsY_, __global double* restrict rhs_, int jsize, int gp22) {
    __global double (*lhsY)[5][3][PROBLEM_SIZE][IMAXP-1][KMAX-1] = (__global double (*)[5][3][PROBLEM_SIZE][IMAXP-1][KMAX-1]) lhsY_;
    __global double (*rhs)[JMAXP+1][IMAXP+1][5] = (__global double (*)[JMAXP+1][IMAXP+1][5]) rhs_;

    int i = get_global_id(0);
    int j, k;

    double pivot, coeff;

    for (j = 1; j <= jsize-1; j++) {
        for (k = 1; k <= gp22; k++) {
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
        }
    }
}

__kernel void y_solve_6(__global double* restrict lhsY_, __global double* restrict rhs_, int jsize) {
    __global double (*lhsY)[5][3][PROBLEM_SIZE][IMAXP-1][KMAX-1] = (__global double (*)[5][3][PROBLEM_SIZE][IMAXP-1][KMAX-1]) lhsY_;
    __global double (*rhs)[JMAXP+1][IMAXP+1][5] = (__global double (*)[JMAXP+1][IMAXP+1][5]) rhs_;

    int i = get_global_id(1);
    int k = get_global_id(0);

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

__kernel void y_solve_7(__global double* restrict lhsY_, int jsize) {
    __global double (*lhsY)[5][3][PROBLEM_SIZE][IMAXP-1][KMAX-1] = (__global double (*)[5][3][PROBLEM_SIZE][IMAXP-1][KMAX-1]) lhsY_;

    int i = get_global_id(1);
    int k = get_global_id(0);

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

__kernel void y_solve_8(__global double* restrict lhsY_, __global double* restrict rhs_, int jsize) {
    __global double (*lhsY)[5][3][PROBLEM_SIZE][IMAXP-1][KMAX-1] = (__global double (*)[5][3][PROBLEM_SIZE][IMAXP-1][KMAX-1]) lhsY_;
    __global double (*rhs)[JMAXP+1][IMAXP+1][5] = (__global double (*)[JMAXP+1][IMAXP+1][5]) rhs_;

    int i = get_global_id(1);
    int k = get_global_id(0);

    double pivot, coeff;

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

__kernel void y_solve_9(__global double* restrict lhsY_, __global double* restrict rhs_, int jsize) {
    __global double (*lhsY)[5][3][PROBLEM_SIZE][IMAXP-1][KMAX-1] = (__global double (*)[5][3][PROBLEM_SIZE][IMAXP-1][KMAX-1]) lhsY_;
    __global double (*rhs)[JMAXP+1][IMAXP+1][5] = (__global double (*)[JMAXP+1][IMAXP+1][5]) rhs_;

    int k = get_global_id(1);
    int i = get_global_id(0);
    int j, m, n;

    for (j = jsize-1; j >= 0; j--) {
        for (m = 0; m < BLOCK_SIZE; m++) {
            for (n = 0; n < BLOCK_SIZE; n++) {
                rhs[k][j][i][m] = rhs[k][j][i][m]
                    - lhsY[m][n][CC][j][i][k]*rhs[k][j+1][i][n];
            }
        }
    }
}

__kernel void z_solve_0(__global double* restrict u_, __global double* restrict fjacZ_, __global double* restrict njacZ_, __global double* restrict qs_, __global double* restrict square_, double c1, double c2, double c3c4, double c1345, double con43, int gp12) {
    __global double (*u)[JMAXP+1][IMAXP+1][5] = (__global double (*)[JMAXP+1][IMAXP+1][5]) u_;
    __global double (*fjacZ)[5][PROBLEM_SIZE+1][IMAXP-1][JMAXP-1] = (__global double (*)[5][PROBLEM_SIZE+1][IMAXP-1][JMAXP-1]) fjacZ_;
    __global double (*njacZ)[5][PROBLEM_SIZE+1][IMAXP-1][JMAXP-1] = (__global double (*)[5][PROBLEM_SIZE+1][IMAXP-1][JMAXP-1]) njacZ_;
    __global double (*qs)[JMAXP+1][IMAXP+1] = (__global double (*)[JMAXP+1][IMAXP+1]) qs_;
    __global double (*square)[JMAXP+1][IMAXP+1] = (__global double (*)[JMAXP+1][IMAXP+1]) square_;

    int k = get_global_id(1);
    int i = get_global_id(0);
    int j;

    double temp1, temp2, temp3;

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

__kernel void z_solve_1(__global double* restrict lhsZ_, int ksize, int gp12) {
    __global double (*lhsZ)[5][3][PROBLEM_SIZE][IMAXP-1][JMAXP-1] = (__global double (*)[5][3][PROBLEM_SIZE][IMAXP-1][JMAXP-1]) lhsZ_;

    int m = get_global_id(2);
    int n = get_global_id(1);
    int i = get_global_id(0);
    int j;

    for (j = 1; j <= gp12; j++) {
        lhsZ[m][n][0][0][i][j] = 0.0;
        lhsZ[m][n][1][0][i][j] = 0.0;
        lhsZ[m][n][2][0][i][j] = 0.0;
        lhsZ[m][n][0][ksize][i][j] = 0.0;
        lhsZ[m][n][1][ksize][i][j] = 0.0;
        lhsZ[m][n][2][ksize][i][j] = 0.0;
    }
}

__kernel void z_solve_2(__global double* restrict lhsZ_, int ksize) {
    __global double (*lhsZ)[5][3][PROBLEM_SIZE][IMAXP-1][JMAXP-1] = (__global double (*)[5][3][PROBLEM_SIZE][IMAXP-1][JMAXP-1]) lhsZ_;

    int m = get_global_id(2);
    int i = get_global_id(1);
    int j = get_global_id(0);

    lhsZ[m][m][1][0][i][j] = 1.0;
    lhsZ[m][m][1][ksize][i][j] = 1.0;
}

__kernel void z_solve_3(__global double* restrict lhsZ_, __global double* restrict fjacZ_, __global double* restrict njacZ_, double dttz1, double dttz2, double dz1, double dz2, double dz3, double dz4, double dz5) {
    __global double (*lhsZ)[5][3][PROBLEM_SIZE][IMAXP-1][JMAXP-1] = (__global double (*)[5][3][PROBLEM_SIZE][IMAXP-1][JMAXP-1]) lhsZ_;
    __global double (*fjacZ)[5][PROBLEM_SIZE+1][IMAXP-1][JMAXP-1] = (__global double (*)[5][PROBLEM_SIZE+1][IMAXP-1][JMAXP-1]) fjacZ_;
    __global double (*njacZ)[5][PROBLEM_SIZE+1][IMAXP-1][JMAXP-1] = (__global double (*)[5][PROBLEM_SIZE+1][IMAXP-1][JMAXP-1]) njacZ_;

    int k = get_global_id(2);
    int i = get_global_id(1);
    int j = get_global_id(0);

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


__kernel void z_solve_4(__global double* restrict lhsZ_, __global double* restrict rhs_) {
    __global double (*lhsZ)[5][3][PROBLEM_SIZE][IMAXP-1][JMAXP-1] = (__global double (*)[5][3][PROBLEM_SIZE][IMAXP-1][JMAXP-1]) lhsZ_;
    __global double (*rhs)[JMAXP+1][IMAXP+1][5] = (__global double (*)[JMAXP+1][IMAXP+1][5]) rhs_;

    int i = get_global_id(1);
    int j = get_global_id(0);

    double pivot, coeff;

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

__kernel void z_solve_5(__global double* restrict lhsZ_, __global double* restrict rhs_, int ksize, int gp12) {
    __global double (*lhsZ)[5][3][PROBLEM_SIZE][IMAXP-1][JMAXP-1] = (__global double (*)[5][3][PROBLEM_SIZE][IMAXP-1][JMAXP-1]) lhsZ_;
    __global double (*rhs)[JMAXP+1][IMAXP+1][5] = (__global double (*)[JMAXP+1][IMAXP+1][5]) rhs_;

    int i = get_global_id(0);
    int j, k;

    double pivot, coeff;

    for (k = 1; k <= ksize-1; k++) {
        for (j = 1; j <= gp12; j++) {
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
        }
    }
}

__kernel void z_solve_6(__global double* restrict lhsZ_, __global double* restrict rhs_, int ksize) {
    __global double (*lhsZ)[5][3][PROBLEM_SIZE][IMAXP-1][JMAXP-1] = (__global double (*)[5][3][PROBLEM_SIZE][IMAXP-1][JMAXP-1]) lhsZ_;
    __global double (*rhs)[JMAXP+1][IMAXP+1][5] = (__global double (*)[JMAXP+1][IMAXP+1][5]) rhs_;

    int i = get_global_id(1);
    int j = get_global_id(0);

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

__kernel void z_solve_7(__global double* restrict lhsZ_, int ksize) {
    __global double (*lhsZ)[5][3][PROBLEM_SIZE][IMAXP-1][JMAXP-1] = (__global double (*)[5][3][PROBLEM_SIZE][IMAXP-1][JMAXP-1]) lhsZ_;

    int i = get_global_id(1);
    int j = get_global_id(0);

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

__kernel void z_solve_8(__global double* restrict lhsZ_, __global double* restrict rhs_, int ksize) {
    __global double (*lhsZ)[5][3][PROBLEM_SIZE][IMAXP-1][JMAXP-1] = (__global double (*)[5][3][PROBLEM_SIZE][IMAXP-1][JMAXP-1]) lhsZ_;
    __global double (*rhs)[JMAXP+1][IMAXP+1][5] = (__global double (*)[JMAXP+1][IMAXP+1][5]) rhs_;

    int i = get_global_id(1);
    int j = get_global_id(0);

    double pivot, coeff;

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

__kernel void z_solve_9(__global double* restrict lhsZ_, __global double* restrict rhs_, int ksize) {
    __global double (*lhsZ)[5][3][PROBLEM_SIZE][IMAXP-1][JMAXP-1] = (__global double (*)[5][3][PROBLEM_SIZE][IMAXP-1][JMAXP-1]) lhsZ_;
    __global double (*rhs)[JMAXP+1][IMAXP+1][5] = (__global double (*)[JMAXP+1][IMAXP+1][5]) rhs_;

    int j = get_global_id(1);
    int i = get_global_id(0);
    int k, m, n;

    for (k = ksize-1; k >= 0; k--) {
        for (m = 0; m < BLOCK_SIZE; m++) {
            for (n = 0; n < BLOCK_SIZE; n++) {
                rhs[k][j][i][m] = rhs[k][j][i][m]
                    - lhsZ[m][n][CC][k][i][j]*rhs[k+1][j][i][n];
            }
        }
    }
}
