#pragma OPENCL EXTENSION cl_khr_fp64 : enable

#define PROBLEM_SIZE 102
#define IMAX      PROBLEM_SIZE
#define JMAX      PROBLEM_SIZE
#define KMAX      PROBLEM_SIZE
#define IMAXP     IMAX/2*2
#define JMAXP     JMAX/2*2

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
