#pragma OPENCL EXTENSION cl_khr_fp64 : enable

#define MAX(X,Y)  (((X) > (Y)) ? (X) : (Y))

#define A         1220703125.0
#define S         271828183.0
#define r23 1.1920928955078125e-07
#define r46 r23 * r23
#define t23 8.388608e+06
#define t46 t23 * t23

__kernel void init_x(__global double* restrict x) {
    int i = get_global_id(0);
    x[i] = -1.0e99;
}

__kernel void init_q(__global double* restrict q) {
    int i = get_global_id(0);
    q[i] = 0.0;
}

__kernel void qq_xx(__global double* restrict qq, __global double* restrict xx, __global double* restrict x, int nq, int nk) {
    int k = get_global_id(0);
    for (int i = 0; i < nq; i++) qq[k * nq + i] = 0.0;
    for (int i = 0; i < 2 * nk; i++) xx[k * 2 * nk + i] = x[i];
}

double randlc_ep(double* x, double a) {
  double t1, t2, t3, t4, a1, a2, x1, x2, z;
  double r;

  t1 = r23 * a;
  a1 = (int) t1;
  a2 = a - t23 * a1;

  t1 = r23 * (*x);
  x1 = (int) t1;
  x2 = *x - t23 * x1;
  t1 = a1 * x2 + a2 * x1;
  t2 = (int) (r23 * t1);
  z = t1 - t23 * t2;
  t3 = t23 * z + a2 * x2;
  t4 = (int) (r46 * t3);
  *x = t3 - t46 * t4;
  r = r46 * (*x);

  return r;
}

__kernel void core(__global double* restrict xx, __global double* restrict qq, int koff, double an, int nk, int blksize, int nq, __global double* restrict sx, __local double* local_sx, __global double* restrict sy, __local double* local_sy, unsigned long gws0) {
    int k = get_global_id(0);
    double t1, t2, t3, t4, a1, a2, x1, x2, z;
    int i, ik, kk, l;
    double in_t1, in_t2, in_t3, in_t4;
    double in_a1, in_a2, in_x1, in_x2, in_z;
    double tmp_sx, tmp_sy;

    kk = k + koff;
    t1 = S;
    t2 = an;

    for (i = 1; i <= 100; i++) {
        ik = kk / 2;
        if ((2 * ik) != kk) t3 = randlc_ep(&t1, t2);
        if (ik == 0) break;
        t3 = randlc_ep(&t2, t2);
        kk = ik;
    }

    in_t1 = r23 * A;
    in_a1 = (int)in_t1;
    in_a2 = A - t23 * in_a1;

    for(i=0; i<2*nk; i++)
    {
        in_t1 = r23 * t1;
        in_x1 = (int)in_t1;
        in_x2 = t1 - t23 * in_x1;
        in_t1 = in_a1 * in_x2 + in_a2 * in_x1;
        in_t2 = (int)(r23 * in_t1);
        in_z = in_t1 - t23 * in_t2;
        in_t3 = t23*in_z + in_a2 *in_x2;
        in_t4 = (int)(r46 * in_t3);
        t1 = in_t3 - t46 * in_t4;
        xx[k*2*nk + i] = r46 * t1;
    }

    tmp_sx = 0.0;
    tmp_sy = 0.0;

    for (i = 0; i < nk; i++) {
        x1 = 2.0 * xx[k*2*nk + 2*i] - 1.0;
        x2 = 2.0 * xx[k*2*nk + (2*i+1)] - 1.0;
        t1 = x1 * x1 + x2 * x2;
        if (t1 <= 1.0) {
            t2   = sqrt(-2.0 * log(t1) / t1);
            t3   = (x1 * t2);
            t4   = (x2 * t2);
            l    = MAX(fabs(t3), fabs(t4));
            qq[k*nq + l] += 1.0;
            tmp_sx   = tmp_sx + t3;
            tmp_sy   = tmp_sy + t4;
        }
    }

    int lid = get_local_id(0);
    local_sx[lid] = tmp_sx;
    local_sy[lid] = tmp_sy;
    for (int s = get_local_size(0) / 2; s > 0; s >>= 1) {
        barrier(CLK_LOCAL_MEM_FENCE);
        if (lid < s) local_sx[lid] += local_sx[lid + s];
    }
    for (int s = get_local_size(0) / 2; s > 0; s >>= 1) {
        barrier(CLK_LOCAL_MEM_FENCE);
        if (lid < s) local_sy[lid] += local_sy[lid + s];
    }
    if (lid == 0) sx[get_group_id(0)] = local_sx[0];
    if (lid == 0) sy[get_group_id(0)] = local_sy[0];
}

__kernel void gc(__global double* restrict qq, __global double* restrict q, int blksize, int nq, __global double* restrict gc, __local double* local_gc, unsigned long gws0) {
    int i = get_global_id(0);
    double sum_qi = 0.0;
    for(int k=0; k<blksize; k++)
        sum_qi = sum_qi + qq[k*nq + i];
    q[i] += sum_qi;

    int lid = get_local_id(0);
    local_gc[lid] = sum_qi;
    for (int s = get_local_size(0) / 2; s > 0; s >>= 1) {
        barrier(CLK_LOCAL_MEM_FENCE);
        if (lid < s) local_gc[lid] += local_gc[lid + s];
    }
    if (lid == 0) gc[get_group_id(0)] = local_gc[0];
}

