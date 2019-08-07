#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__kernel void waxpby_0(__global double* restrict wcoefs, __global double* restrict xcoefs) {
    int i = get_global_id(0);
    wcoefs[i] = xcoefs[i];
}

__kernel void waxpby_1(__global double* restrict wcoefs, __global double* restrict xcoefs, double alpha) {
    int i = get_global_id(0);
    wcoefs[i] = alpha * xcoefs[i];
}

__kernel void waxpby_2(__global double* restrict wcoefs, __global double* restrict xcoefs, __global double* restrict ycoefs, double beta) {
    int i = get_global_id(0);
    wcoefs[i] = xcoefs[i] + beta * ycoefs[i];
}

__kernel void waxpby_3(__global double* restrict wcoefs, __global double* restrict xcoefs, __global double* restrict ycoefs, double alpha, double beta) {
    int i = get_global_id(0);
    wcoefs[i] = alpha * xcoefs[i] + beta * ycoefs[i];
}

__kernel void daxpby_0(__global double* restrict ycoefs, __global double* restrict xcoefs) {
    int i = get_global_id(0);
    ycoefs[i] += xcoefs[i];
}

__kernel void daxpby_1(__global double* restrict ycoefs, __global double* restrict xcoefs, double alpha) {
    int i = get_global_id(0);
    ycoefs[i] += alpha * xcoefs[i];
}

__kernel void daxpby_2(__global double* restrict ycoefs, __global double* restrict xcoefs, double beta) {
    int i = get_global_id(0);
    ycoefs[i] = xcoefs[i] + beta * ycoefs[i];
}

__kernel void daxpby_3(__global double* restrict ycoefs, __global double* restrict xcoefs, double alpha) {
    int i = get_global_id(0);
    ycoefs[i] = alpha * xcoefs[i];
}

__kernel void daxpby_4(__global double* restrict ycoefs, __global double* restrict xcoefs, double alpha, double beta) {
    int i = get_global_id(0);
    ycoefs[i] = alpha * xcoefs[i] + beta * ycoefs[i];
}

__kernel void kernel_dot(__global double* restrict ycoefs, __global double* restrict xcoefs, __global double* restrict result, __local double* local_result, unsigned long gws0) {
    int i = get_global_id(0);
    int lid = get_local_id(0);

    local_result[lid] = i < gws0 ? xcoefs[i] * ycoefs[i] : 0.0;
    
    for (int s = get_local_size(0) / 2; s > 0; s >>= 1) {
        barrier(CLK_LOCAL_MEM_FENCE);
        if (lid < s) local_result[lid] += local_result[lid + s];
    }
    if (lid == 0) result[get_group_id(0)] = local_result[0];
}

__kernel void kernel_dot_r2(__global double* restrict xcoefs, __global double* restrict result, __local double* local_result, unsigned long gws0) {
    int i = get_global_id(0);
    int lid = get_local_id(0);

    local_result[lid] = i < gws0 ? xcoefs[i] * xcoefs[i] : 0.0;
    
    for (int s = get_local_size(0) / 2; s > 0; s >>= 1) {
        barrier(CLK_LOCAL_MEM_FENCE);
        if (lid < s) local_result[lid] += local_result[lid + s];
    }
    if (lid == 0) result[get_group_id(0)] = local_result[0];
}

__kernel void matvec_std_operator(__global int* restrict Arowoffsets, __global int* restrict Acols, __global double* restrict Acoefs, __global double* restrict xcoefs, __global double* restrict ycoefs, double beta) {
    int row = get_global_id(0);
    int row_start = Arowoffsets[row];
    int row_end   = Arowoffsets[row+1];

    double sum = beta * ycoefs[row];

    for(int i = row_start; i < row_end; ++i) {
        sum += Acoefs[i] * xcoefs[Acols[i]];
    }

    ycoefs[row] = sum;
}

