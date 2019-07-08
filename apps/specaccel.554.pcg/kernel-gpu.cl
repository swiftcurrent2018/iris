#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__kernel void loop0(__global double* restrict x) {
    int i = get_global_id(0);
    x[i] = 1.0;
}

__kernel void loop1(__global double* restrict q, __global double* restrict z, __global double* restrict r, __global double* restrict p) {
    int j = get_global_id(0);
    q[j] = 0.0;
    z[j] = 0.0;
    r[j] = 0.0;
    p[j] = 0.0;
}

__kernel void loop2(__global double* restrict x, __global double* restrict z, __global double* restrict norm_temp1, __local double* local_norm_temp1, __global double* restrict norm_temp2, __local double* local_norm_temp2) {
    int j = get_global_id(0);
    int lid = get_local_id(0);
    local_norm_temp1[lid] = x[j] * z[j];
    local_norm_temp2[lid] = z[j] * z[j];
    for (int s = get_local_size(0) / 2; s > 0; s >>= 1) {
        barrier(CLK_LOCAL_MEM_FENCE);
        if (lid < s) local_norm_temp1[lid] += local_norm_temp1[lid + s];
    }
    for (int s = get_local_size(0) / 2; s > 0; s >>= 1) {
        barrier(CLK_LOCAL_MEM_FENCE);
        if (lid < s) local_norm_temp2[lid] += local_norm_temp2[lid + s];
    }
    if (lid == 0) norm_temp1[get_group_id(0)] = local_norm_temp1[0];
    if (lid == 0) norm_temp2[get_group_id(0)] = local_norm_temp2[0];
}

__kernel void loop3(__global double* restrict x, __global double* restrict z, double norm_temp2) {
    int j = get_global_id(0);
    x[j] = norm_temp2 * z[j];
}

__kernel void loop4(__global double* restrict x, __global double* restrict q, __global double* restrict z, __global double* restrict r, __global double* restrict p) {
    int j = get_global_id(0);
    q[j] = 0.0;
    z[j] = 0.0;
    r[j] = x[j];
    p[j] = r[j];
}

__kernel void loop5(__global double* restrict r, __global double* restrict rho, __local double* local_rho) {
    int j = get_global_id(0);
    int lid = get_local_id(0);
    local_rho[lid] = r[j] * r[j];
    for (int s = get_local_size(0) / 2; s > 0; s >>= 1) {
        barrier(CLK_LOCAL_MEM_FENCE);
        if (lid < s) local_rho[lid] += local_rho[lid + s];
    }
    if (lid == 0) rho[get_group_id(0)] = local_rho[0];
}

__kernel void loop6(__global int* restrict rowstr, __global int* restrict colidx, __global double* restrict a, __global double* restrict p, __global double* restrict q) {
    int j = get_global_id(0);
    int tmp1 = rowstr[j];
    int tmp2 = rowstr[j + 1];
    double sum = 0.0;
    for (int k = tmp1; k < tmp2; k++) {
        int tmp3 = colidx[k];
        sum = sum + a[k] * p[tmp3];
    }
    q[j] = sum;
}

__kernel void loop7(__global double* restrict p, __global double* restrict q, __global double* restrict d, __local double* local_d) {
    int j = get_global_id(0);
    int lid = get_local_id(0);
    local_d[lid] = p[j] * q[j];
    for (int s = get_local_size(0) / 2; s > 0; s >>= 1) {
        barrier(CLK_LOCAL_MEM_FENCE);
        if (lid < s) local_d[lid] += local_d[lid + s];
    }
    if (lid == 0) d[get_group_id(0)] = local_d[0];
}

__kernel void loop8(__global double* restrict z, __global double* restrict p, __global double* restrict r, __global double* restrict q, double alpha) {
    int j = get_global_id(0);
    z[j] = z[j] + alpha * p[j];
    r[j] = r[j] - alpha * q[j];
}

__kernel void loop9(__global double* restrict r, __global double* restrict rho, __local double* local_rho) {
    int j = get_global_id(0);
    int lid = get_local_id(0);
    local_rho[lid] = r[j] * r[j];
    for (int s = get_local_size(0) / 2; s > 0; s >>= 1) {
        barrier(CLK_LOCAL_MEM_FENCE);
        if (lid < s) local_rho[lid] += local_rho[lid + s];
    }
    if (lid == 0) rho[get_group_id(0)] = local_rho[0];
}

__kernel void loop10(__global double* restrict p, __global double* restrict r, double beta) {
    int j = get_global_id(0);
    p[j] = r[j] + beta * p[j];
}

__kernel void loop11(__global int* restrict rowstr, __global int* restrict colidx, __global double* restrict a, __global double* restrict z, __global double* restrict r) {
    int j = get_global_id(0);
    int tmp1 = rowstr[j];
    int tmp2 = rowstr[j + 1];
    double d = 0.0;
    for (int k = tmp1; k < tmp2; k++) {
        int tmp3 = colidx[k];
        d = d + a[k] * z[tmp3];
    }
    r[j] = d;
}

__kernel void loop12(__global double* restrict x, __global double* restrict z, __global double* restrict norm_temp1, __local double* local_norm_temp1, __global double* restrict norm_temp2, __local double* local_norm_temp2) {
    int j = get_global_id(0);
    int lid = get_local_id(0);
    local_norm_temp1[lid] = x[j] * z[j];
    local_norm_temp2[lid] = z[j] * z[j];
    for (int s = get_local_size(0) / 2; s > 0; s >>= 1) {
        barrier(CLK_LOCAL_MEM_FENCE);
        if (lid < s) local_norm_temp1[lid] += local_norm_temp1[lid + s];
    }
    for (int s = get_local_size(0) / 2; s > 0; s >>= 1) {
        barrier(CLK_LOCAL_MEM_FENCE);
        if (lid < s) local_norm_temp2[lid] += local_norm_temp2[lid + s];
    }
    if (lid == 0) norm_temp1[get_group_id(0)] = local_norm_temp1[0];
    if (lid == 0) norm_temp2[get_group_id(0)] = local_norm_temp2[0];
}

__kernel void loop13(__global double* restrict x, __global double* restrict z, double norm_temp2) {
    int j = get_global_id(0);
    x[j] = norm_temp2 * z[j];
}

__kernel void loop14(__global double* restrict x) {
    int i = get_global_id(0);
    x[i] = 1.0;
}

__kernel void loop15(__global double* restrict x, __global double* restrict r, __global double* restrict sum, __local double* local_sum) {
    int j = get_global_id(0);
    int lid = get_local_id(0);
    double d = x[j] - r[j];
    local_sum[lid] = d * d;
    for (int s = get_local_size(0) / 2; s > 0; s >>= 1) {
        barrier(CLK_LOCAL_MEM_FENCE);
        if (lid < s) local_sum[lid] += local_sum[lid + s];
    }
    if (lid == 0) sum[get_group_id(0)] = local_sum[0];
}
