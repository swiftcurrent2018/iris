__kernel void init(__global int* restrict A) {
    int id = get_global_id(0);
    A[id] = id;
}

__kernel void reduce_sum(__global int* restrict A, __global unsigned long* restrict sumA, __local unsigned long* local_sumA, unsigned long gws0) {
    int gid = get_global_id(0);
    int gsize = get_global_size(0);
    int lid = get_local_id(0);
    int lsize = get_local_size(0); 

    local_sumA[lid] = gid < gws0 ? A[gid] : 0;

    for (int s = lsize / 2; s > 0; s >>= 1) {
        barrier(CLK_LOCAL_MEM_FENCE);
        if (lid < s) local_sumA[lid] += local_sumA[lid + s];
    }

    if (lid == 0) sumA[get_group_id(0)] = local_sumA[0];
}

__kernel void reduce_max(__global int* restrict A, __global unsigned long* restrict maxA) {
    int id = get_global_id(0);
    int dimx = get_global_size(0);
    if (id == 0) {
        unsigned long m = 0;
        for (int i = 0; i < dimx; i++) if (A[i] > m) m = A[i];
        maxA[0] = m;
    }
}

