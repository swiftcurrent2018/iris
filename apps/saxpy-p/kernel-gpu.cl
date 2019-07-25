__kernel void saxpy(__global float* restrict Z, float A, __global float* restrict X, __global float* restrict Y) {
    int id = get_global_id(0);
    Z[id] = A * X[id] + Y[id];
}

