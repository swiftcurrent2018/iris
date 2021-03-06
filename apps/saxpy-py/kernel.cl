__kernel void saxpy0(__global float* Z, float A, __global float* X) {
    size_t id = get_global_id(0);
    Z[id] = A * X[id];
}

__kernel void saxpy1(__global float* Z, __global float* Y) {
    size_t id = get_global_id(0);
    Z[id] += Y[id];
}

