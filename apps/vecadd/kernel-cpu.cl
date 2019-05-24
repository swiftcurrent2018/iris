__kernel void loop0(__global int* C, __global int* A, __global int* B) {
    int id = get_global_id(0);
    C[id] = A[id] + B[id];
}

__kernel void loop1(__global int* D, __global int* C) {
    int id = get_global_id(0);
    D[id] = C[id] * 10;
}

__kernel void loop2(__global int* E, __global int* D) {
    int id = get_global_id(0);
    E[id] = D[id] * 2;
}
