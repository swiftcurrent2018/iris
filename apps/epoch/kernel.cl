__kernel void loop0(__global int* restrict C, __global int* restrict A, __global int* restrict B) {
  int id = get_global_id(0);
  C[id] += A[id] + B[id];
}

