#define Index3D(_nx,_ny,_i,_j,_k) ((_i)+_nx*((_j)+_ny*(_k)))

__kernel void stencil_3D(float c0, float c1, __global float* restrict A0, __global float* restrict Anext, int nx, int ny, int nz) {
    int i = get_global_id(0) + 1;
    int j = get_global_id(1) + 1;
    int k = get_global_id(2) + 1;

    Anext[Index3D (nx, ny, i, j, k)] = c1 *
    ( A0[Index3D (nx, ny, i, j, k + 1)] +
      A0[Index3D (nx, ny, i, j, k - 1)] +
      A0[Index3D (nx, ny, i, j + 1, k)] +
      A0[Index3D (nx, ny, i, j - 1, k)] +
      A0[Index3D (nx, ny, i + 1, j, k)] +
      A0[Index3D (nx, ny, i - 1, j, k)] )
    - A0[Index3D (nx, ny, i, j, k)] * c0;
}

__kernel void stencil_2D(float c0, float c1, __global float* restrict A0, __global float* restrict Anext, int nx, int ny, int nz) {
    int i = get_global_id(0) + 1;
    int j = get_global_id(1) + 1;

    for (int k = 1; k < nz - 1; k++) {
    Anext[Index3D (nx, ny, i, j, k)] = c1 *
    ( A0[Index3D (nx, ny, i, j, k + 1)] +
      A0[Index3D (nx, ny, i, j, k - 1)] +
      A0[Index3D (nx, ny, i, j + 1, k)] +
      A0[Index3D (nx, ny, i, j - 1, k)] +
      A0[Index3D (nx, ny, i + 1, j, k)] +
      A0[Index3D (nx, ny, i - 1, j, k)] )
    - A0[Index3D (nx, ny, i, j, k)] * c0;
    }
}

__kernel void stencil_1D(float c0, float c1, __global float* restrict A0, __global float* restrict Anext, int nx, int ny, int nz) {
    int i = get_global_id(0) + 1;

    for (int j = 1; j < ny - 1; j++)
    for (int k = 1; k < nz - 1; k++) {
    Anext[Index3D (nx, ny, i, j, k)] = c1 *
    ( A0[Index3D (nx, ny, i, j, k + 1)] +
      A0[Index3D (nx, ny, i, j, k - 1)] +
      A0[Index3D (nx, ny, i, j + 1, k)] +
      A0[Index3D (nx, ny, i, j - 1, k)] +
      A0[Index3D (nx, ny, i + 1, j, k)] +
      A0[Index3D (nx, ny, i - 1, j, k)] )
    - A0[Index3D (nx, ny, i, j, k)] * c0;
    }
}
