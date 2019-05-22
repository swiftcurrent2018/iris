#include <brisbane/brisbane.h>
#include <stdio.h>
#include <stdlib.h>
#include <malloc.h>

int main(int argc, char** argv) {
    int SIZE;
    int* A;
    int* B;
    int* C;
    int* D;

    brisbane_init(&argc, &argv);

    SIZE = argc > 1 ? atoi(argv[1]) : 16;

    A = (int*) malloc(SIZE * sizeof(int));
    B = (int*) malloc(SIZE * sizeof(int));
    C = (int*) malloc(SIZE * sizeof(int));
    D = (int*) malloc(SIZE * sizeof(int));

    for (int i = 0; i < SIZE; i++) {
        A[i] = i;
        B[i] = i * 1000;
    }

    brisbane_region_begin(brisbane_device_cpu);

    brisbane_mem mem_C;
    brisbane_mem_create(SIZE * sizeof(int), &mem_C);

    brisbane_mem mem_A;
    brisbane_mem_create(SIZE * sizeof(int), &mem_A);
    brisbane_mem_h2d(mem_A, 0, SIZE * sizeof(int), A);

    /*
    brisbane_mem mem_B;
    brisbane_mem_create(SIZE * sizeof(int), &mem_B);
    brisbane_mem_h2d(mem_B, 0, SIZE * sizeof(int), B);

    brisbane_kernel kernel_loop0;
    brisbane_kernel_create("loop0", &kernel_loop0);
    size_t kernel_index_loop0[1] = { SIZE };
    brisbane_kernel_setarg(kernel_loop0, 0, sizeof(mem_C), (void*) &mem_C);
    brisbane_kernel_setarg(kernel_loop0, 1, sizeof(mem_A), (void*) &mem_A);
    brisbane_kernel_setarg(kernel_loop0, 2, sizeof(mem_B), (void*) &mem_B);
    brisbane_kernel_launch(kernel_loop0, 1, kernel_index_loop0, BRISBANE_DEVICE_CPU);
    */
    /*
    for (int i = 0; i < SIZE; i++) {
        C[i] = A[i] + B[i];
    }
    */

    /*
    brisbane_mem mem_D;
    brisbane_mem_create(SIZE * sizeof(int), &mem_D);

    brisbane_kernel kernel_loop1;
    brisbane_kernel_create("loop1", &kernel_loop1);
    size_t kernel_index_loop1[1] = { SIZE };
    brisbane_kernel_setarg(kernel_loop1, 0, sizeof(mem_D), (void*) &mem_D);
    brisbane_kernel_setarg(kernel_loop1, 1, sizeof(mem_C), (void*) &mem_C);
    brisbane_kernel_launch(kernel_loop1, 1, kernel_index_loop1, brisbane_device_gpu);
    brisbane_mem_d2h(mem_C, 0, SIZE * sizeof(int), C);
    */
    /*
    for (int i = 0; i < SIZE; i++) {
        D[i] = C[i] * 10;
    }
    */

    for (int i = 0; i < SIZE; i++) {
        printf("[%8d] %8d = (%8d + %8d) * %d\n", i, D[i], A[i], B[i], 10);
    }

    /*
    brisbane_release(kernel_loop0);
    brisbane_release(kernel_loop1);
    brisbane_release(mem_A);
    brisbane_release(mem_B);
    brisbane_release(mem_C);
    brisbane_release(mem_D);
    */

    free(A);
    free(B);
    free(C);
    free(D);

    brisbane_finalize();

    return 0;
}

