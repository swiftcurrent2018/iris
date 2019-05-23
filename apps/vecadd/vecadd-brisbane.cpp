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
        C[i] = 0;
    }

    brisbane_mem mem_C;
    brisbane_mem_create(SIZE * sizeof(int), &mem_C);

    brisbane_mem mem_A;
    brisbane_mem_create(SIZE * sizeof(int), &mem_A);

    brisbane_mem mem_B;
    brisbane_mem_create(SIZE * sizeof(int), &mem_B);

    brisbane_kernel kernel_loop0;
    brisbane_kernel_create("loop0", &kernel_loop0);
    brisbane_kernel_setarg(kernel_loop0, 0, sizeof(mem_C), &mem_C);
    brisbane_kernel_setarg(kernel_loop0, 1, sizeof(mem_A), &mem_A);
    brisbane_kernel_setarg(kernel_loop0, 2, sizeof(mem_B), &mem_B);

    brisbane_task task0;
    brisbane_task_create(&task0);
    brisbane_task_h2d(task0, mem_A, 0, SIZE * sizeof(int), A);
    brisbane_task_h2d(task0, mem_B, 0, SIZE * sizeof(int), B);
    size_t kernel_index_loop0[1] = { SIZE };
    brisbane_task_kernel(task0, kernel_loop0, 1, kernel_index_loop0);
    brisbane_task_d2h(task0, mem_C, 0, SIZE * sizeof(int), C);
    brisbane_task_submit(task0, brisbane_device_gpu);
    brisbane_task_wait(task0);
    /*
    for (int i = 0; i < SIZE; i++) {
        C[i] = A[i] + B[i];
    }
    */

    brisbane_mem mem_D;
    brisbane_mem_create(SIZE * sizeof(int), &mem_D);

    brisbane_kernel kernel_loop1;
    brisbane_kernel_create("loop1", &kernel_loop1);
    brisbane_kernel_setarg(kernel_loop1, 0, sizeof(mem_D), (void*) &mem_D);
    brisbane_kernel_setarg(kernel_loop1, 1, sizeof(mem_C), (void*) &mem_C);

    brisbane_task task1;
    brisbane_task_create(&task1);
    brisbane_task_h2d(task1, mem_C, 0, SIZE * sizeof(int), C);
    size_t kernel_index_loop1[1] = { SIZE };
    brisbane_task_kernel(task1, kernel_loop1, 1, kernel_index_loop1);
    brisbane_task_d2h(task1, mem_D, 0, SIZE * sizeof(int), D);
    brisbane_task_submit(task1, brisbane_device_cpu);
    brisbane_task_wait(task1);
    /*
    for (int i = 0; i < SIZE; i++) {
        D[i] = C[i] * 10;
    }
    */

    for (int i = 0; i < SIZE; i++) {
        printf("[%8d] %8d = (%8d + %8d) * %d\n", i, D[i], A[i], B[i], 10);
    }

    brisbane_task_release(task0);
    brisbane_task_release(task1);
    brisbane_kernel_release(kernel_loop0);
    brisbane_kernel_release(kernel_loop1);
    brisbane_mem_release(mem_A);
    brisbane_mem_release(mem_B);
    brisbane_mem_release(mem_C);
    brisbane_mem_release(mem_D);

    free(A);
    free(B);
    free(C);
    free(D);

    brisbane_finalize();

    return 0;
}

