#include <brisbane/brisbane.h>
#include <stdio.h>
#include <stdlib.h>
#include <malloc.h>

int main(int argc, char** argv) {
    int SIZE;
    int *A, *B, *C, *D, *E;
    int ERROR = 0;

    brisbane_init(&argc, &argv);

    SIZE = argc > 1 ? atoi(argv[1]) : 16;
    printf("SIZE[%d]\n", SIZE);

    A = (int*) malloc(SIZE * sizeof(int));
    B = (int*) malloc(SIZE * sizeof(int));
    C = (int*) malloc(SIZE * sizeof(int));
    D = (int*) malloc(SIZE * sizeof(int));
    E = (int*) malloc(SIZE * sizeof(int));

    for (int i = 0; i < SIZE; i++) {
        A[i] = i;
        B[i] = i * 1000;
    }

    brisbane_mem mem_A;
    brisbane_mem mem_B;
    brisbane_mem mem_C;
    brisbane_mem_create(SIZE * sizeof(int), &mem_A);
    brisbane_mem_create(SIZE * sizeof(int), &mem_B);
    brisbane_mem_create(SIZE * sizeof(int), &mem_C);

    brisbane_kernel kernel_loop0;
    brisbane_kernel_create("loop0", &kernel_loop0);
    brisbane_kernel_setmem(kernel_loop0, 0, mem_C, brisbane_wr);
    brisbane_kernel_setmem(kernel_loop0, 1, mem_A, brisbane_rd);
    brisbane_kernel_setmem(kernel_loop0, 2, mem_B, brisbane_rd);

    brisbane_task task0;
    brisbane_task_create(&task0);
    brisbane_task_h2d(task0, mem_A, 0, SIZE * sizeof(int), A);
    brisbane_task_h2d(task0, mem_B, 0, SIZE * sizeof(int), B);
    size_t kernel_index_loop0[1] = { SIZE };
    brisbane_task_kernel(task0, kernel_loop0, 1, kernel_index_loop0);
    brisbane_task_submit(task0, brisbane_device_fpga, NULL, true);
    /*
#pragma acc parallel loop copyin(A[0:SIZE], B[0:SIZE]) device(gpu)
#pragma omp target teams distribute parallel for map(to:A[0:SIZE], B[0:SIZE]) device(gpu)
#pragma brisbane kernel h2d(A[0:SIZE], B[0:SIZE]) alloc(C[0:SIZE]) device(gpu)
    for (int i = 0; i < SIZE; i++) {
        C[i] = A[i] + B[i];
    }
    */

    brisbane_mem mem_D;
    brisbane_mem_create(SIZE * sizeof(int), &mem_D);

    brisbane_kernel kernel_loop1;
    brisbane_kernel_create("loop1", &kernel_loop1);
    brisbane_kernel_setmem(kernel_loop1, 0, mem_D, brisbane_wr);
    brisbane_kernel_setmem(kernel_loop1, 1, mem_C, brisbane_rd);

    brisbane_task task1;
    brisbane_task_create(&task1);
    brisbane_task_present(task1, mem_C, 0, SIZE * sizeof(int));
    size_t kernel_index_loop1[1] = { SIZE };
    brisbane_task_kernel(task1, kernel_loop1, 1, kernel_index_loop1);
    brisbane_task_submit(task1, brisbane_device_gpu, NULL, true);
    /*
#pragma acc parallel loop present(C[0:SIZE]) device(cpu)
#pragma omp target teams distribute parallel for device(cpu)
#pragma brisbane kernel present(C[0:SIZE]) device(cpu)
    for (int i = 0; i < SIZE; i++) {
        D[i] = C[i] * 10;
    }
    */

    brisbane_mem mem_E;
    brisbane_mem_create(SIZE * sizeof(int), &mem_E);

    brisbane_kernel kernel_loop2;
    brisbane_kernel_create("loop2", &kernel_loop2);
    brisbane_kernel_setmem(kernel_loop2, 0, mem_E, brisbane_wr);
    brisbane_kernel_setmem(kernel_loop2, 1, mem_D, brisbane_rd);

    brisbane_task task2;
    brisbane_task_create(&task2);
    brisbane_task_present(task2, mem_D, 0, SIZE * sizeof(int));
    size_t kernel_index_loop2[1] = { SIZE };
    brisbane_task_kernel(task2, kernel_loop2, 1, kernel_index_loop2);
    brisbane_task_d2h(task2, mem_E, 0, SIZE * sizeof(int), E);
    brisbane_task_submit(task2, brisbane_device_fpga, NULL, true);
    /*
#pragma acc parallel loop present(D[0:SIZE]) device(data)
#pragma omp target teams distribute parallel for map(from:E[0:SIZE]) device(data)
#pragma brisbane kernel d2h(E[0:SIZE]) present(D[0:SIZE]) device(data)
    for (int i = 0; i < SIZE; i++) {
        E[i] = D[i] * 2;
    }
    */

    for (int i = 0; i < SIZE; i++) {
        printf("[%8d] %8d = (%8d + %8d) * %d\n", i, E[i], A[i], B[i], 20);
        if (E[i] != (A[i] + B[i]) * 20) ERROR++;
    }
    printf("ERROR[%d]\n", ERROR);

    brisbane_task_release(task0);
    brisbane_task_release(task1);
    brisbane_task_release(task2);
    brisbane_kernel_release(kernel_loop0);
    brisbane_kernel_release(kernel_loop1);
    brisbane_kernel_release(kernel_loop2);
    brisbane_mem_release(mem_A);
    brisbane_mem_release(mem_B);
    brisbane_mem_release(mem_C);
    brisbane_mem_release(mem_D);
    brisbane_mem_release(mem_E);

    free(A);
    free(B);
    free(C);
    free(D);
    free(E);

    brisbane_finalize();

    return 0;
}
