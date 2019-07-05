#include <brisbane/brisbane.h>
#include <stdio.h>
#include <stdlib.h>
#include <malloc.h>

int main(int argc, char** argv) {
    brisbane_init(&argc, &argv);
    size_t SIZE = argc > 1 ? atol(argv[1]) : 16;
    int *A;
    size_t sumA;
    int ERROR = 0;

    int ndevs;

    brisbane_info_ndevs(&ndevs);

    int nteams = ndevs * 2;
    int chunk_size = SIZE / nteams;

    printf("SIZE[%d] ndevs[%d] nteams[%d] chunk_size[%d]\n", SIZE, ndevs, nteams, chunk_size);

    A = (int*) calloc(SIZE, sizeof(int));
    sumA = 0UL;

    for (int i = 0; i < SIZE; i++) {
        A[i] = i;
    }

    brisbane_mem mem_A;
    brisbane_mem_create(SIZE * sizeof(int), &mem_A);

    brisbane_kernel kernel_reduce_sum;
    brisbane_kernel_create("reduce_sum", &kernel_reduce_sum);
    brisbane_kernel_setmem(kernel_reduce_sum, 0, mem_A, brisbane_rd);

    brisbane_task task0;
    brisbane_task_create(&task0);
    for (int i = 0; i < SIZE; i += chunk_size) {
        brisbane_task subtask;
        brisbane_task_create(&subtask);

        brisbane_mem mem_sumA;
        brisbane_mem_create(sizeof(unsigned long), &mem_sumA);
        brisbane_mem_reduce(mem_sumA, brisbane_sum, brisbane_long);
        brisbane_kernel_setmem(kernel_reduce_sum, 1, mem_sumA, brisbane_wr);

        brisbane_task_h2d(subtask, mem_A, i * sizeof(int), chunk_size * sizeof(int), A + i);
        size_t kernel_reduce_sum_off[1] = { i };
        size_t kernel_reduce_sum_idx[1] = { chunk_size };
        brisbane_task_kernel(subtask, kernel_reduce_sum, 1, kernel_reduce_sum_off, kernel_reduce_sum_idx);
        brisbane_task_d2h(subtask, mem_sumA, 0, sizeof(unsigned long), &sumA);
        brisbane_task_release_mem(subtask, mem_sumA);
        brisbane_task_add_subtask(task0, subtask);
    }
    brisbane_task_submit(task0, brisbane_cpu, NULL, true);

    /*
#pragma omp target teams distribute parallel for reduction(sum:sumA) num_teams(nteams)
    for (int i = 0; i < SIZE; i++) {
        sumA += A[i];
    }
    */

    /*
#pragma omp target teams distribute parallel for reduction(sum:sumA)
    for (int i = 0; i < SIZE; i += chunk_size) {
        for (int j = i; j < i + chunk_size; j++) {
            sumA += A[j];
        }
    }
    */

    printf("sumA[%lu]\n", sumA);

    if (sumA != (SIZE - 1) * (SIZE / 2)) ERROR++;

    printf("ERROR[%d]\n", ERROR);

    free(A);

    brisbane_mem_release(mem_A);

    brisbane_finalize();
    return 0;
}
