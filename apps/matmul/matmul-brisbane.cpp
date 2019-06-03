#include <brisbane/brisbane.h>
#include <stdio.h>
#include <stdlib.h>
#include <malloc.h>

int SIZE;
double *A, *B, *C;

void ijk() {
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            double sum = 0.0;
            for (int k = 0; k < SIZE; k++) {
                sum += A[i * SIZE + k] * B[k * SIZE + j];
            }
            C[i * SIZE + j] = sum;
        }
    }
}

void kij() {
    for (int k = 0; k < SIZE; k++) {
        for (int i = 0; i < SIZE; i++) {
            double a = A[i * SIZE + k];
            for (int j = 0; j < SIZE; j++) {
                C[i * SIZE + j] += a * B[k * SIZE + j];
            }
        }
    }
}

int main(int argc, char** argv) {
    int ERROR = 0;
    int DEV = 2;

    brisbane_init(&argc, &argv);

    SIZE = argc > 1 ? atoi(argv[1]) : 4096;
    DEV = argc > 2 ? atoi(argv[2]) : 2;

    printf("SIZE[%d] DEV[%d]\n", SIZE, DEV);

    A = (double*) malloc(SIZE * SIZE * sizeof(double));
    B = (double*) malloc(SIZE * SIZE * sizeof(double));
    C = (double*) malloc(SIZE * SIZE * sizeof(double));

    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            A[i * SIZE + j] = i + j;
            B[i * SIZE + j] = i * j;
            C[i * SIZE + j] = 0.0;
        }
    }

    brisbane_mem mem_A;
    brisbane_mem mem_B;
    brisbane_mem mem_C;
    brisbane_mem_create(SIZE * SIZE * sizeof(double), &mem_A);
    brisbane_mem_create(SIZE * SIZE * sizeof(double), &mem_B);
    brisbane_mem_create(SIZE * SIZE * sizeof(double), &mem_C);

    brisbane_kernel kernel_ijk;
    brisbane_kernel_create("ijk", &kernel_ijk);
    brisbane_kernel_setmem(kernel_ijk, 0, mem_C, brisbane_wr);
    brisbane_kernel_setmem(kernel_ijk, 1, mem_A, brisbane_rd);
    brisbane_kernel_setmem(kernel_ijk, 2, mem_B, brisbane_rd);
    brisbane_kernel_setarg(kernel_ijk, 3, sizeof(int), &SIZE);

    brisbane_task task0;
    brisbane_task_create(&task0);
    brisbane_task_h2d(task0, mem_A, 0, SIZE * SIZE * sizeof(double), A);
    brisbane_task_h2d(task0, mem_B, 0, SIZE * SIZE * sizeof(double), B);
    brisbane_task_h2d(task0, mem_C, 0, SIZE * SIZE * sizeof(double), C);
    size_t kernel_ijk_index[1] = { SIZE };
    brisbane_task_kernel(task0, kernel_ijk, 1, kernel_ijk_index);
    brisbane_task_d2h(task0, mem_C, 0, SIZE * SIZE * sizeof(double), C);
    brisbane_task_submit(task0, DEV, NULL, true);

    //ijk();

#if 0
    printf("[[ A ]]\n");
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            printf("%5.0lf ", A[i * SIZE + j]);
        }
        printf("\n");
    }

    printf("[[ B ]]\n");
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            printf("%5.0lf ", B[i * SIZE + j]);
        }
        printf("\n");
    }

    printf("[[ C ]]\n");
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            printf("%5.0lf ", C[i * SIZE + j]);
        }
        printf("\n");
    }

    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            double sum = 0.0;
            for (int k = 0; k < SIZE; k++) {
                sum += A[i * SIZE + k] * B[k * SIZE + j];
            }
            if (sum != C[i * SIZE + j]) ERROR++;
        }
    }
#endif

    printf("ERROR[%d]\n", ERROR);

    brisbane_task_release(task0);
    brisbane_kernel_release(kernel_ijk);
    brisbane_mem_release(mem_A);
    brisbane_mem_release(mem_B);
    brisbane_mem_release(mem_C);

    free(A);
    free(B);
    free(C);

    brisbane_finalize();

    return 0;
}
