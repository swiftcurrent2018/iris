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

    SIZE = argc > 1 ? atoi(argv[1]) : 16;
    printf("SIZE[%d]\n", SIZE);

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

    ijk();

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

    printf("ERROR[%d]\n", ERROR);

    free(A);
    free(B);
    free(C);

    return 0;
}
