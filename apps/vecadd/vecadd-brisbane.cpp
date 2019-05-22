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

    for (int i = 0; i < SIZE; i++) {
        C[i] = A[i] + B[i];
    }

    for (int i = 0; i < SIZE; i++) {
        D[i] = C[i] * 10;
    }

    for (int i = 0; i < SIZE; i++) {
        printf("[%8d] %8d = (%8d + %8d) * %d\n", i, D[i], A[i], B[i], 10);
    }

    free(A);
    free(B);
    free(C);
    free(D);

    brisbane_finalize();

    return 0;
}

