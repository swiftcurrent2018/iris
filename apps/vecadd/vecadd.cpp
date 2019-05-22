#include <stdio.h>
#include <stdlib.h>
#include <malloc.h>

int main(int argc, char** argv) {
    int SIZE;
    int* A;
    int* B;
    int* C;
    int* D;

    SIZE = argc > 1 ? atoi(argv[1]) : 1024;

    A = (int*) calloc(SIZE, sizeof(int));
    B = (int*) calloc(SIZE, sizeof(int));
    C = (int*) calloc(SIZE, sizeof(int));
    D = (int*) calloc(SIZE, sizeof(int));

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

    return 0;
}

