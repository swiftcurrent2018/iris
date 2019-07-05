#include <stdio.h>
#include <stdlib.h>
#include <malloc.h>

int main(int argc, char** argv) {
    size_t SIZE = argc > 1 ? atol(argv[1]) : 16;
    int *A;
    size_t sumA;
    int ERROR = 0;

    int nteams = 8;
    int chunk_size = SIZE / nteams;

    printf("SIZE[%d]\n", SIZE);

    A = (int*) calloc(SIZE, sizeof(int));
    sumA = 0UL;

    for (int i = 0; i < SIZE; i++) {
        A[i] = i;
    }

#pragma omp target teams distribute parallel for reduction(sum:sumA)
    for (int i = 0; i < SIZE; i += chunk_size) {
        for (int j = i; j < i + chunk_size; j++) {
            sumA += A[j];
        }
    }

    printf("sumA[%lu]\n", sumA);

    if (sumA != (SIZE - 1) * (SIZE / 2)) ERROR++;

    printf("ERROR[%d]\n", ERROR);

    free(A);

    return 0;
}
