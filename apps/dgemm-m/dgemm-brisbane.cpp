#include <brisbane/brisbane.h>
#include <stdio.h>
#include <stdlib.h>
#include <malloc.h>

size_t SIZE, UNIT;
double *A, *B, *C;
double t0, t1;

void ijk() {
#pragma brisbane kernel h2d(A[0:SIZE*SIZE], B[0:SIZE*SIZE]) d2h(C[0:SIZE*SIZE]) device(all)
#pragma brisbane data access index(i) h2d(A[i*SIZE:SIZE], B[0:SIZE*SIZE]) d2h(C[i*SIZE:SIZE])
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
  int DEV = 0;

  brisbane_init(&argc, &argv, true);

  SIZE = argc > 1 ? atol(argv[1]) : 16;
  UNIT = argc > 2 ? atol(argv[2]) : SIZE / 4;
  DEV = argc > 3 ? atoi(argv[3]) : brisbane_device_any;

  printf("SIZE[%d] MATRIX_SIZE[%lu]MB UNIT[%d] DEV[0x%x]\n", SIZE, SIZE * SIZE * sizeof(double) / 1024 / 1024, UNIT, DEV);

  A = (double*) valloc(SIZE * SIZE * sizeof(double));
  B = (double*) valloc(SIZE * SIZE * sizeof(double));
  C = (double*) valloc(SIZE * SIZE * sizeof(double));

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

  brisbane_timer_now(&t0);

  brisbane_task task0;
  brisbane_task_create(&task0);
  for (size_t i = 0; i < SIZE; i += UNIT) {
    brisbane_task subtask;
    brisbane_task_create(&subtask);
    brisbane_task_present(subtask, mem_A, i * SIZE * sizeof(double), UNIT * SIZE * sizeof(double), A + (i * SIZE));
    brisbane_task_present(subtask, mem_B, 0, SIZE * SIZE * sizeof(double), B);
    size_t kernel_ijk_off[2] = { 0, i };
    size_t kernel_ijk_idx[2] = { SIZE, UNIT };
    brisbane_task_kernel(subtask, kernel_ijk, 2, kernel_ijk_off, kernel_ijk_idx);
    brisbane_task_d2h(subtask, mem_C, i * SIZE * sizeof(double), UNIT * SIZE * sizeof(double), C + (i * SIZE));
    brisbane_task_add_subtask(task0, subtask);
  }
  brisbane_task_submit(task0, DEV, NULL, true);

  brisbane_timer_now(&t1);

  //ijk();

#if 1
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

  printf("Checking errors\n");
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

  printf("ERROR[%d] TIME[%lf]\n", ERROR, t1 - t0);

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
