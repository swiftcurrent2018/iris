#ifndef _Vector_functions_hpp_
#define _Vector_functions_hpp_

//@HEADER
// ************************************************************************
//
// MiniFE: Simple Finite Element Assembly and Solve
// Copyright (2006-2013) Sandia Corporation
//
// Under terms of Contract DE-AC04-94AL85000, there is a non-exclusive
// license for use of this work by or on behalf of the U.S. Government.
//
// This library is free software; you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as
// published by the Free Software Foundation; either version 2.1 of the
// License, or (at your option) any later version.
//
// This library is distributed in the hope that it will be useful, but
// WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, write to the Free Software
// Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307
// USA
//
// ************************************************************************
//@HEADER

#include <brisbane/brisbane.h>
#include <vector>
#include <sstream>
#include <fstream>

#ifdef HAVE_MPI
#include <mpi.h>
#endif

#ifdef MINIFE_HAVE_TBB
#include <LockingVector.hpp>
#endif

#include <TypeTraits.hpp>
#include <Vector.hpp>

#define MINIFE_MIN(X, Y)  ((X) < (Y) ? (X) : (Y))

namespace miniFE {


template<typename VectorType>
void write_vector(const std::string& filename,
                  const VectorType& vec)
{
  int numprocs = 1, myproc = 0;
#ifdef HAVE_MPI
  MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
  MPI_Comm_rank(MPI_COMM_WORLD, &myproc);
#endif

  std::ostringstream osstr;
  osstr << filename << "." << numprocs << "." << myproc;
  std::string full_name = osstr.str();
  std::ofstream ofs(full_name.c_str());

  typedef typename VectorType::ScalarType ScalarType;

  const std::vector<ScalarType>& coefs = vec.coefs;
  for(int p=0; p<numprocs; ++p) {
    if (p == myproc) {
      if (p == 0) {
        ofs << vec.local_size << std::endl;
      }
  
      typename VectorType::GlobalOrdinalType first = vec.startIndex;
      for(size_t i=0; i<vec.local_size; ++i) {
        ofs << first+i << " " << coefs[i] << std::endl;
      }
    }
#ifdef HAVE_MPI
    MPI_Barrier(MPI_COMM_WORLD);
#endif
  }
}

template<typename VectorType>
void sum_into_vector(size_t num_indices,
                     const typename VectorType::GlobalOrdinalType* indices,
                     const typename VectorType::ScalarType* coefs,
                     VectorType& vec)
{
  typedef typename VectorType::GlobalOrdinalType GlobalOrdinal;
  typedef typename VectorType::ScalarType Scalar;

  GlobalOrdinal first = vec.startIndex;
  GlobalOrdinal last = first + vec.local_size - 1;

  std::vector<Scalar>& vec_coefs = vec.coefs;

  for(size_t i=0; i<num_indices; ++i) {
    if (indices[i] < first || indices[i] > last) continue;
    size_t idx = indices[i] - first;

    #pragma omp atomic
    vec_coefs[idx] += coefs[i];
  }
}

#ifdef MINIFE_HAVE_TBB
template<typename VectorType>
void sum_into_vector(size_t num_indices,
                     const typename VectorType::GlobalOrdinalType* indices,
                     const typename VectorType::ScalarType* coefs,
                     LockingVector<VectorType>& vec)
{
  vec.sum_in(num_indices, indices, coefs);
}
#endif

//------------------------------------------------------------
//Compute the update of a vector with the sum of two scaled vectors where:
//
// w = alpha*x + beta*y
//
// x,y - input vectors
//
// alpha,beta - scalars applied to x and y respectively
//
// w - output vector
//
template<typename VectorType>
void
  waxpby(typename VectorType::ScalarType alpha, const VectorType& x,
         typename VectorType::ScalarType beta, const VectorType& y,
         VectorType& w)
{
  typedef typename VectorType::ScalarType ScalarType;

#ifdef MINIFE_DEBUG_OPENMP
  std::cout << "Starting WAXPBY..." << std::endl;
#endif

#ifdef MINIFE_DEBUG
  if (y.local_size < x.local_size || w.local_size < x.local_size) {
    std::cerr << "miniFE::waxpby ERROR, y and w must be at least as long as x." << std::endl;
    return;
  }
#endif

  const int n = x.coefs.size();
  const ScalarType*  xcoefs = &x.coefs[0];
  const ScalarType*  ycoefs = &y.coefs[0];
        ScalarType*  wcoefs = &w.coefs[0];

  printf("alpha[%lf] beta[%lf] n[%d]\n", alpha, beta, n);

  brisbane_mem mem_xcoefs;
  brisbane_mem mem_ycoefs;
  brisbane_mem mem_wcoefs;
  brisbane_mem_create(sizeof(double) * n, &mem_xcoefs);
  brisbane_mem_create(sizeof(double) * n, &mem_ycoefs);
  brisbane_mem_create(sizeof(double) * n, &mem_wcoefs);

  size_t kernel_waxpby_off[1] = { 0 };
  size_t kernel_waxpby_idx[1] = { n };

  if(beta == 0.0) {
	if(alpha == 1.0) {
        brisbane_kernel kernel_waxpby_0;
        brisbane_kernel_create("waxpby_0", &kernel_waxpby_0);
        brisbane_kernel_setmem(kernel_waxpby_0, 0, mem_wcoefs, brisbane_w);
        brisbane_kernel_setmem(kernel_waxpby_0, 1, mem_xcoefs, brisbane_r);

        brisbane_task task0;
        brisbane_task_create(&task0);
        brisbane_task_h2d_full(task0, mem_xcoefs, (void*) xcoefs);
        brisbane_task_kernel(task0, kernel_waxpby_0, 1, kernel_waxpby_off, kernel_waxpby_idx);
        brisbane_task_d2h_full(task0, mem_wcoefs, (void*) wcoefs);
        brisbane_task_submit(task0, brisbane_cpu, NULL, true);
#if 0
  		#pragma omp target teams distribute parallel for \
			map(to: xcoefs[0:n]) map(from: wcoefs[0:n])
  		for(int i=0; i<n; ++i) {
    			wcoefs[i] = xcoefs[i];
  		}
#endif
  	} else {
        brisbane_kernel kernel_waxpby_1;
        brisbane_kernel_create("waxpby_1", &kernel_waxpby_1);
        brisbane_kernel_setmem(kernel_waxpby_1, 0, mem_wcoefs, brisbane_w);
        brisbane_kernel_setmem(kernel_waxpby_1, 1, mem_xcoefs, brisbane_r);
        brisbane_kernel_setarg(kernel_waxpby_1, 2, sizeof(double), (void*) &alpha);

        brisbane_task task1;
        brisbane_task_create(&task1);
        brisbane_task_h2d_full(task1, mem_xcoefs, (void*) xcoefs);
        brisbane_task_kernel(task1, kernel_waxpby_1, 1, kernel_waxpby_off, kernel_waxpby_idx);
        brisbane_task_d2h_full(task1, mem_wcoefs, (void*) wcoefs);
        brisbane_task_submit(task1, brisbane_cpu, NULL, true);
#if 0
  		#pragma omp target teams distribute parallel for \
			map(to: xcoefs[0:n], alpha) map(from: wcoefs[0:n])
  		for(int i=0; i<n; ++i) {
    			wcoefs[i] = alpha * xcoefs[i];
  		}
#endif
  	}
  } else {
	if(alpha == 1.0) {
        brisbane_kernel kernel_waxpby_2;
        brisbane_kernel_create("waxpby_2", &kernel_waxpby_2);
        brisbane_kernel_setmem(kernel_waxpby_2, 0, mem_wcoefs, brisbane_w);
        brisbane_kernel_setmem(kernel_waxpby_2, 1, mem_xcoefs, brisbane_r);
        brisbane_kernel_setmem(kernel_waxpby_2, 2, mem_ycoefs, brisbane_r);
        brisbane_kernel_setarg(kernel_waxpby_2, 3, sizeof(double), (void*) &beta);

        brisbane_task task2;
        brisbane_task_create(&task2);
        brisbane_task_h2d_full(task2, mem_xcoefs, (void*) xcoefs);
        brisbane_task_kernel(task2, kernel_waxpby_2, 1, kernel_waxpby_off, kernel_waxpby_idx);
        brisbane_task_d2h_full(task2, mem_wcoefs, (void*) wcoefs);
        brisbane_task_submit(task2, brisbane_cpu, NULL, true);
#if 0
  		#pragma omp target teams distribute parallel for \
			map(to: xcoefs[0:n], ycoefs[0:n], beta) map(from: wcoefs[0:n])
  		for(int i=0; i<n; ++i) {
    			wcoefs[i] = xcoefs[i] + beta * ycoefs[i];
  		}
#endif
  	} else {
        brisbane_kernel kernel_waxpby_3;
        brisbane_kernel_create("waxpby_3", &kernel_waxpby_3);
        brisbane_kernel_setmem(kernel_waxpby_3, 0, mem_wcoefs, brisbane_w);
        brisbane_kernel_setmem(kernel_waxpby_3, 1, mem_xcoefs, brisbane_r);
        brisbane_kernel_setmem(kernel_waxpby_3, 2, mem_ycoefs, brisbane_r);
        brisbane_kernel_setarg(kernel_waxpby_3, 3, sizeof(double), (void*) &alpha);
        brisbane_kernel_setarg(kernel_waxpby_3, 4, sizeof(double), (void*) &beta);

        brisbane_task task3;
        brisbane_task_create(&task3);
        brisbane_task_h2d_full(task3, mem_xcoefs, (void*) xcoefs);
        brisbane_task_kernel(task3, kernel_waxpby_3, 1, kernel_waxpby_off, kernel_waxpby_idx);
        brisbane_task_d2h_full(task3, mem_wcoefs, (void*) wcoefs);
        brisbane_task_submit(task3, brisbane_cpu, NULL, true);
#if 0
  		#pragma omp target teams distribute parallel for \
			map(to: xcoefs[0:n], ycoefs[0:n], alpha, beta) map(from: wcoefs[0:n])
  		for(int i=0; i<n; ++i) {
    			wcoefs[i] = alpha * xcoefs[i] + beta * ycoefs[i];
  		}
#endif
  	}
  }

  brisbane_mem_release(mem_xcoefs);
  brisbane_mem_release(mem_ycoefs);
  brisbane_mem_release(mem_wcoefs);

#ifdef MINIFE_DEBUG_OPENMP
  std::cout << "Finished WAXPBY." << std::endl;
#endif
}

template<typename VectorType>
void
  daxpby(const MINIFE_SCALAR alpha, 
	const VectorType& x,
	const MINIFE_SCALAR beta, 
	VectorType& y)
{

  const MINIFE_LOCAL_ORDINAL n = MINIFE_MIN(x.coefs.size(), y.coefs.size());
  const MINIFE_SCALAR*  xcoefs = &x.coefs[0];
        MINIFE_SCALAR*  ycoefs = &y.coefs[0];

  brisbane_mem mem_xcoefs;
  brisbane_mem mem_ycoefs;
  brisbane_mem_create(sizeof(double) * n, &mem_xcoefs);
  brisbane_mem_create(sizeof(double) * n, &mem_ycoefs);

  size_t kernel_daxpby_off[1] = { 0 };
  size_t kernel_daxpby_idx[1] = { n };

  if(alpha == 1.0 && beta == 1.0) {
      brisbane_kernel kernel_daxpby_0;
      brisbane_kernel_create("daxpby_0", &kernel_daxpby_0);
      brisbane_kernel_setmem(kernel_daxpby_0, 0, mem_ycoefs, brisbane_rw);
      brisbane_kernel_setmem(kernel_daxpby_0, 1, mem_xcoefs, brisbane_r);

      brisbane_task task0;
      brisbane_task_create(&task0);
      brisbane_task_h2d_full(task0, mem_xcoefs, (void*) xcoefs);
      brisbane_task_h2d_full(task0, mem_ycoefs, (void*) ycoefs);
      brisbane_task_kernel(task0, kernel_daxpby_0, 1, kernel_daxpby_off, kernel_daxpby_idx);
      brisbane_task_d2h_full(task0, mem_ycoefs, (void*) ycoefs);
      brisbane_task_submit(task0, brisbane_cpu, NULL, true);
#if 0
  	  #pragma omp target teams distribute parallel for \
		map(to: xcoefs[0:n]) map(tofrom: ycoefs[0:n])
	  for(int i = 0; i < n; ++i) {
	    ycoefs[i] += xcoefs[i];
  	  }
#endif
  } else if (beta == 1.0) {
      brisbane_kernel kernel_daxpby_1;
      brisbane_kernel_create("daxpby_1", &kernel_daxpby_1);
      brisbane_kernel_setmem(kernel_daxpby_1, 0, mem_ycoefs, brisbane_rw);
      brisbane_kernel_setmem(kernel_daxpby_1, 1, mem_xcoefs, brisbane_r);
      brisbane_kernel_setarg(kernel_daxpby_1, 2, sizeof(double), (void*) &alpha);

      brisbane_task task1;
      brisbane_task_create(&task1);
      brisbane_task_h2d_full(task1, mem_xcoefs, (void*) xcoefs);
      brisbane_task_h2d_full(task1, mem_ycoefs, (void*) ycoefs);
      brisbane_task_kernel(task1, kernel_daxpby_1, 1, kernel_daxpby_off, kernel_daxpby_idx);
      brisbane_task_d2h_full(task1, mem_ycoefs, (void*) ycoefs);
      brisbane_task_submit(task1, brisbane_cpu, NULL, true);
#if 0
  	  #pragma omp target teams distribute parallel for \
		map(to: xcoefs[0:n], alpha) map(tofrom: ycoefs[0:n])
	  for(int i = 0; i < n; ++i) {
	    ycoefs[i] += alpha * xcoefs[i];
  	  }
#endif
  } else if (alpha == 1.0) {
      brisbane_kernel kernel_daxpby_2;
      brisbane_kernel_create("daxpby_2", &kernel_daxpby_2);
      brisbane_kernel_setmem(kernel_daxpby_2, 0, mem_ycoefs, brisbane_rw);
      brisbane_kernel_setmem(kernel_daxpby_2, 1, mem_xcoefs, brisbane_r);
      brisbane_kernel_setarg(kernel_daxpby_2, 2, sizeof(double), (void*) &beta);

      brisbane_task task2;
      brisbane_task_create(&task2);
      brisbane_task_h2d_full(task2, mem_xcoefs, (void*) xcoefs);
      brisbane_task_h2d_full(task2, mem_ycoefs, (void*) ycoefs);
      brisbane_task_kernel(task2, kernel_daxpby_2, 1, kernel_daxpby_off, kernel_daxpby_idx);
      brisbane_task_d2h_full(task2, mem_ycoefs, (void*) ycoefs);
      brisbane_task_submit(task2, brisbane_cpu, NULL, true);
#if 0
  	  #pragma omp target teams distribute parallel for \
		map(to: xcoefs[0:n], beta) map(tofrom: ycoefs[0:n])
	  for(int i = 0; i < n; ++i) {
	    ycoefs[i] = xcoefs[i] + beta * ycoefs[i];
  	  }
#endif
  } else if (beta == 0.0) {
      brisbane_kernel kernel_daxpby_3;
      brisbane_kernel_create("daxpby_3", &kernel_daxpby_3);
      brisbane_kernel_setmem(kernel_daxpby_3, 0, mem_ycoefs, brisbane_w);
      brisbane_kernel_setmem(kernel_daxpby_3, 1, mem_xcoefs, brisbane_r);
      brisbane_kernel_setarg(kernel_daxpby_3, 2, sizeof(double), (void*) &alpha);

      brisbane_task task3;
      brisbane_task_create(&task3);
      brisbane_task_h2d_full(task3, mem_xcoefs, (void*) xcoefs);
      brisbane_task_kernel(task3, kernel_daxpby_3, 1, kernel_daxpby_off, kernel_daxpby_idx);
      brisbane_task_d2h_full(task3, mem_ycoefs, (void*) ycoefs);
      brisbane_task_submit(task3, brisbane_cpu, NULL, true);
#if 0
  	  #pragma omp target teams distribute parallel for \
		map(to: xcoefs[0:n], alpha) map(from: ycoefs[0:n])
	  for(int i = 0; i < n; ++i) {
	    ycoefs[i] = alpha * xcoefs[i];
  	  }
#endif
  } else {
      brisbane_kernel kernel_daxpby_4;
      brisbane_kernel_create("daxpby_4", &kernel_daxpby_4);
      brisbane_kernel_setmem(kernel_daxpby_4, 0, mem_ycoefs, brisbane_rw);
      brisbane_kernel_setmem(kernel_daxpby_4, 1, mem_xcoefs, brisbane_r);
      brisbane_kernel_setarg(kernel_daxpby_4, 2, sizeof(double), (void*) &alpha);
      brisbane_kernel_setarg(kernel_daxpby_4, 3, sizeof(double), (void*) &beta);

      brisbane_task task4;
      brisbane_task_create(&task4);
      brisbane_task_h2d_full(task4, mem_xcoefs, (void*) xcoefs);
      brisbane_task_h2d_full(task4, mem_ycoefs, (void*) ycoefs);
      brisbane_task_kernel(task4, kernel_daxpby_4, 1, kernel_daxpby_off, kernel_daxpby_idx);
      brisbane_task_d2h_full(task4, mem_ycoefs, (void*) ycoefs);
      brisbane_task_submit(task4, brisbane_cpu, NULL, true);
#if 0
  	  #pragma omp target teams distribute parallel for \
		map(to: xcoefs[0:n], alpha, beta) map(tofrom: ycoefs[0:n])
	  for(int i = 0; i < n; ++i) {
	    ycoefs[i] = alpha * xcoefs[i] + beta * ycoefs[i];
  	  }
#endif
  }

  brisbane_mem_release(mem_xcoefs);
  brisbane_mem_release(mem_ycoefs);
}

//-----------------------------------------------------------
//Compute the dot product of two vectors where:
//
// x,y - input vectors
//
// result - return-value
//
template<typename Vector>
typename TypeTraits<typename Vector::ScalarType>::magnitude_type
  dot(const Vector& x,
      const Vector& y)
{
  const MINIFE_LOCAL_ORDINAL n = x.coefs.size();

  typedef typename Vector::ScalarType Scalar;
  typedef typename TypeTraits<typename Vector::ScalarType>::magnitude_type magnitude;

  const Scalar*  xcoefs = &x.coefs[0];
  const Scalar*  ycoefs = &y.coefs[0];
  MINIFE_SCALAR result = 0;

  printf("[%s:%d] n[%d]\n", __FILE__, __LINE__, n);

  brisbane_mem mem_xcoefs;
  brisbane_mem mem_ycoefs;
  brisbane_mem mem_result;
  brisbane_mem_create(sizeof(double) * n, &mem_xcoefs);
  brisbane_mem_create(sizeof(double) * n, &mem_ycoefs);
  brisbane_mem_create(sizeof(double), &mem_result);
  brisbane_mem_reduce(mem_result, brisbane_sum, brisbane_double);

  size_t kernel_dot_off[1] = { 0 };
  size_t kernel_dot_idx[1] = { n };

  brisbane_kernel kernel_dot;
  brisbane_kernel_create("kernel_dot", &kernel_dot);
  brisbane_kernel_setmem(kernel_dot, 0, mem_ycoefs, brisbane_r);
  brisbane_kernel_setmem(kernel_dot, 1, mem_xcoefs, brisbane_r);
  brisbane_kernel_setmem(kernel_dot, 2, mem_result, brisbane_rw);

  brisbane_task task;
  brisbane_task_create(&task);
  brisbane_task_h2d_full(task, mem_xcoefs, (void*) xcoefs);
  brisbane_task_h2d_full(task, mem_ycoefs, (void*) ycoefs);
  brisbane_task_kernel(task, kernel_dot, 1, kernel_dot_off, kernel_dot_idx);
  brisbane_task_d2h(task, mem_result, 0, sizeof(double), &result);
  brisbane_task_submit(task, brisbane_cpu, NULL, true);

  brisbane_mem_release(mem_xcoefs);
  brisbane_mem_release(mem_ycoefs);
  brisbane_mem_release(mem_result);
#if 0
  #pragma omp target teams distribute parallel for reduction(+:result) \
	map(to: xcoefs[0:n], ycoefs[0:n]) map(tofrom: result)
  for(int i=0; i<n; ++i) {
    result += xcoefs[i] * ycoefs[i];
  }
#endif

#ifdef HAVE_MPI
  magnitude local_dot = result, global_dot = 0;
  MPI_Datatype mpi_dtype = TypeTraits<magnitude>::mpi_type();  
  MPI_Allreduce(&local_dot, &global_dot, 1, mpi_dtype, MPI_SUM, MPI_COMM_WORLD);
  return global_dot;
#else
  return result;
#endif
}

template<typename Vector>
typename TypeTraits<typename Vector::ScalarType>::magnitude_type
  dot_r2(const Vector& x)
{
#ifdef MINIFE_DEBUG_OPENMP
 	int myrank;
	MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
	std::cout << "[" << myrank << "] Starting dot..." << std::endl;
#endif

  const MINIFE_LOCAL_ORDINAL n = x.coefs.size();

#ifdef MINIFE_DEBUG
  if (y.local_size < n) {
    std::cerr << "miniFE::dot ERROR, y must be at least as long as x."<<std::endl;
    n = y.local_size;
  }
#endif

  typedef typename Vector::ScalarType Scalar;
  typedef typename TypeTraits<typename Vector::ScalarType>::magnitude_type magnitude;

  const MINIFE_SCALAR*  xcoefs = &x.coefs[0];
  MINIFE_SCALAR result = 0;

  brisbane_mem mem_xcoefs;
  brisbane_mem mem_result;
  brisbane_mem_create(sizeof(double) * n, &mem_xcoefs);
  brisbane_mem_create(sizeof(double), &mem_result);
  brisbane_mem_reduce(mem_result, brisbane_sum, brisbane_double);

  size_t kernel_dot_r2_off[1] = { 0 };
  size_t kernel_dot_r2_idx[1] = { n };

  brisbane_kernel kernel_dot_r2;
  brisbane_kernel_create("kernel_dot_r2", &kernel_dot_r2);
  brisbane_kernel_setmem(kernel_dot_r2, 0, mem_xcoefs, brisbane_r);
  brisbane_kernel_setmem(kernel_dot_r2, 1, mem_result, brisbane_rw);

  brisbane_task task;
  brisbane_task_create(&task);
  brisbane_task_h2d_full(task, mem_xcoefs, (void*) xcoefs);
  brisbane_task_kernel(task, kernel_dot_r2, 1, kernel_dot_r2_off, kernel_dot_r2_idx);
  brisbane_task_d2h(task, mem_result, 0, sizeof(double), &result);
  brisbane_task_submit(task, brisbane_cpu, NULL, true);

  brisbane_mem_release(mem_xcoefs);
  brisbane_mem_release(mem_result);
#if 0
  #pragma omp target teams distribute parallel for reduction(+:result) \
	map(to: xcoefs[0:n]) map(tofrom: result) 
  for(int i=0; i<n; ++i) {
    result += xcoefs[i] * xcoefs[i];
  }
#endif

#ifdef HAVE_MPI
  magnitude local_dot = result, global_dot = 0;
  MPI_Datatype mpi_dtype = TypeTraits<magnitude>::mpi_type();  
  MPI_Allreduce(&local_dot, &global_dot, 1, mpi_dtype, MPI_SUM, MPI_COMM_WORLD);

#ifdef MINIFE_DEBUG_OPENMP
 	std::cout << "[" << myrank << "] Completed dot." << std::endl;
#endif

  return global_dot;
#else
#ifdef MINIFE_DEBUG_OPENMP
 	std::cout << "[" << myrank << "] Completed dot." << std::endl;
#endif
  return result;
#endif
}

}//namespace miniFE

#endif

