/****************************************************
 * hlbl_lm_cuda
 ****************************************************/
#include <cstdio>
#include <cstdlib>
#include <complex>
#include <cmath>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <library_types.h>

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cublas_v2.h>
#include <cuComplex.h>
#include <cublas_api.h>

// using data_type = cuDoubleComplex;

#include "hlbl_lm_cuda.h"

/***********************************************************/
/***********************************************************/

/***********************************************************
 * upload field to gpu
 ***********************************************************/
// ?? extra function ?

#if 0
/***********************************************************/
/***********************************************************/

/***********************************************************
 * compute p = V^H s
 *
 * V is nv x nx (C) = nx x nv (F)
 * s is ns x nx (C) = nx x ns (F)
 *
 * p is [nx x nv]^H x [nx x ns] = nv x ns (F) = ns x nv (C)
 *
 * HOW TO ?
 * - choose which device, cudaSetDevice ?
 ***********************************************************/
int hlbl_lm_reduce ( void * const h_p, void * const h_v, void * const h_s, const int nv, const int nx, const int ns ) 
{

  cublasHandle_t cublasH = NULL;
  cudaStream_t stream = NULL;

  // like in cpu version projec(...)
  const int lda = nx;
  const int ldb = nx;
  const int ldc = nv;

  const data_type alpha = { 1.0, 0.0 };
  const data_type beta = { 0.0, 0.0 };

  data_type *d_v = nullptr;
  data_type *d_s = nullptr;
  data_type *d_p = nullptr;

  cublasOperation_t transa = CUBLAS_OP_C;
  cublasOperation_t transb = CUBLAS_OP_N;

  /* step 1: create cublas handle, bind a stream */
  CUBLAS_CHECK(cublasCreate(&cublasH));

  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  CUBLAS_CHECK(cublasSetStream(cublasH, stream));

  /* step 2: copy data to device */
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_v), sizeof(data_type) * nv * nx ) );
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_s), sizeof(data_type) * ns * nx ) );
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_p), sizeof(data_type) * ns * nv ) );

  CUDA_CHECK(cudaMemcpyAsync(d_v, h_v, sizeof(data_type) * nv * nx, cudaMemcpyHostToDevice, stream));

  CUDA_CHECK(cudaMemcpyAsync(d_s, h_s, sizeof(data_type) * ns * nx, cudaMemcpyHostToDevice, stream));

  /* step 3: compute */
  CUBLAS_CHECK( cublasZgemm(cublasH, transa, transb, nv, ns, nx, &alpha, d_v, lda, d_s, ldb, &beta, d_p, ldc));

  /* step 4: copy data to host */
  CUDA_CHECK(cudaMemcpyAsync( h_p, d_p, sizeof(data_type) * ns * nv, cudaMemcpyDeviceToHost, stream));

  CUDA_CHECK(cudaStreamSynchronize(stream));

  /* free resources */
  CUDA_CHECK(cudaFree(d_v));
  CUDA_CHECK(cudaFree(d_s));
  CUDA_CHECK(cudaFree(d_p));

  CUBLAS_CHECK(cublasDestroy(cublasH));

  CUDA_CHECK(cudaStreamDestroy(stream));

  return(0);
}  // end of hlbl_lm_reduce
#endif

/***********************************************************/
/***********************************************************/

/***********************************************************
 * compute p = V^H s
 *
 * V is nv x nx (C) = nx x nv (F)
 * s is ns x nx (C) = nx x ns (F)
 *
 * p is [nx x nv]^H x [nx x ns] = nv x ns (F) = ns x nv (C)
 *
 ***********************************************************/
int hlbl_lm_reduce ( cudaStream_t stream, cublasHandle_t cublasH, double _Complex * const h_p, 
    cuda_data_type * const d_v, cuda_data_type * const d_s, const int nv, const int nx, const int ns ) 
{
  cuda_data_type *d_p = nullptr;

  // like in cpu version projec(...)
  const int lda = nx;
  const int ldb = nx;
  const int ldc = nv;

  const cuda_data_type alpha = { 1.0, 0.0 };
  const cuda_data_type beta  = { 0.0, 0.0 };

  cublasOperation_t transa = CUBLAS_OP_C;
  cublasOperation_t transb = CUBLAS_OP_N;

  /* copy data to device */
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_p), sizeof(cuda_data_type) * ns * nv ) );

  /* linear algebra computation */
  CUBLAS_CHECK( cublasZgemm(cublasH, transa, transb, nv, ns, nx, &alpha, d_v, lda, d_s, ldb, &beta, d_p, ldc));

  /* step 4: copy data to host */
  CUDA_CHECK(cudaMemcpyAsync( h_p, d_p, sizeof(cuda_data_type) * ns * nv, cudaMemcpyDeviceToHost, stream));

  CUDA_CHECK(cudaStreamSynchronize(stream));

  /* free resources */
  CUDA_CHECK(cudaFree(d_p));

  return(0);
}  // end of hlbl_lm_reduce
