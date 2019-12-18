#include "moorePenroseInversion.h"
#include <vector>
#include <cuda_runtime.h>
#include "cublas_v2.h"
#include "helper_cuda.h"
void moorePenroseInversion_d(Matrix_d& designMatrix_d, Vector_d& harmonics_d, Vector_d& fundamentalFrequencies_d)
{
	int batchSize = fundamentalFrequencies_d.getElementsCount();
	double alpha = 1;
	double beta = 0;
	int m = designMatrix_d.getRows();
	int n = harmonics_d.getElementsCount() * 2;
	cublasStatus_t stat;
	cublasHandle_t handle;
	stat = cublasCreate(&handle);
	Matrix_d** subDesignMatricesList_d = new Matrix_d*[batchSize];
	double** subDesignMatricesElementsList = new double * [batchSize];
	double** subDesignMatricesElementsList_d;

	Matrix_d** correlationMatricesList_d = new Matrix_d * [batchSize];
	double** correlationMatrixElementsList = new double * [batchSize];
	double** correlationMatrixElementsList_d;

	Matrix_d** invCorrelationMatricesList_d = new Matrix_d * [batchSize];
	double** invCorrelationMatrixElementsList = new double* [batchSize];
	double** invCorrelationMatrixElementsList_d;

	cudaMalloc((void**)& subDesignMatricesElementsList_d, batchSize * sizeof(double*));
	cudaMalloc((void**)& correlationMatrixElementsList_d, batchSize * sizeof(double*));
	cudaMalloc((void**)& invCorrelationMatrixElementsList_d, batchSize * sizeof(double*));
	for (int i = 0; i < batchSize; ++i){
		subDesignMatricesList_d[i] = new Matrix_d(m, n,AMatrix::M_NO_INIT);
		designMatrix_d.getSubMatrix(*(subDesignMatricesList_d[i]), 0, -1, n * i, n * (i + 1)-1);
		subDesignMatricesElementsList[i] = (subDesignMatricesList_d[i])->getCMatrix().elements;

		cudaMemcpy(subDesignMatricesElementsList_d, subDesignMatricesElementsList, batchSize*sizeof(double*), cudaMemcpyHostToDevice);

		correlationMatricesList_d[i] = new Matrix_d(n, n, AMatrix::M_ALLOCATE);
		correlationMatrixElementsList[i] = (correlationMatricesList_d[i])->getCMatrix().elements;
	
		cudaMemcpy(correlationMatrixElementsList_d, correlationMatrixElementsList, batchSize * sizeof(double*), cudaMemcpyHostToDevice);

		invCorrelationMatricesList_d[i] = new Matrix_d(n, n, AMatrix::M_ALLOCATE);
		invCorrelationMatrixElementsList[i] = invCorrelationMatricesList_d[i]->getCMatrix().elements;
		cudaMemcpy(invCorrelationMatrixElementsList_d, invCorrelationMatrixElementsList, batchSize * sizeof(double*), cudaMemcpyHostToDevice);
	}
	stat = cublasDgemmBatched(handle,
		CUBLAS_OP_T,
		CUBLAS_OP_N,
		n, n, m,
		&alpha,
		(const double**)subDesignMatricesElementsList_d, subDesignMatricesList_d[0]->getLeadingDimension(),
		(const double**)subDesignMatricesElementsList_d, subDesignMatricesList_d[0]->getLeadingDimension(),
		&beta,
		correlationMatrixElementsList_d, correlationMatricesList_d[0]->getLeadingDimension(), batchSize);
	if (stat != CUBLAS_STATUS_SUCCESS) {
		throw std::exception(_cudaGetErrorEnum(stat));
		return;
	}
	
	int* pivotArray_d;
	int pivotArray[5];
	auto cudaStat = cudaMalloc<int>(& pivotArray_d, n * batchSize * sizeof(int));
	if (cudaStat != cudaSuccess) {
		throw std::exception(cudaGetErrorString(cudaStat));
		return;
	}
	int* infoArray_d;
	int infoArray[5];
	cudaStat = cudaMalloc<int>(&infoArray_d, batchSize * sizeof(int));
	if (cudaStat != cudaSuccess) {
		throw std::exception(cudaGetErrorString(cudaStat));
		return;
	}
	stat = cublasDgetrfBatched(handle,
		n,
		correlationMatrixElementsList_d,
		correlationMatricesList_d[0]->getLeadingDimension(),
		pivotArray_d,
		infoArray_d,
		batchSize);
	if (stat != CUBLAS_STATUS_SUCCESS) {
		throw std::exception(_cudaGetErrorEnum(stat));
		return;
	}
	stat = cublasDgetriBatched(handle,
		n,
		correlationMatrixElementsList_d,
		n,
		pivotArray_d,
		invCorrelationMatrixElementsList_d,
		n,
		infoArray_d,
		batchSize);

	if (stat != CUBLAS_STATUS_SUCCESS) {
		throw std::exception(_cudaGetErrorEnum(stat));
		return;
	}
	cublasDgemmBatched(handle,
		CUBLAS_OP_N,
		CUBLAS_OP_N,
		m, n, n,
		&alpha,
		subDesignMatricesElementsList_d, subDesignMatricesList_d[0]->getLeadingDimension(),
		invCorrelationMatrixElementsList_d, invCorrelationMatricesList_d[0]->getLeadingDimension(),
		&beta,
		subDesignMatricesElementsList_d, subDesignMatricesList_d[0]->getLeadingDimension(), batchSize);
	cublasDestroy(handle);
	designMatrix_d.transpose();
}
