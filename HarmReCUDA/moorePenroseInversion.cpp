#include "moorePenroseInversion.h"
#include <vector>
#include <cuda_runtime.h>
#include "cublas_v2.h"
#include "helper_cuda.h"
void moorePenroseInversion_d(Matrix_d& designMatrix_d, Vector_d& harmonics_d, Vector_d& fundamentalFrequencies_d)
{
	int batchSize = fundamentalFrequencies_d.getElementsCount();
	Matrix_d designMatrixClone_d(designMatrix_d, AMatrix::M_ASSIGN);
	double alpha = 1;
	double beta = 0;
	int m = designMatrix_d.getRows();
	int n = harmonics_d.getElementsCount() * 2;
	cublasStatus_t stat;
	cublasHandle_t handle;
	stat = cublasCreate(&handle);
	Matrix_d** subDesignMatricesList = new Matrix_d*[batchSize];
	double** subDesignMatricesElementsList = new double * [batchSize];

	Matrix_d** subDesignMatricesCloneList = new Matrix_d * [batchSize];
	double** subDesignMatricesCloneElementsList = new double* [batchSize];

	Matrix_d** correlationMatricesList = new Matrix_d * [batchSize];
	double** correlationMatrixElementsList = new double * [batchSize];

	Matrix_d** invCorrelationMatricesList = new Matrix_d * [batchSize];
	double** invCorrelationMatrixElementsList = new double* [batchSize];

	for (int i = 0; i < batchSize; ++i){
		subDesignMatricesList[i] = new Matrix_d(n, n,AMatrix::M_ALLOCATE);
		//designMatrix_d.getSubMatrix(*(subDesignMatricesList[i]), 0, -1, n * i, n * (i + 1)-1);
		subDesignMatricesElementsList[i] = (subDesignMatricesList[i])->getCMatrix().elements;

		subDesignMatricesCloneList[i] = new Matrix_d(n, n, AMatrix::M_ALLOCATE);
		//designMatrix_d.getSubMatrix(*(subDesignMatricesCloneList[i]), 0, -1, n * i, n * (i + 1)-1);
		subDesignMatricesCloneElementsList[i] = (subDesignMatricesCloneList[i])->getCMatrix().elements;

		correlationMatricesList[i] = new Matrix_d(n, n, AMatrix::M_ALLOCATE);
		correlationMatrixElementsList[i] = (correlationMatricesList[i])->getCMatrix().elements;

		invCorrelationMatricesList[i] = new Matrix_d(n, n, AMatrix::M_ALLOCATE);
		invCorrelationMatrixElementsList[i] = invCorrelationMatricesList[i]->getCMatrix().elements;
	}
	stat = cublasDgemmBatched(handle,
		CUBLAS_OP_T,
		CUBLAS_OP_N,
		n, n, n,
		&alpha,
		(const double**)subDesignMatricesElementsList, subDesignMatricesList[0]->getLeadingDimension(),
		(const double**)subDesignMatricesCloneElementsList, subDesignMatricesList[0]->getLeadingDimension(),
		&beta,
		correlationMatrixElementsList, correlationMatricesList[0]->getLeadingDimension(), batchSize);
	if (stat != CUBLAS_STATUS_SUCCESS) {
		throw std::exception(_cudaGetErrorEnum(stat));
		return;
	}
	/*
	int* pivotArray;

	auto cudaStat = cudaMalloc((void**)& pivotArray, n * batchSize * sizeof(int));
	if (cudaStat != cudaSuccess) {
		throw std::exception(cudaGetErrorString(cudaStat));
		return;
	}
	int* infoArray;
	cudaStat = cudaMalloc((void**)& infoArray, n * batchSize * sizeof(int));
	if (cudaStat != cudaSuccess) {
		throw std::exception(cudaGetErrorString(cudaStat));
		return;
	}
	stat = cublasDgetrfBatched(handle,
		n,
		correlationMatrixElementsList,
		n,
		pivotArray,
		infoArray,
		batchSize);
	if (stat != CUBLAS_STATUS_SUCCESS) {
		throw std::exception(_cudaGetErrorEnum(stat));
		return;
	}
	stat = cublasDgetriBatched(handle,
		n,
		correlationMatrixElementsList,
		n,
		pivotArray,
		invCorrelationMatrixElementsList,
		n,
		infoArray,
		batchSize);

	if (stat != CUBLAS_STATUS_SUCCESS) {
		throw std::exception(_cudaGetErrorEnum(stat));
		return;
	}*/
	/*cublasDgemmBatched(handle,
		cublasOperation_t::CUBLAS_OP_T,
		cublasOperation_t::CUBLAS_OP_N,
		n, m, n,
		&alpha,
		subDesignMatricesElementsList, m,
		subDesignMatricesElementsList, m,
		&beta,
		correlationMatrixElementsList, n, batchSize);*/
	cublasDestroy(handle);
}
