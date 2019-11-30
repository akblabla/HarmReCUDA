#include "moorePenroseInversion.h"
#include "cublas_v2.h"
#include "helper_cuda.h"
#include <vector>
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
	Matrix_d** subDesignMatricesList = new Matrix_d*[batchSize];
	double** subDesignMatricesElementsList = new double * [batchSize];
	Matrix_d** correlationMatricesList = new Matrix_d * [batchSize];
	double** correlationMatrixElementsList = new double * [batchSize];
	Matrix_d** invCorrelationMatricesList = new Matrix_d * [batchSize];
	double** invCorrelationMatrixElementsList = new double* [batchSize];
	for (int i = 0; i < batchSize; ++i){
		subDesignMatricesList[i] = new Matrix_d(m, n,AMatrix::M_ALLOCATE);
		designMatrix_d.getSubMatrix(*(subDesignMatricesList[i]), 0, -1, harmonics_d.getElementsCount() * 2 * i, harmonics_d.getElementsCount() * 2 * (i + 1) + 1);
		subDesignMatricesElementsList[i] = correlationMatricesList[i]->getCMatrix().elements;
		correlationMatricesList[i] = new Matrix_d(n, n, AMatrix::M_ALLOCATE);
		correlationMatrixElementsList[i] = correlationMatricesList[i]->getCMatrix().elements;
		invCorrelationMatricesList[i] = new Matrix_d(n, n, AMatrix::M_ALLOCATE);
		invCorrelationMatrixElementsList[i] = invCorrelationMatricesList[i]->getCMatrix().elements;
	}

	cublasDgemmBatched(handle,
		cublasOperation_t::CUBLAS_OP_T,
		cublasOperation_t::CUBLAS_OP_N,
		n, m, n,
		&alpha,
		subDesignMatricesElementsList, m,
		subDesignMatricesElementsList, m,
		&beta,
		correlationMatrixElementsList, n, batchSize);
	auto pivotArray = new int[n*batchSize];
	auto infoArray = new int[n * batchSize];
	cublasDgetrfBatched(handle,
		n,
		correlationMatrixElementsList,
		n,
		pivotArray,
		infoArray,
		batchSize);
	cublasDgetriBatched(handle,
		n,
		correlationMatrixElementsList,
		n,
		pivotArray,
		invCorrelationMatrixElementsList,
		n,
		infoArray,
		batchSize);
	/*cublasDgemmBatched(handle,
		cublasOperation_t::CUBLAS_OP_T,
		cublasOperation_t::CUBLAS_OP_N,
		n, m, n,
		&alpha,
		subDesignMatricesElementsList, m,
		subDesignMatricesElementsList, m,
		&beta,
		correlationMatrixElementsList, n, batchSize);*/
}
