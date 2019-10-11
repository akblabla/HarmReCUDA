#include <cuda_runtime.h>
#include "cublas_v2.h"
#include "LinearAlgebraStructs.h"

__global__
void partialMatrixSummation_kernel(matrix dest_d, matrix src_d, unsigned int rowFactor, unsigned int columnFactor)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int destColumnIndex = i / src_d.rows;
	int destRowIndex = i % src_d.rows;
	dest_d.elements[MATRIX_INDEX(destRowIndex, destColumnIndex, dest_d.rows)] = 0;
	if (destRowIndex< dest_d.rows){ //don't write outside matrix
		for (int srcRowIndex = rowFactor * destRowIndex; srcRowIndex < rowFactor * (destRowIndex + 1); ++srcRowIndex) {
			for (int srcColumnIndex = columnFactor * destColumnIndex; srcColumnIndex < columnFactor * (destColumnIndex + 1); ++srcColumnIndex) {
				dest_d.elements[MATRIX_INDEX(destRowIndex, destColumnIndex, dest_d.rows)] += src_d.elements[MATRIX_INDEX(srcRowIndex, srcColumnIndex, src_d.rows)];
			}
		}
	}
}
/**
*creates a projection matrix on the gpu to the given matrix on the device.
**/
extern "C" void partialMatrixSummation_cuda(matrix dest_d, const matrix src_d, unsigned int rowFactor, unsigned int columnFactor) {
	int N = src_d.rows * src_d.columns; //each thread handles two elements of the matrix
	partialMatrixSummation_kernel <<<(N + 255) / 256, 256 >>> (dest_d, src_d, rowFactor, columnFactor);
}