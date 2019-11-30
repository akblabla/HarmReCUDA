#include <cuda_runtime.h>
#include "cublas_v2.h"
#include "LinearAlgebraStructs.h"

__global__
void partialMatrixSummation_kernel(matrix dest_d, matrix src_d, unsigned int rowFactor, unsigned int columnFactor)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int destColumnIndex = i / dest_d.rows;
	int destRowIndex = i % dest_d.rows;
	if (i< dest_d.rows* dest_d.columns){ //don't write outside matrix
		dest_d.elements[MATRIX_INDEX(destRowIndex, destColumnIndex, dest_d.ld)] = 0;
		for (int srcRowIndex = rowFactor * destRowIndex; srcRowIndex < rowFactor * (destRowIndex + 1); ++srcRowIndex) {
			for (int srcColumnIndex = columnFactor * destColumnIndex; srcColumnIndex < columnFactor * (destColumnIndex + 1); ++srcColumnIndex) {

				dest_d.elements[MATRIX_INDEX(destRowIndex, destColumnIndex, dest_d.ld)] += src_d.elements[MATRIX_INDEX(srcRowIndex, srcColumnIndex, src_d.ld)];
			}
		}
	}
}
/**
*creates a projection matrix on the gpu to the given matrix on the device.
**/
extern "C" void partialMatrixSummation_cuda(matrix dest_d, const matrix src_d, unsigned int rowFactor, unsigned int columnFactor) {
	int N = dest_d.rows * dest_d.columns;
	partialMatrixSummation_kernel <<<(N + 255) / 256, 256 >>> (dest_d, src_d, rowFactor, columnFactor);
}