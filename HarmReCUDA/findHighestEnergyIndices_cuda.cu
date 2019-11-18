#include <cuda_runtime.h>
#include "cublas_v2.h"
#include "LinearAlgebraStructs.h"

__global__
void findHighestEnergyIndices_kernel(matrix dest_d, matrix src_d, unsigned int rowFactor, unsigned int columnFactor)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int destColumnIndex = (i / dest_d.rows) % dest_d.columns;
	int destRowIndex = i % dest_d.rows;
	int srcColumnIndex = ((i / columnFactor) / src_d.rows) % src_d.columns;
	int srcRowIndex = ((i / rowFactor) % src_d.rows);
	if (destColumnIndex < dest_d.columns && srcColumnIndex < src_d.columns) { //don't write outside matrix
		dest_d.elements[MATRIX_INDEX(destRowIndex, destColumnIndex, dest_d.rows)] *= src_d.elements[MATRIX_INDEX(srcRowIndex, srcColumnIndex, src_d.rows)];
	}
}


extern "C" void findHighestEnergyIndices_cuda(matrix dest_d, const matrix src_d, unsigned int rowFactor, unsigned int columnFactor) {
	int N = dest_d.rows * dest_d.columns;
	findHighestEnergyIndices_kernel << <(N + 1023) / 1024, 1024 >> > (dest_d, src_d, rowFactor, columnFactor);
}