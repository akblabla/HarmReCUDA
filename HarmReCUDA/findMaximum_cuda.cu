#include "LinearAlgebraStructs.h"

__global__
void findMaximum_kernel(matrix a_d)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int columnIndex = i;
	if (columnIndex < a_d.columns) {//make sure not to write outside of matrix, incase the number of elements did not have a base of 1024
		double largestValue = -1;
		int largestIndex = 0;
		for (int rowIndex = 0; rowIndex < a_d.rows; ++rowIndex) {
			if (a_d.elements[MATRIX_INDEX(rowIndex, columnIndex, a_d.rows)]>largestValue) {
				a_d.elements[MATRIX_INDEX(largestIndex, columnIndex, a_d.rows)] = 0;
				largestIndex = rowIndex;
				largestValue = a_d.elements[MATRIX_INDEX(rowIndex, columnIndex, a_d.rows)];
				a_d.elements[MATRIX_INDEX(largestIndex, columnIndex, a_d.rows)] = 1;
			}
			else
			{
				a_d.elements[MATRIX_INDEX(rowIndex, columnIndex, a_d.rows)] = 0;
			}
		}
	}
}
/**
*creates a projection matrix on the gpu to the given matrix on the device.
**/
extern "C" void findMaximum_cuda(matrix a_d) {
	int N = a_d.columns; //each thread handles one column each
	findMaximum_kernel << <(N + 255) / 256, 256 >> > (a_d);
}