#include "LinearAlgebraStructs.h"

__global__
void squareElements_kernel(matrix a_d)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	int columnIndex = i / a_d.rows;
	int rowIndex = i % a_d.rows; //skip every second row
	if (columnIndex < a_d.columns) {//make sure not to write outside of matrix, incase the number of elements did not have a base of 1024
		double value = a_d.elements[MATRIX_INDEX(rowIndex, columnIndex, a_d.rows)];
		a_d.elements[MATRIX_INDEX(rowIndex,columnIndex,a_d.rows)] = value * value;
	}
}
/**
*creates a projection matrix on the gpu to the given matrix on the device.
**/
extern "C" void squareElements_d(matrix a_d){
	int N = a_d.rows*a_d.columns; //each thread handles two elements of the matrix
	squareElements_kernel<<<(N+1023)/ 1024, 1024>>>(a_d);
}