#include "LinearAlgebraStructs.h"

__global__
void squareElements_kernel(matrix a_d)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	int column_index = i%a_d.columns;
	int row_index = (i/a_d.columns);
	if (row_index<a_d.rows) { //make sure not to write outside of matrix, incase the number of elements did not have a base of 256
		double value = a_d.elements[a_d.columns*(row_index+1)+column_index];
		a_d.elements[a_d.columns * (row_index + 1) + column_index] = value * value;
	}
}
/**
*creates a projection matrix on the gpu to the given matrix on the device.
**/
void squareElements_d(matrix a_d){
	int N = a_d.rows*a_d.columns; //each thread handles two elements of the matrix
	squareElements_kernel<<<(N+255)/256, 256>>>(a_d);
}