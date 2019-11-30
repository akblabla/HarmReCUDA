#include "LinearAlgebraStructs.h"
#define M_PI   3.14159265358979323846264338327950288

__global__
void blackmanWindow_kernel(matrix a_d)
{
	double a0 = 7938.0 / 18608.0;
	double a1 = 9240.0 / 18608.0;
	double a2 = 1430.0 / 18608.0;
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int columnIndex = i / a_d.rows;
	int rowIndex = i % a_d.rows;
	if (columnIndex < a_d.columns) {//make sure not to write outside of matrix, incase the number of elements did not have a base of 1024
		double value = a_d.elements[MATRIX_INDEX(rowIndex, columnIndex, a_d.ld)];
		double window = (a0 - a1 * cos((2 * M_PI * columnIndex) / a_d.columns) + a2 * cos((4 * M_PI * columnIndex) / a_d.columns)) / a0;
		a_d.elements[MATRIX_INDEX(rowIndex, columnIndex, a_d.ld)] = value * window;
	}
}
/**
*creates a projection matrix on the gpu to the given matrix on the device.
**/
extern "C" void blackmanWindow_cuda(matrix a_d) {
	int N = a_d.rows * a_d.columns; //each thread handles two elements of the matrix
	blackmanWindow_kernel << <(N + 1023) / 1024, 1024 >> > (a_d);
}