#include <cuda_runtime.h>
#define M_PI   3.14159265358979323846264338327950288
#include "cublas_v2.h"
#include "LinearAlgebraStructs.h"

__global__
void generateProjectionMatrix_kernel(matrix dest_d, const matrix freq_d, const matrix time_d, const matrix harmonics_d)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	int columnIndex = (2 * i)/dest_d.rows;
	int rowIndex = ((2 * i) % dest_d.rows); //skip every second row
	double t = time_d.elements[columnIndex];
	int fundamentalFrequencyIndex = rowIndex / (2*harmonics_d.rows);
	int harmonicIndex = (rowIndex/2) % (harmonics_d.rows);
	double fundamentalFrequency = 2 * M_PI *freq_d.elements[fundamentalFrequencyIndex];
	double freq = fundamentalFrequency * harmonics_d.elements[harmonicIndex];
	double phase = freq* t;
	if (columnIndex < dest_d.columns) {//make sure not to write outside of matrix, incase the number of elements did not have a base of 1024
		sincos(
			phase,
			&(dest_d.elements[MATRIX_INDEX(rowIndex + 1, columnIndex, dest_d.ld)]),
			&(dest_d.elements[MATRIX_INDEX(rowIndex, columnIndex, dest_d.ld)])
		);
		//dest_d.elements[MATRIX_INDEX(rowIndex + 1, columnIndex, dest_d.rows)] *= renomalizeFactor;
		//dest_d.elements[MATRIX_INDEX(rowIndex, columnIndex, dest_d.rows)] *= renomalizeFactor;
	}
}
/**
*creates a projection matrix on the gpu to the given matrix on the device.
**/
extern "C" void generateProjectionMatrix_cuda(matrix dest_d, const matrix freq_d, const matrix time_d, const matrix harmonics_d){
	int N = dest_d.rows* dest_d.columns/2; //each thread handles two elements of the matrix
	generateProjectionMatrix_kernel <<< (N+ 1023)/ 1024, 1024 >>>(dest_d, freq_d, time_d, harmonics_d);
}