#include <cuda_runtime.h>
#define M_PI   3.14159265358979323846264338327950288
#include "cublas_v2.h"
#include "LinearAlgebraStructs.h"

__global__
void generateProjectionMatrix_kernel(matrix dest_d, const double minFreq, const double maxFreq, const double startTime, const double deltaTime, const matrix harmonics_d)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	int columnIndex = (2 * i)/dest_d.rows;
	int rowIndex = ((2 * i) % dest_d.rows); //skip every second row
	double t = columnIndex*deltaTime+startTime;
	int fundamentalFrequencyIndex = rowIndex / (2*harmonics_d.rows);
	int fundamentalFrequencyEndIndex = (dest_d.rows-1) / (2 * harmonics_d.rows);
	int harmonicIndex = (rowIndex/2) % (harmonics_d.rows);
	double fundamentalFrequency;
	if (fundamentalFrequencyEndIndex>0){
		fundamentalFrequency = minFreq + ((double) fundamentalFrequencyIndex) * (maxFreq - minFreq) / ((double)fundamentalFrequencyEndIndex);
	}
	else {
		fundamentalFrequency = minFreq;
	}
	double freq = fundamentalFrequency * harmonics_d.elements[harmonicIndex];
	

	double phase = freq*t;
	double finalPhase = freq*(dest_d.columns * deltaTime + startTime);
	double finalWholePhase = ((double)(2 * M_PI)) * __double2int_rd(finalPhase / (2*M_PI));
	double renomalizeFactor = finalPhase / finalWholePhase;
	if (columnIndex < dest_d.columns && phase < finalWholePhase) {//make sure not to write outside of matrix, incase the number of elements did not have a base of 1024
		sincos(
			phase,
			&(dest_d.elements[MATRIX_INDEX(rowIndex + 1, columnIndex, dest_d.rows)]),
			&(dest_d.elements[MATRIX_INDEX(rowIndex, columnIndex, dest_d.rows)])
		);
		dest_d.elements[MATRIX_INDEX(rowIndex + 1, columnIndex, dest_d.rows)] *= renomalizeFactor;
		dest_d.elements[MATRIX_INDEX(rowIndex, columnIndex, dest_d.rows)] *= renomalizeFactor;
	}
}
/**
*creates a projection matrix on the gpu to the given matrix on the device.
**/
extern "C" void generateProjectionMatrix_cuda(matrix dest_d, const double minFreq, const double maxFreq, const double startTime, const double deltaTime, const matrix harmonics_d){
	int N = dest_d.rows* dest_d.columns/2; //each thread handles two elements of the matrix
	generateProjectionMatrix_kernel <<< (N+ 255)/ 256, 256 >>>(dest_d, minFreq, maxFreq, startTime, deltaTime, harmonics_d);
}