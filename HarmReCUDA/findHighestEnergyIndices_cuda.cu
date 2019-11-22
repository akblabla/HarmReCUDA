#include <cuda_runtime.h>
#include "cublas_v2.h"
#include "LinearAlgebraStructs.h"
#define SEARCH_WIDTH 2
__global__
void findHighestEnergyFrequencies_kernel(matrix highestEnergies_d, matrix centralFreq_d, const matrix energies_d, const matrix freq_d)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int energiesColumnIndex = (i / highestEnergies_d.rows) % highestEnergies_d.columns;
	int srcColumnIndex = energiesColumnIndex;

	double maxEnergy = 0;
	double maxIndex = 0;
	for (int srcRowIndex = 0; srcRowIndex < energies_d.rows; srcRowIndex++) {
		double energy = energies_d.elements[MATRIX_INDEX(srcRowIndex, srcColumnIndex, energies_d.rows)];
		if (energy > maxEnergy) {
			maxEnergy = energy;
			maxIndex = srcRowIndex;
		}
	}
	int centralFreqColumnIndex = energiesColumnIndex;
	if (energiesColumnIndex < highestEnergies_d.columns) { //don't write outside matrix
		int centralFreqIndex = maxIndex;
		if (centralFreqIndex < SEARCH_WIDTH) centralFreqIndex = SEARCH_WIDTH;
		if (centralFreqIndex > freq_d.rows-SEARCH_WIDTH-1) centralFreqIndex = freq_d.rows - SEARCH_WIDTH - 1;

		centralFreq_d.elements[centralFreqColumnIndex] = freq_d.elements[centralFreqIndex];

		for (int energiesRowIndex = 0; energiesRowIndex < highestEnergies_d.rows; energiesRowIndex++) {
			highestEnergies_d.elements[MATRIX_INDEX(energiesRowIndex, energiesColumnIndex, highestEnergies_d.rows)] = energies_d.elements[MATRIX_INDEX(energiesRowIndex+centralFreqIndex-SEARCH_WIDTH, energiesColumnIndex, energies_d.rows)];
		}
	}
}


extern "C" void findHighestEnergyFrequencies_cuda(matrix highestEnergies_d, matrix centralFreq_d, const matrix energies_d, const matrix freq_d) {
	int N = energies_d.columns;
	findHighestEnergyFrequencies_kernel << <(N + 1023) / 1024, 1024 >> > (highestEnergies_d, centralFreq_d, highestEnergies_d, freq_d);
}