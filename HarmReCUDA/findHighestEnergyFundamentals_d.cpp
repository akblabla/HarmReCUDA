#include "findHighestEnergyFundamentals_d.hpp"
#include "parabularSearch.hpp"
#include <exception>
#include <stdio.h>
#include "cuda_runtime.h"
#include <iostream>

extern "C" void squareElements_d(matrix a_d);
extern "C" void partialMatrixSummation_cuda(matrix dest_d, const matrix src_d, unsigned int rowFactor, unsigned int columnFactor);
extern "C" void findMaximum_cuda(matrix a_d);
extern "C" void elementWiseMultiplication_cuda(matrix dest_d, const matrix src_d, unsigned int rowFactor, unsigned int columnFactor);
extern "C" void findHighestEnergyFrequencies_cuda(matrix highestEnergies_d, matrix centralFreq_d, const matrix energies_d, const matrix freq_d);

void findHighestEnergyFundamentals_d(Vector_d& fundamentalFrequencies_d, const Matrix_d& amplitudes_d, const Vector_d& freq_d, int harmonicCount)
{
	if (amplitudes_d.getRows() % (2*harmonicCount) != 0) {
		throw std::exception("The harmonic energy matrix must be a factor of the amount of harmonics times 2");
		return;
	}
	/*if (src.getRows() != destination.getRows() || src.getColumns() != destination.getColumns()) {
		throw std::exception("Matrix dimension mismatch.");
		return;
	}*/
	Matrix_d harmonicEnergies_d(amplitudes_d, Matrix::matrixInitialisation::M_ASSIGN);
	squareElements_d(harmonicEnergies_d.getCMatrix());
	#ifdef DEBUG
	printf("Harmonic Energies\n");
	harmonicEnergies_d.print(200, 300, 0, 1);
	printf("\n");
	#endif // DEBUG


	Matrix_d energies_d(harmonicEnergies_d.getRows()/ (2 * harmonicCount), harmonicEnergies_d.getColumns(), Matrix::matrixInitialisation::M_ALLOCATE);
	partialMatrixSummation_cuda(energies_d.getCMatrix(), harmonicEnergies_d.getCMatrix(), (2 * harmonicCount), 1);
	#ifdef DEBUG
	printf("Energies\n");
	energies_d.print(0,-1, 0, 10);
	printf("\n");
	#endif

	//findMaximum_cuda(energies_d.getCMatrix());
	Matrix_d highestEnergy_d(5, energies_d.getColumns(),AMatrix::M_ALLOCATE);
	Vector_d centralFreq_d(energies_d.getColumns(), AMatrix::M_ALLOCATE);

	findHighestEnergyFrequencies_cuda(highestEnergy_d.getCMatrix(), centralFreq_d.getCMatrix(), energies_d.getCMatrix(), freq_d.getCMatrix());
	#ifdef DEBUG
	printf("Mask\n");
	energies_d.print(0, -1, 0, 10);
	printf("\n");
	#endif
	parabularSearch(fundamentalFrequencies_d, highestEnergy_d, centralFreq_d, freq_d);
	harmonicEnergies_d.deallocate();
	energies_d.deallocate();
}
