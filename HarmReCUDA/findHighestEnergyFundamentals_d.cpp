#include "findHighestEnergyFundamentals_d.h"
#include <exception>
#include <stdio.h>
#include "cuda_runtime.h"
#include <iostream>

extern "C" void squareElements_d(matrix a_d);
extern "C" void partialMatrixSummation_cuda(matrix dest_d, const matrix src_d, unsigned int rowFactor, unsigned int columnFactor);
extern "C" void findMaximum_d(matrix a_d);
extern "C" void elementWiseMultiplication_d(matrix dest_d, const matrix src_d, unsigned int rowFactor, unsigned int columnFactor);

void findHighestEnergyFundamentals_d(Matrix_d& destination, const Matrix_d& src, int harmonicCount)
{
	if (src.getRows() % (2*harmonicCount) != 0) {
		throw std::exception("The harmonic energy matrix must be a factor of the amount of harmonics times 2");
		return;
	}
	if (src.getRows() != destination.getRows() || src.getColumns() != destination.getColumns()) {
		throw std::exception("Matrix dimension mismatch.");
		return;
	}
	Matrix_d harmonicEnergies_d(src, Matrix::matrixInitialisation::M_ASSIGN);
	squareElements_d(harmonicEnergies_d.getCMatrix());
	printf("Harmonic Energies\n");
	harmonicEnergies_d.print(200, 300, 0, 1);
	printf("\n");
	Matrix_d energies_d(harmonicEnergies_d.getRows()/ (2 * harmonicCount), harmonicEnergies_d.getColumns(), Matrix::matrixInitialisation::M_ALLOCATE);
	partialMatrixSummation_cuda(energies_d.getCMatrix(), harmonicEnergies_d.getCMatrix(), (2 * harmonicCount), 1);
	printf("Energies\n");
	energies_d.print(0,-1, 0, 1);
	printf("\n");

	findMaximum_d(energies_d.getCMatrix());
	printf("Mask\n");
	energies_d.print(0, -1, 0, 1);
	printf("\n");
	destination.copyFromDevice(src);
	elementWiseMultiplication_d(destination.getCMatrix(), energies_d.getCMatrix(), (2 * harmonicCount), 1);
	harmonicEnergies_d.deallocate();
	energies_d.deallocate();
}
