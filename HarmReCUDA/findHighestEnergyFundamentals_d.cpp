#include "findHighestEnergyFundamentals_d.h"

extern "C" void squareElements_d(matrix a_d);

void findHighestEnergyFundamentals_d(Matrix_d& src, Matrix_d& destination, int harmonicCount)
{
	Matrix_d harmonicEnergies_d(src,matrixInitialisation::assign);
	squareElements_d(harmonicEnergies_d.getCMatrix());
	harmonicEnergies_d.deallocate();
}
