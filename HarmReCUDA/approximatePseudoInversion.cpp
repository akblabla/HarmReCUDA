#include "approximatePseudoInversion.h"
#include "blackmanWindow_d.hpp"

void approximatePseudoInversion(Matrix_d& designMatrix_d)
{
	blackmanWindow_d(designMatrix_d);
	designMatrix_d.multiplyWithScalar(2.0 / designMatrix_d.getRows());
	designMatrix_d.transpose();
}
