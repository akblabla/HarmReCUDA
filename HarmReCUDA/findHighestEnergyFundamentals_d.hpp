#pragma once
#include "LinearAlgebraStructs.h"
#include "Matrix_d.hpp"
#include "Vector_d.hpp"
void findHighestEnergyFundamentals_d(Vector_d& fundamentalFrequencies_d, const Matrix_d& amplitudes_d, const Vector_d& freq_d, int harmonicCount);