#pragma once
#include "LinearAlgebraStructs.h"
#include "Matrix_d.hpp"
#include "Vector_d.hpp"
void parabularSearch(Matrix_d& freqOut_d, const Matrix_d& energies_d, const Vector_d& freq_d, double deltaFundamentalFrequency);