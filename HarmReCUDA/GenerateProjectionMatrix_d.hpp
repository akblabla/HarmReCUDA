#pragma once
#include "LinearAlgebraStructs.h"
#include "Matrix_d.hpp"
#include "Vector_d.hpp"
void generateProjectionMatrix_d(Matrix_d& destination, const Vector_d& fundamentalFrequencies, const Vector_d& sampleTime, const Vector_d& harmonics);