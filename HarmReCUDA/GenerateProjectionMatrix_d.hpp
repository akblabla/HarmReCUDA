#pragma once
#include "LinearAlgebraStructs.h"
#include "Matrix_d.hpp"
#include "Vector_d.hpp"
void generateProjectionMatrix_d(Matrix_d& destination, const double minFreq, const double maxFreq, const double startTime, const double sampleRate, const Vector_d& harmonics);