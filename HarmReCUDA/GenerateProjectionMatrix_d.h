#pragma once
#include "LinearAlgebraStructs.h"
extern "C" void generateProjectionMatrix_cuda(matrix a_d, const double minFreq, const double maxFreq, const double startTime, const double deltaTime, const int harmonics);