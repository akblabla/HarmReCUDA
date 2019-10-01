#include "LinearAlgebraStructs.h"

/**
*creates a projection matrix on the gpu to the given matrix on the device.
**/
extern "C" void generateProjectionMatrix_d(matrix a_d, const double minFreq, const double maxFreq, const double startTime, const double deltaTime, const int harmonics);