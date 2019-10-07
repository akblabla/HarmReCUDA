#include "GenerateProjectionMatrix_d.h"
extern "C" void generateProjectionMatrix_cuda(matrix dest_d, const double minFreq, const double maxFreq, const double startTime, const double deltaTime, const matrix harmonics_d);
void generateProjectionMatrix_d(Matrix_d& destination, const double minFreq, const double maxFreq, const double startTime, const double deltaTime, const Vector_d& harmonics)
{
	generateProjectionMatrix_cuda(destination.getCMatrix(), minFreq, maxFreq, startTime, deltaTime, harmonics.getCMatrix());
}