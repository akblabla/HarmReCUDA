#include "GenerateProjectionMatrix_d.h"
#include <exception>
#define _USE_MATH_DEFINES
#include <math.h>
extern "C" void generateProjectionMatrix_cuda(matrix dest_d, const double minFreq, const double maxFreq, const double startTime, const double deltaTime, const matrix harmonics_d);
void generateProjectionMatrix_d(Matrix_d& destination, const double minFreq, const double maxFreq, const double startTime, const double sampleRate, const Vector_d& harmonics)
{
	if (destination.getRows() % (harmonics.getRows() * 2) != 0) {
		throw std::exception("Matrix rows must be a factor of 2 times the amount of harmonics");
		return;
	}
	generateProjectionMatrix_cuda(destination.getCMatrix(), minFreq* (2*M_PI), maxFreq * (2 * M_PI), startTime, 1/sampleRate, harmonics.getCMatrix());
}