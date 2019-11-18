#include "GenerateProjectionMatrix_d.hpp"
#include <exception>
#define _USE_MATH_DEFINES
#include <math.h>
extern "C" void generateProjectionMatrix_cuda(matrix dest_d, const matrix freq_d, const matrix time_d, const matrix harmonics_d);
void generateProjectionMatrix_d(Matrix_d& destination, const Vector_d& fundamentalFrequencies, const Vector_d& sampleTime, const Vector_d& harmonics)
{
	if (destination.getRows() % (harmonics.getRows() * 2) != 0) {
		throw std::exception("Matrix rows must be a factor of 2 times the amount of harmonics");
		return;
	}
	generateProjectionMatrix_cuda(destination.getCMatrix(), fundamentalFrequencies.getCMatrix(),sampleTime.getCMatrix(), harmonics.getCMatrix());
}