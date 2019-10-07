#pragma once
#include "Matrix.h"
#include "Vector.h"
void harmReCUDA(Matrix& data, double minimumFundamentalFrequency, double maximumFundamentalFrequency, int fundamentalFrequencyResolution, double sampleRate, Vector& Harmonics);