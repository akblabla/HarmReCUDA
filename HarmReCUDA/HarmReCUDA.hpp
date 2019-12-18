#pragma once
#include "Matrix.hpp"
#include "Vector.hpp"
extern void harmReCUDA(Matrix& data, double minimumFundamentalFrequency, double maximumFundamentalFrequency, int fundamentalFrequencyResolution, double sampleRate, Vector& Harmonics);