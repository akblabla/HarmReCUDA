#include "mex.h"
//#include "HarmReCUDA.hpp"
#include <string.h>

void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {
	int measurementCount = mxGetN(prhs[0]);
	int stackCount = mxGetM(prhs[0]);
//	Matrix data(measurementCount, stackCount, AMatrix::M_ALLOCATE);

//	memcpy(data.getCMatrix().elements, mxGetPr(prhs[0]), measurementCount * stackCount * mxGetElementSize(prhs[0]) );
}