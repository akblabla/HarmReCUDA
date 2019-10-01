#pragma once
#ifndef  CUDALA 
#define CUDALA
#endif // ! CUDALA 

#include "LAlib/la.h"
namespace GenerateProjectionMatrix {
	NRMat<double>  GenerateProjectionMatrix(NRVec<double>& harmonics, NRVec<double>& frequencies,int measurementCount);
}