#pragma once
#ifndef  CUDALA 
#define CUDALA
#endif // ! CUDALA 

#include "LAlib/la.h"
NRMat<double> harmReCUDA(NRMat<double> data, NRVec<double> harmonics, NRVec<double> frequencies);