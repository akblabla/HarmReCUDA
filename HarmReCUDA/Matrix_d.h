#pragma once
#include "LinearAlgebraStructs.h"
#include "Matrix_cpu.h"
#include "AMatrix.h"

class Matrix_d : AMatrix
{
public:
	Matrix_d(int rows, int columns);
	virtual void allocateMatrixOnDevice();
	virtual void deallocateMatrixOnDevice();
	virtual void uploadMatrixToDevice(Matrix_cpu src) const;
	virtual void downloadMatrixFromDevice(Matrix_cpu dst) const;
private:
	bool _allocatedToDevice = false;

};

