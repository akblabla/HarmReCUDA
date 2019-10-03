#pragma once
#include "LinearAlgebraStructs.h"
#include "Matrix.h"
class Matrix_d : public Matrix
{
public:
	Matrix_d(int rows, int columns);
	void allocateMatrix() override;
	void deallocateMatrix() override;
	virtual void uploadMatrixToDevice(Matrix& src) const;
	virtual void downloadMatrixFromDevice(Matrix& dst) const;
private:
	bool _allocated = false;

};

