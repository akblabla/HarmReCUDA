#pragma once
#include "LinearAlgebraStructs.h"
#include "AMatrix.h"
class Matrix_cpu : public AMatrix
{
public:
	Matrix_cpu(int rows, int columns);
	virtual void matrixMultiply(Matrix_cpu* ) const;
	virtual ~Matrix_cpu();

};

