#pragma once
#include "Matrix.h"
class Vector : public Matrix
{
public:
	Vector(int length, matrixInitialisation::matrixInitialisation initialisation = matrixInitialisation::no_init);
	virtual ~Vector();
};