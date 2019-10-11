#pragma once
#include "Matrix.h"
class Vector : public Matrix
{
public:
	Vector(int length, matrixInitialisation initialisation = matrixInitialisation::M_NO_INIT);
	virtual ~Vector();
};