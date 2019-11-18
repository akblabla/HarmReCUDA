#pragma once
#include "Matrix.hpp"
extern class Vector : public Matrix
{
public:
	Vector(int length, matrixInitialisation initialisation = matrixInitialisation::M_NO_INIT);
	virtual ~Vector();
};