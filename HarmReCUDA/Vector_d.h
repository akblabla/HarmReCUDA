#pragma once
#include "Matrix_d.h"
class Vector_d :
	public Matrix_d
{
public:
	Vector_d(int length);
	virtual ~Vector_d();
};

