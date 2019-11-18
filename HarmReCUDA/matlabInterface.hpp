#pragma once
#include "Matrix.hpp"
Matrix matLoad(const char* file);
void matSave(const char* file, Matrix & A);
int diagnose(const char* file);