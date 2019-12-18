#pragma once
#include "Matrix.hpp"
Matrix matLoad(const char* file, const char* symbol);
void matSave(const char* file, const char* symbol, Matrix & A);
int diagnose(const char* file);