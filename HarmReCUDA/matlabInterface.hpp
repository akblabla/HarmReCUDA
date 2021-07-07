#pragma once
#include "Matrix.hpp"
#include <string>
Matrix matLoad(const std::string file, const char* symbol);
void matSave(const std::string file, const char* symbol, Matrix & A);
int diagnose(const std::string file);