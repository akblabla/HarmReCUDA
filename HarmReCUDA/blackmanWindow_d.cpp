#include "blackmanWindow_d.h"
extern "C" void blackmanWindow_cuda(matrix a_d);

void blackmanWindow_d(Matrix_d& a_d)
{
	blackmanWindow_cuda(a_d.getCMatrix());
}
