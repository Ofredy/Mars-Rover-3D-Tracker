#include "stdio.h"
#include "rover_ekf.h"
#include "../C-Linear-Algebra/matrix.h"


int main()
{

    matrix *x_0 = rover_init();

    printMatrix(x_0);  

    return 0;
}