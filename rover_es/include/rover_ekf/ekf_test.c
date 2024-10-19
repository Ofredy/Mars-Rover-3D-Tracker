#include "stdio.h"
#include "rover_ekf.h"
#include "../C-Linear-Algebra/matrix.h"


int main()
{
    matrix *x_n = makeMatrix(1, NUM_STATES);
    matrix *A = makeMatrix(NUM_STATES, NUM_STATES);
    matrix *B = makeMatrix(NUM_STATES, NUM_STATES);
    matrix *Q = makeMatrix(NUM_STATES, NUM_STATES);
    matrix *Q_n = makeMatrix(NUM_STATES, NUM_STATES);

    rover_ekf_init(x_n, A, B, Q, Q_n);

    double imu_buffer[3] = { 1, 1, 1 };

    rover_ekf_predict(x_n, Q_n, imu_buffer, A, B, Q);

    rover_ekf_update(x_n, Q_n, 3, 0, A, B, Q);

    printf("x_n");
    printMatrix(x_n);

    printf("Q_n");
    printMatrix(Q_n);

    return 0;
}
