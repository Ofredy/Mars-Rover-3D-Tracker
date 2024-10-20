#include "rover_ekf.h"


int imu_hz = 25;

double measurement_noise_var = 0.01;

double beacons[NUM_BEACONS][3] = {
                                     {-10.0, 10.0, 0.0},  // Beacon 1
                                     {10.0, 10.0, 0.0},   // Beacon 2
                                     {0.0, -10.0, 0.0}    // Beacon 3
                                 };

void rover_ekf_init(matrix *x_n, matrix *A, matrix *B, matrix *Q, matrix *Q_n) 
{
    create_A_matrix(A);
    create_B_matrix(B);
    create_process_noise_matrix(Q);
    
    init_position(x_n);
    init_Q_n(Q_n);
}

void init_position(matrix *x_n) 
{
    double *ptr = x_n->data;
    double array[NUM_STATES] = { 1, 2, 3, 4, 5, 6, 7, 8, 9 };

    memcpy(ptr, array, NUM_STATES * sizeof(double));
}

void init_Q_n(matrix *Q_n) 
{
    double T = 1.0 / imu_hz;
    double tau_a = sqrt(X0_Variance), tau_b = sqrt(X0_Variance);

    double Q0[9] = {
        (T * T * T) / 3.0 * tau_a + (T * T * T * T * T) / 20.0 * tau_b,
        (T * T) / 2.0 * tau_a + (T * T * T * T) / 8.0 * tau_b,
        -(T * T * T) / 6.0 * tau_b,

        (T * T) / 2.0 * tau_a + (T * T * T * T) / 8.0 * tau_b,
        T * tau_a + (T * T * T) / 3.0 * tau_b,
        -(T * T) / 2.0 * tau_b,

        -(T * T * T) / 6.0 * tau_b,
        -(T * T) / 2.0 * tau_b,
        T * tau_b
    };

    double temp_Q[81] = {
        Q0[0], Q0[1], Q0[2], 0, 0, 0, 0, 0, 0,
        Q0[3], Q0[4], Q0[5], 0, 0, 0, 0, 0, 0,
        Q0[6], Q0[7], Q0[8], 0, 0, 0, 0, 0, 0,
        0, 0, 0, Q0[0], Q0[1], Q0[2], 0, 0, 0,
        0, 0, 0, Q0[3], Q0[4], Q0[5], 0, 0, 0,
        0, 0, 0, Q0[6], Q0[7], Q0[8], 0, 0, 0,
        0, 0, 0, 0, 0, 0, Q0[0], Q0[1], Q0[2],
        0, 0, 0, 0, 0, 0, Q0[3], Q0[4], Q0[5],
        0, 0, 0, 0, 0, 0, Q0[6], Q0[7], Q0[8]
    };

    memcpy(Q_n->data, temp_Q, NUM_STATES * NUM_STATES * sizeof(double));
}

void create_A_matrix(matrix *A) 
{
    double T = 1.0 / imu_hz;

    double temp_A[81] = {
        1, T, -(T * T) / 2, 0, 0, 0, 0, 0, 0,
        0, 1, -T, 0, 0, 0, 0, 0, 0,
        0, 0, 1, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 1, T, -(T * T) / 2, 0, 0, 0,
        0, 0, 0, 0, 1, -T, 0, 0, 0,
        0, 0, 0, 0, 0, 1, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 1, T, -(T * T) / 2,
        0, 0, 0, 0, 0, 0, 0, 1, -T,
        0, 0, 0, 0, 0, 0, 0, 0, 1
    };

    memcpy(A->data, temp_A, NUM_STATES * NUM_STATES * sizeof(double));
}

void create_B_matrix(matrix *B) 
{
    double T = 1.0 / imu_hz;

    double temp_B[81] = {
        (T * T) / 2, 0, 0, 0, 0, 0, 0, 0, 0,
        T, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, (T * T) / 2, 0, 0, 0, 0, 0,
        0, 0, 0, T, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, (T * T) / 2, 0, 0,
        0, 0, 0, 0, 0, 0, T, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0
    };

    memcpy(B->data, temp_B, NUM_STATES * NUM_STATES * sizeof(double));
}

void create_process_noise_matrix(matrix *Q) 
{
    double T = 1.0 / imu_hz;
    double tau_a = 0.0001, tau_b = 0.0001;

    double Q0[9] = {
        (T * T * T) / 3.0 * tau_a + (T * T * T * T * T) / 20.0 * tau_b,
        (T * T) / 2.0 * tau_a + (T * T * T * T) / 8.0 * tau_b,
        -(T * T * T) / 6.0 * tau_b,

        (T * T) / 2.0 * tau_a + (T * T * T * T) / 8.0 * tau_b,
        T * tau_a + (T * T * T) / 3.0 * tau_b,
        -(T * T) / 2.0 * tau_b,

        -(T * T * T) / 6.0 * tau_b,
        -(T * T) / 2.0 * tau_b,
        T * tau_b
    };

    double temp_Q[81] = {
        Q0[0], Q0[1], Q0[2], 0, 0, 0, 0, 0, 0,
        Q0[3], Q0[4], Q0[5], 0, 0, 0, 0, 0, 0,
        Q0[6], Q0[7], Q0[8], 0, 0, 0, 0, 0, 0,
        0, 0, 0, Q0[0], Q0[1], Q0[2], 0, 0, 0,
        0, 0, 0, Q0[3], Q0[4], Q0[5], 0, 0, 0,
        0, 0, 0, Q0[6], Q0[7], Q0[8], 0, 0, 0,
         0, 0, 0, 0, 0, 0, Q0[0], Q0[1], Q0[2],
        0, 0, 0, 0, 0, 0, Q0[3], Q0[4], Q0[5],
        0, 0, 0, 0, 0, 0, Q0[6], Q0[7], Q0[8]
    };

    memcpy(Q->data, temp_Q, NUM_STATES * NUM_STATES * sizeof(double));
}

void rover_ekf_predict(matrix *x_n, matrix *Q_n, const double *imu_buffer, matrix *A, matrix *B, matrix *Q) 
{
    rover_state_update(x_n, imu_buffer, A, B);
    update_Q_n_prediction(Q_n, A, Q);
}

void rover_state_update(matrix *x_n, const double *imu_buffer, matrix *A, matrix *B) 
{
    matrix *u = create_u_vector(imu_buffer);

    // Compute temp_1 = A * x_n
    matrix *temp_1 = multiplyMatrix(A, x_n);
    // Compute temp_2 = B * u
    matrix *temp_2 = multiplyMatrix(B, u);

    // Update x_n with the sum of temp_1 and temp_2
    for (int i = 0; i < NUM_STATES; i++) {
        x_n->data[i] = temp_1->data[i] + temp_2->data[i];
    }

    // Free temporary matrices
    freeMatrix(temp_1);
    freeMatrix(temp_2);
    freeMatrix(u);
}

matrix* create_u_vector(const double *imu_buffer) 
{
    matrix *u = makeMatrix(1, NUM_STATES);
    // Initialize u vector based on imu_buffer
    memset(u->data, 0, NUM_STATES * sizeof(double));
    u->data[0] = imu_buffer[0];
    u->data[3] = imu_buffer[1];
    u->data[6] = imu_buffer[2];
    return u;
}

void update_Q_n_prediction(matrix *Q_n, matrix *A, matrix *Q) 
{
    // Compute temp_1 = A * Q_n
    matrix *temp_1 = multiplyMatrix(A, Q_n);
    // Compute temp_2 = temp_1 * A^T
    matrix *temp_2 = multiplyMatrix(temp_1, transposeMatrix(A));

    // Update Q_n with the sum of temp_2 and Q
    for (int i = 0; i < NUM_STATES * NUM_STATES; i++) {
        Q_n->data[i] = temp_2->data[i] + Q->data[i];
    }

    // Free temporary matrices
    freeMatrix(temp_1);
    freeMatrix(temp_2);
}

void rover_ekf_update(matrix *x_n, matrix *Q_n, const double ranging_buffer, const int beacon_idx,
                      matrix *A, matrix *B, matrix *Q) 
{
    double range_estimate = get_range_estimate(x_n, beacon_idx);
    matrix *H = get_observation_jacobian(x_n, beacon_idx, range_estimate);

    matrix *k_n = calculate_kalman_gain(Q_n, H);

    // Update x_n with the new estimate
    matrix *scaled_k_n = scaleMatrix(k_n, (ranging_buffer - range_estimate));
    addMatrixInPlace(x_n, scaled_k_n); // Ensure addVector handles memory properly

    update_Q_n(Q_n, H, k_n);

    // Free temporary matrices
    freeMatrix(H);
    freeMatrix(k_n);
    freeMatrix(scaled_k_n);
}

double get_range_estimate(matrix *x_n, const int beacon_idx) 
{
    double range_estimate = 0;
    double *ptr = x_n->data;

    for (int i = 0; i < NUM_STATES; i += 3) {
        range_estimate += (*ptr - beacons[beacon_idx][i / 3]) * (*ptr - beacons[beacon_idx][i / 3]);
        ptr += 3;
    }

    return sqrt(range_estimate);
}

matrix* get_observation_jacobian(matrix *x_n, const int beacon_idx, const double range_estimate) 
{
    matrix *H = makeMatrix(NUM_STATES, 1);
    double *ptr = H->data;
    double *ptr_x = x_n->data;

    for (int i = 0; i < NUM_STATES; i++) {
        if (i % 3 == 0) { // Check for position indices
            *(ptr++) = (*ptr_x - beacons[beacon_idx][i / 3]) / range_estimate;
        } else {
            *(ptr++) = 0.0;
        }
        ptr_x++;
    }

    return H;
}

matrix* calculate_kalman_gain(matrix *Q_n, matrix *H) 
{
    // temp1 = Q_n * H^T
    matrix *temp1 = multiplyMatrix(Q_n, transposeMatrix(H));
    // temp2 = H * Q_n * H^T
    matrix *temp2 = multiplyMatrix(H, Q_n);
    matrix *temp3 = multiplyMatrix(temp2, transposeMatrix(H));

    double R = measurement_noise_var * sqrt(measurement_noise_var);
    double gain = 1 / (temp3->data[0] + R);

    freeMatrix(temp2);
    freeMatrix(temp3);

    return scaleMatrix(temp1, gain);
}

void update_Q_n(matrix *Q_n, matrix *H, matrix *k_n) 
{
    matrix *I = eyeMatrix(NUM_STATES);
    // temp1 = k_n * H
    matrix *temp_1 = multiplyMatrix(k_n, H);
    // temp2 = I - temp1
    matrix *temp_2 = addMatrix(I, scaleMatrix(temp_1, -1));
    // temp3 = temp2 * Q_n
    matrix *temp_3 = multiplyMatrix(temp_2, Q_n);
    // temp3 = temp3 * temp2^T
    temp_3 = multiplyMatrix(temp_3, transposeMatrix(temp_2));

    // temp4 = k_n * k_n^T
    matrix *temp_4 = multiplyMatrix(k_n, transposeMatrix(k_n));
    double R = measurement_noise_var * sqrt(measurement_noise_var);
    temp_4 = scaleMatrix(temp_4, R);

    // temp5 = temp3 + temp4
    matrix *temp_5 = addMatrix(temp_3, temp_4);

    memcpy(Q_n->data, temp_5->data, NUM_STATES * NUM_STATES * sizeof(double));

    // Free temporary matrices
    freeMatrix(I);
    freeMatrix(temp_1);
    freeMatrix(temp_2);
    freeMatrix(temp_3);
    freeMatrix(temp_4);
    freeMatrix(temp_5);
}