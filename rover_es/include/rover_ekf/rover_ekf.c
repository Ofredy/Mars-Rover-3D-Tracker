#include "rover_ekf.h"


int imu_hz = 25;

double measurement_noise_var = 0.01;

double beacons[NUM_BEACONS][3] = {
                                     {-10.0, 10.0, 0.0},  // Beacon 1
                                     {10.0, 10.0, 0.0},   // Beacon 2
                                     {0.0, -10.0, 0.0}    // Beacon 3
                                 };

void rover_ekf_init(matrix *x_n, matrix* A, matrix* B, matrix* Q, matrix* Q_n)
{

    create_A_matrix(A);
    create_B_matrix(B);
    create_process_noise_matrix(Q);

    init_position(x_n);
    init_Q_n(Q, Q_n);
}

void init_position(matrix *x_n)
{
    double *ptr = x_n->data;

    for( int state_idx=0; state_idx<NUM_STATES; state_idx++ )
    {
        *(ptr++) = generate_normal_random(0.0, sqrt(X0_Variance));
    }
}

void init_Q_n(matrix* Q, matrix *Q_n)
{
    Q_n = scaleMatrix(Q, sqrt(X0_Variance));
}

void create_A_matrix(matrix* A)
{
    double T = 1.0 / imu_hz;

    double temp_A[81] = {
                            // First 3x9 block
                            1, T, -(T * T) / 2,  0, 0, 0,  0, 0, 0,
                            0, 1, -T,            0, 0, 0,  0, 0, 0,
                            0, 0, 1,             0, 0, 0,  0, 0, 0,

                            // Second 3x9 block
                            0, 0, 0,  1, T, -(T * T) / 2,  0, 0, 0,
                            0, 0, 0,  0, 1, -T,            0, 0, 0,
                            0, 0, 0,  0, 0, 1,             0, 0, 0,

                            // Third 3x9 block
                            0, 0, 0,  0, 0, 0,  1, T, -(T * T) / 2,
                            0, 0, 0,  0, 0, 0,  0, 1, -T,
                            0, 0, 0,  0, 0, 0,  0, 0, 1
                        };

    double *ptr = A->data;

    for( int i=0; i<NUM_STATES*NUM_STATES; i++ )
    {
        *(ptr++) = temp_A[i];
    }
}

void create_B_matrix(matrix* B)
{
    double T = 1.0 / imu_hz;

    double temp_B[81] = {
                            // First 3x9 block
                            (T * T) / 2, 0, 0,  0, 0, 0,  0, 0, 0,
                            T, 0, 0,              0, 0, 0,  0, 0, 0,
                            0, 0, 0,              0, 0, 0,  0, 0, 0,

                            // Second 3x9 block
                            0, 0, 0,  (T * T) / 2, 0, 0,  0, 0, 0,
                            0, 0, 0,  T, 0, 0,              0, 0, 0,
                            0, 0, 0,  0, 0, 0,              0, 0, 0,

                            // Third 3x9 block
                            0, 0, 0,  0, 0, 0,  (T * T) / 2, 0, 0,
                            0, 0, 0,  0, 0, 0,  T, 0, 0,
                            0, 0, 0,  0, 0, 0,  0, 0, 0
                        };

    double *ptr = B->data;

    for( int i=0; i<NUM_STATES*NUM_STATES; i++ )
    {
        *(ptr++) = temp_B[i];
    }
}

void create_process_noise_matrix(matrix* Q)
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
    T * tau_b };

    double temp_Q[81] = {
                        // First block for the x-axis (Q0 block)
                        Q0[0], Q0[1], Q0[2], 0, 0, 0, 0, 0, 0,
                        Q0[3], Q0[4], Q0[5], 0, 0, 0, 0, 0, 0,
                        Q0[6], Q0[7], Q0[8], 0, 0, 0, 0, 0, 0,

                        // Second block for the y-axis (zero matrix followed by Q0 block)
                        0, 0, 0, Q0[0], Q0[1], Q0[2], 0, 0, 0,
                        0, 0, 0, Q0[3], Q0[4], Q0[5], 0, 0, 0,
                        0, 0, 0, Q0[6], Q0[7], Q0[8], 0, 0, 0,

                        // Third block for the z-axis (two zero matrices followed by Q0 block)
                        0, 0, 0, 0, 0, 0, Q0[0], Q0[1], Q0[2],
                        0, 0, 0, 0, 0, 0, Q0[3], Q0[4], Q0[5],
                        0, 0, 0, 0, 0, 0, Q0[6], Q0[7], Q0[8]
                    };

    double *ptr = Q->data;

    for( int i=0; i<NUM_STATES*NUM_STATES; i++ )
    {
        *(ptr++) = temp_Q[i];
    }
}

void rover_ekf_predict(matrix *x_n, matrix *Q_n, const double *imu_buffer, matrix *A, matrix *B, matrix *Q)
{
    rover_state_update(x_n, imu_buffer, A, B);
    update_Q_n_prediction(Q_n, A, Q);
}

void rover_state_update(matrix *x_n, const double *imu_buffer, matrix *A, matrix *B)
{
    matrix *u = create_u_vector(imu_buffer);

    matrix *temp_1 = makeMatrix(1, NUM_STATES);
    matrix *temp_2 = makeMatrix(1, NUM_STATES);

    temp_1 = multiplyMatrix(A, x_n);
    temp_2 = multiplyMatrix(B, u);

    x_n = addVector(temp_1, temp_2);
}

matrix* create_u_vector(const double *imu_buffer)
{
    double temp_u[NUM_STATES] = { 0 };
    temp_u[0] = imu_buffer[0];
    temp_u[3] = imu_buffer[1];
    temp_u[6] = imu_buffer[2];

    matrix *u = makeMatrix(1, NUM_STATES);

    double *ptr = u->data;

    for( int i=0; i<NUM_STATES*NUM_STATES; i++ )
    {
        *(ptr++) = temp_u[i];
    }

    return u;
}

void update_Q_n_prediction(matrix *Q_n, matrix *A, matrix *Q)
{

    matrix *temp_1 = makeMatrix(NUM_STATES, NUM_STATES);
    matrix *temp_2 = makeMatrix(NUM_STATES, NUM_STATES);

    temp_1 = multiplyMatrix(A, Q_n);
    temp_2 = multiplyMatrix(temp_1, transposeMatrix(A));

    Q_n = addMatrix(temp_2, Q);
}

void rover_ekf_update(matrix *x_n, matrix *Q_n, const double ranging_buffer, const double range_measurement, const int beacon_idx,
                         matrix *A, matrix *B, matrix *Q)
{
    double range_estimate = get_range_estimate(x_n, beacon_idx);
    matrix *H = get_observation_jacobian(x_n, beacon_idx, range_estimate);
    
    matrix *k_n = calculate_kalman_gain(Q_n, H);

    x_n = addVector(x_n, scaleMatrix(k_n, (ranging_buffer - range_estimate)));
}

double get_range_estimate(matrix *x_n, const int beacon_idx)
{
    double range_estimate = 0;
    double *ptr = x_n->data;
    int position_idx = 0;

    for( int i=0; i<NUM_STATES; i+=3 )
    {
        range_estimate += (*ptr - beacons[beacon_idx][position_idx]) * (*ptr - beacons[beacon_idx][position_idx]);
        ptr += 3;
        position_idx += 1;
    }

    return sqrt(range_estimate);
}

matrix* get_observation_jacobian(matrix *x_n, const int beacon_idx, const double range_estimate)
{
    matrix *H = makeMatrix(NUM_STATES, 1);

    double *ptr = H->data;
    double *ptr_x = x_n->data;

    int position_idx = 0;

    for( int i=0; i<NUM_STATES; i++ )
    {
        if( i == 0 || i == 3 || i == 6 )
        {
            *(ptr++) = (*ptr_x - beacons[beacon_idx][position_idx]) / range_estimate;
            position_idx += 1;
        }
        else
        {
            *(ptr++) = 0.0;
        }
        ptr_x++;
    }

    return H;
}

matrix* calculate_kalman_gain(matrix *Q_n, matrix *H)
{
    matrix *temp1 = makeMatrix(1, NUM_STATES);
    matrix *temp2 = makeMatrix(NUM_STATES, 1);
    matrix *temp3 = makeMatrix(1, 1);

    temp1 = multiplyMatrix(Q_n, transposeMatrix(H));

    temp2 = multiplyMatrix(H, Q_n);
    temp3 = multiplyMatrix(temp2, transposeMatrix(H));

    double *ptr = temp3->data;

    return scaleMatrix(temp1, ( 1 /( *ptr + measurement_noise_var )));
}

void update_Q_n(matrix *Q_n, matrix *H, matrix *k_n)
{
    
}
