#ifndef _ROVER_EKF_
#define _ROVER_EKF_

#include <stdio.h>
#include <stdlib.h>
#include <string.h>  

#include "../C-Linear-Algebra/matrix.h"
#include "../C-Linear-Algebra/matrixadv.h"
#include "helper_functions.h"

#define NUM_BEACONS 3
#define X0_Variance 2

extern int imu_hz;

extern double measurement_noise_var;

extern double beacons[NUM_BEACONS][3];

void rover_ekf_init(matrix *x_n, matrix* A, matrix* B, matrix* Q, matrix* Q_n);

void init_position(matrix *x_n);

void init_Q_n(matrix *Q_n);

void create_process_noise_matrix(matrix* Q);

void create_A_matrix(matrix* A);

void create_B_matrix(matrix* B);

void rover_ekf_predict(matrix *x_n, matrix *Q_n, const double *imu_buffer, matrix *A, matrix *B, matrix *Q);

void rover_state_update(matrix *x_n, const double *imu_buffer, matrix *A, matrix *B);

matrix* create_u_vector(const double *imu_buffer);

void update_Q_n_prediction(matrix *Q_n, matrix *A, matrix *Q);

void rover_ekf_update(matrix *x_n, matrix *Q_n, const double ranging_buffer, const int beacon_idx,
                      matrix *A, matrix *B, matrix *Q);

double get_range_estimate(matrix *x_n, const int beacon_idx);

matrix* get_observation_jacobian(matrix *x_n, const int beacon_idx, const double range_estimate);

matrix* calculate_kalman_gain(matrix *Q_n, matrix *H);

void update_Q_n(matrix *Q_n, matrix *H, matrix *k_n);

#endif