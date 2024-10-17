#ifndef _ROVER_EKF_
#define _ROVER_EKF_

#include "../C-Linear-Algebra/matrix.h"
#include "../C-Linear-Algebra/matrixadv.h"
#include "helper_functions.h"


#define X0_Variance 2


matrix* rover_init();

matrix* rover_state_update();

matrix* rover_jacobian();

matrix* rover_ekf_predict();

matrix* get_range_estimate();

matrix* observation_jacobian();

matrix* rover_ekf_update();

#endif