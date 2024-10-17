#include "rover_ekf.h"

matrix* rover_init()
{

    matrix *x_0 = makeMatrix(1, NUM_STATES);
    double *ptr = x_0->data;

    for( int state_idx=0; state_idx<NUM_STATES; state_idx++ )
    {
        *(ptr++) = generate_normal_random(0.0, sqrt(X0_Variance));
    }

    return x_0;
}

matrix* rover_state_update()
{

}

matrix* rover_jacobian()
{

}

matrix* rover_ekf_predict()
{

}

matrix* get_range_estimate()
{

}

matrix* observation_jacobian()
{

}

matrix* rover_ekf_update()
{

}
