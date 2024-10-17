#ifndef _HELPER_FUNCTIONS_
#define _HELPER_FUNCTIONS_

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define NUM_STATES 9

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

double generate_normal_random(double mean, double stddev);

#endif