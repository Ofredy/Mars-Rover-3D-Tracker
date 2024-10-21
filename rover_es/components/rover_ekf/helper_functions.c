#include "helper_functions.h"

double generate_normal_random(double mean, double stddev) 
{
    static double z1 = 0.0;
    static int generate = 0;

    if (!generate) {
        double u1, u2;
        do {
            u1 = (double)rand() / RAND_MAX;
            u2 = (double)rand() / RAND_MAX;
        } while (u1 <= 0.0 || u2 <= 0.0);

        z1 = sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
        generate = 1;
        return z1 * stddev + mean;
    } else {
        generate = 0;
        return z1 * stddev + mean;
    }
}