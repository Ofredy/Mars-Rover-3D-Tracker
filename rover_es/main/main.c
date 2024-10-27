/*
 * SPDX-FileCopyrightText: 2010-2022 Espressif Systems (Shanghai) CO LTD
 *
 * SPDX-License-Identifier: CC0-1.0
 */

#include <stdio.h>
#include <inttypes.h>
#include "sdkconfig.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_chip_info.h"
#include "esp_flash.h"

#include "../components/C-Linear-Algebra/matrix.h"
#include "../components/rover_ekf/rover_ekf.h"
#include "../components/hardware/xbee.h"

#define NUM_STATES 9

static void xbee_test_task(void *arg)
{
    xbee_init();

    while (1) 
    {
        xbee_send("test\n");
        vTaskDelay(pdMS_TO_TICKS(1000));
    }
}

void app_main(void)
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

    xTaskCreate(xbee_test_task, "xbee_task", XBEE_TASK_SIZE, NULL, 10, NULL);
}
