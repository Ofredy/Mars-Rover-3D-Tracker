#ifndef _XBEE_
#define _XBEE_

#include <stdio.h>
#include "freertos/FreeRTOS.h"
#include "driver/uart.h"
#include "driver/gpio.h"
#include "sdkconfig.h"
#include "string.h"

#define XBEE_TXD    4
#define XBEE_RXD    5
#define XBEE_RTS    18
#define XBEE_CTS    19

#define XBEE_PORT_NUM      (UART_NUM_0)
#define XBEE_BAUD_RATE     (CONFIG_CONSOLE_UART_BAUDRATE)
#define XBEE_TASK_SIZE     2048

#define BUF_SIZE (1024)


void xbee_init();

char* xbee_read();

void xbee_send(const char* message);

#endif