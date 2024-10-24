#include "xbee.h"


void xbee_init()
{
    /* Configure parameters of an UART driver,
     * communication pins and install the driver */
    uart_config_t uart_config = {
        .baud_rate = XBEE_BAUD_RATE,
        .data_bits = UART_DATA_8_BITS,
        .parity    = UART_PARITY_DISABLE,
        .stop_bits = UART_STOP_BITS_1,
        .flow_ctrl = UART_HW_FLOWCTRL_DISABLE,
        .source_clk = UART_SCLK_DEFAULT,
    };
    int intr_alloc_flags = 0;

    #if CONFIG_UART_ISR_IN_IRAM
        intr_alloc_flags = ESP_INTR_FLAG_IRAM;
    #endif

    ESP_ERROR_CHECK(uart_driver_install(XBEE_PORT_NUM, BUF_SIZE * 2, 0, 0, NULL, intr_alloc_flags));
    ESP_ERROR_CHECK(uart_param_config(XBEE_PORT_NUM, &uart_config));
    ESP_ERROR_CHECK(uart_set_pin(XBEE_PORT_NUM, XBEE_TXD, XBEE_RXD, XBEE_RTS, XBEE_CTS));
}

char* xbee_read()
{
    // Configure a temporary buffer for the incoming data
    uint8_t *data = (uint8_t *) malloc(BUF_SIZE);

    // Read data from the UART
    return (char*) uart_read_bytes(XBEE_PORT_NUM, data, (BUF_SIZE - 1), 20 / portTICK_PERIOD_MS);
}

void xbee_send(const char* message)
{
    uart_write_bytes(XBEE_PORT_NUM, message, sizeof(message)/sizeof(char));
}
