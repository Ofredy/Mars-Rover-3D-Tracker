idf_component_register(SRCS "main.c"
                       INCLUDE_DIRS "../components/C-Linear-Algebra"
                       "../components/rover_ekf"
                       "../components/hardware"
                       REQUIRES C-Linear-Algebra rover_ekf hardware
                       PRIV_REQUIRES nvs_flash spi_flash)
