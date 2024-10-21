idf_component_register(SRCS "main.c"
                       INCLUDE_DIRS "../components/C-Linear-Algebra"
                       "../components/rover_ekf"
                       REQUIRES C-Linear-Algebra rover_ekf
                       PRIV_REQUIRES nvs_flash spi_flash)
