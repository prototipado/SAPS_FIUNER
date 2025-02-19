# Reconocimiento de Gestos

Este proyecto implementa un sistema de reconocimiento de gestos utilizando un acelerómetro triaxial. El sistema es capaz de adquirir datos de aceleración, procesarlos y clasificarlos en diferentes gestos utilizando técnicas de Machine Learning. Además, cuenta con modos de operación para la adquisición de datos con y sin umbral, grabación de gestos y modo de inferencia.

## Cómo usar el ejemplo

### Hardware requerido

* ESP-EDU
* Acelerómetro ADXL335

#### Conexiones

 |      ADXL335     |       ESP-EDU     |
 |:----------------:|:------------------|
 |      VCC         |       3V3         |
 |      X           |       GPIO_1/CH1  |
 |      Y           |       GPIO_2/CH2  |
 |      Z           |       GPIO_3/CH3  |
 |      GND         |       GND         |

### Modos de Funcionamiento

El sistema cuenta con cuatro modos de funcionamiento: Adquisición sin Umbral, Adquisición con Umbral, Grabación e Inferencia. El modo de funcionamiento activado es representado mediante el LED RGB de la placa ESP32-C6-DevKitC-1, y puede ser modificado utilizando las teclas 1 y 2 de la placa ESP-EDU, las cuales cambian de modo de manera secuencial.

* Modo de Adquisición sin Umbral (LED RGB Verde 🟢): Adquiere datos del acelerómetro de manera continua, a la fecuencia de muestreo configurada, y luego envía por puerto serie los datos de acleración en los ejes `x`, `y` y `z` respectivamente (separados por comas) seguidos del módulo de la aceleración total y el caracter de fin de línea (formato compatible con graficador serie). En caso de estar activados los filtros digitales se envía 8 valores: `x_filt`, `y_filt`, `z_filt`, `x`, `y`, `z`, `mag_filt`, `mag`. El LED 2 de la ESP-EDU parpadea con cada nueva adquisición.

    ```PowerShell
    Pasando a modo DATALOGGING sin THRESHOLD 
    -0.43,-0.06,1.01,1.10
    -0.48,-0.01,1.01,1.12
    -0.49,-0.11,1.01,1.12
    -0.49,-0.06,0.96,1.08
    -0.48,-0.00,0.96,1.07
    ...
    ```

* Modo de Adquisición con Umbral (LED RGB Azul 🔵): Similar al modo de Adquisición sin Umbral, pero en este caso sólo se envía datos si la magnitud de la aceleración es mayor que un umbral configurado.

* Modo de Grabación (LED RGB Rojo 🔴): Adquiere datos del acelerómetro de manera continua, a la fecuencia de muestreo configurada, y si detecta el inicio de un movimiento (magnitud de la aceleración es mayor que un umbral configurado) envía por puerto serie los datos registrados durante la duración del movimiento (configurable en el firmware). Los datos son enviados en un formato compatible para ser guardados en un archivo `.csv`, en el cual por cada fila se almacenará un movimiento, alternando los valores de los ejes `x`, `y` y `z` (`x0, y0, z0, x1, y1, z1, ..., xN, yN, zN`). El LED 2 de la ESP-EDU parpadea con cada nueva adquisición mientras se esten enviando datos por el puerto serie, y una vez finalizado se enciende el LED 3 durante un período refractario durante el cual se ingnoran nuevos movimientos.

    ```PowerShell
    Pasando a modo GRABACION 
    0.32,0.10,1.60,0.37,0.10,1.70,0.42,-0.00,1.70,0.31,-0.01,1.65,0.21,-0.00,1.49,0.16,-0.00,1.38,0.05,-0.06,1.38,-0.17,-0.11,1.38,-0.38,-0.22,1.33,-0.59,-0.22,1.33,-0.65,-0.27,1.38,-0.70,-0.38,1.49,-0.70,-0.59,1.70,-0.59,-0.70,1.92,-0.54,-0.70,2.07,-0.43,-0.70,2.29,-0.43,-0.59,2.55,-0.43,-0.48,2.71,-0.48,-0.65,2.71,-0.54,-0.49,2.61,-0.64,-0.33,
    ...
    ```

* Modo de Inferencia (LED RGB Amarillo 🟡): Similar al modo de Grabación, pero en este caso se envía el resultado de la inferencia, imprimiendose el nombre del gesto detectado. El LED 2 de la ESP-EDU parpadea con cada nueva adquisición mientras se esten enviando datos por el puerto serie, y una vez finalizado se enciende el LED 3 durante un período refractario durante el cual se ingnoran nuevos movimientos.

    ```PowerShell
    Pasando a modo INFERENCIA 
    espera
    globo
    reves
    espera
    ...
    ```

### Configurar el proyecto

Todas las secciones del código que se deben ir modificando a lo largo del proyecto están indicadas con una sección comentada que comienza con el mensaje `TODO`, por ejemplo:

```c
/* TODO: Modificar tiempo y frecuencia de muestreo según diseño */
#define SAMPLE_FREC 500  /**< @brief Frecuencia de muestreo en Hz*/
#define TIME_SAMPLE 3.0  /**< @brief Duración del movimiento en segundos */
```

### Ejecutar la aplicación

Para poder utilizar la aplicación primero debe conectar la placa `ESP-EDU` a la PC utlizando el puerto USB inicado con `UART` y luego abrir un monitor serie, configurando el baudrate en `230400`.
