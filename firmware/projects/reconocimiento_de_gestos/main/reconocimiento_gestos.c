/*! @mainpage Reconocimiento de Gestos
 *
 * @section genDesc General Description
 *
 * Este proyecto implementa un sistema de reconocimiento de gestos utilizando un acelerómetro triaxial.
 * El sistema es capaz de adquirir datos de aceleración, procesarlos y clasificarlos en diferentes gestos
 * utilizando técnicas de Machine Learning. Además, cuenta con modos de operación para la adquisición de datos
 * con y sin umbral, grabación de gestos y modo de inferencia.
 *
 * @section changelog Changelog
 *
 * |   Date	    | Description                                           |
 * |:----------:|:------------------------------------------------------|
 * | 29/03/2022 | Creación del documento                                |
 * | 01/08/2023 | Correción de errores                                  |
 * | 01/07/2024 | Migración a ESP-IDF  	                                |
 * | 14/02/2025 | Cambios para realizar extracción de características   |
 *
 * @author Juan Ignacio Cerrudo (juan.cerrudo@uner.edu.ar)
 * @author Albano Peñalva (albano.penalva@uner.edu.ar)
 *
 */

/*==================[inclusions]=============================================*/

#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_dsp.h"

#include "uart_mcu.h"
#include "led.h"
#include "switch.h"
#include "ADXL335.h"
#include "neopixel_stripe.h"
#include "timer_mcu.h"

/* TODO: Incluir header con los coeficientes del filtro a utilizar */
// #include "butter.h"
/* TODO: Incluir header con el algoritmo de clasificación */
// #include "classifier.h"

/*==================[macros and definitions]=================================*/
/* TODO: Luego de obtenido los coeficientes del filtro descomentar la siguiente 
linea para agregar filtrado digital a las señales */
// #define FILTER                              /**< @brief Filtrado digital activado */

/* TODO: Luego de obtenido el modelo de inferencia descomentar la siguiente linea
 para activar la clasificación de señales */
// #define INFERRINGML                         /**< @brief Inferencia mediante ML activada */

/* TODO: Modificar tiempo y frecuencia de muestreo según diseño */
#define SAMPLE_FREC		500                 /**< @brief Frecuencia de muestreo en Hz*/
#define TIME_SAMPLE		3.0                 /**< @brief Duración del movimiento en segundos */

#ifdef INFERRINGML
/* TODO: Modificar cantidad de características a utilizar. En este ejemplo son 3 (mean_x, mean_y y mean_z) */
#define FEATURES_QTY    3                   /**< @brief Cantidad de características */
#endif

#define NUM_AXES		3                   /**< @brief Número de ejes */
#define REFRACT_TIME	1500                /**< @brief Periodo refractario */
#define NUM_SAMPLES 	(uint16_t)(TIME_SAMPLE*SAMPLE_FREC) /**< @brief Cantidad de muestras */
#define ACCEL_THRESHOLD	1.7                 /**< @brief Umbral inicio de movimiento */

/** @brief Enumeraciones de la máquina de estado principal */
typedef enum state {
    DATALOGGING = 0,        /**< @brief Modo de adquisición de datos sin umbral. */    
    INFERRING,              /**< @brief Modo de inferencia de gestos. */
    RECORD,                 /**< @brief Modo de grabación de gestos. */
    DATALOGGING_THR,        /**< @brief Modo de adquisición de datos con umbral. */
} state_t;
    
/** @brief Enumeraciones de la máquina de estado de grabación de gestos */
typedef enum {
    WAITING_REC = 0,        /**< @brief Espera de inicio de movimiento. */
    END_REC,                /**< @brief Fin de grabación de movimiento. */
    LOGGING_REC,            /**< @brief Grabación de movimiento. */
    REFRACT_REC             /**< @brief Periodo refractario. */
} state_rec_t;

/*==================[internal data definition]===============================*/

TaskHandle_t process_task_handle = NULL;    /**< Handle tarea "process" */

float x_G;      /**< @brief Aceleración triaxial, eje x, flotante. */
float y_G;      /**< @brief Aceleración triaxial, eje y, flotante. */
float z_G;      /**< @brief Aceleración triaxial, eje z, flotante. */

float x_G_filt; /**< @brief Aceleración triaxial filtrada, eje x.  */
float y_G_filt; /**< @brief Aceleración triaxial filtrada, eje y.  */
float z_G_filt; /**< @brief Aceleración triaxial filtrada, eje z.  */

serial_config_t uart_usb;   /**< @brief Estructura de configuración de la UART. */

state_t state;              /**< @brief Variable para máquina de estado principal. */
state_rec_t record_state;   /**< @brief Variable para máquina de estado de grabación de movimientos. */

uint16_t frame =0;          /**< @brief Variable auxiliar, lleva la cuenta de los "frames" triaxiales enviados. */

#ifdef INFERRINGML
float x_chunk[(int)(NUM_SAMPLES)];  /**< @brief Buffer de muestras de aceleración en X. */
float y_chunk[(int)(NUM_SAMPLES)];  /**< @brief Buffer de muestras de aceleración en Y. */
float z_chunk[(int)(NUM_SAMPLES)];  /**< @brief Buffer de muestras de aceleración en Z. */
float features[FEATURES_QTY];       /**< @brief Vector de características. */
/* TODO: Declarar variables auxiliares para la extracción de características */
float x_mean, y_mean, z_mean;       /**< @brief Medias de las señales. */
float x_max, y_max, z_max;          /**< @brief Maximos de las señales. */
float x_min, y_min, z_min;          /**< @brief Minimos de las señales. */
float x_rms, y_rms, z_rms;          /**< @brief RMS de las señales. */
#endif

#ifdef FILTER
float delay_x[2*STAGES] = {0}; /**< @brief Arreglo para los estados del filtro del eje x (dos por cada etapa). */
float delay_y[2*STAGES] = {0}; /**< @brief Arreglo para los estados del filtro del eje y (dos por cada etapa). */
float delay_z[2*STAGES] = {0}; /**< @brief Arreglo para los estados del filtro del eje z (dos por cada etapa). */
#endif
/*==================[internal functions declaration]=========================*/

/** @fn bool motionDetected()
 * @brief Función para detección de movimiento
 * @param[in] ax Valores de aceleración en X
 * @param[in] ay Valores de aceleración en Y
 * @param[in] az Valores de aceleración en Z
 * @return True o False
 */
bool motionDetected(float ax, float ay, float az) {
	float thr = sqrt((ax*ax) + (ay*ay) + (az*az));
    return  (thr > ACCEL_THRESHOLD);
}

/**
 * @brief Notifica a la tarea Process cada 1/SAMPLE_FREC seg
 */
void TimerProcessISR(void* param){
    xTaskNotifyGive(process_task_handle);
}

/** @fn void SwitchesTask()
 * @brief Tarea encargada de  la lectura de teclas para el cambio de modo de funcionamiento.
 */
void SwitchesTask(void *pvParameter){
    uint8_t teclas;

    while(true){
        teclas  = SwitchesRead();
    	switch(teclas){
    		case SWITCH_1:
                if(state == DATALOGGING_THR){
                    state = DATALOGGING;
                }
                else{
                    state++;
                }
            break;
            case SWITCH_2:
                if(state == DATALOGGING){
                    state = DATALOGGING_THR;
                }
                else{
                    state--;
                }
    		break;
    	}
        if(teclas){
            switch(state){
                case DATALOGGING:
                    NeoPixelAllColor(NEOPIXEL_COLOR_GREEN);
                    printf("\n Pasando a modo DATALOGGING sin THRESHOLD \n");
                break;
                case DATALOGGING_THR:
                    NeoPixelAllColor(NEOPIXEL_COLOR_BLUE);
                    printf("\n Pasando a modo DATALOGGING con THRESHOLD \n");
                break;
                case RECORD:
                    NeoPixelAllColor(NEOPIXEL_COLOR_RED);
                    printf("\n Pasando a modo GRABACION \n");
                break;
                case INFERRING:
                    NeoPixelAllColor(NEOPIXEL_COLOR_YELLOW);
                    printf("\n Pasando a modo INFERENCIA \n");
                break;
            }
            vTaskDelay(300 / portTICK_PERIOD_MS);
        }
		vTaskDelay(100 / portTICK_PERIOD_MS);
    }
}

/** @fn void ProcessTask()
 * @brief Tarea encargada de la adquisición y procesamiento de los datos.
 */
void ProcessTask(void *pvParameter){
    float mag;
    while(true){
        ulTaskNotifyTake(pdTRUE, portMAX_DELAY);
        x_G = ReadXValue();
        y_G = ReadYValue();
        z_G = ReadZValue();
        mag = sqrt((x_G*x_G) + (y_G*y_G) + (z_G*z_G));

        #ifdef FILTER
        float mag_filt;
        dsps_biquad_f32(&x_G, &x_G_filt, 1, ba_coeff, delay_x);
        dsps_biquad_f32(&y_G, &y_G_filt, 1, ba_coeff, delay_y);
        dsps_biquad_f32(&z_G, &z_G_filt, 1, ba_coeff, delay_z);
        if(STAGES > 1){
            for(uint8_t i=1; i<STAGES; i++){
                dsps_biquad_f32(&x_G_filt, &x_G_filt, 1, &ba_coeff[i*5], &delay_x[i*2]);
                dsps_biquad_f32(&y_G_filt, &y_G_filt, 1, &ba_coeff[i*5], &delay_y[i*2]);
                dsps_biquad_f32(&z_G_filt, &z_G_filt, 1, &ba_coeff[i*5], &delay_z[i*2]);
            }
        }
        mag_filt = sqrt((x_G_filt*x_G_filt) + (y_G_filt*y_G_filt) + (z_G_filt*z_G_filt));
        #endif

        switch(state){
        /* Modo Grabación */
        case RECORD:
            switch(record_state){
            case WAITING_REC:
                if(motionDetected(x_G, y_G, z_G)){
                    record_state = LOGGING_REC;
                }
                break;
            case LOGGING_REC:
                LedToggle(LED_2);
                #ifdef FILTER
                printf("%1.2f,", x_G_filt);
                printf("%1.2f,", y_G_filt);
                printf("%1.2f,", z_G_filt);
                #else
                printf("%1.2f,", x_G);
                printf("%1.2f,", y_G);
                printf("%1.2f,", z_G);
                #endif
                frame++;
                if(frame > (NUM_SAMPLES - 2)){
                    record_state = END_REC;
                }
                break;
            case END_REC:
                LedOff(LED_2);
                #ifdef FILTER
                printf("%1.2f,", x_G_filt);
                printf("%1.2f,", y_G_filt);
                printf("%1.2f\n", z_G_filt);
                #else
                printf("%1.2f,", x_G);
                printf("%1.2f,", y_G);
                printf("%1.2f\n", z_G);
                #endif
                frame = 0;
                record_state = REFRACT_REC;
                break;
            case REFRACT_REC:
                LedOn(LED_3);
                vTaskDelay(REFRACT_TIME / portTICK_PERIOD_MS);
                LedOff(LED_3);
                record_state = WAITING_REC;
                break;
            } 
            break; /* Fin del modo Grabación*/
        /* Modo Adquisición de datos sin Umbral*/
        case DATALOGGING:
            LedToggle(LED_2);
            #ifdef FILTER
            printf("%1.2f,", 	x_G_filt);
            printf("%1.2f,", 	y_G_filt);
            printf("%1.2f,",	z_G_filt);
            #endif
            printf("%1.2f,", 	x_G);
            printf("%1.2f,", 	y_G);
            printf("%1.2f,",	z_G);
            #ifdef FILTER
            printf("%1.2f,",	mag_filt);
            #endif
            printf("%1.2f\n",	mag);
            break; /* Fin del modo Adquisición de datos sin Umbral*/
        /* Modo Adquisición de datos con Umbral*/
        case DATALOGGING_THR:  
            if(motionDetected(x_G, y_G, z_G)){
                LedToggle(LED_2);
                #ifdef FILTER
                printf("%1.2f,", 	x_G_filt);
                printf("%1.2f,", 	y_G_filt);
                printf("%1.2f,",	z_G_filt);
                #endif
                printf("%1.2f,", 	x_G);
                printf("%1.2f,",    y_G);
                printf("%1.2f,",	z_G);
                printf("%1.2f\n",	mag);
            }
            else{
                LedOff(LED_2);
            }
            break; /* Fin del modo Adquisición de datos con Umbral*/
        /* Modo Inferencia */
        case INFERRING:
            #ifdef INFERRINGML
            switch(record_state){
                case WAITING_REC:
                    if(motionDetected(x_G,y_G,z_G)){
                        record_state = LOGGING_REC;
                    }
                    break;
                case LOGGING_REC:
                    x_chunk[frame] = x_G_filt;
                    y_chunk[frame] = y_G_filt;
                    z_chunk[frame] = z_G_filt;
                    frame++;
                    if(frame > (NUM_SAMPLES - 1)){
                        record_state = END_REC;
                    }
                    LedToggle(LED_2);
                    break;
                case END_REC:
                    /* TODO: Agregar el código necesario para el cálculo de las características.
                    A modo de ejemplo, se muestra el cálculo de las medias de las señales */
                    x_mean = 0;
                    y_mean = 0;
                    z_mean = 0;
                    for(uint16_t i=0; i<NUM_SAMPLES; i++){
                        x_mean += x_chunk[i];
                        y_mean += y_chunk[i];
                        z_mean += z_chunk[i];
                    }
                    features[0] = x_mean / NUM_SAMPLES;
                    features[1] = y_mean / NUM_SAMPLES;
                    features[2] = z_mean / NUM_SAMPLES;
                    /* Fin de cálculo de features */
                    printf("%s\n", predictLabel(features));
                    frame = 0;
                    record_state = REFRACT_REC;
                    LedOff(LED_2);
                    break;
                case REFRACT_REC:
                    LedOn(LED_3);
                    vTaskDelay(REFRACT_TIME / portTICK_PERIOD_MS);
                    LedOff(LED_3);
                    record_state = WAITING_REC;
                break;
            } /* End switch record_state*/
            #else
                printf("Falta implementar el modelo para inferencia \n");
            #endif
            break; /* Fin del modo Inferencia */
        } /* Fin de switch(state) */ 
    } /* Fin while*/
}

/*==================[external functions definition]==========================*/

void app_main(void){
    /* Inicialización de perifécricos */
	/* Configuración de UART */
	uart_usb.baud_rate = 230400;
	uart_usb.port = UART_PC;
	uart_usb.func_p = NULL;
	UartInit(&uart_usb);

	/* Configuración Timer de muestreo y procesamiento */
	timer_config_t timer_process = {
	    .timer = TIMER_A,
	    .period = (uint32_t)(1000*1000/SAMPLE_FREC),    /* Período de muestreo en useg */
	    .func_p = TimerProcessISR,
    };

	/* Configuración de teclas */
	SwitchesInit();

	/* Inicialización de acelerómetro */
	ADXL335Init();
    
    /* Secuencia de LEDs: inicialización */
	LedsInit();
	vTaskDelay(250 / portTICK_PERIOD_MS);
	LedOn(LED_3);
	LedOn(LED_2);
	LedOn(LED_1);
	vTaskDelay(250 / portTICK_PERIOD_MS);
	LedOff(LED_3);
	LedOff(LED_2);
	LedOff(LED_1);
	vTaskDelay(250 / portTICK_PERIOD_MS);
	LedOn(LED_3);
	LedOn(LED_2);
	LedOn(LED_1);
	vTaskDelay(250 / portTICK_PERIOD_MS);
	LedOff(LED_3);
	LedOff(LED_2);
	LedOff(LED_1);
    /* Estado inicial */
    state = DATALOGGING;
    record_state = WAITING_REC;
	/* LED estado incial */
    static neopixel_color_t color;
	NeoPixelInit(BUILT_IN_RGB_LED_PIN, BUILT_IN_RGB_LED_LENGTH, &color);
    NeoPixelAllColor(NEOPIXEL_COLOR_GREEN);
	printf("Inicio Firmware SAPS\n");

    /* Creación de tareas */
    xTaskCreate(ProcessTask, "Process", 4096, NULL, 5, &process_task_handle);
    xTaskCreate(SwitchesTask, "Switches", 4096, NULL, 5, NULL);

	/* Inicialización del Timer */
	TimerInit(&timer_process);
	TimerStart(TIMER_A);
}
/*==================[end of file]============================================*/
