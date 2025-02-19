/** @brief driver que maneja la lectura de datos obtenida a partir de un acelerómetro.
 **
 **/

/*
 * Initials     Name
 * ---------------------------
 *
 */

/*
 * modification history (new versions first)
 * -----------------------------------------------------------
 * 20210609 v0.0.1 initials initial version
 */

/*==================[inclusions]=============================================*/
#include "ADXL335.h"
#include <stddef.h>
/*==================[macros and definitions]=================================*/
/** @def MAX_VOLTAGE
 * @brief Máximo valor de voltaje en mV que puede medir el driver
 */
#define MAX_VOLTAGE 3300
/** @def MAX_VALUE
 * @brief Máximo valor de bits
 */
#define MAX_VALUE 1024
/** @def OFFSET
 * @brief Mitad del valor máximo de voltaje correspondiente en bits
 */
#define OFFSET (3300.0/2.0)
/** @def SENSITIVITY
 * @brief Valor de la sensibilidad dado por el driver
 */
#define SENSITIVITY 300.0

/*==================[internal data declaration]==============================*/

bool conversion; /**< booleando si realiza o no la conversión*/
uint16_t valor; /**< Valor convertido*/
uint8_t canal; /**< Canal de la placa*/

/*==================[internal functions declaration]=========================*/

/**@fn float UnitConvert(uint16_t value)
 * @brief  Función que convierte el valor leído en unidades de gravedad
 * @return valor convertido
 */
float UnitConvert(float value);

/*==================[internal data definition]===============================*/


/*==================[external data definition]===============================*/

analog_input_config_t my_ad_x = {CH1, ADC_SINGLE, NULL, NULL, 0}; /**< Typedef configuración puerto analógico/digital canal x*/
analog_input_config_t my_ad_y = {CH2, ADC_SINGLE, NULL, NULL, 0}; /**< Typedef configuración puerto analógico/digital canal y*/
analog_input_config_t my_ad_z = {CH3, ADC_SINGLE, NULL, NULL, 0}; /**< Typedef configuración puerto analógico/digital canal z*/

/*==================[internal functions definition]==========================*/

float UnitConvert(float value){

	float new_value;

	new_value = value - OFFSET;
	new_value = new_value / SENSITIVITY;

	return new_value;
}

/*==================[external functions definition]==========================*/

bool ADXL335Init(){
    AnalogInputInit(&my_ad_x);
    AnalogInputInit(&my_ad_y);
    AnalogInputInit(&my_ad_z);
	return 1;
}

int ReadXValueInt(){
	AnalogInputReadSingle(my_ad_x.input, &valor);
	return valor;
}

float ReadXValue(){
	AnalogInputReadSingle(my_ad_x.input, &valor);
	return UnitConvert((float)valor);
}

int ReadYValueInt(){
    AnalogInputReadSingle(my_ad_y.input, &valor);
	return valor;
}

float ReadYValue(){
	AnalogInputReadSingle(my_ad_y.input, &valor);
	return UnitConvert((float)valor);
}

int ReadZValueInt(){
    AnalogInputReadSingle(my_ad_z.input, &valor);
	return (int)((float)valor); /* Resistor divider with serial resistor in ADXL335 (32K) and resistor to GND for HCSR-04 (10K): multiply by 4.2*/
}

float ReadZValue(){
	AnalogInputReadSingle(my_ad_z.input, &valor);
	return UnitConvert((float)valor); /* Resistor divider with serial resistor in ADXL335 (32K) and resistor to GND for HCSR-04 (10K): multiply by 4.2*/
}

bool ADXL335DeInit(gpio_t gSelect1, gpio_t gSelect2){

	return 1;
}
/** @} doxygen end group definition */
/** @} doxygen end group definition */
/** @} doxygen end group definition */
/*==================[end of file]============================================*/