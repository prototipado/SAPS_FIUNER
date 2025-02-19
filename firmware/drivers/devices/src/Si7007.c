/* Copyright 2021,
 * Maria Casablanca
 * mariacasabla@hotmail.com
 * Facultad de Ingeniería
 * Universidad Nacional de Entre Ríos
 * Argentina
 *
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 *    contributors may be used to endorse or promote products derived from this
 *    software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 *
 */

/** \brief Driver for SI7007 H/T sensor.
 **
 **/

/*
 * Initials     Name
 * ---------------------------
 *	MC         Maria Casablanca
 */

/*
 * modification history (new versions first)
 * -----------------------------------------------------------
 * 20210901 v0.1 initials initial version Maria Casablanca
 * 20242703 v1.1 converted to ESP IDF by JC
 */

/*==================[inclusions]=============================================*/
#include "Si7007.h"

#include "gpio_mcu.h"
#include <stdio.h>
#include <stdint.h>

/*==================[macros and definitions]=================================*/

#define V_REF 3.3                /**< Tensión de referencia*/
#define TOTAL_BITS 1024          /**< Cantidad total de bits*/

/*==================[internal data declaration]==============================*/

analog_input_config_t temp_config;
analog_input_config_t hum_config;

/*==================[internal functions declaration]=========================*/

/*==================[internal data definition]===============================*/

/*==================[external data definition]===============================*/

/*==================[internal functions definition]==========================*/

/*==================[external functions definition]==========================*/

bool Si7007Init(Si7007_config *pins){


	GPIOInit(pins->select, GPIO_OUTPUT);
	GPIOOn(pins->select); //Lo pongo en 1 para que PWM 2 sea temperatura y PWM 1 humedad.

	temp_config.input = pins->PWM_2;;
	temp_config.mode = ADC_SINGLE;

	hum_config.input = pins->PWM_1;
	hum_config.mode = ADC_SINGLE;

	AnalogInputInit(&temp_config);  
	AnalogInputInit(&hum_config);
	return true;
}

float Si7007MeasureTemperature(void){

	uint16_t value;
	float temperature = 0;
	float valor = 0;
	
	AnalogInputReadSingle(temp_config.input, &value);
	valor = (value/1000.0)/V_REF;
	temperature = -46.85 + (valor*175.71);
	return temperature;

}

float Si7007MeasureHumidity(void){

	uint16_t value;
	float humidity = 0;
	float valor = 0;
	
	AnalogInputReadSingle(hum_config.input, &value);
	valor = (value/1000.0)/V_REF;
	humidity = -6 + (valor*125);
	return humidity;
}

bool Si7007Deinit(Si7007_config *pins){
	return true;
}


/*==================[end of file]============================================*/
