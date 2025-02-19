#ifndef SI7007_H
#define SI7007_H
/** \addtogroup Drivers_Programable Drivers Programable
 ** @{ */
/** \addtogroup Drivers_Devices Drivers devices
 ** @{ */
/** \addtogroup Sensor_TyH Sensor Temperatura y Humedad
 ** @{ */

/** @brief Bare Metal header for Si7007
 **
 ** This is a driver for Si7007
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
 * 20211006 v0.1 initials initial version Maria Casablanca
 */

/*==================[inclusions]=============================================*/

#include <stdint.h>
#include "gpio_mcu.h"
#include "analog_io_mcu.h"

/*==================[macros]=================================================*/

/*==================[typedef]================================================*/

typedef struct {				/*!< Si7007 Inputs config struct*/
	    gpio_t select;			/*!< Determines the output on each pin*/
	    uint8_t PWM_1;		    /*!< Channel connected to the PWM1 output pin on the device*/
	    uint8_t PWM_2;	        /*!< Channel connected to the PWM2 output pin on the device*/
} Si7007_config;

/*==================[external data declaration]==============================*/


/*==================[external functions declaration]=========================*/

/** @fn bool Si7007Init(Si7007_config *pins);
 * @brief Initialization function of Si7007.
 * @param[in] *pins
 * @return TRUE if no error.
 */
bool Si7007Init(Si7007_config *pins);

/** @fn uint16_t Si7007MeasureTemperature(void)
 * @brief Measures the current temperature
 * @param[in] No Parameter
 * @return value of temperature in °C
 */
float Si7007MeasureTemperature(void);

/** @fn uint16_t Si7007MeasureHumidity(void)
 * @brief Measures the current relative humidity
 * @param[in] No Parameter
 * @return value of relative humidity in %
 */
float Si7007MeasureHumidity(void);


/** @fn bool Si7007dEInit(Si7007_config *pins);
 * @brief deinitialization function of Si7007.
 * @param[in] *pins
 * @return TRUE if no error.
 */
bool Si7007Deinit(Si7007_config *pins);

/** @} doxygen end group definition */
/** @} doxygen end group definition */
/** @} doxygen end group definition */
/*==================[end of file]============================================*/
#endif /* #ifndef SI7007_H */

