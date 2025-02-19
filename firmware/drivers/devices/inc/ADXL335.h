
#ifndef ADXL335_H
#define ADXL335_H


/** \addtogroup Drivers_Programable Drivers Programable
 ** @{ */
/** \addtogroup Drivers_Devices Drivers devices
 ** @{ */
/** \addtogroup ADXL335
 ** @{ */

/** \brief driver que maneja la lectura de datos obtenida a partir de un acelerómetro.
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

#include "gpio_mcu.h"
#include "analog_io_mcu.h"


/*==================[macros]=================================================*/

/*==================[typedef]================================================*/

/*==================[external data declaration]==============================*/

/*==================[external functions declaration]=========================*/
/** @fn bool ADXL335Init()
 * @brief Función que inicializa los pines de selección del driver
 * @param[in] gSelect1
 * @return 1 (true) if no error
 */
bool ADXL335Init();
/** @fn float ReadXValue()
 * @brief Función que lee el pin x del driver y devuelve el valor convertido de analógico a digital en unidades de gravedad.
 * @param[in] No hay parámetros
 * @return Aceleración en el eje x
 */
float ReadXValue();
/** @fn float ReadYValue()
 * @brief Función que lee el pin y del driver y devuelve el valor convertido de analógico a digital en unidades de gravedad.
 * @param[in] No hay parámetros
 * @return Aceleración en el eje y
 */
float ReadYValue();
/** @fn float ReadZValue()
 * @brief Función que lee el pin z del driver y devuelve el valor convertido de analógico a digital en unidades de gravedad.
 * @param[in] No hay parámetros
 * @return Aceleración en el eje z
 */
float ReadZValue();
/** @fn bool ADXL335Deinit(gpio_t gSelect1, gpio_t gSelect2)
 * @brief Función que de-inicializa los pines de selección del driver
 * @return 1 (true) if no error
 */
bool ADXL335Deinit();

int ReadXValueInt();
int ReadYValueInt();
int ReadZValueInt();

/** @} doxygen end group definition */
/** @} doxygen end group definition */
/** @} doxygen end group definition */
/*==================[end of file]============================================*/
#endif /* #ifndef MMA7260Q_H */