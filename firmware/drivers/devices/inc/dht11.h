/** \addtogroup Drivers_Programable Drivers Programable
 ** @{ */
/** \addtogroup Drivers_Devices Drivers devices
 ** @{ */
/** \addtogroup DHT11 DHT11
 ** @{ 
 */
 
#ifndef _DHT11_H_
#define _DHT11_H_

#include "gpio_mcu.h"

/**
 * @brief 
 * 
 * @param gpio GPIO number where data pin (out) will be connected
 */
void dht11Init(gpio_t gpio);

/**
 * @brief 
 * 
 * @param phum      Humidity variable pointer (in %)
 * @param ptemp     Temperature variable pointer (in ºC)
 * @return int      TRUE if no error
 */
int dht11Read( float *phum, float *ptemp );

#endif /* _DHT11_H_ */

 /** @} doxygen end group definition */
 /** @} doxygen end group definition */
 /** @} doxygen end group definition */