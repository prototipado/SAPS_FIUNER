/***************************************************
 This is a library written for the Maxim MAX30105 Optical Smoke Detector
 It should also work with the MAX30102. However, the MAX30102 does not have a Green LED.
 These sensors use I2C to communicate, as well as a single (optional)
 interrupt line that is not currently supported in this driver.

 Written by Peter Jansen and Nathan Seidle (SparkFun)
 BSD license, all text above must be included in any redistribution.
 *****************************************************/

#ifndef _DRIVERS_DEVICES_INC_MAX3010X_H_
#define _DRIVERS_DEVICES_INC_MAX3010X_H_

#include <stdbool.h>
#include <stdint.h>

#define MAX30105_ADDRESS          0x57 //7-bit I2C Address
//Note that MAX30102 has the same I2C address and Part ID

#define I2C_SPEED_STANDARD        100000
#define I2C_SPEED_FAST            400000

//Define the size of the I2C buffer based on the platform the user has
#if defined(__AVR_ATmega328P__) || defined(__AVR_ATmega168__)

  //I2C_BUFFER_LENGTH is defined in Wire.H
  #define I2C_BUFFER_LENGTH BUFFER_LENGTH

#elif defined(__SAMD21G18A__)

  //SAMD21 uses RingBuffer.h
  #define I2C_BUFFER_LENGTH SERIAL_BUFFER_SIZE

#else

  //The catch-all default is 32
  #define I2C_BUFFER_LENGTH 32

#endif


  bool MAX3010X_begin(void);

  uint32_t MAX3010X_getRed(void); //Returns immediate red value
  uint32_t MAX3010X_getIR(void); //Returns immediate IR value
  uint32_t MAX3010X_getGreen(void); //Returns immediate green value
  bool MAX3010X_safeCheck(uint8_t maxTimeToCheck); //Given a max amount of time, check for new data

  // Configuration
  void MAX3010X_softReset();
  void MAX3010X_shutDown();
  void MAX3010X_wakeUp();

  void MAX3010X_setLEDMode(uint8_t mode);

  void MAX3010X_setADCRange(uint8_t adcRange);
  void MAX3010X_setSampleRate(uint8_t sampleRate);
  void MAX3010X_setPulseWidth(uint8_t pulseWidth);

  void MAX3010X_setPulseAmplitudeRed(uint8_t value);
  void MAX3010X_setPulseAmplitudeIR(uint8_t value);
  void MAX3010X_setPulseAmplitudeGreen(uint8_t value);
  void MAX3010X_setPulseAmplitudeProximity(uint8_t value);

  void MAX3010X_setProximityThreshold(uint8_t threshMSB);

  //Multi-led configuration mode (page 22)
  void MAX3010X_enableSlot(uint8_t slotNumber, uint8_t device); //Given slot number, assign a device to slot
  void MAX3010X_disableSlots(void);

  // Data Collection

  //Interrupts (page 13, 14)
  uint8_t MAX3010X_getINT1(void); //Returns the main interrupt group
  uint8_t MAX3010X_getINT2(void); //Returns the temp ready interrupt
  void MAX3010X_enableAFULL(void); //Enable/disable individual interrupts
  void MAX3010X_disableAFULL(void);
  void MAX3010X_enableDATARDY(void);
  void MAX3010X_disableDATARDY(void);
  void MAX3010X_enableALCOVF(void);
  void MAX3010X_disableALCOVF(void);
  void MAX3010X_enablePROXINT(void);
  void MAX3010X_disablePROXINT(void);
  void MAX3010X_enableDIETEMPRDY(void);
  void MAX3010X_disableDIETEMPRDY(void);

  //FIFO Configuration (page 18)
  void MAX3010X_setFIFOAverage(uint8_t samples);
  void MAX3010X_enableFIFORollover();
  void MAX3010X_disableFIFORollover();
  void MAX3010X_setFIFOAlmostFull(uint8_t samples);

  //FIFO Reading
  uint16_t MAX3010X_check(void); //Checks for new data and fills FIFO
  uint8_t MAX3010X_available(void); //Tells caller how many new samples are available (head - tail)
  void MAX3010X_nextSample(void); //Advances the tail of the sense array
  uint32_t MAX3010X_getFIFORed(void); //Returns the FIFO sample pointed to by tail
  uint32_t MAX3010X_getFIFOIR(void); //Returns the FIFO sample pointed to by tail
  uint32_t MAX3010X_getFIFOGreen(void); //Returns the FIFO sample pointed to by tail

  uint8_t MAX3010X_getWritePointer(void);
  uint8_t MAX3010X_getReadPointer(void);
  void MAX3010X_clearFIFO(void); //Sets the read/write pointers to zero

  //Proximity Mode Interrupt Threshold
  void MAX3010X_setPROXINTTHRESH(uint8_t val);

  // Die Temperature
  float MAX3010X_readTemperature();
  float MAX3010X_readTemperatureF();

  // Detecting ID/Revision
  uint8_t MAX3010X_getRevisionID();
  uint8_t MAX3010X_readPartID();

  // Setup the IC with user selectable settings
  //void MAX3010X_setup(byte powerLevel = 0x1F, byte sampleAverage = 4, byte ledMode = 3, int sampleRate = 400, int pulseWidth = 411, int adcRange = 4096);
  void MAX3010X_setup(uint8_t powerLevel, uint8_t sampleAverage, uint8_t ledMode, int sampleRate , int pulseWidth, int adcRange);

  // Low-level I2C communication
  uint8_t readRegister8( uint8_t reg);
  void writeRegister8(uint8_t reg, uint8_t value);

  //activeLEDs is the number of channels turned on, and can be 1 to 3. 2 is common for Red+IR.
  

  void MAX3010X_readRevisionID();

  void bitMask(uint8_t reg, uint8_t mask, uint8_t thing);

   #define STORAGE_SIZE 4 //Each long is 4 bytes so limit this to fit on your micro
  typedef struct Record
  {
    uint32_t red[STORAGE_SIZE];
    uint32_t IR[STORAGE_SIZE];
    uint32_t green[STORAGE_SIZE];
    uint8_t head;
    uint8_t tail;
  } sense_struct; //This is our circular buffer of readings from the sensor

  

#endif /* _DRIVERS_DEVICES_INC_MAX3010X_H_ */
