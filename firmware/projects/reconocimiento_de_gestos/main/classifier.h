#ifndef _INC_MODEL_H_
#define _INC_MODEL_H_

#include <stdarg.h>
#include <stdint.h>

int predict(float *x);
const char* idxToLabel(uint8_t classIdx);
const char* predictLabel(float *x);
float compute_kernel(float *x, ...);

#endif /* _INC_MODEL_H_ */