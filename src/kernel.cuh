/**
 *
 */

#ifndef _KERNEL_H_
#define _KERNEL_H_

__global__ void vecAdd(const float* A, const float* B, float* C, int N);

#endif // #ifndef _KERNEL_H_