#ifndef __CIRCUITSIM__
#define __CIRCUITSIM__
#define NOCPP
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <type_traits>
#include "circuitConfig.h"

inline constexpr uint32_t ROUND_UP_TO_16(uint32_t x){

    return (((x) + 15) & ~15);
}  

bool compare_and_copy_bits(uint32_t& dst, uint32_t src, uint32_t pos, uint32_t num_bits) ;
void setBit(uint32_t &num, uint8_t bitIndex, bool value);
void update_x_cur_and_u_with_new_u(float* x_cur_and_u, float *u_input );

#endif
